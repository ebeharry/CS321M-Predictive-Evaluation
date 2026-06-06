"""Tests for NCF and NCFHead."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from ncf import NCF, NCFHead


# ---------------------------------------------------------------------------
# Stub encoder so tests don't download real model weights
# ---------------------------------------------------------------------------

class _StubEncoder:
    """Deterministic fake SentenceTransformer returning fixed-size embeddings."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, sentences, convert_to_tensor=False, batch_size=256,
               show_progress_bar=False, device="cpu"):
        if isinstance(sentences, str):
            n = 1
        else:
            n = len(sentences)
        arr = np.ones((n, self.dim), dtype=np.float32) * 0.1
        if convert_to_tensor:
            t = torch.tensor(arr)
            return t.squeeze(0) if n == 1 else t
        return arr


DIM = 16


@pytest.fixture
def stub_ncf():
    enc = _StubEncoder(DIM)
    model = NCF(encoder=enc, embedding_dim=DIM, hidden_dim=32, n_layers=2, dropout=0.0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# NCFHead tests
# ---------------------------------------------------------------------------

class TestNCFHead:
    def test_output_shape_batch(self):
        head = NCFHead(input_dim=32, hidden_dim=64, n_layers=2, dropout=0.0)
        x = torch.randn(8, 32)
        out = head(x)
        assert out.shape == (8,), f"expected (8,), got {out.shape}"

    def test_output_shape_single(self):
        head = NCFHead(input_dim=32, hidden_dim=64, n_layers=1, dropout=0.0)
        x = torch.randn(1, 32)
        out = head(x)
        assert out.shape == (1,), f"expected (1,), got {out.shape}"

    def test_output_is_scalar_logit(self):
        # output should be unbounded (raw logit, no sigmoid)
        head = NCFHead(input_dim=4, hidden_dim=8, n_layers=2, dropout=0.0)
        x = torch.randn(100, 4)
        out = head(x)
        assert out.min().item() < 0 or out.max().item() > 1, (
            "expected unbounded logits"
        )

    def test_n_layers_one(self):
        head = NCFHead(input_dim=8, hidden_dim=16, n_layers=1, dropout=0.0)
        x = torch.randn(4, 8)
        out = head(x)
        assert out.shape == (4,)

    def test_deterministic_eval(self):
        head = NCFHead(input_dim=8, hidden_dim=16, n_layers=2, dropout=0.5)
        head.eval()
        x = torch.randn(4, 8)
        with torch.no_grad():
            a = head(x)
            b = head(x)
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# NCF._raw_prob tests
# ---------------------------------------------------------------------------

class TestNCFRawProb:
    def test_returns_float_in_unit_interval(self, stub_ncf):
        p = stub_ncf._raw_prob("subject text", "item text")
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_deterministic(self, stub_ncf):
        p1 = stub_ncf._raw_prob("subject", "item")
        p2 = stub_ncf._raw_prob("subject", "item")
        assert p1 == p2


# ---------------------------------------------------------------------------
# NCF.predict tests
# ---------------------------------------------------------------------------

class TestNCFPredict:
    def test_no_labels_returns_raw_prob(self, stub_ncf):
        data = {"subject_content": "GPT-4", "item_content": "What is 2+2?"}
        p = stub_ncf.predict(data, labeled=[])
        assert 0.0 <= p <= 1.0
        assert not stub_ncf._round_calibrated

    def test_with_labels_triggers_calibration(self, stub_ncf):
        labeled = [
            {"subject_content": "GPT-4", "item_content": "What is 2+2?", "label": 1.0},
            {"subject_content": "GPT-3", "item_content": "What is 3+3?", "label": 0.0},
        ]
        data = {"subject_content": "GPT-4", "item_content": "What is 2+2?"}
        p = stub_ncf.predict(data, labeled=labeled)
        assert stub_ncf._round_calibrated
        assert 0.0 <= p <= 1.0

    def test_calibration_only_fits_once_per_round(self, stub_ncf):
        labeled = [
            {"subject_content": "s", "item_content": "i", "label": 1.0},
        ]
        data = {"subject_content": "s", "item_content": "i"}
        stub_ncf.predict(data, labeled=labeled)
        a1, b1 = stub_ncf._platt_a, stub_ncf._platt_b

        # second call — should NOT refit
        stub_ncf.predict(data, labeled=labeled * 5)
        assert stub_ncf._platt_a == a1
        assert stub_ncf._platt_b == b1

    def test_output_clipped(self, stub_ncf):
        # Even with extreme Platt params, output stays in (0, 1)
        stub_ncf._round_calibrated = True
        stub_ncf._platt_a = 1000.0
        stub_ncf._platt_b = 1000.0
        data = {"subject_content": "s", "item_content": "i"}
        p = stub_ncf.predict(data, labeled=[])
        assert p < 1.0
        assert p > 0.0

    def test_none_labeled_treated_as_empty(self, stub_ncf):
        # labeled=None should not crash; calibration stays False
        data = {"subject_content": "s", "item_content": "i"}
        p = stub_ncf.predict(data, labeled=None)
        assert isinstance(p, float)
        assert not stub_ncf._round_calibrated


# ---------------------------------------------------------------------------
# NCF._fit_platt tests
# ---------------------------------------------------------------------------

class TestFitPlatt:
    def test_empty_labeled_is_noop(self, stub_ncf):
        stub_ncf._fit_platt([])
        assert stub_ncf._platt_a == 1.0
        assert stub_ncf._platt_b == 0.0

    def test_sets_round_calibrated(self, stub_ncf):
        labeled = [{"subject_content": "s", "item_content": "i", "label": 1.0}]
        stub_ncf._fit_platt(labeled)
        assert stub_ncf._round_calibrated

    def test_all_positive_labels_shifts_b_positive(self, stub_ncf):
        # With all positive labels the calibration should bias toward p=1
        labeled = [
            {"subject_content": "s", "item_content": "i", "label": 1.0}
            for _ in range(10)
        ]
        stub_ncf._fit_platt(labeled)
        cal_logit = stub_ncf._platt_a * 0.0 + stub_ncf._platt_b
        p = 1.0 / (1.0 + math.exp(-cal_logit))
        assert p > 0.5

    def test_platt_params_are_floats(self, stub_ncf):
        labeled = [{"subject_content": "s", "item_content": "i", "label": 0.0}]
        stub_ncf._fit_platt(labeled)
        assert isinstance(stub_ncf._platt_a, float)
        assert isinstance(stub_ncf._platt_b, float)


# ---------------------------------------------------------------------------
# NCF.encode_batch tests
# ---------------------------------------------------------------------------

class TestEncodeBatch:
    def test_returns_tensors_of_correct_shape(self, stub_ncf):
        subjects = ["s1", "s2", "s3"]
        items = ["i1", "i2", "i3"]
        u, v = stub_ncf.encode_batch(subjects, items)
        assert u.shape == (3, DIM)
        assert v.shape == (3, DIM)
