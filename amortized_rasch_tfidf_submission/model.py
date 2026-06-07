"""Amortized IRT (1PL/Rasch, TF-IDF) submission for the Predictive AI Evaluation Challenge."""

import math
import os
import re

import numpy as np
import torch
from torch_measure.models import AmortizedIRT

_dir = os.path.dirname(__file__)

_ckpt = torch.load(
    os.path.join(_dir, "amortized_rasch_tfidf.pt"),
    weights_only=True,
    map_location="cpu",
)

_tfidf_data = np.load(os.path.join(_dir, "tfidf_arrays.npz"), allow_pickle=True)
_vocab: dict[str, int] = {w: i for i, w in enumerate(_tfidf_data["vocab"])}
_idf: np.ndarray = _tfidf_data["idf"].astype("float32")

_model = AmortizedIRT(
    n_subjects=_ckpt["n_subjects"],
    n_items=_ckpt["n_items"],
    embedding_dim=_ckpt["embedding_dim"],
    hidden_dim=_ckpt["hidden_dim"],
    n_layers=_ckpt["n_layers"],
    pl=_ckpt["pl"],  # 1
)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_ability = _model.ability.detach()
_subj_to_idx = {s: i for i, s in enumerate(_ckpt["subjects_list"])}
_mean_ability = _ckpt["mean_ability"]

_TOKENIZER = re.compile(r"(?u)\b\w\w+\b")

print(f"AmortizedIRT (Rasch, TF-IDF) loaded: {_ckpt['n_subjects']} subjects, {_ckpt['n_items']} items")


def _tfidf_transform(text: str) -> torch.Tensor:
    tokens = _TOKENIZER.findall(text.lower())
    counts: dict[int, int] = {}
    for tok in tokens:
        idx = _vocab.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    vec = np.zeros(len(_idf), dtype="float32")
    for idx, tf in counts.items():
        vec[idx] = (1.0 + math.log(tf)) * _idf[idx]
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return torch.tensor(vec).unsqueeze(0)


def predict(input: dict, labeled=None) -> float:
    item_key = f"{input['item_content']}\n[Condition: {input['condition']}]"
    emb = _tfidf_transform(item_key)

    with torch.no_grad():
        params = _model.item_net(emb)
        b = params[0, 0]  # difficulty only; Rasch fixes discrimination at 1

    s = _subj_to_idx.get(input["subject_content"])
    theta = _ability[s] if s is not None else _mean_ability

    return torch.clamp(torch.sigmoid(theta - b), 0.001, 0.999).item()
