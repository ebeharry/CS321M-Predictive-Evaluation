"""Amortized IRT (2PL, sentence-encoder) submission for the Predictive AI Evaluation Challenge."""

import os

import torch
from sentence_transformers import SentenceTransformer
from torch_measure.models import AmortizedIRT

_dir = os.path.dirname(__file__)

_ckpt = torch.load(
    os.path.join(_dir, "amortized_irt.pt"),
    weights_only=True,
    map_location="cpu",
)

_model = AmortizedIRT(
    n_subjects=_ckpt["n_subjects"],
    n_items=_ckpt["n_items"],
    embedding_dim=_ckpt["embedding_dim"],
    hidden_dim=_ckpt["hidden_dim"],
    n_layers=_ckpt["n_layers"],
    pl=_ckpt["pl"],
)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_ability = _model.ability.detach()
_subj_to_idx = {s: i for i, s in enumerate(_ckpt["subjects_list"])}
_mean_ability = _ckpt["mean_ability"]

_encoder = SentenceTransformer(_ckpt["encoder_name"])

print(f"AmortizedIRT loaded: {_ckpt['n_subjects']} subjects, {_ckpt['n_items']} items, encoder={_ckpt['encoder_name']}")


def predict(input: dict, labeled=None) -> float:
    item_key = f"{input['item_content']}\n[Condition: {input['condition']}]"
    emb = torch.tensor(_encoder.encode([item_key], convert_to_numpy=True).astype("float32"))

    with torch.no_grad():
        params = _model.item_net(emb)
        b = params[0, 0]
        a = torch.exp(params[0, 1])

    s = _subj_to_idx.get(input["subject_content"])
    theta = _ability[s] if s is not None else _mean_ability

    return torch.clamp(torch.sigmoid(a * (theta - b)), 0.001, 0.999).item()
