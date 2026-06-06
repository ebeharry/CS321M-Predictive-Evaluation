"""Sample code submission for the Predictive AI Evaluation Challenge.

Your submission must define a single function:

    predict(input: dict, labeled: list[dict] | None = None) -> float

The ingestion program calls predict() once per hidden (model_id, item_id) pair. Module-level code
runs once when the container starts. Load weights, tokenizers, prompt
templates here. Heavy training must be done OFFLINE (e.g. publish a model
to HuggingFace and load it at module init).

`input` keys
------------
    benchmark        Benchmark identifier (e.g. "mmlupro", "ai2d_test").
    condition        Test condition (e.g. "zero-shot"). Literal "none" when
                     no condition applies.
    subject_content  Text description of the AI subject being evaluated,
                     beginning with a "Name:" line.
    item_content     The question / prompt / task text the subject is asked.

`labeled` (optional)
--------------------
    A list of dicts shaped like `input` plus a `label` field (0 or 1).
    These are revealed via adaptive labeling (see labeling.py). May be None
    or empty.

Return value
------------
    A single float in [0, 1], the predicted probability that the subject
    answers the item correctly.
"""

import os

import torch
from sentence_transformers import SentenceTransformer
from torch_measure.models import AmortizedIRT


# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# ---------------------------------------------------------------------------

_dir = os.path.dirname(__file__)

_ckpt = torch.load(
    os.path.join(_dir, "amortized_irt.pt"),
    weights_only=True,
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

_ability = _model.ability.detach()  # (n_subjects,)
_subj_to_idx = {s: i for i, s in enumerate(_ckpt["subjects_list"])}
_mean_ability = _ckpt["mean_ability"]

_encoder = SentenceTransformer(_ckpt["encoder_name"])

print(
    f"AmortizedIRT loaded: {_ckpt['n_subjects']} subjects, "
    f"{_ckpt['n_items']} items, mean_ability={_mean_ability:.4f}"
)


# ---------------------------------------------------------------------------
# Prediction Model
# ---------------------------------------------------------------------------

def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return the predicted probability that the subject answers correctly."""
    item_key = f"{input['item_content']}\n[Condition: {input['condition']}]"
    emb_np = _encoder.encode([item_key], convert_to_numpy=True).astype("float32")  # (1, dim)
    emb = torch.tensor(emb_np)

    with torch.no_grad():
        params = _model.item_net(emb)  # (1, 2) for 2PL
        b = params[0, 0]              # difficulty
        a = torch.exp(params[0, 1])  # discrimination

    idx = _subj_to_idx.get(input["subject_content"])
    theta = _ability[idx] if idx is not None else _mean_ability

    return torch.sigmoid(a * (theta - b)).item()
