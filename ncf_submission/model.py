"""Neural Collaborative Filter (NCF) submission for the Predictive AI Evaluation Challenge."""

import os
import pickle
import re

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


def _naturalize_condition(cond: str) -> str:
    """Convert structured condition key=value string to readable English for the sentence encoder."""
    if not cond or cond == "none":
        return ""
    if cond.startswith("aspect="):
        return f"evaluated for {cond[len('aspect='):].replace('_', ' ')}"
    m = re.match(r"judge=(\d+);(.*)", cond)
    if m:
        detail = m.group(2).replace("_", " ").replace("=", " ")
        return f"scored by judge {m.group(1)} on {detail}"
    if cond.startswith("metric="):
        return f"{cond[len('metric='):].replace('_', ' ')} metric"
    if cond.startswith("mode="):
        return f"{cond[len('mode='):].replace('_', ' ')} mode"
    if cond.startswith("skill="):
        return f"testing {cond[len('skill='):].replace('_', ' ')} skill"
    if cond.startswith("source="):
        return f"sourced from {cond[len('source='):].replace('_', ' ')}"
    if cond.startswith("subset="):
        return f"{cond[len('subset='):].replace('_', ' ')} subset"
    return cond.replace("=", " ").replace(";", ", ").replace("_", " ")

_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "ncf_meta_slim.pkl"), "rb") as f:
    _meta = pickle.load(f)


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        if n_layers == 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ELU()])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _NCFHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.net = _MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


_head = _NCFHead(
    input_dim=_meta["embedding_dim"] * 2,
    hidden_dim=_meta["hidden_dim"],
    n_layers=_meta["n_layers"],
    dropout=_meta.get("dropout", 0.1),
)
_head.load_state_dict(
    torch.load(os.path.join(_dir, "ncf_head.pt"), weights_only=True, map_location="cpu")
)
_head.eval()

_encoder = SentenceTransformer(_meta["encoder_name"])

_subj_to_idx = {s: i for i, s in enumerate(_meta["subjects_list"])}
print(f"Pre-encoding {len(_meta['subjects_list'])} subjects...")
_subj_emb = torch.tensor(
    _encoder.encode(
        _meta["subjects_list"],
        batch_size=256,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
)
_mean_subj_emb = _subj_emb.mean(0)

print(f"NCF loaded: {len(_meta['subjects_list'])} subjects, encoder={_meta['encoder_name']}")


def predict(input: dict, labeled=None) -> float:
    cond_text = _naturalize_condition(input["condition"])
    item_key = f"{input['item_content']}\n[Condition: {cond_text}]" if cond_text else input["item_content"]

    s = _subj_to_idx.get(input["subject_content"])
    u = _subj_emb[s] if s is not None else _mean_subj_emb

    v = torch.tensor(_encoder.encode(item_key, convert_to_numpy=True).astype("float32"))

    with torch.no_grad():
        x = torch.cat([u, v], dim=-1).unsqueeze(0)
        logit = _head(x)

    return torch.clamp(torch.sigmoid(logit), 0.001, 0.999).item()
