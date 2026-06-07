"""Evaluate all trained submission models against their respective val splits.

Run from starting_kit/:
    python evaluate_models.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
_val_pkl = ROOT / "amortized_irt_submission" / "val_split.pkl"

results = []


def report(name, preds, val_labels):
    preds = np.clip(preds, 0.001, 0.999)
    auc = roc_auc_score(val_labels, preds)
    nll = -log_loss(val_labels, preds)
    print(f"{name:<30s}  AUC={auc:.4f}  NegLogLoss={nll:.4f}")
    return {"model": name, "auc": auc, "neg_log_loss": nll}


def load_val(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Rasch
# ---------------------------------------------------------------------------

_d = ROOT / "rasch_submission"
if (_d / "rasch.pt").exists() and (_d / "val_split.pkl").exists():
    split = load_val(_d / "val_split.pkl")
    ckpt = torch.load(_d / "rasch.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    preds = torch.sigmoid(ckpt["ability"][vs] - ckpt["difficulty"][vi]).numpy()
    results.append(report("Rasch", preds, lbl))
else:
    print("Rasch: checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# 2PL
# ---------------------------------------------------------------------------

_d = ROOT / "twopl_submission"
if (_d / "twopl.pt").exists() and (_d / "val_split.pkl").exists():
    split = load_val(_d / "val_split.pkl")
    ckpt = torch.load(_d / "twopl.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    preds = torch.sigmoid(ckpt["discrimination"][vi] * (ckpt["ability"][vs] - ckpt["difficulty"][vi])).numpy()
    results.append(report("2PL", preds, lbl))
else:
    print("2PL: checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# 3PL
# ---------------------------------------------------------------------------

_d = ROOT / "threepl_submission"
if (_d / "threepl.pt").exists() and (_d / "val_split.pkl").exists():
    split = load_val(_d / "val_split.pkl")
    ckpt = torch.load(_d / "threepl.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    a, b, c = ckpt["discrimination"][vi], ckpt["difficulty"][vi], ckpt["guessing"][vi]
    preds = (c + (1 - c) * torch.sigmoid(a * (ckpt["ability"][vs] - b))).numpy()
    results.append(report("3PL", preds, lbl))
else:
    print("3PL: checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# AmortizedIRT (Sentence Encoder)
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_irt_submission"
if (_d / "amortized_irt.pt").exists() and (_d / "val_split.pkl").exists():
    from sentence_transformers import SentenceTransformer
    from torch_measure.models import AmortizedIRT

    split = load_val(_d / "val_split.pkl")
    ckpt = torch.load(_d / "amortized_irt.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encoder = SentenceTransformer(ckpt["encoder_name"])
    print(f"Encoding {len(items_list)} items for AmortizedIRT eval...")
    all_emb = torch.tensor(
        encoder.encode(items_list, batch_size=256, convert_to_numpy=True).astype("float32")
    )
    with torch.no_grad():
        params = model.item_net(all_emb)
        b_all, a_all = params[:, 0], torch.exp(params[:, 1])
        preds = torch.sigmoid(a_all[vi] * (model.ability[vs] - b_all[vi])).numpy()
    results.append(report("AmortizedIRT (Sentence)", preds, lbl))
else:
    print("AmortizedIRT (Sentence): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# AmortizedIRT TF-IDF
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_tfidf_submission"
if (_d / "amortized_irt_tfidf.pt").exists() and (_d / "val_split.pkl").exists():
    import joblib
    from torch_measure.models import AmortizedIRT

    split = load_val(_d / "val_split.pkl")
    ckpt = torch.load(_d / "amortized_irt_tfidf.pt", weights_only=True, map_location="cpu")
    tfidf = joblib.load(_d / "tfidf_vectorizer.pkl")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_emb = torch.tensor(tfidf.transform(items_list).toarray().astype("float32"))
    with torch.no_grad():
        params = model.item_net(all_emb)
        b_all, a_all = params[:, 0], torch.exp(params[:, 1])
        preds = torch.sigmoid(a_all[vi] * (model.ability[vs] - b_all[vi])).numpy()
    results.append(report("AmortizedIRT (TF-IDF)", preds, lbl))
else:
    print("AmortizedIRT (TF-IDF): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# MultiFacet2PL
# ---------------------------------------------------------------------------

_d = ROOT / "multifacet_submission"
if (_d / "multifacet_2pl.pt").exists() and (_d / "multifacet_val_split.pkl").exists():
    split = load_val(_d / "multifacet_val_split.pkl")
    ckpt = torch.load(_d / "multifacet_2pl.pt", weights_only=True, map_location="cpu")
    vs = torch.tensor(split["val_s_idx"], dtype=torch.long)
    vi = torch.tensor(split["val_i_idx"], dtype=torch.long)
    vg = torch.tensor(split["val_g_idx"], dtype=torch.long)
    lbl = split["val_labels"]
    with torch.no_grad():
        logit = ckpt["discrimination"][vi] * (
            (ckpt["ability"][vs] + ckpt["delta"][vs, vg])
            - (ckpt["difficulty"][vi] + ckpt["gamma"][vg] + ckpt["tau"][vi, vg])
        )
        preds = np.clip(torch.sigmoid(logit).numpy(), 0.001, 0.999)
    results.append(report("MultiFacet2PL", preds, lbl))
else:
    print("MultiFacet2PL: checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# NCF
# ---------------------------------------------------------------------------

_d = ROOT / "ncf_submission"
if all((_d / f).exists() for f in ["ncf_head.pt", "ncf_embeddings.pt", "ncf_meta.pkl", "ncf_val_split.pkl"]):
    from torch_measure.models._network import MLP

    class _NCFHead(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, dropout):
            super().__init__()
            self.net = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1,
                           n_layers=n_layers, dropout=dropout)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    split = load_val(_d / "ncf_val_split.pkl")
    with open(_d / "ncf_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    emb = torch.load(_d / "ncf_embeddings.pt", weights_only=True, map_location="cpu")
    head = _NCFHead(
        input_dim=meta["embedding_dim"] * 2,
        hidden_dim=meta["hidden_dim"],
        n_layers=meta["n_layers"],
        dropout=meta.get("dropout", 0.1),
    )
    head.load_state_dict(torch.load(_d / "ncf_head.pt", weights_only=True, map_location="cpu"))
    head.eval()

    vs = torch.tensor(split["val_s_idx"], dtype=torch.long)
    vi = torch.tensor(split["val_i_idx"], dtype=torch.long)
    lbl = split["val_labels"]
    with torch.no_grad():
        x = torch.cat([emb["subject_embeddings"][vs], emb["item_embeddings"][vi]], dim=-1)
        preds = np.clip(torch.sigmoid(head(x)).numpy(), 0.001, 0.999)
    results.append(report("NCF", preds, lbl))
else:
    print("NCF: checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# AmortizedRasch TF-IDF
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_rasch_tfidf_submission"
_val_pkl = ROOT / "amortized_irt_submission" / "val_split.pkl"
if (_d / "amortized_rasch_tfidf.pt").exists() and _val_pkl.exists():
    import math, re
    from torch_measure.models import AmortizedIRT

    split = load_val(_val_pkl)
    ckpt = torch.load(_d / "amortized_rasch_tfidf.pt", weights_only=True, map_location="cpu")
    tfidf_data = np.load(_d / "tfidf_arrays.npz", allow_pickle=True)
    vocab = {w: i for i, w in enumerate(tfidf_data["vocab"])}
    idf = tfidf_data["idf"].astype("float32")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _tok = re.compile(r"(?u)\b\w\w+\b")
    def _npz_tfidf(texts):
        vecs = np.zeros((len(texts), len(idf)), dtype="float32")
        for r, text in enumerate(texts):
            counts = {}
            for t in _tok.findall(text.lower()):
                idx = vocab.get(t)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, tf in counts.items():
                vecs[r, idx] = (1.0 + math.log(tf)) * idf[idx]
            n = np.linalg.norm(vecs[r])
            if n > 0:
                vecs[r] /= n
        return torch.tensor(vecs)

    print(f"Encoding {len(items_list)} items for AmortizedRasch (TF-IDF)...")
    all_emb = _npz_tfidf(items_list)
    with torch.no_grad():
        b_all = model.item_net(all_emb)[:, 0]
        preds = torch.sigmoid(model.ability[vs] - b_all[vi]).numpy()
    results.append(report("AmortizedRasch (TF-IDF)", preds, lbl))
else:
    print("AmortizedRasch (TF-IDF): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# Amortized3PL TF-IDF
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_3pl_tfidf_submission"
if (_d / "amortized_3pl_tfidf.pt").exists() and _val_pkl.exists():
    from torch_measure.models import AmortizedIRT

    split = load_val(_val_pkl)
    ckpt = torch.load(_d / "amortized_3pl_tfidf.pt", weights_only=True, map_location="cpu")
    tfidf_data = np.load(_d / "tfidf_arrays.npz", allow_pickle=True)
    vocab = {w: i for i, w in enumerate(tfidf_data["vocab"])}
    idf = tfidf_data["idf"].astype("float32")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Encoding {len(items_list)} items for Amortized3PL (TF-IDF)...")
    all_emb = _npz_tfidf(items_list)
    with torch.no_grad():
        params = model.item_net(all_emb)
        b_all, a_all, c_all = params[:, 0], torch.exp(params[:, 1]), torch.sigmoid(params[:, 2])
        preds = (c_all[vi] + (1 - c_all[vi]) * torch.sigmoid(a_all[vi] * (model.ability[vs] - b_all[vi]))).numpy()
    results.append(report("Amortized3PL (TF-IDF)", preds, lbl))
else:
    print("Amortized3PL (TF-IDF): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# Amortized3PL Sentence
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_3pl_sentence_submission"
if (_d / "amortized_3pl_sentence.pt").exists() and _val_pkl.exists():
    from sentence_transformers import SentenceTransformer
    from torch_measure.models import AmortizedIRT

    split = load_val(_val_pkl)
    ckpt = torch.load(_d / "amortized_3pl_sentence.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encoder = SentenceTransformer(ckpt["encoder_name"])
    print(f"Encoding {len(items_list)} items for Amortized3PL (Sentence)...")
    all_emb = torch.tensor(
        encoder.encode(items_list, batch_size=256, convert_to_numpy=True).astype("float32")
    )
    with torch.no_grad():
        params = model.item_net(all_emb)
        b_all, a_all, c_all = params[:, 0], torch.exp(params[:, 1]), torch.sigmoid(params[:, 2])
        preds = (c_all[vi] + (1 - c_all[vi]) * torch.sigmoid(a_all[vi] * (model.ability[vs] - b_all[vi]))).numpy()
    results.append(report("Amortized3PL (Sentence)", preds, lbl))
else:
    print("Amortized3PL (Sentence): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# AmortizedRasch Sentence
# ---------------------------------------------------------------------------

_d = ROOT / "amortized_irt_rasch_submission"
if (_d / "amortized_irt_rasch.pt").exists() and _val_pkl.exists():
    from sentence_transformers import SentenceTransformer
    from torch_measure.models import AmortizedIRT

    split = load_val(_val_pkl)
    ckpt = torch.load(_d / "amortized_irt_rasch.pt", weights_only=True, map_location="cpu")
    vs, vi, lbl = split["val_s_idx"], split["val_i_idx"], split["val_labels"]
    items_list = split["items_list"]

    model = AmortizedIRT(
        n_subjects=ckpt["n_subjects"], n_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        n_layers=ckpt["n_layers"], pl=ckpt["pl"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encoder = SentenceTransformer(ckpt["encoder_name"])
    print(f"Encoding {len(items_list)} items for AmortizedRasch (Sentence)...")
    all_emb = torch.tensor(
        encoder.encode(items_list, batch_size=256, convert_to_numpy=True).astype("float32")
    )
    with torch.no_grad():
        b_all = model.item_net(all_emb)[:, 0]
        preds = torch.sigmoid(model.ability[vs] - b_all[vi]).numpy()
    results.append(report("AmortizedRasch (Sentence)", preds, lbl))
else:
    print("AmortizedRasch (Sentence): checkpoint not found, skipping")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

df = pd.DataFrame(results).sort_values("neg_log_loss", ascending=False)
print("\n--- Summary (sorted by NegLogLoss) ---")
print(df.to_string(index=False))

out_path = ROOT / "eval_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
