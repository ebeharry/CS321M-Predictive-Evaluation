"""Train an Amortized IRT (3PL, TF-IDF) model.

Run from starting_kit/:  python amortized_3pl_tfidf_submission/train.py
"""

import os

import numpy as np
import pandas as pd
import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_measure.models import AmortizedIRT

_dir = os.path.dirname(os.path.abspath(__file__))

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
HIDDEN_DIM = 256
N_LAYERS = 3
PL = 3
TFIDF_MAX_FEATURES = 4096

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    f for f in repo_files
    if f.endswith(".parquet") and f not in REGISTRY_FILES and not f.endswith("_traces.parquet")
)

responses = load_dataset(
    REPO_ID,
    data_files=response_files,
    features=Features({
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    }),
    split="train",
)

items_by_id = {
    row["item_id"]: row
    for row in pd.read_parquet(hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset")).to_dict("records")
}
subjects_by_id = {
    row["subject_id"]: row
    for row in pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset")).to_dict("records")
}


def render_subject_content(subject, fallback_id):
    lines = [f"Name: {subject.get('display_name') or fallback_id}"]
    for key, label in (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ):
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


df = responses.to_pandas()
df["subject_content"] = df["subject_id"].map(
    {sid: render_subject_content(subjects_by_id.get(sid, {}), sid) for sid in df["subject_id"].unique()}
)
df["item_content"] = df["item_id"].map({iid: items_by_id.get(iid, {}).get("content") for iid in df["item_id"].unique()})
df = df.dropna(subset=["item_content"])
df = df.sort_values("trial").drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
df["item_key"] = df["item_content"] + "\n[Condition: " + df["test_condition"].fillna("none") + "]"

subjects_list = df["subject_content"].unique().tolist()
items_list = df["item_key"].unique().tolist()
subj_idx = {s: i for i, s in enumerate(subjects_list)}
item_idx = {it: i for i, it in enumerate(items_list)}
n_subjects, n_items = len(subjects_list), len(items_list)
print(f"Building matrix: {n_subjects} subjects x {n_items} items")

s_idx_arr = df["subject_content"].map(subj_idx).values
i_idx_arr = df["item_key"].map(item_idx).values
labels_arr = (df["response"].values > 0).astype("float32")

matrix = torch.full((n_subjects, n_items), float("nan"))
matrix[s_idx_arr, i_idx_arr] = torch.tensor(labels_arr, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Encode items with TF-IDF and train
# ---------------------------------------------------------------------------

print(f"Fitting TF-IDF on {n_items} items...")
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
embeddings_np = tfidf.fit_transform(items_list).toarray().astype("float32")
embeddings = torch.tensor(embeddings_np)
print(f"Embeddings shape: {embeddings.shape}")

model = AmortizedIRT(
    n_subjects=n_subjects,
    n_items=n_items,
    embedding_dim=embeddings.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    pl=PL,
    dropout=0.1,
)
history = model.fit(matrix, embeddings, max_epochs=300, lr=1e-3, verbose=True)
print(f"Final loss — amortized IRT (3PL, TF-IDF): {history['losses'][-1]:.4f}")

est_ability = model.ability.detach()
print(f"Mean ability: {est_ability.mean():.4f}, std: {est_ability.std():.4f}")

save_path = os.path.join(_dir, "amortized_3pl_tfidf.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "subjects_list": subjects_list,
        "mean_ability": est_ability.mean().item(),
        "n_subjects": n_subjects,
        "n_items": n_items,
        "embedding_dim": int(embeddings.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "pl": PL,
        "encoder_type": "tfidf",
    },
    save_path,
)

# Save vocab and IDF arrays for inference
vocab = np.array(tfidf.get_feature_names_out())
idf = tfidf.idf_.astype("float32")
np.savez(os.path.join(_dir, "tfidf_arrays.npz"), vocab=vocab, idf=idf)

print(f"Saved to {save_path}")
