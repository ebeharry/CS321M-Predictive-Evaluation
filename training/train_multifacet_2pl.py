import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import roc_auc_score
from torch_measure.models import MultiFacet2PL

from OLD.sample_code_submission._data import render_subject_content

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
VAL_FRAC = 0.1
SPLIT_SEED = 42


def condition_to_group(cond: str) -> str:
    """Map a fine-grained condition string to one of ~10 coarse facet groups.

    215 raw conditions → ~10 groups keeps the tau (item × facet) matrix
    at ~1M params instead of ~22M, and ensures each cell has enough observations.
    """
    if cond == "none" or not cond:
        return "none"
    if cond.startswith("aspect="):
        return "aspect"
    if cond.startswith("judge=1;"):
        return "judge_1"
    if cond.startswith("judge=2;"):
        return "judge_2"
    if cond.startswith("metric=security"):
        return "metric_security"
    if cond.startswith("metric=utility"):
        return "metric_utility"
    if cond.startswith("mode="):
        return "mode"
    if cond.startswith("skill="):
        return "skill"
    if cond.startswith("source="):
        return "source"
    if cond.startswith("subset="):
        return "subset"
    return "other"


def load_multifacet_data():
    """Load data with separate subject, base-item, and coarse-condition (facet) indices.

    Unlike _data.py, condition is NOT baked into item_key — it is mapped to a
    coarse group and used as a separate facet dimension for MultiFacet2PL.
    """
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )

    response_features = Features({
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    })

    responses = load_dataset(REPO_ID, data_files=response_files, features=response_features, split="train")
    items_df = pd.read_parquet(hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset"))
    subjects_df = pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset"))

    items_by_id = items_df.set_index("item_id").to_dict("index")
    subjects_by_id = subjects_df.set_index("subject_id").to_dict("index")

    df = responses.to_pandas()
    df["subject_content"] = df["subject_id"].map(
        lambda sid: render_subject_content(subjects_by_id.get(sid, {}), sid)
    )
    df["item_content"] = df["item_id"].map(
        lambda iid: items_by_id.get(iid, {}).get("content")
    )
    df = df.dropna(subset=["item_content"])
    df["test_condition"] = df["test_condition"].fillna("none")
    df = (
        df
        .sort_values("trial")
        .drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
    )

    df["facet_group"] = df["test_condition"].map(condition_to_group)

    subjects_list = df["subject_content"].unique().tolist()
    items_list = df["item_content"].unique().tolist()
    groups_list = sorted(df["facet_group"].unique().tolist())

    subj_map = {s: i for i, s in enumerate(subjects_list)}
    item_map = {it: i for i, it in enumerate(items_list)}
    group_map = {g: i for i, g in enumerate(groups_list)}

    n_subjects = len(subjects_list)
    n_items = len(items_list)
    n_groups = len(groups_list)

    s_arr = df["subject_content"].map(subj_map).values
    i_arr = df["item_content"].map(item_map).values
    g_arr = df["facet_group"].map(group_map).values
    labels_arr = (df["response"].values > 0).astype("float32")

    group_counts = df["facet_group"].value_counts().to_dict()
    print(f"Subjects: {n_subjects}, Items: {n_items}, Facet groups: {n_groups}")
    print(f"Group counts: {group_counts}")
    print(f"Observations: {len(labels_arr)}, positive rate: {labels_arr.mean():.4f}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(labels_arr))
    split = int((1 - VAL_FRAC) * len(perm))
    train_perm, val_perm = perm[:split], perm[split:]

    print(f"Train obs: {len(train_perm)}, Val obs: {len(val_perm)}")

    return (
        s_arr[train_perm], i_arr[train_perm], g_arr[train_perm], labels_arr[train_perm],
        s_arr[val_perm], i_arr[val_perm], g_arr[val_perm], labels_arr[val_perm],
        subjects_list, items_list, groups_list,
        n_subjects, n_items, n_groups,
    )


(
    train_s, train_i, train_g, train_labels,
    val_s, val_i, val_g, val_labels,
    subjects_list, items_list, groups_list,
    n_subjects, n_items, n_groups,
) = load_multifacet_data()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

model = MultiFacet2PL(
    n_subjects=n_subjects,
    n_items=n_items,
    n_facet_levels=n_groups,
)

history = model.fit(
    subject_idx=torch.tensor(train_s, dtype=torch.long),
    item_idx=torch.tensor(train_i, dtype=torch.long),
    facet_idx=torch.tensor(train_g, dtype=torch.long),
    response=torch.tensor(train_labels, dtype=torch.float32),
    max_epochs=4000,
    verbose=True,
)

print(f"Final loss: {history['losses'][-1]:.4f}")

# ---------------------------------------------------------------------------
# Extract posterior means
# ---------------------------------------------------------------------------

post = history["posterior"]
ability = torch.tensor(np.array(post["ability"]), dtype=torch.float32)               # (n_subjects,)
difficulty = torch.tensor(np.array(post["difficulty"]), dtype=torch.float32)         # (n_items,)
discrimination = torch.tensor(np.array(post["discrimination"]), dtype=torch.float32) # (n_items,)
gamma = torch.tensor(np.array(post["gamma"]), dtype=torch.float32)                   # (n_groups,)
tau = torch.tensor(np.array(post["tau"]), dtype=torch.float32)                       # (n_items, n_groups)
delta = torch.tensor(np.array(post["delta"]), dtype=torch.float32)                   # (n_subjects, n_groups)

print(f"Mean ability: {ability.mean():.4f}, mean difficulty: {difficulty.mean():.4f}")
print(f"Discrimination — mean: {discrimination.mean():.4f}, std: {discrimination.std():.4f}")
print(f"Gamma (group shifts): {dict(zip(groups_list, gamma.tolist()))}")

# ---------------------------------------------------------------------------
# Val AUC
# ---------------------------------------------------------------------------

vs = torch.tensor(val_s, dtype=torch.long)
vi = torch.tensor(val_i, dtype=torch.long)
vg = torch.tensor(val_g, dtype=torch.long)

with torch.no_grad():
    logit = discrimination[vi] * (
        (ability[vs] + delta[vs, vg]) - (difficulty[vi] + gamma[vg] + tau[vi, vg])
    )
    val_preds = np.clip(torch.sigmoid(logit).numpy(), 0.001, 0.999)

val_auc = roc_auc_score(val_labels, val_preds)
print(f"Val AUC: {val_auc:.4f}")

# ---------------------------------------------------------------------------
# Save checkpoint and val split
# ---------------------------------------------------------------------------

_dir = os.path.dirname(os.path.abspath(__file__))

torch.save(
    {
        "ability": ability,
        "difficulty": difficulty,
        "discrimination": discrimination,
        "gamma": gamma,
        "tau": tau,
        "delta": delta,
        "subjects_list": subjects_list,
        "items_list": items_list,
        "groups_list": groups_list,
        "mean_ability": ability.mean().item(),
        "mean_difficulty": difficulty.mean().item(),
        "mean_discrimination": discrimination.mean().item(),
        "mean_delta": delta.mean(dim=0),  # (n_groups,) fallback for unseen subjects
        "mean_tau": tau.mean(dim=0),      # (n_groups,) fallback for unseen items
    },
    os.path.join(_dir, "multifacet_2pl.pt"),
)
print("Saved multifacet_2pl.pt")

with open(os.path.join(_dir, "multifacet_val_split.pkl"), "wb") as f:
    pickle.dump({
        "val_s_idx": val_s,
        "val_i_idx": val_i,
        "val_g_idx": val_g,
        "val_labels": val_labels,
        "subjects_list": subjects_list,
        "items_list": items_list,
        "groups_list": groups_list,
    }, f)
print("Saved multifacet_val_split.pkl")
