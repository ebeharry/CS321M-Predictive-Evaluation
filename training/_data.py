import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
VAL_FRAC = 0.1
SPLIT_SEED = 42


def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    for key, label in (("provider", "Organization"), ("params", "Parameters"), ("release_date", "Released"), ("family", "Family")):
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _load_raw():
    """Download and process responses into index arrays. Shared by both public functions."""
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
    items = pd.read_parquet(hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset")).to_dict("records")
    subjects = pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset")).to_dict("records")

    items_by_id = {row["item_id"]: row for row in items}
    subjects_by_id = {row["subject_id"]: row for row in subjects}

    responses_df = responses.to_pandas()

    subject_content_map = {
        sid: render_subject_content(subjects_by_id.get(sid, {}), sid)
        for sid in responses_df["subject_id"].unique()
    }
    item_content_map = {
        iid: items_by_id.get(iid, {}).get("content")
        for iid in responses_df["item_id"].unique()
    }

    responses_df["subject_content"] = responses_df["subject_id"].map(subject_content_map)
    responses_df["item_content"] = responses_df["item_id"].map(item_content_map)
    responses_df = responses_df.dropna(subset=["item_content"])
    responses_df = (
        responses_df
        .sort_values("trial")
        .drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
    )
    responses_df["item_key"] = (
        responses_df["item_content"] + "\n[Condition: " + responses_df["test_condition"].fillna("none") + "]"
    )

    subjects_list = responses_df["subject_content"].unique().tolist()
    items_list = responses_df["item_key"].unique().tolist()
    subj_idx = {s: i for i, s in enumerate(subjects_list)}
    item_idx = {it: i for i, it in enumerate(items_list)}

    n_subjects = len(subjects_list)
    n_items = len(items_list)

    s_idx_arr = responses_df["subject_content"].map(subj_idx).values
    i_idx_arr = responses_df["item_key"].map(item_idx).values
    labels_arr = (responses_df["response"].values > 0).astype("float32")

    return s_idx_arr, i_idx_arr, labels_arr, subjects_list, items_list, n_subjects, n_items


def load_training_data():
    """Return full matrix (no split). Kept for backwards compatibility."""
    s_idx_arr, i_idx_arr, labels_arr, subjects_list, items_list, n_subjects, n_items = _load_raw()
    print(f"Building matrix: {n_subjects} subjects x {n_items} items")
    matrix = torch.full((n_subjects, n_items), float("nan"))
    matrix[s_idx_arr, i_idx_arr] = torch.tensor(labels_arr.copy(), dtype=torch.float32)
    print(f"Matrix shape: {matrix.shape}")
    return matrix, subjects_list, items_list, n_subjects, n_items


def load_train_val_split():
    """Return 90/10 train/val split (seed=42). Saves val_split.pkl alongside this file."""
    s_idx_arr, i_idx_arr, labels_arr, subjects_list, items_list, n_subjects, n_items = _load_raw()

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(labels_arr))
    split = int((1 - VAL_FRAC) * len(perm))
    train_perm, val_perm = perm[:split], perm[split:]

    train_s, val_s = s_idx_arr[train_perm], s_idx_arr[val_perm]
    train_i, val_i = i_idx_arr[train_perm], i_idx_arr[val_perm]
    train_labels, val_labels = labels_arr[train_perm], labels_arr[val_perm]
    print(f"Building train matrix: {n_subjects} subjects x {n_items} items")
    print(f"Train obs: {len(train_perm)}, Val obs: {len(val_perm)}")

    train_matrix = torch.full((n_subjects, n_items), float("nan"))
    train_matrix[train_s, train_i] = torch.tensor(train_labels, dtype=torch.float32)

    val_split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_split.pkl")
    with open(val_split_path, "wb") as f:
        pickle.dump({
            "val_s_idx": val_s,
            "val_i_idx": val_i,
            "val_labels": val_labels,
            "subjects_list": subjects_list,
            "items_list": items_list,
        }, f)
    print(f"Val split saved to {val_split_path}")

    return train_matrix, val_s, val_i, val_labels, subjects_list, items_list, n_subjects, n_items
