"""Modal training app for MultiFacet2PL and NCF.

Run with:
    modal run modal_train.py                        # both jobs in parallel
    modal run modal_train.py::train_multifacet      # single job
    modal run modal_train.py::train_ncf             # single job
"""

import os
from pathlib import Path

import modal

APP_NAME = "irt-training"
VOLUME_NAME = "irt-artifacts"
REMOTE_OUT = "/out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "torch-measure",
        "pyro-ppl",
        "sentence-transformers",
        "datasets",
        "huggingface_hub",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
    )
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
SPLIT_SEED = 42
VAL_FRAC = 0.1


# ---------------------------------------------------------------------------
# MultiFacet2PL
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="h100", timeout=4 * 3600, volumes={REMOTE_OUT: vol})
def train_multifacet():
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from sklearn.metrics import roc_auc_score
    from torch_measure.models import MultiFacet2PL

    def render_subject_content(subject, fallback_id):
        display_name = subject.get("display_name") or fallback_id
        lines = [f"Name: {display_name}"]
        for key, label in (
            ("provider", "Organization"), ("params", "Parameters"),
            ("release_date", "Released"), ("family", "Family"),
        ):
            value = subject.get(key)
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def condition_to_group(cond):
        if not cond or cond == "none":
            return "none"
        if cond.startswith("aspect="):       return "aspect"
        if cond.startswith("judge=1;"):      return "judge_1"
        if cond.startswith("judge=2;"):      return "judge_2"
        if cond.startswith("metric=security"): return "metric_security"
        if cond.startswith("metric=utility"):  return "metric_utility"
        if cond.startswith("mode="):         return "mode"
        if cond.startswith("skill="):        return "skill"
        if cond.startswith("source="):       return "source"
        if cond.startswith("subset="):       return "subset"
        return "other"

    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f for f in repo_files
        if f.endswith(".parquet") and f not in REGISTRY_FILES and not f.endswith("_traces.parquet")
    )
    responses = load_dataset(
        REPO_ID,
        data_files=response_files,
        features=Features({
            "subject_id": Value("string"), "item_id": Value("string"),
            "benchmark_id": Value("string"), "trial": Value("int64"),
            "test_condition": Value("string"), "response": Value("float64"),
            "correct_answer": Value("string"), "trace": Value("string"),
        }),
        split="train",
    )
    items_by_id = (
        pd.read_parquet(hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset"))
        .set_index("item_id").to_dict("index")
    )
    subjects_by_id = (
        pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset"))
        .set_index("subject_id").to_dict("index")
    )

    df = responses.to_pandas()
    df["subject_content"] = df["subject_id"].map(lambda s: render_subject_content(subjects_by_id.get(s, {}), s))
    df["item_content"] = df["item_id"].map(lambda i: items_by_id.get(i, {}).get("content"))
    df = df.dropna(subset=["item_content"])
    df["test_condition"] = df["test_condition"].fillna("none")
    df = df.sort_values("trial").drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
    df["facet_group"] = df["test_condition"].map(condition_to_group)

    subjects_list = df["subject_content"].unique().tolist()
    items_list = df["item_content"].unique().tolist()
    groups_list = sorted(df["facet_group"].unique().tolist())
    s_arr = df["subject_content"].map({s: i for i, s in enumerate(subjects_list)}).values
    i_arr = df["item_content"].map({it: i for i, it in enumerate(items_list)}).values
    g_arr = df["facet_group"].map({g: i for i, g in enumerate(groups_list)}).values
    labels_arr = (df["response"].values > 0).astype("float32")

    n_subjects, n_items, n_groups = len(subjects_list), len(items_list), len(groups_list)
    print(f"Subjects: {n_subjects}, Items: {n_items}, Facet groups: {n_groups}: {groups_list}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(labels_arr))
    cut = int((1 - VAL_FRAC) * len(perm))
    tr, vl = perm[:cut], perm[cut:]
    print(f"Train obs: {len(tr)}, Val obs: {len(vl)}")

    model = MultiFacet2PL(n_subjects=n_subjects, n_items=n_items, n_facet_levels=n_groups)
    history = model.fit(
        subject_idx=torch.tensor(s_arr[tr], dtype=torch.long),
        item_idx=torch.tensor(i_arr[tr], dtype=torch.long),
        facet_idx=torch.tensor(g_arr[tr], dtype=torch.long),
        response=torch.tensor(labels_arr[tr], dtype=torch.float32),
        max_epochs=4000,
        verbose=True,
    )
    print(f"Final loss: {history['losses'][-1]:.4f}")

    post = history["posterior"]
    ability       = torch.tensor(np.array(post["ability"]),       dtype=torch.float32)
    difficulty    = torch.tensor(np.array(post["difficulty"]),    dtype=torch.float32)
    discrimination = torch.tensor(np.array(post["discrimination"]), dtype=torch.float32)
    gamma         = torch.tensor(np.array(post["gamma"]),         dtype=torch.float32)
    tau           = torch.tensor(np.array(post["tau"]),           dtype=torch.float32)
    delta         = torch.tensor(np.array(post["delta"]),         dtype=torch.float32)

    print(f"Mean ability: {ability.mean():.4f}  mean difficulty: {difficulty.mean():.4f}")
    print(f"Gamma: {dict(zip(groups_list, gamma.tolist()))}")

    vs = torch.tensor(s_arr[vl], dtype=torch.long)
    vi = torch.tensor(i_arr[vl], dtype=torch.long)
    vg = torch.tensor(g_arr[vl], dtype=torch.long)
    with torch.no_grad():
        logit = discrimination[vi] * ((ability[vs] + delta[vs, vg]) - (difficulty[vi] + gamma[vg] + tau[vi, vg]))
        val_preds = np.clip(torch.sigmoid(logit).numpy(), 0.001, 0.999)
    print(f"Val AUC: {roc_auc_score(labels_arr[vl], val_preds):.4f}")

    os.makedirs(REMOTE_OUT, exist_ok=True)
    torch.save({
        "ability": ability, "difficulty": difficulty, "discrimination": discrimination,
        "gamma": gamma, "tau": tau, "delta": delta,
        "subjects_list": subjects_list, "items_list": items_list, "groups_list": groups_list,
        "mean_ability": ability.mean().item(), "mean_difficulty": difficulty.mean().item(),
        "mean_discrimination": discrimination.mean().item(),
        "mean_delta": delta.mean(dim=0), "mean_tau": tau.mean(dim=0),
    }, f"{REMOTE_OUT}/multifacet_2pl.pt")
    with open(f"{REMOTE_OUT}/multifacet_val_split.pkl", "wb") as f:
        pickle.dump({
            "val_s_idx": s_arr[vl], "val_i_idx": i_arr[vl], "val_g_idx": g_arr[vl],
            "val_labels": labels_arr[vl],
            "subjects_list": subjects_list, "items_list": items_list, "groups_list": groups_list,
        }, f)
    vol.commit()
    print("MultiFacet2PL artifacts saved.")


# ---------------------------------------------------------------------------
# NCF
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="h100", timeout=2 * 3600, volumes={REMOTE_OUT: vol})
def train_ncf():
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    from torch_measure.models._network import MLP

    ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    HIDDEN_DIM = 256
    N_LAYERS = 3
    BATCH_SIZE = 1024
    MAX_EPOCHS = 100
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class NCFHead(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, n_layers=3, dropout=0.1):
            super().__init__()
            self.net = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1,
                           n_layers=n_layers, dropout=dropout)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def render_subject_content(subject, fallback_id):
        display_name = subject.get("display_name") or fallback_id
        lines = [f"Name: {display_name}"]
        for key, label in (
            ("provider", "Organization"), ("params", "Parameters"),
            ("release_date", "Released"), ("family", "Family"),
        ):
            value = subject.get(key)
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    # Replicate _load_raw() with baked item keys (condition included in item string)
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f for f in repo_files
        if f.endswith(".parquet") and f not in REGISTRY_FILES and not f.endswith("_traces.parquet")
    )
    responses = load_dataset(
        REPO_ID,
        data_files=response_files,
        features=Features({
            "subject_id": Value("string"), "item_id": Value("string"),
            "benchmark_id": Value("string"), "trial": Value("int64"),
            "test_condition": Value("string"), "response": Value("float64"),
            "correct_answer": Value("string"), "trace": Value("string"),
        }),
        split="train",
    )
    items_by_id = (
        pd.read_parquet(hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset"))
        .set_index("item_id").to_dict("index")
    )
    subjects_by_id = (
        pd.read_parquet(hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset"))
        .set_index("subject_id").to_dict("index")
    )

    df = responses.to_pandas()
    df["subject_content"] = df["subject_id"].map(lambda s: render_subject_content(subjects_by_id.get(s, {}), s))
    df["item_content"] = df["item_id"].map(lambda i: items_by_id.get(i, {}).get("content"))
    df = df.dropna(subset=["item_content"])
    df["test_condition"] = df["test_condition"].fillna("none")
    df = df.sort_values("trial").drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
    df["item_key"] = df["item_content"] + "\n[Condition: " + df["test_condition"] + "]"

    subjects_list = df["subject_content"].unique().tolist()
    items_list = df["item_key"].unique().tolist()
    s_arr = df["subject_content"].map({s: i for i, s in enumerate(subjects_list)}).values
    i_arr = df["item_key"].map({it: i for i, it in enumerate(items_list)}).values
    labels_arr = (df["response"].values > 0).astype("float32")

    n_subjects, n_items = len(subjects_list), len(items_list)
    print(f"Subjects: {n_subjects}, Items: {n_items}, Device: {DEVICE}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(labels_arr))
    cut = int((1 - VAL_FRAC) * len(perm))
    tr, vl = perm[:cut], perm[cut:]
    print(f"Train obs: {len(tr)}, Val obs: {len(vl)}")

    encoder = SentenceTransformer(ENCODER_NAME, device=DEVICE)
    print(f"Encoding {n_subjects} subjects...")
    subj_emb = encoder.encode(subjects_list, batch_size=256, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    print(f"Encoding {n_items} items...")
    item_emb = encoder.encode(items_list, batch_size=256, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    subj_emb_t = torch.tensor(subj_emb, device=DEVICE)
    item_emb_t = torch.tensor(item_emb, device=DEVICE)
    embedding_dim = subj_emb.shape[1]

    train_s_t = torch.tensor(s_arr[tr], dtype=torch.long, device=DEVICE)
    train_i_t = torch.tensor(i_arr[tr], dtype=torch.long, device=DEVICE)
    train_labels_t = torch.tensor(labels_arr[tr], device=DEVICE)
    val_s_t = torch.tensor(s_arr[vl], dtype=torch.long, device=DEVICE)
    val_i_t = torch.tensor(i_arr[vl], dtype=torch.long, device=DEVICE)
    n_train = len(train_labels_t)

    head = NCFHead(input_dim=embedding_dim * 2, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(MAX_EPOCHS):
        head.train()
        epoch_perm = torch.randperm(n_train, device=DEVICE)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = epoch_perm[start:start + BATCH_SIZE]
            logit = head(torch.cat([subj_emb_t[train_s_t[idx]], item_emb_t[train_i_t[idx]]], dim=-1))
            loss = criterion(logit, train_labels_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            head.eval()
            with torch.no_grad():
                preds = np.clip(
                    torch.sigmoid(
                        head(torch.cat([subj_emb_t[val_s_t], item_emb_t[val_i_t]], dim=-1))
                    ).cpu().numpy(),
                    0.001, 0.999,
                )
            val_auc = roc_auc_score(labels_arr[vl], preds)
            print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  loss={total_loss/n_batches:.4f}  val_auc={val_auc:.4f}")

    os.makedirs(REMOTE_OUT, exist_ok=True)
    torch.save(head.state_dict(), f"{REMOTE_OUT}/ncf_head.pt")
    torch.save(
        {"subject_embeddings": torch.tensor(subj_emb), "item_embeddings": torch.tensor(item_emb)},
        f"{REMOTE_OUT}/ncf_embeddings.pt",
    )
    with open(f"{REMOTE_OUT}/ncf_meta.pkl", "wb") as f:
        pickle.dump({
            "subjects_list": subjects_list, "items_list": items_list,
            "encoder_name": ENCODER_NAME, "embedding_dim": embedding_dim,
            "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": 0.1,
        }, f)
    with open(f"{REMOTE_OUT}/ncf_val_split.pkl", "wb") as f:
        pickle.dump({
            "val_s_idx": s_arr[vl], "val_i_idx": i_arr[vl], "val_labels": labels_arr[vl],
            "subjects_list": subjects_list, "items_list": items_list,
        }, f)
    vol.commit()
    print("NCF artifacts saved.")


# ---------------------------------------------------------------------------
# Local entrypoint: launch both jobs in parallel, then download artifacts
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    root = Path(__file__).parent.parent.parent  # starting_kit/
    local_dir = Path(__file__).parent            # fallback for non-submission artifacts

    destinations = {
        "multifacet_2pl.pt":        root / "multifacet_submission" / "multifacet_2pl.pt",
        "multifacet_val_split.pkl": local_dir / "multifacet_val_split.pkl",
        "ncf_head.pt":              root / "ncf_submission" / "ncf_head.pt",
        "ncf_meta.pkl":             root / "ncf_submission" / "ncf_meta_slim.pkl",
        "ncf_val_split.pkl":        local_dir / "ncf_val_split.pkl",
        "ncf_embeddings.pt":        local_dir / "ncf_embeddings.pt",
    }

    print("Launching MultiFacet2PL and NCF training in parallel...")
    h_mf = train_multifacet.spawn()
    h_ncf = train_ncf.spawn()

    print("Waiting for MultiFacet2PL...")
    h_mf.get()
    print("MultiFacet2PL done.")

    print("Waiting for NCF...")
    h_ncf.get()
    print("NCF done.")

    print("Downloading artifacts...")
    for entry in vol.listdir("/"):
        name = Path(entry.path).name
        dest = destinations.get(name, local_dir / name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"".join(vol.read_file(entry.path)))
        print(f"  {entry.path} -> {dest}")
    print("Done.")
