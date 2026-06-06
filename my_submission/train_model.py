from huggingface_hub import HfApi, hf_hub_download
from torch_measure.models import AmortizedIRT
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Features, Value, load_dataset

plt.rcParams["figure.dpi"] = 100
print(f"torch_measure imported successfully")

# ---------------------------------------------------------------------------
# Loading and Processing The Public Training Data (ReadMe Tutorial Code)
# ---------------------------------------------------------------------------

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    name
    for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)

response_features = Features(
    {
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    }
)

responses = load_dataset(
    REPO_ID,
    data_files=response_files,
    features=response_features,
    split="train",
)
items = pd.read_parquet(
    hf_hub_download(REPO_ID, "items.parquet", repo_type="dataset")
).to_dict("records")
subjects = pd.read_parquet(
    hf_hub_download(REPO_ID, "subjects.parquet", repo_type="dataset")
).to_dict("records")
benchmarks = pd.read_parquet(
    hf_hub_download(REPO_ID, "benchmarks.parquet", repo_type="dataset")
).to_dict("records")

items_by_id = {row["item_id"]: row for row in items}
subjects_by_id = {row["subject_id"]: row for row in subjects}
benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}

def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def to_training_example(row):
    item = items_by_id.get(row["item_id"], {})
    subject = subjects_by_id.get(row["subject_id"], {})
    benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
    benchmark_id = row["benchmark_id"]
    if "benchmark_id" in benchmark and benchmark["benchmark_id"]:
        benchmark_id = benchmark["benchmark_id"]

    return {
        "benchmark": benchmark_id,
        "condition": row["test_condition"] or "none",
        "subject_content": render_subject_content(subject, row["subject_id"]),
        "item_content": item.get("content"),
        "label": row["response"],
    }

# ---------------------------------------------------------------------------
# Loading and Processing The Public Training Data with Pandas (Claude)
# ---------------------------------------------------------------------------

# Convert to DataFrame once — vastly faster than row-by-row Python iteration
responses_df = responses.to_pandas()

# Render content strings only for unique subjects/items, then map
unique_subject_ids = responses_df["subject_id"].unique()
subject_content_map = {
    sid: render_subject_content(subjects_by_id.get(sid, {}), sid)
    for sid in unique_subject_ids
}
unique_item_ids = responses_df["item_id"].unique()
item_content_map = {
    iid: items_by_id.get(iid, {}).get("content")
    for iid in unique_item_ids
}

responses_df["subject_content"] = responses_df["subject_id"].map(subject_content_map)
responses_df["item_content"] = responses_df["item_id"].map(item_content_map)
responses_df = responses_df.dropna(subset=["item_content"])

# Keep only the smallest trial per (subject, item, condition) — discard repeats
responses_df = (
    responses_df
    .sort_values("trial")
    .drop_duplicates(subset=["subject_id", "item_id", "test_condition"], keep="first")
)

# Include test_condition so different conditions are treated as separate item variants
responses_df["item_key"] = (
    responses_df["item_content"] + "\n[Condition: " + responses_df["test_condition"].fillna("none") + "]"
)

# Build index mappings
subjects_list = responses_df["subject_content"].unique().tolist()
items_list = responses_df["item_key"].unique().tolist()
subj_idx = {s: i for i, s in enumerate(subjects_list)}
item_idx = {it: i for i, it in enumerate(items_list)}

n_subjects = len(subjects_list)
n_items = len(items_list)
print(f"Building matrix: {n_subjects} subjects x {n_items} items")

# Vectorized fill using index arrays
s_idx_arr = responses_df["subject_content"].map(subj_idx).values
i_idx_arr = responses_df["item_key"].map(item_idx).values
# Binarize: any credit (> 0) → correct; hidden scoring is binary
labels_arr = (responses_df["response"].values > 0).astype("float32")

matrix = torch.full((n_subjects, n_items), float("nan"))
matrix[s_idx_arr, i_idx_arr] = torch.tensor(labels_arr, dtype=torch.float32)

print(matrix.shape)

# ---------------------------------------------------------------------------
# Train an amortized IRT model and save the estimates
# ---------------------------------------------------------------------------

n_subjects = matrix.shape[0]
n_items = matrix.shape[1]

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HIDDEN_DIM = 256
N_LAYERS = 3
PL = 2

print(f"Encoding {n_items} items with {ENCODER_NAME}...")
encoder = SentenceTransformer(ENCODER_NAME)
embeddings_np = encoder.encode(items_list, batch_size=256, convert_to_numpy=True).astype("float32")
embeddings = torch.tensor(embeddings_np)
print(f"Embeddings shape: {embeddings.shape}")

amort = AmortizedIRT(
    n_subjects=n_subjects,
    n_items=n_items,
    embedding_dim=embeddings.shape[1],
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    pl=PL,
    dropout=0.1,
)
history = amort.fit(matrix, embeddings, max_epochs=300, lr=1e-3, verbose=True)
print(f"Final loss — amortized IRT: {history['losses'][-1]:.4f}")

est_ability = amort.ability.detach()
print(f"Mean ability: {est_ability.mean():.4f}, std: {est_ability.std():.4f}")

save_path = "sample_code_submission/amortized_irt.pt"
torch.save(
    {
        "model_state_dict": amort.state_dict(),
        "subjects_list": subjects_list,
        "mean_ability": est_ability.mean().item(),
        "n_subjects": n_subjects,
        "n_items": n_items,
        "embedding_dim": int(embeddings.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "pl": PL,
        "encoder_name": ENCODER_NAME,
    },
    save_path,
)
print(f"Saved to {save_path}")