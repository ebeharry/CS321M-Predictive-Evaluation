import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from torch_measure.models._network import MLP

from OLD.sample_code_submission._data import load_train_val_split

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
        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


train_matrix, val_s, val_i, val_labels, subjects_list, items_list, n_subjects, n_items = load_train_val_split()

# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

encoder = SentenceTransformer(ENCODER_NAME, device=DEVICE)

print(f"Encoding {n_subjects} subjects with {ENCODER_NAME}...")
subj_emb = encoder.encode(
    subjects_list, batch_size=256, convert_to_numpy=True, show_progress_bar=True,
).astype("float32")

print(f"Encoding {n_items} items...")
item_emb = encoder.encode(
    items_list, batch_size=256, convert_to_numpy=True, show_progress_bar=True,
).astype("float32")

subj_emb_t = torch.tensor(subj_emb, device=DEVICE)
item_emb_t = torch.tensor(item_emb, device=DEVICE)
embedding_dim = subj_emb.shape[1]
print(f"Embedding dim: {embedding_dim}")

# ---------------------------------------------------------------------------
# Build train triples from matrix
# ---------------------------------------------------------------------------

train_s_idx, train_i_idx = torch.where(~torch.isnan(train_matrix))
train_labels = train_matrix[train_s_idx, train_i_idx]
train_s_idx = train_s_idx.to(DEVICE)
train_i_idx = train_i_idx.to(DEVICE)
train_labels = train_labels.to(DEVICE)
n_train = len(train_labels)
print(f"Train triples: {n_train}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

head = NCFHead(
    input_dim=embedding_dim * 2,
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    dropout=0.1,
).to(DEVICE)
optimizer = torch.optim.Adam(head.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

val_s_t = torch.tensor(val_s, dtype=torch.long, device=DEVICE)
val_i_t = torch.tensor(val_i, dtype=torch.long, device=DEVICE)

for epoch in range(MAX_EPOCHS):
    head.train()
    perm = torch.randperm(n_train, device=DEVICE)
    total_loss, n_batches = 0.0, 0
    for start in range(0, n_train, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        u = subj_emb_t[train_s_idx[idx]]
        v = item_emb_t[train_i_idx[idx]]
        logit = head(torch.cat([u, v], dim=-1))
        loss = criterion(logit, train_labels[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        head.eval()
        with torch.no_grad():
            u_val = subj_emb_t[val_s_t]
            v_val = item_emb_t[val_i_t]
            preds = np.clip(torch.sigmoid(head(torch.cat([u_val, v_val], dim=-1))).cpu().numpy(), 0.001, 0.999)
        val_auc = roc_auc_score(val_labels, preds)
        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  loss={total_loss/n_batches:.4f}  val_auc={val_auc:.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

_dir = os.path.dirname(os.path.abspath(__file__))

torch.save(head.state_dict(), os.path.join(_dir, "ncf_head.pt"))
torch.save(
    {
        "subject_embeddings": torch.tensor(subj_emb),
        "item_embeddings": torch.tensor(item_emb),
    },
    os.path.join(_dir, "ncf_embeddings.pt"),
)
with open(os.path.join(_dir, "ncf_meta.pkl"), "wb") as f:
    pickle.dump({
        "subjects_list": subjects_list,
        "items_list": items_list,
        "encoder_name": ENCODER_NAME,
        "embedding_dim": embedding_dim,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "dropout": 0.1,
    }, f)
with open(os.path.join(_dir, "ncf_val_split.pkl"), "wb") as f:
    pickle.dump({
        "val_s_idx": val_s,
        "val_i_idx": val_i,
        "val_labels": val_labels,
        "subjects_list": subjects_list,
        "items_list": items_list,
    }, f)

print("Saved: ncf_head.pt, ncf_embeddings.pt, ncf_meta.pkl, ncf_val_split.pkl")
