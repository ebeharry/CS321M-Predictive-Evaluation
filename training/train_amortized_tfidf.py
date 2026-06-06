import os

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_measure.models import AmortizedIRT

from OLD.sample_code_submission._data import load_train_val_split

train_matrix, val_s, val_i, val_labels, subjects_list, items_list, n_subjects, n_items = load_train_val_split()

HIDDEN_DIM = 256
N_LAYERS = 3
PL = 2
TFIDF_MAX_FEATURES = 4096

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
history = model.fit(train_matrix, embeddings, max_epochs=300, lr=1e-3, verbose=True)
print(f"Final loss — amortized IRT (TF-IDF): {history['losses'][-1]:.4f}")

est_ability = model.ability.detach()
print(f"Mean ability: {est_ability.mean():.4f}, std: {est_ability.std():.4f}")

_dir = os.path.dirname(__file__)
save_path = os.path.join(_dir, "amortized_irt_tfidf.pt")
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
tfidf_path = os.path.join(_dir, "tfidf_vectorizer.pkl")
joblib.dump(tfidf, tfidf_path)
print(f"Saved to {save_path} and {tfidf_path}")
