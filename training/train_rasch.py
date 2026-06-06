import os

import torch
from torch_measure.models import Rasch

from OLD.sample_code_submission._data import load_train_val_split

train_matrix, val_s, val_i, val_labels, subjects_list, items_list, n_subjects, n_items = load_train_val_split()

model = Rasch(n_subjects=n_subjects, n_items=n_items)
history = model.fit(train_matrix, max_epochs=300, verbose=True)
print(f"Final loss — Rasch: {history['losses'][-1]:.4f}")

est_ability = model.ability.detach()
est_difficulty = model.difficulty.detach()
print(f"Mean ability: {est_ability.mean():.4f}, mean difficulty: {est_difficulty.mean():.4f}")

save_path = os.path.join(os.path.dirname(__file__), "rasch.pt")
torch.save(
    {
        "ability": est_ability,
        "difficulty": est_difficulty,
        "subjects_list": subjects_list,
        "items_list": items_list,
        "mean_ability": est_ability.mean().item(),
    },
    save_path,
)
print(f"Saved to {save_path}")
