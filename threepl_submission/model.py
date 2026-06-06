"""3PL IRT submission for the Predictive AI Evaluation Challenge."""

import os
import torch

_dir = os.path.dirname(__file__)
_ckpt = torch.load(os.path.join(_dir, "threepl.pt"), weights_only=True, map_location="cpu")

_c = _ckpt["guessing"].mean()
_mean_prob = float(torch.clamp(
    _c + (1 - _c) * torch.sigmoid(_ckpt["discrimination"].mean() * (torch.tensor(_ckpt["mean_ability"]) - _ckpt["difficulty"].mean())),
    0.001, 0.999,
))

print(f"3PL loaded: mean_prob={_mean_prob:.4f}")


def predict(input: dict, labeled=None) -> float:
    return _mean_prob
