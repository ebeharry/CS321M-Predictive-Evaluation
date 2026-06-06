"""MultiFacet 2PL IRT submission for the Predictive AI Evaluation Challenge."""

import os
import re

import torch

_dir = os.path.dirname(__file__)
_ckpt = torch.load(
    os.path.join(_dir, "multifacet_2pl.pt"),
    weights_only=True,
    map_location="cpu",
)

_subj_to_idx = {s: i for i, s in enumerate(_ckpt["subjects_list"])}
_item_to_idx = {it: i for i, it in enumerate(_ckpt["items_list"])}
_group_to_idx = {g: i for i, g in enumerate(_ckpt["groups_list"])}

_ability = _ckpt["ability"]
_delta = _ckpt["delta"]
_difficulty = _ckpt["difficulty"]
_discrimination = _ckpt["discrimination"]
_tau = _ckpt["tau"]
_gamma = _ckpt["gamma"]
_mean_delta = _ckpt["mean_delta"]
_mean_tau = _ckpt["mean_tau"]
_mean_ability = torch.tensor(_ckpt["mean_ability"])
_mean_difficulty = torch.tensor(_ckpt["mean_difficulty"])
_mean_discrimination = torch.tensor(_ckpt["mean_discrimination"])

print(
    f"MultiFacet2PL loaded: {len(_subj_to_idx)} subjects, "
    f"{len(_item_to_idx)} items, {len(_group_to_idx)} facet groups"
)


def _condition_to_fine_group(cond: str) -> str:
    """Fine-grained group: preserves the specific aspect or judge criterion."""
    if not cond or cond == "none":
        return "none"
    if cond.startswith("aspect="):
        return f"aspect_{cond[len('aspect='):]}"
    m = re.match(r"judge=(\d+);(.*)", cond)
    if m:
        return f"judge_{m.group(1)}_{m.group(2)}"
    if cond.startswith("metric=security"): return "metric_security"
    if cond.startswith("metric=utility"):  return "metric_utility"
    if cond.startswith("mode="):           return "mode"
    if cond.startswith("skill="):          return "skill"
    if cond.startswith("source="):         return "source"
    if cond.startswith("subset="):         return "subset"
    return "other"


def _condition_to_coarse_group(cond: str) -> str:
    """Coarse fallback group — used when fine group is absent from the checkpoint."""
    if not cond or cond == "none":         return "none"
    if cond.startswith("aspect="):         return "aspect"
    if cond.startswith("judge=1;"):        return "judge_1"
    if cond.startswith("judge=2;"):        return "judge_2"
    if cond.startswith("metric=security"): return "metric_security"
    if cond.startswith("metric=utility"):  return "metric_utility"
    if cond.startswith("mode="):           return "mode"
    if cond.startswith("skill="):          return "skill"
    if cond.startswith("source="):         return "source"
    if cond.startswith("subset="):         return "subset"
    return "other"


def predict(input: dict, labeled=None) -> float:
    cond = input["condition"]
    grp = _group_to_idx.get(
        _condition_to_fine_group(cond),
        _group_to_idx.get(_condition_to_coarse_group(cond), 0),
    )

    s = _subj_to_idx.get(input["subject_content"])
    theta = _ability[s] if s is not None else _mean_ability
    d_s = _delta[s, grp] if s is not None else _mean_delta[grp]

    it = _item_to_idx.get(input["item_content"])
    b = _difficulty[it] if it is not None else _mean_difficulty
    disc = _discrimination[it] if it is not None else _mean_discrimination
    tau_it = _tau[it, grp] if it is not None else _mean_tau[grp]

    logit = disc * ((theta + d_s) - (b + _gamma[grp] + tau_it))
    return torch.clamp(torch.sigmoid(logit), 0.001, 0.999).item()
