"""Masked heatmap loss (plan Step 5.3).

``L = Σ_ij w_ij · mean_uv (Ĥ_ij − H_ij)² / max(1, Σ_ij w_ij)``: the proposal's loss (a sum
over pixels) divided by the heatmap size 64 x 48; same optimum (plan §10, 5.3).
``masked_mse`` uses only tensor methods, so this module imports without torch;
``masked_mse_np`` is the numpy reference the torch version is tested against (D13).
"""

from __future__ import annotations

import numpy as np


def masked_mse_np(pred, target, weight) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    per_joint = ((pred - target) ** 2).mean(axis=(-2, -1))  # [N, K]
    return float((weight * per_joint).sum() / max(1.0, weight.sum()))


def masked_mse(pred, target, weight):
    """Torch tensors ``[N,K,H,W]``, ``[N,K,H,W]``, ``[N,K]`` -> scalar tensor."""
    per_joint = ((pred - target) ** 2).mean(dim=(-2, -1))
    return (weight * per_joint).sum() / weight.sum().clamp(min=1.0)
