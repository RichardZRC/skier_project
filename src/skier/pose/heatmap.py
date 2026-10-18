"""Heatmap targets and decoding (plan Step 3.2).

Convention: heatmap coordinate = input coordinate / 4, with no half-pixel shift
(MMPose's MSRA codec, which the COCO checkpoint uses).

- ``encode`` centers each Gaussian exactly at ``xy / 4`` (sub-pixel). MMPose's default
  MSRA target rounds the center to a whole heatmap pixel; ours is more precise and only
  affects training (decision D12).
- ``decode`` copies MMPose 1.x's MSRA decoder (``get_heatmap_maximum`` +
  ``refine_keypoints``) including its edge conditions, so E0 matches the checkpoint's
  own inference. The AWS cross-check (``scripts/crosscheck_mmpose.py``) confirms it.
"""

from __future__ import annotations

import numpy as np

STRIDE = 4
HM_W, HM_H = 48, 64


def encode(xy_input, mask, hm_w: int = HM_W, hm_h: int = HM_H, sigma: float = 2.0,
           stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian targets ``[K, hm_h, hm_w]`` and weights ``[K]``.

    A point gets weight 1 only if it is masked in, finite, and inside the
    ``(hm_w * stride) x (hm_h * stride)`` input crop; otherwise its target stays zero.
    """
    xy = np.asarray(xy_input, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool).copy()
    with np.errstate(invalid="ignore"):
        keep &= np.isfinite(xy).all(axis=1)
        keep &= (xy[:, 0] >= 0) & (xy[:, 0] < hm_w * stride) & (xy[:, 1] >= 0) & (xy[:, 1] < hm_h * stride)
    target = np.zeros((len(xy), hm_h, hm_w), dtype=np.float32)
    xs, ys = np.arange(hm_w), np.arange(hm_h)
    for k in np.flatnonzero(keep):
        cx, cy = xy[k] / stride
        gx = np.exp(-((xs - cx) ** 2) / (2 * sigma**2))
        gy = np.exp(-((ys - cy) ** 2) / (2 * sigma**2))
        target[k] = np.outer(gy, gx)
    return target, keep.astype(np.float32)


def decode(heatmaps, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """``[N, K, H, W]`` heatmaps -> input-pixel coordinates ``[N, K, 2]`` and scores ``[N, K]``.

    Argmax, then a quarter-pixel step toward the larger neighbor (only away from the
    edges, exactly as MMPose's ``refine_keypoints``), then times the stride. Where the
    maximum is <= 0 the location is NaN (MMPose writes -1), i.e. a missing prediction.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    if hm.ndim != 4:
        raise ValueError(f"heatmaps must be [N, K, H, W], got {hm.shape}")
    n, k, h, w = hm.shape
    flat = hm.reshape(n, k, h * w)
    idx = flat.argmax(axis=-1)  # first maximum on ties, as np.argmax in MMPose
    score = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
    x, y = idx % w, idx // w
    valid = score > 0

    def at(yy, xx):
        lin = np.clip(yy, 0, h - 1) * w + np.clip(xx, 0, w - 1)
        return np.take_along_axis(flat, lin[..., None], axis=-1)[..., 0]

    # MMPose's conditions, verbatim: 1 < x < W-1 and 0 < y < H for dx; 1 < y < H-1 and 0 < x < W for dy.
    cond_x = valid & (x > 1) & (x < w - 1) & (y > 0) & (y < h)
    cond_y = valid & (y > 1) & (y < h - 1) & (x > 0) & (x < w)
    dx = np.where(cond_x, at(y, x + 1) - at(y, x - 1), 0.0)
    dy = np.where(cond_y, at(y + 1, x) - at(y - 1, x), 0.0)
    xy = np.stack([x + 0.25 * np.sign(dx), y + 0.25 * np.sign(dy)], axis=-1).astype(np.float64) * stride
    xy[~valid] = np.nan
    return xy, score.astype(np.float64)
