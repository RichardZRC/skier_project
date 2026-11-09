"""Pose-guided crop refinement (Step 7c, D26): predict on the tracker crop, then crop again around the prediction.

Pass 1 is the usual tracker crop (``propagation.tracker_crop_boxes``). The refined box is the
tight box of the pass-1 body + ski predictions (poles excluded, as in ``geometry.reference_box``)
that are finite and score at least ``s_min``, scaled by ``k_refine``. Pass 2 crops it with the
model's usual padding and predicts again. One refinement step only.

The refined box is built from predictions alone, never from records, so no label reaches the
crop. Rows with fewer than ``N_MIN`` qualifying points, or a box diagonal under 1 px, fall back:
they keep the pass-1 prediction and the pass-1 crop, so cut-off counts follow the crop used.

The setting is chosen on val by the D26 rule: the highest E2 seed-mean of refined (body + ski)
PCK@0.1 / 2; within ``REFINE_TIE`` points of the best is a tie, broken by ``k_refine`` closest to
1.0 and then the higher ``s_min``. Refinement becomes the system default only if the pick beats
the single tracker crop by more than ``REFINE_GATE`` points.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.geometry import body_ski_mask, box_diagonal, box_from_points, expand_box
from skier.pose.infer import HeatmapFn, predict_records
from skier.schema import Schema

S_MIN_GRID = (0.1, 0.2, 0.3, 0.5)
K_REFINE_GRID = (1.0, 1.15, 1.3)
REFINE_GRID = tuple((s, k) for s in S_MIN_GRID for k in K_REFINE_GRID)
N_MIN = 4
REFINE_TIE = 0.5  # points
REFINE_GATE = 0.5  # points over the single tracker crop
REFINE_RULE = (f"highest E2 seed-mean of refined (body + ski) PCK@0.1 / 2; within {REFINE_TIE} points of the best "
               "is a tie, broken by k_refine closest to 1.0, then the higher s_min; adopted as the system default "
               f"only if it beats the single tracker crop by more than {REFINE_GATE} points")


def setting_key(s_min: float, k_refine: float) -> str:
    return f"s{s_min:g}_k{k_refine:g}"


def refined_boxes(xy: np.ndarray, scores: np.ndarray, schema: Schema, s_min: float, k_refine: float,
                  n_min: int = N_MIN) -> np.ndarray:
    """``[N, 4]`` refined boxes from pass-1 predictions ``xy [N, K, 2]`` and heatmap maxima ``scores [N, K]``.

    NaN rows fall back (fewer than ``n_min`` qualifying points, or a diagonal under 1 px).
    """
    xy = np.asarray(xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    out = np.full((len(xy), 4), np.nan)
    ok = body_ski_mask(schema)[None, :] & np.isfinite(xy).all(axis=2) & (np.nan_to_num(scores, nan=-np.inf) >= s_min)
    for i in range(len(xy)):
        if ok[i].sum() < n_min:
            continue
        box = box_from_points(xy[i], ok[i])
        if box_diagonal(box) >= 1.0:
            out[i] = expand_box(box, k_refine)
    return out


def refine_on(records: Sequence[PoseRecord], heatmap_fn: HeatmapFn, pass1_xy: np.ndarray, pass1_scores: np.ndarray,
              pass1_boxes: np.ndarray, schema: Schema, s_min: float, k_refine: float, expand: float,
              batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(xy [N, K, 2], crop boxes [N, 4], refined [N])``: pass 2 where a refined box exists, else pass 1."""
    boxes = refined_boxes(pass1_xy, pass1_scores, schema, s_min, k_refine)
    used = np.isfinite(boxes).all(axis=1)
    xy = np.array(pass1_xy, dtype=np.float64, copy=True)
    crop = np.array(pass1_boxes, dtype=np.float64, copy=True)
    idx = np.flatnonzero(used)
    if len(idx):
        xy[idx] = predict_records([records[i] for i in idx], heatmap_fn, expand, batch_size, boxes=boxes[idx]).xy
        crop[idx] = boxes[idx]
    return xy, crop, used


def choose_refine(table: Sequence[dict], baseline: dict) -> dict:
    """``table``: rows ``{"s_min", "k_refine", "body", "ski"}`` (E2 refined PCK@0.1, mean over seeds);
    ``baseline``: ``{"body", "ski"}`` for the single tracker crop."""
    scored = [(float(r["s_min"]), float(r["k_refine"]), (r["body"] + r["ski"]) / 2) for r in table]
    best = max(m for _, _, m in scored)
    s, k, m = min((row for row in scored if row[2] >= best - REFINE_TIE), key=lambda r: (abs(r[1] - 1.0), -r[0]))
    base = (baseline["body"] + baseline["ski"]) / 2
    return {"rule": REFINE_RULE, "pick": {"s_min": s, "k_refine": k}, "pick_mean": m, "best_mean": best,
            "baseline_mean": base, "gain": m - base, "adopt": m - base > REFINE_GATE,
            "table": [{"s_min": s_, "k_refine": k_, "mean_body_ski": m_} for s_, k_, m_ in scored]}
