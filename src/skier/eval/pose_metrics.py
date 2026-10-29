"""Pose metrics (plan Step 2.5): PCK normalized by the reference-box diagonal.

Rules (also in docs/evaluation_protocol.md):
- Only reliable labels (``visible == 1``) count.
- A hit is a finite prediction within ``thr x diagonal`` (``<=``) of the label.
- A missing prediction (NaN) is a miss and stays in the denominator.
- Confidence thresholds are never applied here.
- Counts are always reported next to percentages; ``pct`` is None when nothing counts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.schema import Schema

THRESHOLDS = (0.1, 0.05)


def _check(pred_xy, gt_xy, reliable, ref_diag):
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    rel = np.asarray(reliable, dtype=bool)
    diag = np.asarray(ref_diag, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 2:
        raise ValueError(f"pred and gt must both be [N, K, 2], got {pred.shape} and {gt.shape}")
    if rel.shape != pred.shape[:2] or diag.shape != pred.shape[:1]:
        raise ValueError(f"reliable must be [N, K] and ref_diag [N]; got {rel.shape}, {diag.shape}")
    if not (np.isfinite(diag) & (diag > 0))[rel.any(axis=1)].all():
        raise ValueError("ref_diag must be finite and > 0 for every sample with reliable labels")
    return pred, gt, rel, diag


def _result(hits: int, total: int) -> dict:
    return {"hits": int(hits), "total": int(total), "pct": 100.0 * hits / total if total else None}


def sample_hits(pred_xy, gt_xy, reliable, ref_diag, joint_ids: Sequence[int], thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: ``(hits [N], totals [N])`` over the reliable labels of ``joint_ids``."""
    pred, gt, rel, diag = _check(pred_xy, gt_xy, reliable, ref_diag)
    j = list(joint_ids)
    dist = np.linalg.norm(pred[:, j] - gt[:, j], axis=-1)  # NaN where the prediction is missing
    counted = rel[:, j]
    with np.errstate(invalid="ignore"):
        hit = counted & np.isfinite(dist) & (dist <= thr * diag[:, None])
    return hit.sum(axis=1), counted.sum(axis=1)


def pck(pred_xy, gt_xy, reliable, ref_diag, joint_ids: Sequence[int], thr: float) -> dict:
    """``{"hits", "total", "pct"}`` over the reliable labels of ``joint_ids``."""
    hits, totals = sample_hits(pred_xy, gt_xy, reliable, ref_diag, joint_ids, thr)
    return _result(hits.sum(), totals.sum())


def per_joint_pck(pred_xy, gt_xy, reliable, ref_diag, thr: float) -> list[dict]:
    k = np.asarray(gt_xy).shape[1]
    return [pck(pred_xy, gt_xy, reliable, ref_diag, [i], thr) for i in range(k)]


def group_pck(pred_xy, gt_xy, reliable, ref_diag, schema: Schema, thr: float) -> dict:
    return {g: pck(pred_xy, gt_xy, reliable, ref_diag, members, thr) for g, members in schema.groups.items()}


def per_video_pck(pred_xy, gt_xy, reliable, ref_diag, video_ids: Sequence[str], joint_ids, thr: float) -> dict:
    vids = np.asarray(video_ids)
    out = {}
    for v in sorted(set(vids.tolist())):
        m = vids == v
        out[v] = pck(np.asarray(pred_xy)[m], np.asarray(gt_xy)[m], np.asarray(reliable)[m],
                     np.asarray(ref_diag)[m], joint_ids, thr)
    return out


def normalized_error(pred_xy, gt_xy, reliable, ref_diag, joint_ids: Sequence[int]) -> dict:
    """Mean ``distance / diagonal`` over reliable labels with a prediction, plus the missing rate."""
    pred, gt, rel, diag = _check(pred_xy, gt_xy, reliable, ref_diag)
    j = list(joint_ids)
    dist = np.linalg.norm(pred[:, j] - gt[:, j], axis=-1)
    counted = rel[:, j]
    available = counted & np.isfinite(dist)
    n_counted, n_available = int(counted.sum()), int(available.sum())
    norm = (dist / diag[:, None])[available]
    return {
        "mean_norm_error": float(norm.mean()) if n_available else None,
        "n_available": n_available,
        "n_missing": n_counted - n_available,
        "missing_rate": (n_counted - n_available) / n_counted if n_counted else None,
    }


def pose_summary(pred_xy, gt_xy, reliable, ref_diag, schema: Schema, video_ids: Sequence[str] | None = None,
                 thresholds: Sequence[float] = THRESHOLDS) -> dict:
    """Group PCK at each threshold, per-joint PCK@first threshold, errors, and per-video PCK."""
    out = {
        "n_samples": int(np.asarray(gt_xy).shape[0]),
        "groups": {f"{thr:g}": group_pck(pred_xy, gt_xy, reliable, ref_diag, schema, thr) for thr in thresholds},
        "per_joint": {
            schema.names[i]: r for i, r in enumerate(per_joint_pck(pred_xy, gt_xy, reliable, ref_diag, thresholds[0]))
        },
        "errors": {g: normalized_error(pred_xy, gt_xy, reliable, ref_diag, m) for g, m in schema.groups.items()},
    }
    if video_ids is not None:
        out["per_video"] = {
            g: per_video_pck(pred_xy, gt_xy, reliable, ref_diag, video_ids, m, thresholds[0])
            for g, m in schema.groups.items()
        }
    return out


def gt_arrays(records: Sequence[PoseRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """``(gt_xy [N,K,2], reliable [N,K], ref_diag [N], video_ids)`` from records.

    Excluded records are refused rather than skipped, so callers filter (and report)
    them explicitly.
    """
    excluded = [r.sample_id for r in records if r.excluded]
    if excluded:
        raise ValueError(f"{len(excluded)} excluded records passed to metrics (e.g. {excluded[0]}); filter them first")
    gt = np.stack([r.keypoints_xy for r in records])
    rel = np.stack([r.reliable for r in records])
    boxes = np.stack([r.reference_box_xyxy for r in records])
    diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return gt, rel, diag, [r.source_video_id for r in records]
