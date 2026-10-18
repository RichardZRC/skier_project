"""Replay metrics (plan Step 4.3).

A static full-frame crop scores perfectly on containment and smoothness, so crop area
and centering are always reported next to them (``replay_summary``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from skier.schema import Schema

SKI_ENDPOINTS = (16, 19, 20, 23)  # ski tips and tails


def _finite_rows(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a).all(axis=-1)


def box_containment(crops, boxes) -> dict:
    """Per frame ``area(box ∩ crop) / area(box)``; frames without a finite box are skipped."""
    crops, boxes = np.asarray(crops, float), np.asarray(boxes, float)
    use = _finite_rows(crops) & _finite_rows(boxes)
    c, b = crops[use], boxes[use]
    iw = np.clip(np.minimum(c[:, 2], b[:, 2]) - np.maximum(c[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(c[:, 3], b[:, 3]) - np.maximum(c[:, 1], b[:, 1]), 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    frac = np.where(area > 0, iw * ih / np.where(area > 0, area, 1), 0.0)
    return {"mean": float(frac.mean()) if len(frac) else None,
            "min": float(frac.min()) if len(frac) else None,
            "frac_fully_inside": float((frac >= 1 - 1e-9).mean()) if len(frac) else None,
            "n_frames": int(len(frac))}


def _inside(crops, xy):
    return ((xy[..., 0] >= crops[:, None, 0]) & (xy[..., 0] <= crops[:, None, 2])
            & (xy[..., 1] >= crops[:, None, 1]) & (xy[..., 1] <= crops[:, None, 3]))


def landmark_containment(crops, xy, mask, schema: Schema) -> dict:
    """Reliable body+ski points inside the crop, and ski endpoints separately (with counts)."""
    crops = np.asarray(crops, float)
    xy = np.asarray(xy, float)
    mask = np.asarray(mask, bool) & _finite_rows(xy) & _finite_rows(crops)[:, None]
    with np.errstate(invalid="ignore"):
        inside = _inside(crops, xy) & mask
    out = {}
    groups = {"body_ski": list(schema.groups["body"]) + list(schema.groups["ski"]), "ski_endpoints": list(SKI_ENDPOINTS)}
    for name, idx in groups.items():
        total, hits = int(mask[:, idx].sum()), int(inside[:, idx].sum())
        out[name] = {"inside": hits, "total": total, "pct": 100.0 * hits / total if total else None}
    return out


def centering_error(crops, boxes) -> dict:
    """Distance from box centre to crop centre / crop diagonal."""
    crops, boxes = np.asarray(crops, float), np.asarray(boxes, float)
    use = _finite_rows(crops) & _finite_rows(boxes)
    c, b = crops[use], boxes[use]
    d = np.linalg.norm((c[:, :2] + c[:, 2:]) / 2 - (b[:, :2] + b[:, 2:]) / 2, axis=1)
    e = d / np.linalg.norm(c[:, 2:] - c[:, :2], axis=1)
    return {"mean": float(e.mean()) if len(e) else None,
            "p95": float(np.percentile(e, 95)) if len(e) else None, "n_frames": int(len(e))}


def segments(timestamps, valid, max_gap_s: float) -> list[np.ndarray]:
    """Index runs with valid rows and no timestamp gap above ``max_gap_s``."""
    ts = np.asarray(timestamps, float)
    out, cur = [], []
    for i in range(len(ts)):
        if not valid[i] or (cur and ts[i] - ts[cur[-1]] > max_gap_s):
            if cur:
                out.append(np.array(cur))
            cur = [i] if valid[i] else []
        else:
            cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def crop_acceleration(crops, timestamps, frame_size, max_gap_s: float = 1.0) -> dict:
    """|second difference of the crop centre| / frame diagonal, in 1/s², within continuous segments.

    Uneven time steps are handled exactly: ``a = 2 (v1 - v0) / (dt0 + dt1)``.
    """
    crops = np.asarray(crops, float)
    ts = np.asarray(timestamps, float)
    centre = (crops[:, :2] + crops[:, 2:]) / 2
    diag = float(np.hypot(*frame_size))
    acc = []
    for seg in segments(ts, _finite_rows(crops), max_gap_s):
        if len(seg) < 3:
            continue
        c, t = centre[seg], ts[seg]
        dt = np.diff(t)
        v = np.diff(c, axis=0) / dt[:, None]
        a = 2 * (v[1:] - v[:-1]) / (dt[1:] + dt[:-1])[:, None]
        acc.append(np.linalg.norm(a, axis=1) / diag)
    a = np.concatenate(acc) if acc else np.array([])
    return {"mean": float(a.mean()) if len(a) else None,
            "p95": float(np.percentile(a, 95)) if len(a) else None, "n": int(len(a))}


def zoom_stats(crops, frame_size, timestamps, max_gap_s: float = 1.0) -> dict:
    """Mean crop-area fraction of the frame, and mean |d log(height) / dt|."""
    crops = np.asarray(crops, float)
    ts = np.asarray(timestamps, float)
    valid = _finite_rows(crops)
    w, h = crops[:, 2] - crops[:, 0], crops[:, 3] - crops[:, 1]
    area = (w * h)[valid] / (frame_size[0] * frame_size[1])
    rates = []
    for seg in segments(ts, valid, max_gap_s):
        if len(seg) >= 2:
            rates.append(np.abs(np.diff(np.log(h[seg])) / np.diff(ts[seg])))
    r = np.concatenate(rates) if rates else np.array([])
    return {"area_fraction_mean": float(area.mean()) if len(area) else None,
            "log_zoom_rate_mean": float(r.mean()) if len(r) else None}


def replay_summary(crops, timestamps, frame_size, target_boxes=None, target_label: str = "tracker",
                   landmarks: tuple | None = None, schema: Schema | None = None, max_gap_s: float = 1.0) -> dict:
    """All replay metrics. ``target_boxes`` are GT boxes when available, else tracker boxes
    (``target_label`` says which). ``landmarks`` = ``(xy[T,K,2], mask[T,K])``, NaN rows on
    frames without labels."""
    out = {
        "n_frames": int(len(crops)),
        "zoom": zoom_stats(crops, frame_size, timestamps, max_gap_s),
        "acceleration": crop_acceleration(crops, timestamps, frame_size, max_gap_s),
    }
    if target_boxes is not None:
        out["target"] = target_label
        out["box_containment"] = box_containment(crops, target_boxes)
        out["centering_error"] = centering_error(crops, target_boxes)
    if landmarks is not None:
        if schema is None:
            raise ValueError("landmark containment needs the schema")
        out["landmark_containment"] = landmark_containment(crops, *landmarks, schema)
    return out
