"""The SkiTB toolkit's official tracking metrics (plan Step 8), ported from the vendored toolkit.

Source: SkiTB-toolkit ``b402f9a``, ``toolkit/experiments/skitb.py`` (``report``, ``_calc_metrics``,
``_calc_curves_robustness``) and ``toolkit/utils/metrics.py`` (``rect_iou``, ``_intersection``).
Ported, not imported, because the toolkit's loader doesn't run here: numpy 1.26 refuses
``np.loadtxt(delimiter='\\n')``, and its SC mode lists ``ST/`` instead of ``SC/``. The
cross-check test runs the real ``report()`` on synthetic clips.

Conventions, as in the toolkit:
- Boxes are ``x, y, w, h``. Frame 0 is forced to the GT box.
- Confidences sit on the result file's ``%.8f`` grid; our init row gets 1.0, as the
  toolkit's base tracker gives it.
- Thresholds are 0.00 … 0.99 (float32).
- At a threshold ``t``: precision = Σ IoU over frames with confidence ≥ t (visible or not) ÷
  their number; recall = the same sum ÷ the number of visible frames; F = 2PR / (P + R).
- A clip's F is the max of its curve. The overall F is the max of the mean curve over clips
  (one shared threshold).
- GSR uses visible frames only.
SiamFC's scores are about 1e-5, so every t ≥ 0.01 keeps only frame 0, and its F is the
value at t = 0 (all frames).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

CONF_THRESHOLDS = np.arange(0, 100, dtype=np.float32) / 100
NBINS_GSR = 51


def _intersection(rects1, rects2):
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2], rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3], rects2[..., 1] + rects2[..., 3])
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)
    return np.stack([x1, y1, w, h]).T


def rect_iou(rects1, rects2):
    assert rects1.shape == rects2.shape
    areas_inter = np.prod(_intersection(rects1, rects2)[..., 2:], axis=-1)
    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter
    eps = np.finfo(float).eps
    return np.clip(areas_inter / (areas_union + eps), 0.0, 1.0)


def _ious(boxes, anno) -> np.ndarray:
    """``_calc_metrics``: per-frame IoU, skipping frames whose GT is all negative."""
    out = [rect_iou(np.array([b]), np.array([a]))[0] for b, a in zip(boxes, anno)
           if not (a[0] < 0 and a[1] < 0 and a[2] < 0 and a[3] < 0)]
    return np.array(out)


def gsr_curve(ious: np.ndarray) -> np.ndarray:
    """``_calc_curves_robustness``: per threshold, the share of frames before the first IoU ≤ threshold."""
    n = ious.shape[0]
    curve = np.zeros(NBINS_GSR)
    for i, th in enumerate(np.linspace(0, 0.5, NBINS_GSR)):
        below = np.flatnonzero(ious <= th)
        curve[i] = float(below[0]) / n if len(below) else 1.0
    return curve


def to_toolkit(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Our track rows (xyxy, NaN score on the init row) -> the toolkit's ``(boxes xywh %.3f, confidences %.8f)``."""
    boxes = np.array([[r["x1"], r["y1"], r["x2"] - r["x1"], r["y2"] - r["y1"]] for r in rows], dtype=np.float64)
    conf = np.array([1.0 if r["status"] == "init" else r["score"] for r in rows], dtype=np.float64)
    if not np.isfinite(conf).all():
        raise ValueError("a tracked row has no confidence; the toolkit needs one per frame")
    return np.round(boxes, 3), np.array([float(f"{c:.8f}") for c in conf])


def clip_metrics(boxes: np.ndarray, confidences: np.ndarray, anno: np.ndarray, visibilities: np.ndarray) -> dict:
    """The toolkit's per-clip curves and scores for one SC clip (``boxes``/``anno`` xywh, one row per GT row)."""
    boxes = np.array(boxes, dtype=np.float64)
    anno = np.asarray(anno, dtype=np.float64)
    visibilities = np.asarray(visibilities)
    confidences = np.asarray(confidences, dtype=np.float64)
    if not len(boxes) == len(confidences) == len(anno) == len(visibilities):
        raise ValueError(f"lengths differ: boxes {len(boxes)}, confidences {len(confidences)}, anno {len(anno)}, "
                         f"visibilities {len(visibilities)}")
    boxes[0] = anno[0, :]
    gsr = gsr_curve(_ious(boxes[visibilities > 0], anno[visibilities > 0]))
    n_t = len(CONF_THRESHOLDS)
    p, r, f = np.zeros(n_t), np.zeros(n_t), np.zeros(n_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        for ct, t in enumerate(CONF_THRESHOLDS):
            keep = confidences >= t
            s = np.sum(_ious(boxes[keep], anno[keep]))
            p[ct] = s / np.sum(keep)
            r[ct] = s / np.sum(visibilities > 0)
            f[ct] = (2.0 * p[ct] * r[ct]) / (p[ct] + r[ct])
    best = int(np.argmax(f))
    return {"f_curve": f, "p_curve": p, "r_curve": r, "gsr_curve": gsr,
            "f_score": float(np.max(f)), "precision": float(p[best]), "recall": float(r[best]),
            "f_score_thresh": float(CONF_THRESHOLDS[best]), "gsr_score": float(np.mean(gsr))}


def overall(clips: Mapping[str, dict], names: Sequence[str] | None = None) -> dict:
    """The toolkit's ``overall`` block: the max of the mean F curve, with P and R at the same threshold."""
    names = list(clips) if names is None else list(names)
    f = np.mean([clips[n]["f_curve"] for n in names], axis=0)
    p = np.mean([clips[n]["p_curve"] for n in names], axis=0)
    r = np.mean([clips[n]["r_curve"] for n in names], axis=0)
    gsr = np.mean([clips[n]["gsr_curve"] for n in names], axis=0)
    best = int(np.argmax(f))
    return {"f_score": float(np.max(f)), "precision": float(p[best]), "recall": float(r[best]),
            "f_score_thresh": float(CONF_THRESHOLDS[best]), "gsr_score": float(np.mean(gsr)), "n_clips": len(names)}


def skitb_clip_inputs(clip, rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(boxes, confidences, anno, visibilities)`` for one of our SkiTB clips; anno is read raw (xywh)."""
    from skier.data.skitb import BOXES_FILE, SINGLE_CAMERA_DIR, parse_boxes_xywh

    anno = parse_boxes_xywh(clip.frames_dir.parent / SINGLE_CAMERA_DIR / clip.clip_id / BOXES_FILE)
    if rows[0]["frame_index"] != 0 or len(rows) != len(anno):
        raise ValueError(f"{clip.sequence_id}/{clip.clip_id}: the toolkit scores every GT row from frame 0; the track "
                         f"has {len(rows)} rows from frame {rows[0]['frame_index']} for {len(anno)} GT rows")
    boxes, conf = to_toolkit(rows)
    return boxes, conf, anno, clip.visible.astype(np.float64)
