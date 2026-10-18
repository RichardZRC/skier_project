"""Tracking metrics (plan Step 2.5).

- IoU is averaged over **visible annotated** frames, excluding the initialization
  frame (the tracker is given the true box there). A missing tracker box counts as 0.
- Pooled mean = over all such frames; per-clip mean = average of clip means.
- Success curve: fraction of frames with IoU > threshold, thresholds 0:0.05:1; AUC = its mean.
- Failure episode (our diagnostic, not the toolkit's metric): IoU < 0.1 on visible frames
  for at least 0.5 s. Invisible frames neither break nor end an episode; its duration is
  ``(last_frame - first_frame + 1) / fps`` over its failing visible frames.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from skier.geometry import iou_xyxy

SUCCESS_THRESHOLDS = np.round(np.linspace(0.0, 1.0, 21), 2)


def frame_ious(pred_xyxy, gt_xyxy) -> np.ndarray:
    pred = np.asarray(pred_xyxy, dtype=np.float64)
    gt = np.asarray(gt_xyxy, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError(f"boxes must both be [T, 4], got {pred.shape} and {gt.shape}")
    return np.array([iou_xyxy(p, g) for p, g in zip(pred, gt)])


def scored_mask(visible, init_index: int | None = 0) -> np.ndarray:
    """Frames that count: visible and not the initialization frame."""
    m = np.asarray(visible, dtype=bool).copy()
    if init_index is not None:
        m[init_index] = False
    return m


def clip_iou(pred_xyxy, gt_xyxy, visible, init_index: int | None = 0) -> dict:
    ious = frame_ious(pred_xyxy, gt_xyxy)
    m = scored_mask(visible, init_index)
    return {"mean_iou": float(ious[m].mean()) if m.any() else None, "n_frames": int(m.sum()), "ious": ious[m]}


def pooled_iou(clips: Sequence[dict]) -> dict:
    """``clips``: outputs of ``clip_iou``. Returns pooled and per-clip means."""
    used = [c for c in clips if c["n_frames"] > 0]
    all_ious = np.concatenate([c["ious"] for c in used]) if used else np.array([])
    return {
        "pooled_mean_iou": float(all_ious.mean()) if len(all_ious) else None,
        "per_clip_mean_iou": float(np.mean([c["mean_iou"] for c in used])) if used else None,
        "n_frames": int(len(all_ious)),
        "n_clips": len(used),
    }


def success_curve(ious, thresholds=SUCCESS_THRESHOLDS) -> dict:
    ious = np.asarray(ious, dtype=np.float64)
    if len(ious) == 0:
        return {"thresholds": list(map(float, thresholds)), "success": None, "auc": None}
    success = np.array([(ious > t).mean() for t in thresholds])
    return {"thresholds": list(map(float, thresholds)), "success": success.tolist(), "auc": float(success.mean())}


def failure_episodes(ious, visible, frame_indices, fps: float, iou_thr: float = 0.1,
                     min_duration_s: float = 0.5, init_index: int | None = 0) -> list[dict]:
    ious = np.asarray(ious, dtype=np.float64)
    frames = np.asarray(frame_indices)
    use = scored_mask(visible, init_index)
    episodes, start, last = [], None, None

    def close():
        if start is not None:
            duration = (last - start + 1) / fps
            if duration >= min_duration_s - 1e-9:
                episodes.append({"start_frame": int(start), "end_frame": int(last), "duration_s": round(duration, 4)})

    for i in range(len(ious)):
        if not use[i]:
            continue  # invisible / init: neither breaks nor extends
        if ious[i] < iou_thr:
            if start is None:
                start = frames[i]
            last = frames[i]
        else:
            close()
            start = last = None
    close()
    return episodes
