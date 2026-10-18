"""Top-down pose inference on records (plan Step 3.4).

The model enters only through ``heatmap_fn``: a plain function from a float32 batch
``[N, 3, 256, 192]`` to heatmaps ``[N, K, 64, 48]``. The real one wraps the torch model
(``skier.pose.model.torch_heatmap_fn``); tests pass a fake one. Everything else here
is numpy and OpenCV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from skier.data.records import PoseRecord
from skier.geometry import apply_affine, invert_affine
from skier.pose.heatmap import decode
from skier.pose.preprocess import make_pose_crop

HeatmapFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Predictions:
    sample_ids: list[str]
    xy: np.ndarray  # [N, K, 2] original-image pixels; NaN = missing
    scores: np.ndarray  # [N, K] heatmap maxima
    timing: dict = field(default_factory=dict)


def predict_boxes(images: Sequence[np.ndarray], boxes: Sequence, heatmap_fn: HeatmapFn,
                  expand: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """One batch: crops around ``boxes`` -> heatmaps -> decoded points in original pixels."""
    crops, Ms = zip(*(make_pose_crop(img, box, expand=expand) for img, box in zip(images, boxes)))
    heatmaps = np.asarray(heatmap_fn(np.stack(crops)))
    xy_in, scores = decode(heatmaps)
    xy = np.stack([apply_affine(invert_affine(M), p) for M, p in zip(Ms, xy_in)])
    return xy, scores


def predict_records(records: Sequence[PoseRecord], heatmap_fn: HeatmapFn, expand: float = 1.25,
                    batch_size: int = 32, boxes: Sequence | None = None) -> Predictions:
    """Predict every record, cropping around its reference box (or ``boxes`` if given).

    Excluded records are refused (they have no reference box); filter and report them first.
    """
    excluded = [r.sample_id for r in records if r.excluded]
    if excluded and boxes is None:
        raise ValueError(f"{len(excluded)} excluded records (e.g. {excluded[0]}); filter them first")
    boxes = list(boxes) if boxes is not None else [r.reference_box_xyxy for r in records]
    all_xy, all_scores = [], []
    t_load = t_model = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        t0 = time.perf_counter()
        images = []
        for r in batch:
            img = cv2.imread(str(r.image_path))
            if img is None:
                raise FileNotFoundError(f"cannot read {r.image_path}")
            images.append(img)
        t1 = time.perf_counter()
        xy, scores = predict_boxes(images, boxes[start:start + batch_size], heatmap_fn, expand)
        t2 = time.perf_counter()
        t_load, t_model = t_load + (t1 - t0), t_model + (t2 - t1)
        all_xy.append(xy)
        all_scores.append(scores)
    n = len(records)
    return Predictions(
        sample_ids=[r.sample_id for r in records],
        xy=np.concatenate(all_xy) if all_xy else np.zeros((0, 0, 2)),
        scores=np.concatenate(all_scores) if all_scores else np.zeros((0, 0)),
        timing={"n_images": n, "load_s": round(t_load, 3), "crop_model_decode_s": round(t_model, 3),
                "ms_per_image_total": round(1000 * (t_load + t_model) / n, 2) if n else None},
    )
