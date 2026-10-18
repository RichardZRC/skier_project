"""Training samples (plan Step 5.1) and the optional crop cache.

``make_sample`` and ``PoseDataset`` are numpy-only (tested without torch). PyTorch's
``DataLoader`` accepts any object with ``__len__``/``__getitem__``, so only ``collate``
(numpy -> tensors) needs torch, imported lazily (D13).

``weight = reliable AND inside the augmented crop``. Training samples whose weights are
all zero are dropped in ``collate`` (the plan's "skipped"). Evaluation is different:
reliable points a crop cuts off still count as misses (handled in the metrics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from skier.data.records import PoseRecord
from skier.data.ski2dpose_raw import safe_name
from skier.geometry import apply_affine, box_center_size, box_from_center_size
from skier.io_utils import ensure_dir, read_json, write_json
from skier.pose.augment import AugConfig, AugParams, compose, crop_transform, flip_labels, sample_aug
from skier.pose.heatmap import encode
from skier.pose.preprocess import INPUT_H, INPUT_W, normalize, pose_crop_box
from skier.schema import Schema


@dataclass
class Sample:
    x: np.ndarray  # [3, 256, 192] float32, normalized RGB
    target: np.ndarray  # [K, 64, 48] float32
    weight: np.ndarray  # [K] float32
    M: np.ndarray  # [2, 3] original pixels -> model input (flip included)
    sample_id: str
    keypoints_input: np.ndarray  # [K, 2] labels in model-input pixels, after any left/right swap


def read_image(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    return img


def make_sample(record: PoseRecord, image_bgr: np.ndarray, params: AugParams, schema: Schema,
                offset: tuple[float, float] = (0.0, 0.0), expand: float = 1.25, sigma: float = 2.0) -> Sample:
    """One training/eval sample. ``offset`` = where ``image_bgr`` sits in the original image (cache)."""
    if record.reference_box_xyxy is None:
        raise ValueError(f"{record.sample_id}: excluded record ({record.excluded_reason})")
    M = crop_transform(record.reference_box_xyxy, params, expand)
    M_img = M if offset == (0.0, 0.0) else compose(M, np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]]]))
    crop = cv2.warpAffine(image_bgr, M_img, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    xy = apply_affine(M, record.keypoints_xy)
    reliable = record.reliable
    if params.flip:
        xy, reliable = flip_labels(xy, reliable, schema)
    target, weight = encode(xy, reliable, sigma=sigma)
    return Sample(normalize(crop), target, weight, M, record.sample_id, xy)


class PoseDataset:
    """Map-style dataset of numpy samples. Randomness depends only on (seed, epoch, index),
    so results don't depend on the number of DataLoader workers."""

    def __init__(self, records: Sequence[PoseRecord], schema: Schema, train: bool, aug_cfg: AugConfig | None = None,
                 seed: int = 230, image_loader: Callable = read_image, cache: "CropCache | None" = None,
                 expand: float = 1.25, sigma: float = 2.0):
        excluded = [r.sample_id for r in records if r.excluded]
        if excluded:
            raise ValueError(f"{len(excluded)} excluded records (e.g. {excluded[0]}); filter them first")
        self.records = list(records)
        self.schema, self.train = schema, train
        self.aug_cfg = aug_cfg or AugConfig()
        self.seed, self.epoch = seed, 0
        self.image_loader, self.cache = image_loader, cache
        self.expand, self.sigma = expand, sigma
        if cache is not None and train:
            cache.check_covers(self.aug_cfg, expand)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def params_for(self, index: int) -> AugParams:
        if not self.train:
            return AugParams.identity()
        return sample_aug(np.random.default_rng([self.seed, self.epoch, index]), self.aug_cfg)

    def __getitem__(self, index: int) -> dict:
        r = self.records[index]
        if self.cache is not None:
            image, offset = self.cache.load(r)
        else:
            image, offset = self.image_loader(r.image_path), (0.0, 0.0)
        s = make_sample(r, image, self.params_for(index), self.schema, offset, self.expand, self.sigma)
        return {"x": s.x, "target": s.target, "weight": s.weight, "M": s.M, "sample_id": s.sample_id, "index": index}


def collate(samples: list[dict]):
    """Stack numpy samples into tensors, dropping samples with no valid points. ``None`` if none are left."""
    import torch

    keep = [s for s in samples if s["weight"].sum() > 0]
    if not keep:
        return None
    return {
        "x": torch.from_numpy(np.stack([s["x"] for s in keep])),
        "target": torch.from_numpy(np.stack([s["target"] for s in keep])),
        "weight": torch.from_numpy(np.stack([s["weight"] for s in keep])),
        "M": np.stack([s["M"] for s in keep]),
        "sample_ids": [s["sample_id"] for s in keep],
        "n_dropped": len(samples) - len(keep),
    }


# --- optional crop cache ------------------------------------------------------------------

def required_region(ref_box_xyxy, aug: AugConfig, expand: float, margin_px: float = 4.0) -> np.ndarray:
    """Original-image region that contains every crop the augmentation can produce."""
    if aug.jitter:
        raise NotImplementedError("extend required_region for crop jitter in Step 6")
    crop = pose_crop_box(ref_box_xyxy, expand, max(1.0, *aug.scale_range))
    c, (w, h) = box_center_size(crop)
    th = math.radians(abs(aug.rot_deg))
    rw = w * math.cos(th) + h * math.sin(th)  # bounding box of the rotated crop
    rh = w * math.sin(th) + h * math.cos(th)
    return box_from_center_size(c, (rw + 2 * margin_px, rh + 2 * margin_px))


class CropCache:
    """Each image cropped once (lossless PNG) to its ``required_region``; offsets kept so
    coordinates stay in original-image space."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = read_json(self.root / "index.json")
        self.offsets = {k: tuple(v) for k, v in meta["offsets"].items()}
        self.bounds = meta["bounds"]

    @classmethod
    def build(cls, root: str | Path, records: Sequence[PoseRecord], aug: AugConfig, expand: float = 1.25,
              image_loader: Callable = read_image) -> "CropCache":
        root = ensure_dir(root)
        offsets = {}
        for r in records:
            img = image_loader(r.image_path)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = required_region(r.reference_box_xyxy, aug, expand)
            x0, y0 = max(0, math.floor(x1)), max(0, math.floor(y1))
            xe, ye = min(w, math.ceil(x2)), min(h, math.ceil(y2))
            cv2.imwrite(str(root / f"{safe_name(r.sample_id)}.png"), img[y0:ye, x0:xe], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            offsets[r.sample_id] = [x0, y0]
        bounds = {"expand": expand, "scale_max": max(1.0, *aug.scale_range), "rot_deg": abs(aug.rot_deg)}
        write_json({"offsets": offsets, "bounds": bounds}, root / "index.json")
        return cls(root)

    def check_covers(self, aug: AugConfig, expand: float) -> None:
        b = self.bounds
        if aug.jitter or expand > b["expand"] + 1e-9 or max(1.0, *aug.scale_range) > b["scale_max"] + 1e-9 \
                or abs(aug.rot_deg) > b["rot_deg"] + 1e-9:
            raise ValueError(f"crop cache built for {b} doesn't cover this augmentation; rebuild it")

    def load(self, record: PoseRecord) -> tuple[np.ndarray, tuple[float, float]]:
        if record.sample_id not in self.offsets:
            raise KeyError(f"{record.sample_id} not in the crop cache")
        img = read_image(self.root / f"{safe_name(record.sample_id)}.png")
        ox, oy = self.offsets[record.sample_id]
        return img, (float(ox), float(oy))
