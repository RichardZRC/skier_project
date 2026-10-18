"""Training augmentation (plan Step 5.2; Step 6 adds crop jitter, blur and occlusion).

Every geometric augmentation goes through the single crop transform, so each training
sample is warped exactly once. A horizontal flip mirrors the crop (``x -> W - 1 - x``,
MMPose's convention) **and** swaps left/right labels with the schema's flip index:
after mirroring, the skier's right ski looks like a left ski and must be labeled so.
No half-body augmentation; no test-time flip.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from skier.geometry import crop_affine
from skier.pose.preprocess import INPUT_H, INPUT_W, pose_crop_box
from skier.schema import Schema

FLIP = np.array([[-1.0, 0.0, INPUT_W - 1.0], [0.0, 1.0, 0.0]])  # mirror in model-input space


@dataclass(frozen=True)
class AugConfig:
    flip_p: float = 0.5
    rot_deg: float = 15.0  # rotation ~ U(-rot_deg, rot_deg)
    scale_range: tuple[float, float] = (0.8, 1.2)  # crop size factor; > 1 zooms out
    # Step 6 hooks (plan §11). Off (None) in E1; setting them before Step 6 raises.
    jitter: dict | None = None
    blur: dict | None = None
    occlusion: dict | None = None

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AugConfig":
        raw = dict(raw or {})
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown aug keys: {sorted(unknown)}")
        if "scale_range" in raw:
            raw["scale_range"] = tuple(float(v) for v in raw["scale_range"])
        return cls(**raw)

    @classmethod
    def none(cls) -> "AugConfig":
        return cls(flip_p=0.0, rot_deg=0.0, scale_range=(1.0, 1.0))


@dataclass(frozen=True)
class AugParams:
    flip: bool = False
    rot_deg: float = 0.0
    scale: float = 1.0

    @classmethod
    def identity(cls) -> "AugParams":
        return cls()


def sample_aug(rng: np.random.Generator, cfg: AugConfig) -> AugParams:
    if cfg.jitter or cfg.blur or cfg.occlusion:
        raise NotImplementedError("crop jitter / blur / occlusion are Step 6 (plan §11)")
    lo, hi = cfg.scale_range
    return AugParams(
        flip=bool(rng.random() < cfg.flip_p),
        rot_deg=float(rng.uniform(-cfg.rot_deg, cfg.rot_deg)) if cfg.rot_deg > 0 else 0.0,
        scale=float(rng.uniform(lo, hi)) if hi > lo else float(lo),
    )


def crop_transform(ref_box_xyxy, params: AugParams, expand: float = 1.25) -> np.ndarray:
    """Original pixels -> model input, including the flip (one 2x3 matrix)."""
    M = crop_affine(pose_crop_box(ref_box_xyxy, expand, params.scale), INPUT_W, INPUT_H, params.rot_deg)
    if params.flip:
        M = compose(FLIP, M)
    return M


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The 2x3 affine ``A ∘ B`` (apply B, then A)."""
    h = np.array([[0.0, 0.0, 1.0]])
    return (np.vstack([A, h]) @ np.vstack([B, h]))[:2]


def flip_labels(xy: np.ndarray, reliable: np.ndarray, schema: Schema) -> tuple[np.ndarray, np.ndarray]:
    """Swap left/right labels after mirroring: new label ``i`` takes old label ``flip_index[i]``."""
    idx = list(schema.flip_index)
    return np.asarray(xy)[idx], np.asarray(reliable)[idx]
