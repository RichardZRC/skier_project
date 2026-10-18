"""Training augmentation (plan Steps 5.2 and 6).

Geometry (flip, rotation, scale and the Step 6 crop jitter) goes through the single crop
transform, so each training sample is warped exactly once. A horizontal flip mirrors the
crop (``x -> W - 1 - x``, MMPose's convention) **and** swaps left/right labels with the
schema's flip index: after mirroring, the skier's right ski looks like a left ski and must
be labeled so. No half-body augmentation; no test-time flip.

Step 6 also adds two pixel augmentations, applied to the warped crop in this order:
motion blur, then one occlusion patch. Neither changes the targets or the weights. A
point under the patch stays supervised, because its true position is known; unreliable
labels stay masked as always (D16).

Paired design: ``sample_aug`` maps one fixed-length vector of uniforms onto the
parameters, slot by slot. So for the same ``(seed, epoch, index)`` every experiment gets
the same flip, rotation and scale. Experiments that share an augmentation also get the
same values for it (E2 and E4 the same jitter; E3 and E4 the same blur and occlusion).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np

from skier.geometry import box_center_size, box_from_center_size, crop_affine
from skier.pose.preprocess import INPUT_H, INPUT_W, pose_crop_box
from skier.schema import Schema

FLIP = np.array([[-1.0, 0.0, INPUT_W - 1.0], [0.0, 1.0, 0.0]])  # mirror in model-input space

# Slots of the uniform vector drawn by ``sample_aug``. Never reorder: that would change
# every experiment's augmentation stream.
N_DRAWS = 18
_FLIP, _ROT, _SCALE = 0, 1, 2
_J_ON, _J_DX, _J_DY, _J_SCALE = 3, 4, 5, 6
_B_ON, _B_LEN, _B_ANGLE = 7, 8, 9
_O_ON, _O_AREA, _O_ASPECT, _O_X, _O_Y = 10, 11, 12, 13, 14
_O_COLOR = slice(15, 18)


def _sub_from_dict(cls, raw: dict | None, name: str):
    raw = dict(raw or {})
    unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown aug.{name} keys: {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def _check_p(p: float, name: str) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"aug.{name}.p must be in [0, 1], got {p}")


def _check_range(r, name: str, positive: bool = True) -> None:
    if len(r) != 2 or r[0] > r[1] or (positive and r[0] <= 0):
        raise ValueError(f"aug.{name} must be an increasing [low, high] pair{' of positive numbers' if positive else ''}, got {r}")


@dataclass(frozen=True)
class JitterConfig:
    """Crop jitter (E2, E4): the reference box's center moves by up to ``±shift`` of its
    width and height (uniform, per axis), and its size by an extra factor in ``scale_range``."""

    p: float = 1.0
    shift: float = 0.10
    scale_range: tuple[float, float] = (0.9, 1.1)

    def __post_init__(self):
        _check_p(self.p, "jitter")
        if not 0.0 <= self.shift < 0.5:
            raise ValueError(f"aug.jitter.shift must be in [0, 0.5), got {self.shift}")
        _check_range(self.scale_range, "jitter.scale_range")


@dataclass(frozen=True)
class BlurConfig:
    """Linear motion blur at model resolution (E3, E4): a random direction and a kernel
    length from ``lengths``. Odd lengths only: an even kernel has no center pixel and would
    shift the image half a pixel against the labels."""

    p: float = 0.5
    lengths: tuple[int, ...] = (3, 5, 7, 9)

    def __post_init__(self):
        _check_p(self.p, "blur")
        if not self.lengths or any(not isinstance(n, int) or n < 1 or n % 2 == 0 for n in self.lengths):
            raise ValueError(f"aug.blur.lengths must be odd positive integers, got {self.lengths}")


@dataclass(frozen=True)
class OcclusionConfig:
    """One solid random-color rectangle (E3, E4) covering ``area_range`` of the crop, with
    width/height ratio log-uniform in ``aspect_range``, placed fully inside the crop."""

    p: float = 0.5
    area_range: tuple[float, float] = (0.05, 0.15)
    aspect_range: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self):
        _check_p(self.p, "occlusion")
        _check_range(self.area_range, "occlusion.area_range")
        _check_range(self.aspect_range, "occlusion.aspect_range")
        a_max, area = self.area_range[1], INPUT_W * INPUT_H
        if a_max * area * self.aspect_range[1] > INPUT_W ** 2 or a_max * area / self.aspect_range[0] > INPUT_H ** 2:
            raise ValueError("aug.occlusion: the largest patch would not fit inside the crop")


_SUB_CONFIGS = {"jitter": JitterConfig, "blur": BlurConfig, "occlusion": OcclusionConfig}


@dataclass(frozen=True)
class AugConfig:
    flip_p: float = 0.5
    rot_deg: float = 15.0  # rotation ~ U(-rot_deg, rot_deg)
    scale_range: tuple[float, float] = (0.8, 1.2)  # crop size factor; > 1 zooms out
    # Step 6 (plan §11). None = off (E1). A dict is converted to the typed config.
    jitter: JitterConfig | None = None
    blur: BlurConfig | None = None
    occlusion: OcclusionConfig | None = None

    def __post_init__(self):
        for name, cls in _SUB_CONFIGS.items():
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, _sub_from_dict(cls, value, name))
            elif value is not None and not isinstance(value, cls):
                raise ValueError(f"aug.{name} must be a mapping or null, got {value!r}")

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
    shift: tuple[float, float] = (0.0, 0.0)  # jitter, in reference-box widths/heights
    jitter_scale: float = 1.0
    blur_length: int = 0  # 0 = no blur
    blur_angle: float = 0.0  # degrees
    occlusion: tuple[int, int, int, int] | None = None  # x, y, w, h in model-input pixels
    occlusion_color: tuple[int, int, int] = (0, 0, 0)  # BGR

    @classmethod
    def identity(cls) -> "AugParams":
        return cls()


def sample_aug(rng: np.random.Generator, cfg: AugConfig) -> AugParams:
    """Parameters for one sample from one fixed-length uniform vector (see the module doc)."""
    u = rng.random(N_DRAWS)
    lo, hi = cfg.scale_range
    kw = dict(flip=bool(u[_FLIP] < cfg.flip_p), rot_deg=float((2 * u[_ROT] - 1) * cfg.rot_deg),
              scale=float(lo + u[_SCALE] * (hi - lo)))
    j = cfg.jitter
    if j is not None and u[_J_ON] < j.p:
        jlo, jhi = j.scale_range
        kw.update(shift=(float((2 * u[_J_DX] - 1) * j.shift), float((2 * u[_J_DY] - 1) * j.shift)),
                  jitter_scale=float(jlo + u[_J_SCALE] * (jhi - jlo)))
    b = cfg.blur
    if b is not None and u[_B_ON] < b.p:
        kw.update(blur_length=b.lengths[min(int(u[_B_LEN] * len(b.lengths)), len(b.lengths) - 1)],
                  blur_angle=float(180.0 * u[_B_ANGLE]))
    o = cfg.occlusion
    if o is not None and u[_O_ON] < o.p:
        area = (o.area_range[0] + u[_O_AREA] * (o.area_range[1] - o.area_range[0])) * INPUT_W * INPUT_H
        la, lb = math.log(o.aspect_range[0]), math.log(o.aspect_range[1])
        aspect = math.exp(la + u[_O_ASPECT] * (lb - la))
        w = min(INPUT_W, max(1, round(math.sqrt(area * aspect))))
        h = min(INPUT_H, max(1, round(math.sqrt(area / aspect))))
        x = min(INPUT_W - w, int(u[_O_X] * (INPUT_W - w + 1)))
        y = min(INPUT_H - h, int(u[_O_Y] * (INPUT_H - h + 1)))
        kw.update(occlusion=(x, y, w, h), occlusion_color=tuple(int(255.999 * c) for c in u[_O_COLOR]))
    return AugParams(**kw)


def jitter_box(ref_box_xyxy, dx: float, dy: float, scale: float) -> np.ndarray:
    """Move the box center by ``(dx * w, dy * h)`` and scale its size by ``scale``."""
    c, s = box_center_size(ref_box_xyxy)
    return box_from_center_size(c + np.array([dx, dy]) * s, s * scale)


def crop_transform(ref_box_xyxy, params: AugParams, expand: float = 1.25) -> np.ndarray:
    """Original pixels -> model input, including jitter and flip (one 2x3 matrix)."""
    box = ref_box_xyxy
    if params.shift != (0.0, 0.0) or params.jitter_scale != 1.0:
        box = jitter_box(ref_box_xyxy, params.shift[0], params.shift[1], params.jitter_scale)
    M = crop_affine(pose_crop_box(box, expand, params.scale), INPUT_W, INPUT_H, params.rot_deg)
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


def blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """``length x length`` line kernel through the exact center, summing to 1."""
    if length < 1 or length % 2 == 0:
        raise ValueError(f"blur length must be odd and positive, got {length}")
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    c = (length - 1) / 2.0  # the center pixel, so the rotation can't shift the kernel
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((c, c), angle_deg, 1.0), (length, length), flags=cv2.INTER_LINEAR)
    return k / k.sum()


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return img
    return cv2.filter2D(img, -1, blur_kernel(length, angle_deg), borderType=cv2.BORDER_REFLECT)


def occlude(img: np.ndarray, rect: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > img.shape[1] or y + h > img.shape[0]:
        raise ValueError(f"occlusion patch {rect} is not inside the {img.shape[1]}x{img.shape[0]} crop")
    out = img.copy()
    out[y:y + h, x:x + w] = color
    return out


def apply_pixel_aug(crop_bgr: np.ndarray, params: AugParams) -> np.ndarray:
    """Blur, then occlusion, on the warped ``uint8`` crop. Labels are untouched."""
    out = motion_blur(crop_bgr, params.blur_length, params.blur_angle)
    if params.occlusion is not None:
        out = occlude(out, params.occlusion, params.occlusion_color)
    return out
