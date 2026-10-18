"""Box and crop geometry (plan Step 2.1).

Conventions: boxes are ``xyxy`` floats in original-image pixels; pixel centers sit
at integer coordinates (OpenCV). ``crop_affine`` matches MMPose's non-UDP
``get_warp_matrix`` (box center -> output center, uniform scale), the convention the
COCO HRNet checkpoint was trained with.
"""

from __future__ import annotations

import cv2
import numpy as np

from skier.schema import Schema


def box_from_points(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tight ``xyxy`` box of ``xy[mask]``. Raises if fewer than 2 points are selected."""
    pts = np.asarray(xy, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(pts) < 2:
        raise ValueError(f"need at least 2 points for a box, got {len(pts)}")
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def body_ski_mask(schema: Schema) -> np.ndarray:
    """Body + ski landmarks (poles excluded): the points that define the reference box."""
    mask = np.zeros(schema.num_keypoints, dtype=bool)
    mask[list(schema.groups["body"]) + list(schema.groups["ski"])] = True
    return mask


def box_diagonal(xyxy) -> float:
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float64)
    return float(np.hypot(x2 - x1, y2 - y1))


def reference_box(xy_px: np.ndarray, reliable: np.ndarray, schema: Schema) -> np.ndarray | None:
    """Box of reliable body + ski points (poles excluded); its diagonal is the PCK normalizer.

    Returns None if fewer than 2 such points exist or the diagonal is under 1 px.
    """
    xy_px = np.asarray(xy_px, dtype=np.float64)
    mask = body_ski_mask(schema) & np.asarray(reliable, dtype=bool) & np.isfinite(xy_px).all(axis=1)
    if mask.sum() < 2:
        return None
    box = box_from_points(xy_px, mask)
    if box_diagonal(box) < 1.0:
        return None
    return box


def box_center_size(xyxy) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(xyxy, dtype=np.float64)
    return (b[:2] + b[2:]) / 2, b[2:] - b[:2]


def box_from_center_size(center, size) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(size, dtype=np.float64)
    return np.concatenate([c - s / 2, c + s / 2])


def expand_box(xyxy, factor: float) -> np.ndarray:
    """Scale width and height by ``factor`` about the box center."""
    c, s = box_center_size(xyxy)
    return box_from_center_size(c, s * factor)


def fit_aspect(xyxy, w_over_h: float) -> np.ndarray:
    """Grow the short side (about the center) until width / height == ``w_over_h``.

    Never shrinks and never stretches: the result always contains the input box.
    """
    if w_over_h <= 0:
        raise ValueError("w_over_h must be positive")
    c, (w, h) = box_center_size(xyxy)
    if w < h * w_over_h:
        w = h * w_over_h
    else:
        h = w / w_over_h
    return box_from_center_size(c, (w, h))


def crop_affine(xyxy, out_w: int, out_h: int, rot_deg: float = 0.0) -> np.ndarray:
    """2x3 matrix mapping the box onto an ``out_w x out_h`` image.

    The box center goes to ``(out_w / 2, out_h / 2)`` with uniform scale
    ``out_w / box_w``. The box must already have the output's aspect ratio (use
    ``fit_aspect``), so a crop is never stretched. ``rot_deg`` rotates the content
    about the box center; positive is counter-clockwise on screen, as in
    ``cv2.getRotationMatrix2D``. A box partly outside the image is fine:
    ``cv2.warpAffine`` pads.
    """
    c, (w, h) = box_center_size(xyxy)
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate box {np.asarray(xyxy).tolist()}")
    if not np.isclose(w / h, out_w / out_h, rtol=1e-6):
        raise ValueError(f"box aspect {w / h:.6f} != output aspect {out_w / out_h:.6f}; call fit_aspect first")
    scale = out_w / w
    theta = np.deg2rad(rot_deg)
    cos, sin = np.cos(theta) * scale, np.sin(theta) * scale
    # y points down in images, so an on-screen counter-clockwise rotation has +sin in
    # the x row and -sin in the y row (the same matrix cv2.getRotationMatrix2D builds).
    linear = np.array([[cos, sin], [-sin, cos]])
    translation = np.array([out_w / 2, out_h / 2]) - linear @ c
    return np.hstack([linear, translation[:, None]])


def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float64))


def apply_affine(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Map ``[..., 2]`` points through a 2x3 affine. NaN points stay NaN."""
    M = np.asarray(M, dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ M[:, :2].T + M[:, 2]


def iou_xyxy(a, b) -> float:
    """Standard IoU on continuous coordinates; 0 for disjoint, empty or non-finite boxes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
