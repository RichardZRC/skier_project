"""Pose-crop preprocessing (plan Step 3.3), matching MMPose's top-down pipeline.

Order, as in MMPose: pad the tight box by ``expand`` (1.25) once, then grow the short
side to the 192:256 aspect, then warp (linear, black border), BGR -> RGB, normalize with
the ImageNet mean/std, channels first. The model is called directly (``PoseNet``), so
MMPose's data preprocessor never normalizes a second time.
"""

from __future__ import annotations

import cv2
import numpy as np

from skier.geometry import crop_affine, expand_box, fit_aspect

INPUT_W, INPUT_H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def pose_crop_box(ref_box_xyxy, expand: float = 1.25, scale: float = 1.0) -> np.ndarray:
    """The crop region in original pixels. ``scale`` > 1 zooms out (augmentation)."""
    return fit_aspect(expand_box(ref_box_xyxy, expand * scale), INPUT_W / INPUT_H)


def normalize(crop_bgr: np.ndarray) -> np.ndarray:
    """``H x W x 3`` uint8 BGR -> ``3 x H x W`` float32, RGB, ``(v - mean) / std``."""
    rgb = crop_bgr[..., ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))


def make_pose_crop(image_bgr: np.ndarray, ref_box_xyxy, expand: float = 1.25, rot_deg: float = 0.0,
                   scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Model input ``[3, 256, 192]`` float32 and the 2x3 affine ``M`` (original px -> input px)."""
    M = crop_affine(pose_crop_box(ref_box_xyxy, expand, scale), INPUT_W, INPUT_H, rot_deg)
    crop = cv2.warpAffine(image_bgr, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return normalize(crop), M
