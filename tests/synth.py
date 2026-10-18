"""Synthetic inputs for tests: labels JSON, blank images, moving-square videos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_annotation(rng: np.random.Generator, n_unreliable: int = 2) -> list[list[float]]:
    """24 x [x, y, v]; x/y uniform in [0.2, 0.8]; all v=1 except ``n_unreliable`` zeros."""
    xy = rng.uniform(0.2, 0.8, size=(24, 2))
    vis = np.ones(24)
    vis[rng.choice(24, size=n_unreliable, replace=False)] = 0
    return np.column_stack([xy, vis]).tolist()


def make_labels(videos: int = 2, segments: int = 2, images: int = 3, seed: int = 230) -> dict:
    """Same nesting as the real labels JSON; IDs like "v0", "s0", "i0"; frame_idx = 10 * i."""
    rng = np.random.default_rng(seed)
    labels: dict = {}
    for v in range(videos):
        labels[f"v{v}"] = {}
        for s in range(segments):
            labels[f"v{v}"][f"s{s}"] = {
                f"i{i}": {"annotation": make_annotation(rng), "frame_idx": 10 * i} for i in range(images)
            }
    return labels


def write_labels(labels: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels), encoding="utf-8")
    return path


def make_blank_images(
    root: Path,
    labels: dict,
    size: tuple[int, int] = (64, 48),
    template: str = "{video_id}/{segment_id}/{image_id}.png",
    value: int = 0,
) -> None:
    """One image of ``size`` (w, h) per label entry."""
    w, h = size
    img = np.full((h, w, 3), value, dtype=np.uint8)
    for vid, segs in labels.items():
        for seg, imgs in segs.items():
            for image_id in imgs:
                path = root / template.format(video_id=vid, segment_id=seg, image_id=image_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), img)


def square_frame(k: int, size=(640, 360), step: int = 8, square: int = 40, x0: int = 20) -> np.ndarray:
    w, h = size
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    x = x0 + step * k
    y = (h - square) // 2
    frame[y:y + square, x:x + square] = 255
    return frame


def make_moving_square_video(
    path: Path,
    n: int = 60,
    size: tuple[int, int] = (640, 360),
    step: int = 8,
    square: int = 40,
    save_frames: tuple[int, ...] = (),
    frames_dir: Path | None = None,
    fps: float = 25.0,
) -> dict[int, Path]:
    """MJPG .avi of a white square moving ``step`` px/frame on gray (no wrap-around).

    Frames listed in ``save_frames`` are also saved as PNGs from the generated arrays
    (not re-decoded) as ``frames_dir/frame_XXX.png``.
    """
    assert 20 + step * (n - 1) + square <= size[0], "square would leave the frame"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV cannot write MJPG"
    saved: dict[int, Path] = {}
    for k in range(n):
        frame = square_frame(k, size, step, square)
        writer.write(frame)
        if k in save_frames:
            assert frames_dir is not None
            frames_dir.mkdir(parents=True, exist_ok=True)
            saved[k] = frames_dir / f"frame_{k:03d}.png"
            cv2.imwrite(str(saved[k]), frame)
    writer.release()
    return saved
