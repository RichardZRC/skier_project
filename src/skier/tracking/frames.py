"""Sequential frame sources: video files and image folders. Never random seeking."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

DEFAULT_FPS = 25.0


class VideoFrames:
    """``(frame_index, frame_bgr)`` from a video file, decoded in order from frame 0."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        cap.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield idx, frame
                idx += 1
        finally:
            cap.release()


class ImageFrames:
    """``(position, frame_bgr)`` for an ordered list of image files in one folder."""

    def __init__(self, frames_dir: str | Path, files: Sequence[str], fps: float = DEFAULT_FPS):
        self.frames_dir = Path(frames_dir)
        self.files = list(files)
        self.fps = fps

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, name in enumerate(self.files):
            img = cv2.imread(str(self.frames_dir / name))
            if img is None:
                raise FileNotFoundError(f"cannot read frame {self.frames_dir / name}")
            yield i, img
