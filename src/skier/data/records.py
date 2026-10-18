"""Canonical pose record shared by every dataset reader, model and metric (plan §5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PoseRecord:
    sample_id: str  # "{source_video_id}/{segment_id}/{image_id}"
    dataset: str
    source_video_id: str
    segment_id: str
    image_id: str
    frame_index: int
    image_path: Path  # absolute
    image_width: int
    image_height: int
    keypoints_xy: np.ndarray  # [24, 2] original-image pixels
    reliable: np.ndarray  # [24] bool: visible == 1
    raw_visibility: np.ndarray  # [24] as labeled
    reference_box_xyxy: np.ndarray | None  # reliable body + ski points; None if excluded
    discipline: str | None = None
    view_angle: str | None = None
    split: str | None = None
    excluded_reason: str | None = None  # kept and reported, never silently dropped

    @property
    def excluded(self) -> bool:
        return self.excluded_reason is not None
