"""Raw reader for the Ski 2DPose labels JSON.

Nesting (verified 2026-10-17):
``VideoID -> SplitID -> ImageID -> {"annotation": [[x, y, visible] x 24], "frame_idx": i}``.
Their "SplitID" is a camera sequence, which this code calls ``segment_id``.
Coordinates are normalized 0-1 with (0, 0) at the top-left.

This module reports what is in the file and validates nothing beyond the nesting;
the audit (``skier.data.audit``) is where problems are found and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

NUM_KEYPOINTS = 24

# --- FILLED IN AT 1E (2026-10-17; code spec §13) -----------------------------
# Image path relative to the images root ($SKIER_DATA/ski2dpose/images), from the
# images zip listing and the official example loader. Image IDs are full stems,
# e.g. "uLW74013Wp0_0_00001".
IMAGE_PATH_TEMPLATE: str | None = "Images_jpg/{video_id}/{segment_id}/{image_id}.jpg"

VIEW_FLAGS = ("front_view", "side_view", "back_view")


def info_for(info: Any, video_id: str, segment_id: str) -> dict:
    """Return ``{"discipline": str | None, "view_angle": str | None}`` for one sequence.

    Info JSON layout: ``VideoID -> {"video_info": {..., "type": str},
    "splits_info": {SplitID: {"front_view": bool, "side_view": bool, "back_view": bool, ...}}}``.
    ``discipline`` is the per-video type ("Slalom", "Giant Slalom", "Super G", "Downhill",
    "Combined Slalom", "Mixed", "Follow Cam"). ``view_angle`` joins the true view flags,
    e.g. "front+side"; several can be true for one sequence.
    """
    video = info.get(video_id) if isinstance(info, dict) else None
    if not isinstance(video, dict):
        return {"discipline": None, "view_angle": None}
    discipline = (video.get("video_info") or {}).get("type")
    split = (video.get("splits_info") or {}).get(segment_id)
    view = None
    if isinstance(split, dict):
        view = "+".join(flag.removesuffix("_view") for flag in VIEW_FLAGS if split.get(flag)) or None
    return {"discipline": discipline, "view_angle": view}
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RawLabel:
    video_id: str
    segment_id: str
    image_id: str
    annotation: Any  # as found in the JSON; None if the key or entry is missing
    frame_idx: Any  # None if the key is missing

    @property
    def sample_id(self) -> str:
        return make_sample_id(self.video_id, self.segment_id, self.image_id)


def make_sample_id(video_id: str, segment_id: str, image_id: str) -> str:
    return f"{video_id}/{segment_id}/{image_id}"


def safe_name(sample_id: str) -> str:
    """A filename-safe form of a sample ID."""
    return sample_id.replace("/", "__")


def iter_raw_labels(labels: dict) -> Iterator[RawLabel]:
    """Yield every image entry, in sorted-key order at every level."""
    if not isinstance(labels, dict):
        raise ValueError("labels JSON: top level must be an object keyed by video ID")
    for video_id in sorted(labels):
        segments = labels[video_id]
        if not isinstance(segments, dict):
            raise ValueError(f"labels JSON: video {video_id!r} must map segment IDs to objects")
        for segment_id in sorted(segments):
            images = segments[segment_id]
            if not isinstance(images, dict):
                raise ValueError(
                    f"labels JSON: segment {video_id!r}/{segment_id!r} must map image IDs to objects"
                )
            for image_id in sorted(images):
                entry = images[image_id]
                if isinstance(entry, dict):
                    yield RawLabel(
                        video_id, segment_id, image_id,
                        entry.get("annotation"), entry.get("frame_idx"),
                    )
                else:
                    yield RawLabel(video_id, segment_id, image_id, None, None)


def annotation_array(raw: RawLabel) -> np.ndarray | None:
    """The annotation as float64 ``(24, 3)``, or None if it can't be converted to that."""
    try:
        arr = np.asarray(raw.annotation, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (NUM_KEYPOINTS, 3):
        return None
    return arr


def reliable_mask(ann: np.ndarray) -> np.ndarray:
    """``visible == 1``: visible or confidently labeled. ``0`` is an occluded best guess."""
    return np.asarray(ann)[:, 2] == 1


def to_pixels(xy_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized ``[..., 2]`` coordinates to pixels of an image of the given size."""
    return np.asarray(xy_norm, dtype=np.float64) * np.array([width, height], dtype=np.float64)


def image_path_for(
    images_root: str | Path,
    video_id: str,
    segment_id: str,
    image_id: str,
    template: str | None = None,
) -> Path:
    tpl = template if template is not None else IMAGE_PATH_TEMPLATE
    if tpl is None:
        raise NotImplementedError("image path template not set: fill in at 1E (code spec §13)")
    return Path(images_root) / tpl.format(
        video_id=video_id, segment_id=segment_id, image_id=image_id
    )


def info_preview(info: Any, max_keys: int = 5, depth: int = 4) -> Any:
    """Describe a JSON value's layout by key names and types only, never values."""
    if isinstance(info, dict):
        if depth == 0:
            return f"dict[{len(info)}]"
        keys = sorted(info)[:max_keys]
        return {
            "type": "dict",
            "n_keys": len(info),
            "keys": {k: info_preview(info[k], max_keys, depth - 1) for k in keys},
        }
    if isinstance(info, list):
        if depth == 0 or not info:
            return f"list[{len(info)}]"
        return {"type": "list", "len": len(info), "first": info_preview(info[0], max_keys, depth - 1)}
    return type(info).__name__
