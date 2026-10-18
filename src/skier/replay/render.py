"""Replay rendering (plan Step 4.2): sequential frames -> warped 16:9 crops at fixed size."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from skier.geometry import apply_affine, crop_affine
from skier.io_utils import read_csv, write_csv
from skier.replay.controller import CropTrack
from skier.schema import Schema
from skier.viz import FONT, draw_box, draw_edges, draw_points

RAW_COLOR = (0, 255, 0)  # green: raw tracker box
CROP_COLOR = (0, 255, 255)  # yellow: smoothed replay crop
CROP_COLUMNS = ["frame_index", "timestamp", "x1", "y1", "x2", "y2", "raw_x1", "raw_y1", "raw_x2", "raw_y2",
                "urgent", "lost", "clamped", "reset"]


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    return w


def _label(img, text: str) -> None:
    h = img.shape[0]
    cv2.putText(img, text, (10, h - 12), FONT, max(0.5, h / 1080), (255, 255, 255), 1, cv2.LINE_AA)


def render_replay(frames: Iterable[tuple[int, np.ndarray]], frame_indices: Sequence[int], track: CropTrack,
                  out_path: str | Path, fps: float, out_size: tuple[int, int] = (1280, 720),
                  draw_raw_box: bool = False, keypoints: Mapping[int, tuple] | None = None,
                  schema: Schema | None = None) -> int:
    """Write the replay video. ``keypoints``: frame_index -> (xy[K,2] original px, reliable[K]).

    Overlays are drawn after warping, so lines keep their thickness. Returns frames written.
    """
    pos = {int(f): i for i, f in enumerate(frame_indices)}
    last = max(pos) if pos else -1
    out_w, out_h = out_size
    writer, n = None, 0
    try:
        for idx, frame in frames:
            if idx > last:
                break
            if idx not in pos:
                continue
            i = pos[idx]
            M = crop_affine(track.crops[i], out_w, out_h)
            out = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            raw = track.raw_boxes[i]
            if draw_raw_box and np.isfinite(raw).all():
                (x1, y1), (x2, y2) = apply_affine(M, raw.reshape(2, 2))
                draw_box(out, (x1, y1, x2, y2), RAW_COLOR, thickness=2)
            if keypoints is not None and idx in keypoints:
                if schema is None:
                    raise ValueError("drawing keypoints needs the schema")
                xy, reliable = keypoints[idx]
                mapped = apply_affine(M, np.asarray(xy, float))
                draw_edges(out, mapped, reliable, schema)
                draw_points(out, mapped, reliable, schema, label_indices=False)
            _label(out, f"frame {idx}" + ("  LOST" if track.lost[i] else ""))
            if writer is None:
                writer = _writer(Path(out_path), fps, (out_w, out_h))
            writer.write(out)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n


def render_debug(frames: Iterable[tuple[int, np.ndarray]], frame_indices: Sequence[int], track: CropTrack,
                 out_path: str | Path, fps: float, max_width: int = 1280) -> int:
    """The full frame with the raw tracker box (green) and the smoothed crop (yellow)."""
    pos = {int(f): i for i, f in enumerate(frame_indices)}
    last = max(pos) if pos else -1
    writer, n = None, 0
    try:
        for idx, frame in frames:
            if idx > last:
                break
            if idx not in pos:
                continue
            i = pos[idx]
            img = frame.copy()
            if np.isfinite(track.raw_boxes[i]).all():
                draw_box(img, track.raw_boxes[i], RAW_COLOR, thickness=3)
            draw_box(img, track.crops[i], CROP_COLOR, thickness=3)
            flags = [name for name in ("urgent", "lost", "clamped", "reset") if getattr(track, name)[i]]
            _label(img, f"frame {idx}  " + " ".join(flags))
            scale = min(1.0, max_width / img.shape[1])
            if scale < 1:
                img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = _writer(Path(out_path), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    return n


def write_crops_csv(track: CropTrack, frame_indices: Sequence[int], path: str | Path) -> None:
    def f(v):
        return "" if not math.isfinite(v) else f"{v:.3f}"

    rows = []
    for i, idx in enumerate(frame_indices):
        c, r = track.crops[i], track.raw_boxes[i]
        rows.append({
            "frame_index": int(idx), "timestamp": f"{track.timestamps[i]:.6f}",
            **{k: f(v) for k, v in zip(("x1", "y1", "x2", "y2"), c)},
            **{k: f(v) for k, v in zip(("raw_x1", "raw_y1", "raw_x2", "raw_y2"), r)},
            **{k: int(getattr(track, k)[i]) for k in ("urgent", "lost", "clamped", "reset")},
        })
    write_csv(rows, path, CROP_COLUMNS)


def read_crops_csv(path: str | Path) -> tuple[list[int], CropTrack]:
    rows = read_csv(path)

    def arr(keys):
        return np.array([[float(r[k]) if r[k] else np.nan for k in keys] for r in rows]).reshape(-1, len(keys))

    crops = arr(("x1", "y1", "x2", "y2"))
    track = CropTrack(
        crops=crops, desired=np.full_like(crops, np.nan), raw_boxes=arr(("raw_x1", "raw_y1", "raw_x2", "raw_y2")),
        timestamps=np.array([float(r["timestamp"]) for r in rows]),
        **{k: np.array([r[k] == "1" for r in rows], dtype=bool) for k in ("urgent", "lost", "clamped", "reset")},
    )
    return [int(r["frame_index"]) for r in rows], track
