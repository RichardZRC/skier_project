"""Replay crop controller (plan Step 4.1).

Turns per-frame tracker boxes into smooth, centered 16:9 crops. State is
``(cx, cy, log_h)``; width = height x aspect, so a separate ``log_w`` would be redundant.
Every frame: desired crop ``q`` = tracker box expanded by ``k`` and fit to 16:9 (within
size limits), then exponential smoothing ``s = a q + (1 - a) s_prev`` with
``a = 1 - exp(-dt / tau)``, which is consistent across frame rates.

Rules:
- Centre uses ``tau_pos``; size uses ``tau_grow`` when the crop must get bigger and
  ``tau_shrink`` (slower) when it may get smaller.
- Urgent: if the tracker box (padded by ``urgent_margin`` of the crop size) is not inside
  the previous crop, i.e. the target nears the *replay crop's* border, both use ``tau_urgent``.
- Lost (NaN box, status "lost", or score below ``lost_score``): hold the centre and zoom
  out at ``lost_zoom_out_per_s`` (log-height per second) up to the largest crop.
- The *output* crop is shifted to stay inside the image; the internal state is not clamped.
- The first frame, and the first after a gap longer than ``reset_gap_s``, start from ``q``.
The raw tracker boxes are always kept alongside.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from skier.geometry import box_center_size, expand_box, fit_aspect


@dataclass(frozen=True)
class ReplayConfig:
    k: float = 2.0
    aspect: float = 16 / 9
    tau_pos: float = 0.2
    tau_grow: float = 0.1
    tau_shrink: float = 0.6
    tau_urgent: float = 0.03
    urgent_margin: float = 0.05
    min_crop_h_px: float = 240.0
    max_zoom: float = 3.0
    lost_zoom_out_per_s: float = 0.5
    lost_score: float | None = None
    reset_gap_s: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReplayConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown replay config keys: {sorted(unknown)}")
        return cls(**raw)

    def with_(self, **changes) -> "ReplayConfig":
        return dataclasses.replace(self, **changes)

    def unsmoothed(self) -> "ReplayConfig":
        """The tau = 0 baseline: crops follow the desired crop exactly (still clamped)."""
        return self.with_(tau_pos=0.0, tau_grow=0.0, tau_shrink=0.0, tau_urgent=0.0)


@dataclass
class CropTrack:
    crops: np.ndarray  # [T, 4] xyxy, inside the image
    desired: np.ndarray  # [T, 4] q_t (NaN on lost frames)
    raw_boxes: np.ndarray  # [T, 4] tracker output, untouched
    timestamps: np.ndarray  # [T] seconds
    urgent: np.ndarray  # [T] bool
    lost: np.ndarray
    clamped: np.ndarray
    reset: np.ndarray


def alpha(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)


def size_limits(frame_w: float, frame_h: float, cfg: ReplayConfig) -> tuple[float, float]:
    """(min, max) crop height: max = the largest crop of this aspect inside the frame."""
    h_max = min(frame_h, frame_w / cfg.aspect)
    h_min = min(h_max, max(cfg.min_crop_h_px, frame_h / cfg.max_zoom))
    return h_min, h_max


def desired_state(box, frame_size, cfg: ReplayConfig) -> np.ndarray:
    h_min, h_max = size_limits(*frame_size, cfg)
    c, (_, h) = box_center_size(fit_aspect(expand_box(box, cfg.k), cfg.aspect))
    return np.array([c[0], c[1], math.log(min(max(h, h_min), h_max))])


def state_to_box(state, aspect: float) -> np.ndarray:
    cx, cy, log_h = state
    h = math.exp(log_h)
    w = h * aspect
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def clamp_box(box, frame_w: float, frame_h: float) -> tuple[np.ndarray, bool]:
    """Shift (never resize) the box to lie inside the frame."""
    x1, y1, x2, y2 = box
    dx = -x1 if x1 < 0 else (frame_w - x2 if x2 > frame_w else 0.0)
    dy = -y1 if y1 < 0 else (frame_h - y2 if y2 > frame_h else 0.0)
    return np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy]), bool(dx or dy)


def is_urgent(box, crop, margin: float) -> bool:
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    pad_x, pad_y = margin * w, margin * h
    return bool(box[0] - pad_x < crop[0] or box[1] - pad_y < crop[1]
                or box[2] + pad_x > crop[2] or box[3] + pad_y > crop[3])


def run_controller(boxes: np.ndarray, timestamps: Sequence[float], frame_size: tuple[float, float],
                   cfg: ReplayConfig = ReplayConfig(), status: Sequence[str] | None = None,
                   scores: Sequence[float] | None = None) -> CropTrack:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) != len(boxes):
        raise ValueError(f"{len(boxes)} boxes but {len(ts)} timestamps")
    if len(ts) > 1 and (np.diff(ts) <= 0).any():
        raise ValueError("timestamps must be strictly increasing")
    frame_w, frame_h = frame_size
    _, h_max = size_limits(frame_w, frame_h, cfg)
    n = len(boxes)
    crops = np.zeros((n, 4))
    desired = np.full((n, 4), np.nan)
    flags = {name: np.zeros(n, dtype=bool) for name in ("urgent", "lost", "clamped", "reset")}

    state = prev_t = last_out = None
    for t in range(n):
        box = boxes[t]
        lost = (not np.isfinite(box).all()
                or (status is not None and status[t] == "lost")
                or (cfg.lost_score is not None and scores is not None
                    and np.isfinite(scores[t]) and scores[t] < cfg.lost_score))
        dt = ts[t] - prev_t if prev_t is not None else 0.0
        gap = prev_t is not None and dt > cfg.reset_gap_s
        if lost:
            flags["lost"][t] = True
            if state is None:
                state = np.array([frame_w / 2, frame_h / 2, math.log(h_max)])
            else:
                state = state.copy()
                state[2] = min(math.log(h_max), state[2] + cfg.lost_zoom_out_per_s * dt)
        else:
            q = desired_state(box, frame_size, cfg)
            desired[t] = state_to_box(q, cfg.aspect)
            if state is None or gap:
                state = q
                flags["reset"][t] = True
            else:
                urgent = is_urgent(box, last_out, cfg.urgent_margin)
                flags["urgent"][t] = urgent
                tau_c = cfg.tau_urgent if urgent else cfg.tau_pos
                tau_s = cfg.tau_urgent if urgent else (cfg.tau_grow if q[2] > state[2] else cfg.tau_shrink)
                a_c, a_s = alpha(dt, tau_c), alpha(dt, tau_s)
                state = np.array([a_c * q[0] + (1 - a_c) * state[0],
                                  a_c * q[1] + (1 - a_c) * state[1],
                                  a_s * q[2] + (1 - a_s) * state[2]])
        out, clamped = clamp_box(state_to_box(state, cfg.aspect), frame_w, frame_h)
        crops[t], flags["clamped"][t] = out, clamped
        last_out, prev_t = out, ts[t]
    return CropTrack(crops, desired, boxes.copy(), ts.copy(), **flags)
