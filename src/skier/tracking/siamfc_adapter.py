"""SiamFC tracker adapter (plan Step 3.5).

``Tracker.update`` returns ``(box_xyxy, score)``. All conventions of the wrapped
``siamfc-pytorch`` tracker (RGB input, 1-based ``xywh`` boxes) are converted here, at the
edge. torch and the fork are imported only when a ``SiamFCTracker`` is constructed.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class Tracker(Protocol):
    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None: ...

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]: ...


def xyxy_to_xywh1(box) -> np.ndarray:
    """``xyxy`` to 1-based ``[x, y, w, h]`` (GOT-10k / upstream siamfc-pytorch convention)."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return np.array([x1 + 1, y1 + 1, x2 - x1, y2 - y1])


def xywh1_to_xyxy(box) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=np.float64)
    return np.array([x - 1, y - 1, x - 1 + w, y - 1 + h])


class SiamFCTracker:
    """Pretrained SiamFC (GOT-10k weights). The device is chosen by the tracker itself."""

    # --- Filled in 2026-10-26 from matteo-dunnhofer/siamfc-pytorch @ a87f140 (siamfc/siamfc.py) ---
    # ``TrackerSiamFC(net_path, return_conf)``: ``init(rgb, xywh1)``; ``update(rgb)`` returns the
    # 1-based xywh box, or ``(box, conf)`` with ``return_conf=True``. ``conf`` is the peak of the
    # sum-normalized, Hann-windowed response map: a relative score, not a probability.
    MODULE = "siamfc"
    CLASS = "TrackerSiamFC"
    RETURN_CONF = True
    # ------------------------------------------------------------------------------------------------

    def __init__(self, siamfc_repo: Path, weights: Path):
        sys.path.insert(0, str(Path(siamfc_repo).resolve()))
        cls = getattr(importlib.import_module(self.MODULE), self.CLASS)
        self._tracker = cls(net_path=str(weights), return_conf=self.RETURN_CONF)

    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None:
        self._tracker.init(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), xyxy_to_xywh1(box_xyxy))

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        out = self._tracker.update(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        box, score = out if self.RETURN_CONF else (out, math.nan)
        return xywh1_to_xyxy(box), float(score)
