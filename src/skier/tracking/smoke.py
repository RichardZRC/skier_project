"""SiamFC tracker smoke test (guide 1H, code spec §11).

A minimal direct use of the tracker, replaced by the proper adapter in Step 3.
The tracker is initialized once and never reset. torch/siamfc are imported lazily.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from skier.io_utils import ensure_dir, read_csv, write_csv
from skier.paths import results_root
from skier.viz import FONT, draw_box

BOX_COLUMNS = ["frame_index", "x1", "y1", "x2", "y2"]


class BoxTracker(Protocol):
    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None: ...

    def update(self, frame_bgr: np.ndarray) -> np.ndarray: ...  # xyxy


def xyxy_to_xywh1(box) -> np.ndarray:
    """``xyxy`` to 1-based ``[x, y, w, h]`` (GOT-10k / upstream siamfc-pytorch convention)."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return np.array([x1 + 1, y1 + 1, x2 - x1, y2 - y1])


def xywh1_to_xyxy(box) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=np.float64)
    return np.array([x - 1, y - 1, x - 1 + w, y - 1 + h])


class SiamFCTracker:
    """Wraps the siamfc-pytorch tracker. Conversions (BGR→RGB, xyxy↔xywh1) happen here.

    The device is chosen by the tracker itself (upstream uses cuda:0 when available).
    """

    # --- FILL IN AT 1D/1H (code spec §13): confirm against the fork's source ---
    MODULE = "siamfc"
    CLASS = "TrackerSiamFC"
    # ---------------------------------------------------------------------------

    def __init__(self, siamfc_repo: Path, weights: Path):
        sys.path.insert(0, str(Path(siamfc_repo).resolve()))
        cls = getattr(importlib.import_module(self.MODULE), self.CLASS)
        self._tracker = cls(net_path=str(weights))

    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None:
        self._tracker.init(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), xyxy_to_xywh1(box_xyxy))

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        return xywh1_to_xyxy(self._tracker.update(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def run_on_video(tracker: BoxTracker, video_path, init_frame: int, init_box_xyxy, max_frames: int) -> list[tuple]:
    """Decode sequentially; init once at ``init_frame``; update until ``max_frames`` rows or EOF."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    rows: list[tuple] = []
    idx = 0
    try:
        while len(rows) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx == init_frame:
                box = np.asarray(init_box_xyxy, dtype=np.float64)
                tracker.init(frame, box)
                rows.append((idx, *map(float, box)))
            elif idx > init_frame:
                rows.append((idx, *map(float, tracker.update(frame))))
            idx += 1
    finally:
        cap.release()
    if not rows:
        raise ValueError(f"init frame {init_frame} is past the end of {video_path} ({idx} frames)")
    return rows


def write_boxes_csv(rows: list[tuple], path) -> None:
    write_csv(
        [{"frame_index": f, "x1": f"{x1:.3f}", "y1": f"{y1:.3f}", "x2": f"{x2:.3f}", "y2": f"{y2:.3f}"}
         for f, x1, y1, x2, y2 in rows],
        path, BOX_COLUMNS,
    )


def read_boxes_csv(path) -> list[tuple]:
    return [(int(r["frame_index"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            for r in read_csv(path)]


def render_track(video_path, rows: list[tuple], out_path) -> None:
    """Draw each row's box on its frame; frames ``rows[0]..rows[-1]``; mp4v at the source FPS."""
    boxes = {r[0]: r[1:] for r in rows}
    first, last = rows[0][0], rows[-1][0]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    idx = 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= first:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if idx in boxes:
                    draw_box(frame, boxes[idx], (0, 255, 0), thickness=2)
                h = frame.shape[0]
                cv2.putText(frame, f"frame {idx}", (10, h - 12), FONT, max(0.5, h / 1080), (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run pretrained SiamFC on one video (qualitative smoke test).")
    p.add_argument("--siamfc-repo", type=Path, required=True, help="clone of the siamfc-pytorch fork")
    p.add_argument("--weights", type=Path, required=True, help="siamfc_alexnet_e50.pth")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--init-frame", type=int, required=True)
    p.add_argument("--init-box", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/smoke_tracker")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = ensure_dir(args.out or results_root() / "step1" / "smoke_tracker")
    tracker = SiamFCTracker(args.siamfc_repo, args.weights)
    try:
        rows = run_on_video(tracker, args.video, args.init_frame, args.init_box, args.max_frames)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_boxes_csv(rows, out / "boxes.csv")
    render_track(args.video, rows, out / "track.mp4")
    print(f"tracked {len(rows)} frames → {out}")
    return 0
