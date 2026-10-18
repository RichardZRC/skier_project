"""SiamFC tracker smoke test (guide 1H, code spec §11).

A minimal run of the tracker on one video. Since Step 3 it uses the proper adapter
(``skier.tracking.siamfc_adapter``) and ``track_clip``; the tracker is initialized once
and never reset. torch/siamfc are imported lazily.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from skier.io_utils import ensure_dir, read_csv, write_csv
from skier.paths import results_root
from skier.tracking.frames import VideoFrames
from skier.tracking.siamfc_adapter import SiamFCTracker, Tracker, xywh1_to_xyxy, xyxy_to_xywh1  # noqa: F401
from skier.viz import FONT, draw_box

BOX_COLUMNS = ["frame_index", "x1", "y1", "x2", "y2"]


def run_on_video(tracker: Tracker, video_path, init_frame: int, init_box_xyxy, max_frames: int) -> list[tuple]:
    """Decode sequentially; init once at ``init_frame``; update until ``max_frames`` rows or EOF."""
    from skier.tracking.run import track_clip

    try:
        rows = track_clip(tracker, VideoFrames(video_path), init_frame, init_box_xyxy, max_frames)
    except ValueError as exc:
        raise ValueError(f"init frame {init_frame} is past the end of {video_path}") from exc
    return [(r["frame_index"], r["x1"], r["y1"], r["x2"], r["y2"]) for r in rows]


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
