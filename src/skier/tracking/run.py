"""Run a tracker over clips and evaluate it (plan Step 3.5).

The tracker is initialized once per clip, and ground truth is never fed back after
initialization. One CSV per clip: ``frame_index, x1, y1, x2, y2, score, status`` with
``status`` = ``init`` | ``track``. Ski 2DPose landmarks are not tracking ground truth,
so Ski 2DPose segments get CSVs (for the replay) but no IoU metrics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from skier.data.skitb import FPS_TABLE, fps_for, frame_file, load_fps_table, load_skitb_clips
from skier.eval.track_metrics import clip_iou, failure_episodes, frame_ious, pooled_iou, success_curve
from skier.io_utils import ensure_dir, read_csv, read_id_list, read_json, write_csv, write_json
from skier.paths import data_root, repo_root, results_root
from skier.pose.crop_error import FINAL_ONLY
from skier.tracking.frames import ImageFrames, VideoFrames
from skier.tracking.siamfc_adapter import Tracker

TRACK_COLUMNS = ["frame_index", "x1", "y1", "x2", "y2", "score", "status"]
POSE_SPLIT = "data/manifests/pose_split_v1.json"


def track_clip(tracker: Tracker, frames: Iterable[tuple[int, np.ndarray]], init_index: int, init_box,
               max_frames: int | None = None) -> list[dict]:
    """Init at ``init_index`` (a frame index of ``frames``), then update on every later frame."""
    rows: list[dict] = []
    for idx, frame in frames:
        if max_frames is not None and len(rows) >= max_frames:
            break
        if idx == init_index:
            box = np.asarray(init_box, dtype=np.float64)
            tracker.init(frame, box)
            rows.append(_row(idx, box, math.nan, "init"))
        elif idx > init_index:
            box, score = tracker.update(frame)
            rows.append(_row(idx, box, score, "track"))
    if not rows:
        raise ValueError(f"init frame {init_index} never reached")
    return rows


def _row(idx: int, box, score: float, status: str) -> dict:
    x1, y1, x2, y2 = (float(v) for v in box)
    return {"frame_index": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score, "status": status}


def write_track_csv(rows: list[dict], path: Path) -> None:
    # scores keep 7 significant digits: SiamFC's lie around 1e-5 (Step 7 confidence check)
    fmt = [{**r, **{k: f"{r[k]:.3f}" for k in ("x1", "y1", "x2", "y2")},
            "score": "" if math.isnan(r["score"]) else f"{r['score']:.6e}"} for r in rows]
    write_csv(fmt, path, TRACK_COLUMNS)


def read_track_csv(path: Path) -> list[dict]:
    return [{"frame_index": int(r["frame_index"]), **{k: float(r[k]) for k in ("x1", "y1", "x2", "y2")},
             "score": float(r["score"]) if r["score"] else math.nan, "status": r["status"]}
            for r in read_csv(path)]


def boxes_of(rows: list[dict]) -> np.ndarray:
    return np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows], dtype=np.float64).reshape(-1, 4)


def evaluate_clip(rows: list[dict], gt_xyxy: np.ndarray, visible: np.ndarray, init_pos: int, fps: float) -> dict:
    """Metrics for one clip whose rows cover positions ``init_pos..`` of the ground truth."""
    n = len(rows)
    gt, vis = gt_xyxy[init_pos:init_pos + n], visible[init_pos:init_pos + n]
    pred = boxes_of(rows)
    stats = clip_iou(pred, gt, vis, init_index=0)
    ious = frame_ious(pred, gt)
    stats["failure_episodes"] = failure_episodes(ious, vis, [r["frame_index"] for r in rows], fps)
    return stats


def summarize(clip_stats: dict[str, dict]) -> dict:
    pooled = pooled_iou(list(clip_stats.values()))
    all_ious = np.concatenate([s["ious"] for s in clip_stats.values()]) if clip_stats else np.array([])
    return {
        **pooled,
        "success": success_curve(all_ious),
        "per_clip": {k: {"mean_iou": s["mean_iou"], "n_frames": s["n_frames"], "fps": s.get("fps"),
                         "failure_episodes": s["failure_episodes"]} for k, s in clip_stats.items()},
        "n_failure_episodes": sum(len(s["failure_episodes"]) for s in clip_stats.values()),
        "note": "our metrics; the SkiTB toolkit's official metrics are reported separately",
    }


def run_skitb(tracker_factory: Callable[[], Tracker], skitb_root: Path, split_json: Path, partition: str,
              out: Path, fps: float | Mapping[str, float], discipline: str = "AL",
              max_frames: int | None = None) -> dict:
    """Every SC clip of a partition: init at its first visible frame with the GT box.

    ``fps`` is one rate for all clips, or a per-sequence table (D22).
    """
    clip_stats = {}
    for clip in load_skitb_clips(skitb_root, split_json, partition, discipline):
        visible = np.flatnonzero(clip.visible)
        if len(visible) == 0:
            continue
        clip_fps = fps_for(fps, clip.sequence_id) if isinstance(fps, Mapping) else float(fps)
        init_pos = int(visible[0])
        frames = ImageFrames(clip.frames_dir, [frame_file(e) for e in clip.frame_files], fps=clip_fps)
        rows = track_clip(tracker_factory(), frames, init_pos, clip.boxes_xyxy[init_pos], max_frames)
        name = f"{clip.sequence_id}__{clip.clip_id}"
        write_track_csv(rows, out / f"{name}.csv")
        clip_stats[name] = {**evaluate_clip(rows, clip.boxes_xyxy, clip.visible, init_pos, clip_fps), "fps": clip_fps}
    summary = summarize(clip_stats)
    summary.update(dataset="skitb", discipline=discipline, partition=partition,
                   fps="per sequence (D22)" if isinstance(fps, Mapping) else float(fps))
    write_json(summary, out / "summary.json")
    return summary


def run_ski2dpose_segments(tracker_factory: Callable[[], Tracker], records, segments: list[tuple[str, str]],
                           videos_root: Path, out: Path, max_frames: int | None = None) -> list[str]:
    """Ski 2DPose segments: init at the first labeled frame with its reference box. CSVs only."""
    from skier.data.alignment import locate_segment

    written = []
    for vid, seg in segments:
        recs = sorted((r for r in records if r.source_video_id == vid and r.segment_id == seg and not r.excluded),
                      key=lambda r: r.frame_index)
        if not recs:
            continue
        video, start = locate_segment(videos_root, vid, seg)
        rows = track_clip(tracker_factory(), VideoFrames(video), start + recs[0].frame_index,
                          recs[0].reference_box_xyxy, max_frames)
        name = f"{vid}__{seg}"
        write_track_csv(rows, out / f"{name}.csv")
        written.append(name)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the fixed SiamFC tracker over clips (plan Step 3.5).")
    p.add_argument("--siamfc-repo", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--skitb-root", type=Path, help="SkiTB root (with --split-json, --partition)")
    src.add_argument("--pilot-segments", action="store_true", help="all segments of the pilot videos (Ski 2DPose)")
    src.add_argument("--ski2dpose-partition", metavar="PARTITION",
                     help="all segments of one pose-split partition (Ski 2DPose; Step 7), e.g. val")
    p.add_argument("--split-json", type=Path)
    p.add_argument("--pose-split", type=Path, help=f"default: {POSE_SPLIT} (with --ski2dpose-partition)")
    p.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8, once)")
    p.add_argument("--partition", default="val")
    p.add_argument("--fps-table", type=Path, help=f"per-sequence SkiTB frame rates; default: {FPS_TABLE} (D22)")
    p.add_argument("--fps", type=float, help="one SkiTB frame rate for every clip, instead of the table")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--out", type=Path,
                   help="default: $SKIER_RESULTS/tracking/<skitb-PARTITION | pilots | ski2dpose-PARTITION>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from skier.tracking.siamfc_adapter import SiamFCTracker

    def factory():
        return SiamFCTracker(args.siamfc_repo, args.weights)

    if args.skitb_root:
        if args.split_json is None:
            print("error: --skitb-root needs --split-json", file=sys.stderr)
            return 1
        if args.partition in FINAL_ONLY and not args.final_test:
            print(f"error: {args.partition!r} is tracked once, in Step 8; pass --final-test then", file=sys.stderr)
            return 1
        if args.fps is not None and args.fps_table is not None:
            print("error: give --fps or --fps-table, not both", file=sys.stderr)
            return 1
        fps = args.fps if args.fps is not None else load_fps_table(args.fps_table or repo_root() / FPS_TABLE)
        out = ensure_dir(args.out or results_root() / "tracking" / f"skitb-{args.partition}")
        s = run_skitb(factory, args.skitb_root, args.split_json, args.partition, out, fps,
                      max_frames=args.max_frames)
        print(f"{s['n_clips']} clips, pooled mean IoU {s['pooled_mean_iou']}, AUC {s['success']['auc']} → {out}")
        return 0

    from skier.data.ski2dpose import load_records

    part = args.ski2dpose_partition
    if part in FINAL_ONLY and not args.final_test:
        print(f"error: {part!r} is used once, in Step 8; pass --final-test then", file=sys.stderr)
        return 1
    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images")
    if part is None:
        pilots = set(read_id_list(repo_root() / "data/manifests/pilot_videos.txt"))
        segments = sorted({(r.source_video_id, r.segment_id) for r in records if r.source_video_id in pilots})
    else:
        from skier.splits import apply_split

        manifest = read_json(args.pose_split or repo_root() / POSE_SPLIT)
        if part not in manifest["partitions"]:
            print(f"error: no partition {part!r} in the pose split", file=sys.stderr)
            return 1
        apply_split(records, manifest)
        segments = sorted({(r.source_video_id, r.segment_id) for r in records if r.split == part})
    out = ensure_dir(args.out or results_root() / "tracking" / ("pilots" if part is None else f"ski2dpose-{part}"))
    names = run_ski2dpose_segments(factory, records, segments, s2d / "videos", out, args.max_frames)
    print(f"tracked {len(names)} {'pilot' if part is None else part} segments → {out}")
    return 0
