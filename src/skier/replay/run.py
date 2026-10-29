"""Replay runner (plan Step 4.4): tracker CSV -> crops, replay/debug videos, metrics; or a tau/k grid.

Every single-clip run reports the smoothed controller **and** the unsmoothed (tau = 0)
baseline side by side. Choose defaults with ``--grid`` on validation clips only.
Grid tau: ``tau_pos = tau``, ``tau_grow = tau / 2``, ``tau_shrink = 3 tau`` (the default
ratios), and tau = 0 is the fully unsmoothed baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from skier.eval.replay_metrics import replay_summary
from skier.io_utils import ensure_dir, write_csv, write_json
from skier.paths import configs_dir, repo_root, results_root
from skier.replay.controller import CropTrack, ReplayConfig, run_controller
from skier.replay.render import render_debug, render_replay, write_crops_csv
from skier.schema import Schema, load_schema
from skier.tracking.frames import ImageFrames, VideoFrames
from skier.tracking.run import boxes_of, read_track_csv

GRID_TAUS = (0.0, 0.1, 0.2, 0.4)
GRID_KS = (1.5, 2.0, 2.5, 3.0)
GRID_COLUMNS = ["tau", "k", "n_clips", "box_containment_mean", "frac_fully_inside", "ski_endpoints_inside_pct",
                "acceleration_mean", "acceleration_p95", "area_fraction_mean", "centering_error_mean"]


@dataclass
class ClipInput:
    name: str
    rows: list[dict]  # tracker CSV rows
    fps: float
    frame_size: tuple[float, float]
    gt_boxes: np.ndarray | None = None  # [T, 4] aligned with rows; NaN where not visible
    landmarks: tuple | None = None  # (xy [T,K,2], mask [T,K]) aligned with rows


def controller_inputs(rows: list[dict]):
    idx = np.array([r["frame_index"] for r in rows])
    return idx, boxes_of(rows), [r["status"] for r in rows], np.array([r["score"] for r in rows])


def compute_track(clip: ClipInput, cfg: ReplayConfig) -> tuple[np.ndarray, CropTrack]:
    idx, boxes, status, scores = controller_inputs(clip.rows)
    return idx, run_controller(boxes, idx / clip.fps, clip.frame_size, cfg, status, scores)


def clip_metrics(clip: ClipInput, track: CropTrack, schema: Schema) -> dict:
    target, label = (clip.gt_boxes, "ground truth") if clip.gt_boxes is not None else (track.raw_boxes, "tracker")
    return replay_summary(track.crops, track.timestamps, clip.frame_size, target, label,
                          clip.landmarks, schema if clip.landmarks is not None else None)


def grid_config(cfg: ReplayConfig, tau: float, k: float) -> ReplayConfig:
    if tau == 0:
        return cfg.unsmoothed().with_(k=k)
    return cfg.with_(k=k, tau_pos=tau, tau_grow=tau / 2, tau_shrink=3 * tau)


def run_grid(clips: Sequence[ClipInput], cfg: ReplayConfig, schema: Schema,
             taus: Sequence[float] = GRID_TAUS, ks: Sequence[float] = GRID_KS) -> list[dict]:
    """Pooled trade-off table: containment vs. acceleration vs. crop area, per (tau, k)."""
    table = []
    for tau in taus:
        for k in ks:
            ms = [clip_metrics(c, compute_track(c, grid_config(cfg, tau, k))[1], schema) for c in clips]

            def mean_of(get):
                vals = [v for v in (get(m) for m in ms) if v is not None]
                return round(float(np.mean(vals)), 5) if vals else None

            table.append({
                "tau": tau, "k": k, "n_clips": len(clips),
                "box_containment_mean": mean_of(lambda m: m.get("box_containment", {}).get("mean")),
                "frac_fully_inside": mean_of(lambda m: m.get("box_containment", {}).get("frac_fully_inside")),
                "ski_endpoints_inside_pct": mean_of(lambda m: m.get("landmark_containment", {}).get("ski_endpoints", {}).get("pct")),
                "acceleration_mean": mean_of(lambda m: m["acceleration"]["mean"]),
                "acceleration_p95": mean_of(lambda m: m["acceleration"]["p95"]),
                "area_fraction_mean": mean_of(lambda m: m["zoom"]["area_fraction_mean"]),
                "centering_error_mean": mean_of(lambda m: m.get("centering_error", {}).get("mean")),
            })
    return table


def run_clip(clip: ClipInput, frames_factory: Callable[[], Iterable], cfg: ReplayConfig, out: Path,
             schema: Schema, videos: bool = True, keypoints: dict | None = None) -> dict:
    out = ensure_dir(out)
    idx, track = compute_track(clip, cfg)
    _, baseline = compute_track(clip, cfg.unsmoothed())
    write_crops_csv(track, idx, out / "crops.csv")
    metrics = {"clip": clip.name, "fps": clip.fps, "frame_size": list(clip.frame_size),
               "config": dataclasses.asdict(cfg),
               "smoothed": clip_metrics(clip, track, schema),
               "unsmoothed_baseline": clip_metrics(clip, baseline, schema)}
    if videos:
        metrics["replay_frames"] = render_replay(frames_factory(), idx, track, out / "replay.mp4", clip.fps,
                                                 draw_raw_box=True, keypoints=keypoints, schema=schema)
        metrics["debug_frames"] = render_debug(frames_factory(), idx, track, out / "debug.mp4", clip.fps)
    write_json(metrics, out / "metrics.json")
    return metrics


EVAL_COLUMNS = ["clip", "source", "sequence", "n_frames", "box_target", "box_containment_mean", "box_containment_min",
                "frac_fully_inside", "centering_error_mean", "centering_error_p95", "acceleration_mean",
                "acceleration_p95", "area_fraction_mean", "log_zoom_rate_mean", "body_ski_inside", "body_ski_total",
                "ski_end_inside", "ski_end_total"]


def flat_metrics(m: dict) -> dict:
    """One ``replay_summary`` as a flat row (``EVAL_COLUMNS`` minus the clip identity)."""
    lm = m.get("landmark_containment", {})
    return {"n_frames": m["n_frames"], "box_target": m.get("target"),
            "box_containment_mean": m.get("box_containment", {}).get("mean"),
            "box_containment_min": m.get("box_containment", {}).get("min"),
            "frac_fully_inside": m.get("box_containment", {}).get("frac_fully_inside"),
            "centering_error_mean": m.get("centering_error", {}).get("mean"),
            "centering_error_p95": m.get("centering_error", {}).get("p95"),
            "acceleration_mean": m["acceleration"]["mean"], "acceleration_p95": m["acceleration"]["p95"],
            "area_fraction_mean": m["zoom"]["area_fraction_mean"], "log_zoom_rate_mean": m["zoom"]["log_zoom_rate_mean"],
            "body_ski_inside": lm.get("body_ski", {}).get("inside"), "body_ski_total": lm.get("body_ski", {}).get("total"),
            "ski_end_inside": lm.get("ski_endpoints", {}).get("inside"),
            "ski_end_total": lm.get("ski_endpoints", {}).get("total")}


def evaluate_clips(clips: Sequence[tuple[str, str, ClipInput]], cfg: ReplayConfig, schema: Schema) -> dict:
    """The partition-wide replay table (plan Step 8): per clip, the replay at ``cfg`` and its τ = 0 baseline.

    ``clips``: ``(source, sequence, clip)``; the sequence is the bootstrap cluster (D25).
    """
    rows = {"smoothed": [], "unsmoothed_baseline": []}
    for source, sequence, c in clips:
        for key, conf in (("smoothed", cfg), ("unsmoothed_baseline", cfg.unsmoothed())):
            rows[key].append({"clip": c.name, "source": source, "sequence": sequence,
                              **flat_metrics(clip_metrics(c, compute_track(c, conf)[1], schema))})
    return rows


def _frame_size_of(img: np.ndarray) -> tuple[float, float]:
    return float(img.shape[1]), float(img.shape[0])


def video_source(path: Path, fps: float | None):
    vf = VideoFrames(path)
    cap = cv2.VideoCapture(str(path))
    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (lambda: iter(VideoFrames(path))), (fps or vf.fps), size


def image_source(frames_dir: Path, files: list[str], fps: float):
    first = cv2.imread(str(frames_dir / files[0]))
    if first is None:
        raise FileNotFoundError(f"cannot read {frames_dir / files[0]}")
    return (lambda: iter(ImageFrames(frames_dir, files, fps))), fps, _frame_size_of(first)


def ski2dpose_landmarks(rows, video_id: str, segment_id: str, offset: int = 0, records=None):
    """GT landmarks aligned with tracker rows (labeled frames only; NaN elsewhere) + keypoints for drawing.

    ``records``: already loaded Ski 2DPose records (default: load them).
    """
    if records is None:
        from skier.data.ski2dpose import load_records
        from skier.paths import data_root

        s2d = data_root() / "ski2dpose"
        records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images")
    recs = [r for r in records if r.source_video_id == video_id and r.segment_id == segment_id and not r.excluded]
    pos = {r["frame_index"]: i for i, r in enumerate(rows)}
    xy = np.full((len(rows), 24, 2), np.nan)
    mask = np.zeros((len(rows), 24), dtype=bool)
    keypoints = {}
    for r in recs:
        f = r.frame_index + offset  # locate_segment: clip starts at frame_idx 0
        if f in pos:
            xy[pos[f]], mask[pos[f]] = r.keypoints_xy, r.reliable
            keypoints[f] = (r.keypoints_xy, r.reliable)
    return (xy, mask), keypoints


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Smoothed, centered replay from tracker output (plan Step 4).")
    p.add_argument("--config", type=Path, help="default: configs/replay/default.yaml")
    p.add_argument("--fps", type=float, help="frame rate (default: the video's; for SkiTB, the D22 table)")
    p.add_argument("--fps-table", type=Path, help="per-sequence SkiTB frame rates; default: "
                                                  "data/manifests/skitb_fps_v1.csv (D22)")
    p.add_argument("--out", type=Path)
    one = p.add_argument_group("single clip")
    one.add_argument("--track-csv", type=Path)
    one.add_argument("--video", type=Path)
    one.add_argument("--frames-dir", type=Path)
    one.add_argument("--frames-list", type=Path, help="text file: one image filename per tracker row")
    one.add_argument("--gt-clip-dir", type=Path, help="SkiTB SC clip folder: GT boxes for containment")
    one.add_argument("--ski2dpose-segment", metavar="VIDEO/SEGMENT", help="landmark containment on labeled frames")
    one.add_argument("--alignment-verified", action="store_true", help="the 1G alignment check passed for this video")
    one.add_argument("--apply-offset", type=int, default=0)
    one.add_argument("--no-video", action="store_true")
    g = p.add_argument_group("grid (validation clips only)")
    g.add_argument("--grid", action="store_true")
    g.add_argument("--track-dir", type=Path, help="CSVs from run_tracker.py --skitb-root")
    g.add_argument("--skitb-root", type=Path)
    g.add_argument("--split-json", type=Path)
    g.add_argument("--partition", default="val")
    e = p.add_argument_group("evaluate a whole partition at the chosen settings (Step 8)")
    e.add_argument("--evaluate", action="store_true",
                   help="SkiTB clips (--track-dir, --skitb-root, --split-json) and/or Ski 2DPose segments "
                        "(--ski2dpose-track-dir); low_conf frames handled as in the integration config (D24)")
    e.add_argument("--ski2dpose-track-dir", type=Path, help="CSVs from run_tracker.py --ski2dpose-partition")
    e.add_argument("--pose-split", type=Path, help="default: data/manifests/pose_split_v1.json")
    e.add_argument("--integration-config", type=Path, help="default: configs/integration/default.yaml")
    e.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8, once)")
    return p


def evaluate_main(args, cfg: ReplayConfig, schema: Schema, skitb_fps) -> int:
    """``--evaluate``: per-clip replay metrics over a whole partition, written as CSV and JSON."""
    from skier.integration.confidence import replay_status_rows
    from skier.integration.config import IntegrationConfig
    from skier.io_utils import read_json
    from skier.pose.crop_error import FINAL_ONLY

    if args.partition in FINAL_ONLY and not args.final_test:
        print(f"error: {args.partition!r} is used once, in Step 8; pass --final-test then", file=sys.stderr)
        return 1
    skitb = bool(args.track_dir and args.skitb_root and args.split_json)
    if not skitb and args.ski2dpose_track_dir is None:
        print("error: --evaluate needs --track-dir with --skitb-root and --split-json, and/or --ski2dpose-track-dir",
              file=sys.stderr)
        return 1
    icfg = IntegrationConfig.from_yaml(args.integration_config or configs_dir() / "integration" / "default.yaml")
    try:
        icfg.require("conf_signal", "replay_uses_status")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    clips = []
    if skitb:
        from skier.data.skitb import frame_file, load_skitb_clips

        for c in load_skitb_clips(args.skitb_root, args.split_json, args.partition):
            csv = args.track_dir / f"{c.sequence_id}__{c.clip_id}.csv"
            if not csv.is_file():
                print(f"error: no track CSV for SkiTB clip {c.sequence_id}/{c.clip_id} in {args.track_dir}", file=sys.stderr)
                return 1
            rows = replay_status_rows(read_track_csv(csv), icfg)
            pos = np.array([r["frame_index"] for r in rows])
            gt = np.where(c.visible[pos][:, None], c.boxes_xyxy[pos], np.nan)
            first = cv2.imread(str(c.frames_dir / frame_file(c.frame_files[0])))
            clips.append(("skitb", c.sequence_id,
                          ClipInput(csv.stem, rows, skitb_fps(c.sequence_id), _frame_size_of(first), gt)))
    if args.ski2dpose_track_dir is not None:
        from skier.data.alignment import locate_segment
        from skier.data.ski2dpose import load_records
        from skier.paths import data_root
        from skier.splits import apply_split

        s2d = data_root() / "ski2dpose"
        records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images")
        apply_split(records, read_json(args.pose_split or repo_root() / "data/manifests/pose_split_v1.json"))
        for vid, seg in sorted({(r.source_video_id, r.segment_id) for r in records
                                if r.split == args.partition and not r.excluded}):
            csv = args.ski2dpose_track_dir / f"{vid}__{seg}.csv"
            if not csv.is_file():
                print(f"error: no track CSV for Ski 2DPose segment {vid}/{seg} in {args.ski2dpose_track_dir}",
                      file=sys.stderr)
                return 1
            rows = replay_status_rows(read_track_csv(csv), icfg)
            _, fps, size = video_source(locate_segment(s2d / "videos", vid, seg)[0], None)
            landmarks, _ = ski2dpose_landmarks(rows, vid, seg, records=records)
            clips.append(("ski2dpose", f"{vid}/{seg}", ClipInput(csv.stem, rows, fps, size, None, landmarks)))
    result = evaluate_clips(clips, cfg, schema)
    out = ensure_dir(args.out or results_root() / "replay" / f"eval-{args.partition}")
    for key, rows in result.items():
        write_csv(rows, out / f"{key}.csv", EVAL_COLUMNS)
    write_json({"partition": args.partition, "config": dataclasses.asdict(cfg),
                "integration_config": dataclasses.asdict(icfg),
                "n_clips": {s: sum(c[0] == s for c in clips) for s in ("skitb", "ski2dpose")}, **result},
               out / "replay_eval.json")
    print(f"replay over {len(clips)} clips ({args.partition}) → {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ReplayConfig.from_yaml(args.config or configs_dir() / "replay" / "default.yaml")
    schema = load_schema()

    if args.fps is not None and args.fps_table is not None:
        print("error: give --fps or --fps-table, not both", file=sys.stderr)
        return 1

    def skitb_fps(sequence_id: str) -> float:
        """``--fps`` if given, else the sequence's rate from the D22 table."""
        from skier.data.skitb import FPS_TABLE, fps_for, load_fps_table

        if args.fps is not None:
            return args.fps
        return fps_for(load_fps_table(args.fps_table or repo_root() / FPS_TABLE), sequence_id)

    if args.evaluate:
        return evaluate_main(args, cfg, schema, skitb_fps)
    if args.grid:
        if not (args.track_dir and args.skitb_root and args.split_json):
            print("error: --grid needs --track-dir, --skitb-root and --split-json", file=sys.stderr)
            return 1
        if args.partition == "test":
            print("error: never tune on test", file=sys.stderr)
            return 1
        from skier.data.skitb import frame_file, load_skitb_clips

        clips = []
        for c in load_skitb_clips(args.skitb_root, args.split_json, args.partition):
            csv = args.track_dir / f"{c.sequence_id}__{c.clip_id}.csv"
            if not csv.is_file():
                continue
            rows = read_track_csv(csv)
            pos = np.array([r["frame_index"] for r in rows])
            gt = np.where(c.visible[pos][:, None], c.boxes_xyxy[pos], np.nan)
            first = cv2.imread(str(c.frames_dir / frame_file(c.frame_files[0])))
            clips.append(ClipInput(csv.stem, rows, skitb_fps(c.sequence_id), _frame_size_of(first), gt))
        table = run_grid(clips, cfg, schema)
        out = ensure_dir(args.out or results_root() / "replay" / f"grid-{args.partition}")
        write_csv(table, out / "grid.csv", GRID_COLUMNS)
        write_json({"partition": args.partition, "n_clips": len(clips), "table": table}, out / "grid.json")
        print(f"grid over {len(clips)} clips → {out / 'grid.csv'}")
        return 0

    if args.track_csv is None or (args.video is None and args.frames_dir is None):
        print("error: give --track-csv and --video (or --frames-dir with --frames-list), or use --grid", file=sys.stderr)
        return 1
    rows = read_track_csv(args.track_csv)
    if args.video is not None:
        factory, fps, size = video_source(args.video, args.fps)
    else:
        if args.frames_list is None or (args.fps is None and args.gt_clip_dir is None):
            print("error: --frames-dir needs --frames-list, and --fps unless --gt-clip-dir names a SkiTB clip",
                  file=sys.stderr)
            return 1
        files = [line.strip() for line in args.frames_list.read_text().splitlines() if line.strip()]
        fps = args.fps if args.fps is not None else skitb_fps(args.gt_clip_dir.parents[1].name)
        factory, fps, size = image_source(args.frames_dir, files, fps)

    gt = landmarks = keypoints = None
    if args.gt_clip_dir is not None:
        from skier.data.skitb import load_clip

        c = load_clip(args.gt_clip_dir, args.gt_clip_dir.parents[1].name, "unknown")
        pos = np.array([r["frame_index"] for r in rows])
        gt = np.where(c.visible[pos][:, None], c.boxes_xyxy[pos], np.nan)
    if args.ski2dpose_segment:
        vid, seg = args.ski2dpose_segment.split("/", 1)
        landmarks, keypoints = ski2dpose_landmarks(rows, vid, seg, args.apply_offset)
    clip = ClipInput(args.track_csv.stem, rows, fps, size, gt, landmarks)
    out = args.out or results_root() / "replay" / args.track_csv.stem
    m = run_clip(clip, factory, cfg, out, schema, videos=not args.no_video, keypoints=keypoints)
    if args.ski2dpose_segment:
        m["alignment_verified"] = bool(args.alignment_verified)
        if not args.alignment_verified:
            m["note"] = "landmark containment assumes frame_idx alignment that has not been verified (guide 1G)"
        write_json(m, Path(out) / "metrics.json")
    s, b = m["smoothed"], m["unsmoothed_baseline"]
    print(f"accel mean {s['acceleration']['mean']:.4f} (unsmoothed {b['acceleration']['mean']:.4f}), "
          f"area {s['zoom']['area_fraction_mean']:.2f} → {out}")
    return 0
