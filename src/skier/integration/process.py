"""Process one clip end to end (plan Step 7): tracker → pose on the tracker crop → replay.

Stages (so another tracker can be swapped in without touching the pose stack):
1. Tracking writes ``track.csv``, in process with SiamFC or given with ``--track-csv``.
   Either way the later stages read the CSV, so both paths give the same result.
2. A status per frame (``skier.integration.confidence.tracking_status``).
3. Pose on every tracked frame. The crop is ``expand_box(tracker_box, k_pose)`` with the
   model's padding, cut from the original frame, so the replay never changes pose. With
   ``pose_refine`` (D26) a second pass predicts on the box around the confident first-pass
   landmarks (``skier.integration.refine``); ``pose_crop_M`` is the crop actually used.
4. The replay controller (D23 defaults; low_conf frames drive it as lost only if
   ``replay_uses_status``), the replay video with the skeleton, and the debug video.

``frames.jsonl`` holds one line per tracked frame (plan §12 fields) with the raw
predictions. Hiding the overlay on low_conf frames is display-only: ``raw_keypoints_xy``
is never filtered.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from skier.eval.replay_metrics import replay_summary
from skier.geometry import crop_affine
from skier.integration.config import IntegrationConfig, config_sha256
from skier.integration.confidence import replay_status_rows, tracking_status
from skier.integration.propagation import tracker_crop_boxes
from skier.integration.refine import refined_boxes
from skier.io_utils import ensure_dir, read_json, sha256_file, write_json
from skier.paths import configs_dir, data_root, repo_root, results_root
from skier.pose.crop_error import FINAL_ONLY
from skier.pose.infer import HeatmapFn, predict_boxes
from skier.pose.preprocess import INPUT_H, INPUT_W, pose_crop_box
from skier.replay.controller import ReplayConfig, run_controller
from skier.replay.render import render_debug, render_replay, write_crops_csv
from skier.schema import Schema, load_schema
from skier.tracking.run import boxes_of, read_track_csv, track_clip, write_track_csv
from skier.tracking.trackers import (add_tracker_args, check_tracker, tracker_factory, tracker_info,
                                     write_tracker_info)

JSONL_FIELDS = ["video_id", "frame_index", "timestamp", "tracker_box", "tracker_score", "tracking_status",
                "pose_crop_M", "pose_refined", "raw_keypoints_xy", "raw_keypoint_scores", "replay_crop",
                "overlay_visible", "checkpoint_sha256", "config_sha256"]
STATUSES = ("init", "track", "low_conf")
SHA = re.compile(r"[0-9a-f]{64}")


def track_stage(tracker, frames: Iterable, init_index: int, init_box, path: Path) -> list[dict]:
    """Track, write ``path``, and return the rows as read back (what ``--track-csv`` would give)."""
    write_track_csv(track_clip(tracker, frames, init_index, init_box), path)
    return read_track_csv(path)


def pose_on_frames(frames: Iterable[tuple[int, np.ndarray]], rows: Sequence[dict], heatmap_fn: HeatmapFn,
                   num_keypoints: int, k_pose: float, expand: float, batch_size: int = 16,
                   refine: tuple[float, float] | None = None, schema: Schema | None = None):
    """``(xy [T,K,2], scores [T,K], M [T,2,3], refined [T])`` for the tracked rows; NaN where there is no box.

    ``refine = (s_min, k_refine)`` runs the D26 second pass (``schema`` required); ``M`` is the crop used.
    """
    boxes = tracker_crop_boxes(boxes_of(rows), k_pose)
    n = len(rows)
    xy, scores, Ms = np.full((n, num_keypoints, 2), np.nan), np.full((n, num_keypoints), np.nan), np.full((n, 2, 3), np.nan)
    refined = np.zeros(n, dtype=bool)
    pos = {r["frame_index"]: i for i, r in enumerate(rows)}
    last = max(pos) if pos else -1
    images, todo = [], []

    def flush():
        if todo:
            p, s = predict_boxes(images, [boxes[i] for i in todo], heatmap_fn, expand)
            if refine is not None:
                rb = refined_boxes(p, s, schema, *refine)
                ok = np.flatnonzero(np.isfinite(rb).all(axis=1))
                if len(ok):
                    p[ok], s[ok] = predict_boxes([images[j] for j in ok], rb[ok], heatmap_fn, expand)
                    for j in ok:
                        Ms[todo[j]] = crop_affine(pose_crop_box(rb[j], expand), INPUT_W, INPUT_H)
                        refined[todo[j]] = True
            xy[todo], scores[todo] = p, s
            images.clear()
            todo.clear()

    for idx, frame in frames:
        if idx > last:
            break
        i = pos.get(idx)
        if i is None or not np.isfinite(boxes[i]).all():
            continue
        Ms[i] = crop_affine(pose_crop_box(boxes[i], expand), INPUT_W, INPUT_H)
        images.append(frame)
        todo.append(i)
        if len(todo) == batch_size:
            flush()
    flush()
    return xy, scores, Ms, refined


def _num(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _nums(a) -> list:
    return [_nums(x) for x in a] if np.ndim(a) > 0 else _num(a)


def process_clip(frames_factory: Callable[[], Iterable], rows: Sequence[dict], fps: float,
                 frame_size: tuple[float, float], heatmap_fn: HeatmapFn, num_keypoints: int, expand: float,
                 icfg: IntegrationConfig, rcfg: ReplayConfig, schema: Schema, out: Path, video_id: str,
                 checkpoint_sha256: str, videos: bool = True, batch_size: int = 16) -> dict:
    icfg.require("k_pose", "conf_signal", "replay_uses_status", "pose_refine")
    out = ensure_dir(out)
    status = tracking_status(rows, icfg)
    refine = (icfg.refine_min_score, icfg.refine_k) if icfg.pose_refine else None
    xy, kp_scores, Ms, refined = pose_on_frames(frames_factory(), rows, heatmap_fn, num_keypoints, icfg.k_pose,
                                                expand, batch_size, refine, schema)
    idx = np.array([r["frame_index"] for r in rows])
    ts = idx / fps
    drive = [r["status"] for r in replay_status_rows(rows, icfg)]
    track = run_controller(boxes_of(rows), ts, frame_size, rcfg, drive, [r["score"] for r in rows])
    finite = np.isfinite(xy).all(axis=2)
    overlay = [s != "low_conf" and bool(finite[i].any()) for i, s in enumerate(status)]
    cfg_sha = config_sha256(icfg, rcfg, {"expand": expand})

    with open(out / "frames.jsonl", "w") as f:
        for i, r in enumerate(rows):
            box = boxes_of([r])[0]
            line = {"video_id": video_id, "frame_index": int(idx[i]), "timestamp": float(ts[i]),
                    "tracker_box": _nums(box) if np.isfinite(box).all() else None,
                    "tracker_score": _num(r["score"]), "tracking_status": status[i],
                    "pose_crop_M": _nums(Ms[i]) if np.isfinite(Ms[i]).all() else None,
                    "pose_refined": bool(refined[i]), "raw_keypoints_xy": _nums(xy[i]), "raw_keypoint_scores": _nums(kp_scores[i]),
                    "replay_crop": _nums(track.crops[i]), "overlay_visible": overlay[i],
                    "checkpoint_sha256": checkpoint_sha256, "config_sha256": cfg_sha}
            f.write(json.dumps(line) + "\n")
    write_crops_csv(track, idx, out / "crops.csv")
    keypoints = {int(idx[i]): (xy[i], finite[i]) for i in range(len(rows)) if overlay[i]}
    metrics = {"video_id": video_id, "fps": fps, "frame_size": list(frame_size), "n_frames": len(rows),
               "status_counts": dict(Counter(status)), "overlay_frames": int(sum(overlay)),
               "refined_frames": int(refined.sum()),
               "integration_config": dataclasses.asdict(icfg), "replay_config": dataclasses.asdict(rcfg),
               "expand": expand, "config_sha256": cfg_sha, "checkpoint_sha256": checkpoint_sha256,
               "replay": replay_summary(track.crops, track.timestamps, frame_size, track.raw_boxes, "tracker")}
    if videos:
        metrics["replay_frames"] = render_replay(frames_factory(), idx, track, out / "replay.mp4", fps,
                                                 draw_raw_box=True, keypoints=keypoints, schema=schema)
        metrics["debug_frames"] = render_debug(frames_factory(), idx, track, out / "debug.mp4", fps)
    write_json(metrics, out / "metrics.json")
    return metrics


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec(v, n: int, nullable: bool = False) -> bool:
    return isinstance(v, list) and len(v) == n and all(_is_num(x) or (nullable and x is None) for x in v)


def validate_jsonl(path: str | Path, num_keypoints: int) -> int:
    """Check ``frames.jsonl`` line by line; returns the number of lines. Raises ``ValueError`` on the first problem."""
    prev_idx = prev_t = None
    n = 0
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        d = json.loads(text)

        def bad(msg):
            raise ValueError(f"{path} line {n}: {msg}")

        if list(d) != JSONL_FIELDS:
            bad(f"fields {list(d)}")
        if not isinstance(d["frame_index"], int) or (prev_idx is not None and d["frame_index"] <= prev_idx):
            bad("frame_index must be an increasing integer")
        if not _is_num(d["timestamp"]) or (prev_t is not None and d["timestamp"] <= prev_t):
            bad("timestamp must increase")
        if d["tracking_status"] not in STATUSES or (d["tracking_status"] == "init") != (n == 1):
            bad(f"tracking_status {d['tracking_status']!r} (init on the first line only)")
        if not (d["tracker_box"] is None or _vec(d["tracker_box"], 4)):
            bad("tracker_box must be 4 numbers or null")
        if not (d["tracker_score"] is None or _is_num(d["tracker_score"])):
            bad("tracker_score must be a number or null")
        M = d["pose_crop_M"]
        if (M is None) != (d["tracker_box"] is None) or not (M is None or (isinstance(M, list) and len(M) == 2
                                                                              and all(_vec(r, 3) for r in M))):
            bad("pose_crop_M must be 2x3 exactly when there is a tracker box")
        if not isinstance(d["pose_refined"], bool) or (d["pose_refined"] and M is None):
            bad("pose_refined must be a bool, and false without a pose crop")
        kp = d["raw_keypoints_xy"]
        if not (isinstance(kp, list) and len(kp) == num_keypoints and all(_vec(p, 2, nullable=True) for p in kp)):
            bad(f"raw_keypoints_xy must be {num_keypoints} [x, y] pairs")
        if not _vec(d["raw_keypoint_scores"], num_keypoints, nullable=True):
            bad(f"raw_keypoint_scores must have {num_keypoints} entries")
        if not _vec(d["replay_crop"], 4):
            bad("replay_crop must be 4 numbers")
        if not isinstance(d["overlay_visible"], bool) or (d["overlay_visible"] and d["tracking_status"] == "low_conf"):
            bad("overlay_visible must be a bool, and false on low_conf frames")
        if not all(isinstance(d[k], str) and SHA.fullmatch(d[k]) for k in ("checkpoint_sha256", "config_sha256")):
            bad("checkpoint_sha256 and config_sha256 must be sha256 hex digests")
        prev_idx, prev_t = d["frame_index"], d["timestamp"]
    if n == 0:
        raise ValueError(f"{path} is empty")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One clip end to end (plan Step 7): tracker → pose on the tracker crop "
                                            "→ replay with the skeleton; writes frames.jsonl, replay.mp4, debug.mp4.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--skitb-clip", metavar="SEQUENCE/CAMERA", help="a SkiTB SC clip, e.g. AL0027/3")
    src.add_argument("--ski2dpose-segment", metavar="VIDEO/SEGMENT", help="a Ski 2DPose segment clip")
    src.add_argument("--video", type=Path, help="any video, with --init-frame and --init-box")
    p.add_argument("--init-frame", type=int)
    p.add_argument("--init-box", help="x1,y1,x2,y2 in original pixels (with --video)")
    p.add_argument("--skitb-root", type=Path)
    p.add_argument("--split-json", type=Path, help="SkiTB split file (with --skitb-clip)")
    p.add_argument("--pose-split", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--partition", default="val", help="the clip's partition (checked)")
    p.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8, once)")
    p.add_argument("--track-csv", type=Path, help="use this tracker output instead of running a tracker")
    add_tracker_args(p, weights_required=False)
    p.add_argument("--checkpoint", type=Path, required=True, help="our best.pt (Step 7 demos: E2-s230)")
    p.add_argument("--hrnet-config", type=Path, required=True)
    p.add_argument("--config", type=Path, help="default: configs/integration/default.yaml")
    p.add_argument("--replay-config", type=Path, help="default: configs/replay/default.yaml")
    p.add_argument("--k-pose", type=float, help="override the config")
    p.add_argument("--conf-signal", choices=("none", "raw", "relative"), help="override the config")
    p.add_argument("--conf-threshold", type=float, help="override the config")
    p.add_argument("--replay-uses-status", choices=("true", "false"), help="override the config")
    p.add_argument("--pose-refine", choices=("true", "false"), help="override the config (D26)")
    p.add_argument("--fps", type=float, help="with --video: override the file's frame rate")
    p.add_argument("--fps-table", type=Path, help="SkiTB frame rates; default: data/manifests/skitb_fps_v1.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/integration/clips/<name>")
    return p


def _source(args):
    """``(name, video_id, frames_factory, fps, frame_size, init_index, init_box)``; refuses held-out partitions."""
    from skier.replay.run import image_source, video_source

    if args.partition in FINAL_ONLY and not args.final_test:
        raise ValueError(f"{args.partition!r} is used once, in Step 8; pass --final-test then")
    if args.skitb_clip:
        from skier.data.skitb import FPS_TABLE, fps_for, frame_file, load_fps_table, load_skitb_clips

        if not (args.skitb_root and args.split_json):
            raise ValueError("--skitb-clip needs --skitb-root and --split-json")
        seq, cam = args.skitb_clip.split("/", 1)
        clip = next((c for c in load_skitb_clips(args.skitb_root, args.split_json, args.partition)
                     if c.sequence_id == seq and c.clip_id == cam), None)
        if clip is None:
            raise ValueError(f"no SkiTB clip {args.skitb_clip} in partition {args.partition!r}")
        fps = fps_for(load_fps_table(args.fps_table or repo_root() / FPS_TABLE), seq)
        factory, fps, size = image_source(clip.frames_dir, [frame_file(e) for e in clip.frame_files], fps)
        init = int(np.flatnonzero(clip.visible)[0])
        return f"{seq}__{cam}", f"skitb/{seq}/{cam}", factory, fps, size, init, clip.boxes_xyxy[init]
    if args.ski2dpose_segment:
        from skier.data.alignment import locate_segment
        from skier.data.ski2dpose import load_records
        from skier.splits import apply_split

        vid, seg = args.ski2dpose_segment.split("/", 1)
        s2d = data_root() / "ski2dpose"
        records = [r for r in load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images")
                   if r.source_video_id == vid and r.segment_id == seg]
        apply_split(records, read_json(args.pose_split or repo_root() / "data/manifests/pose_split_v1.json"))
        if not records or records[0].split != args.partition:
            raise ValueError(f"{vid}/{seg} is not a segment of partition {args.partition!r}")
        first = min((r for r in records if not r.excluded), key=lambda r: r.frame_index)
        video, start = locate_segment(s2d / "videos", vid, seg)
        factory, fps, size = video_source(video, None)
        return f"{vid}__{seg}", f"ski2dpose/{vid}/{seg}", factory, fps, size, start + first.frame_index, \
            first.reference_box_xyxy
    if args.init_frame is None or args.init_box is None:
        raise ValueError("--video needs --init-frame and --init-box")
    factory, fps, size = video_source(args.video, args.fps)
    box = np.array([float(v) for v in args.init_box.split(",")])
    return args.video.stem, f"video/{args.video.stem}", factory, fps, size, args.init_frame, box


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    icfg = IntegrationConfig.from_yaml(args.config or configs_dir() / "integration" / "default.yaml")
    overrides = {k: v for k, v in {"k_pose": args.k_pose, "conf_signal": args.conf_signal,
                                   "conf_threshold": args.conf_threshold,
                                   "replay_uses_status": None if args.replay_uses_status is None
                                   else args.replay_uses_status == "true",
                                   "pose_refine": None if args.pose_refine is None
                                   else args.pose_refine == "true"}.items() if v is not None}
    try:
        icfg = icfg.with_(**overrides).require("k_pose", "conf_signal", "replay_uses_status", "pose_refine")
        name, video_id, factory, fps, size, init_index, init_box = _source(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rcfg = ReplayConfig.from_yaml(args.replay_config or configs_dir() / "replay" / "default.yaml")
    out = ensure_dir(args.out or results_root() / "integration" / "clips" / name)

    if args.track_csv is not None:
        try:
            check_tracker(args.track_csv.parent, icfg.tracker)  # the gating threshold is the tracker's own (D28)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        rows = read_track_csv(args.track_csv)
        if rows[0]["status"] != "init" or rows[0]["frame_index"] != init_index:
            print(f"error: {args.track_csv} doesn't start with an init row at frame {init_index}", file=sys.stderr)
            return 1
    else:
        try:
            if args.tracker != icfg.tracker:
                raise ValueError(f"--tracker {args.tracker}, but the integration config is for {icfg.tracker} (D28)")
            tracker = tracker_factory(args)
            info = tracker_info(args)
        except ValueError as exc:
            print(f"error: {exc}; or give --track-csv", file=sys.stderr)
            return 1
        rows = track_stage(tracker(), factory(), init_index, init_box, out / "track.csv")
        write_tracker_info(info, out)

    schema = load_schema()
    from skier.pose.model import build_hrnet, torch_heatmap_fn  # torch + mmpose from here on
    from skier.pose.train import load_checkpoint

    model, _ = build_hrnet(schema.num_keypoints, args.hrnet_config, None, device=args.device)
    ck = load_checkpoint(args.checkpoint, model, schema=schema, map_location=args.device)
    expand = float(ck.get("config", {}).get("expand", 1.25))
    m = process_clip(factory, rows, fps, size, torch_heatmap_fn(model, args.device), schema.num_keypoints, expand,
                     icfg, rcfg, schema, out, video_id, sha256_file(args.checkpoint), videos=not args.no_video,
                     batch_size=args.batch_size)
    m.update(checkpoint=str(args.checkpoint), checkpoint_epoch=ck.get("epoch"), partition=args.partition,
             track_csv=str(args.track_csv or out / "track.csv"))
    write_json(m, out / "metrics.json")
    print(f"{name}: {m['n_frames']} frames, status {m['status_counts']}, overlay on {m['overlay_frames']} → {out}")
    return 0
