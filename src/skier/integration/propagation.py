"""Propagation evaluation (plan Step 7): how much tracker error costs pose accuracy.

For each usable segment of a Ski 2DPose partition, the tracker is initialized once at the
first labelled frame with that frame's reference box (``run_tracker.py
--ski2dpose-partition``) and runs on every frame. At every later labelled frame, the same
model sees two crops of the same dataset image:
- the reference crop, from the labels (as in Steps 5–6);
- the tracker crop: ``expand_box(tracker_box, k_pose)``, then the model's usual padding.

Both are scored against the same labels with the same denominators, and the normalizer is
always the reference-box diagonal. So ``PCK_reference − PCK_tracked`` isolates the crop.
Labels the tracker crop cuts off are misses, and their share is reported.
Initialization frames are excluded.

A segment is usable when every labelled image's best NCC match (guide 1G) is its own frame
(offset 0, score ≥ ``MIN_SCORE``) and the clip's frame size equals the image size. The
margin rule may fail in near-static stretches, as it did in 1G.

``k_pose`` is chosen on val by the rule fixed in the Step 7 plan: the highest mean of
tracked body and ski PCK@0.1 for the recipe, averaged over its seeds. Values within
``K_TIE`` points of the best tie, and a tie goes to the smallest ``k_pose`` (higher
resolution).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from skier.data.records import PoseRecord
from skier.eval.pose_metrics import THRESHOLDS, group_pck, gt_arrays
from skier.geometry import expand_box, iou_xyxy
from skier.integration.config import IntegrationConfig
from skier.io_utils import ensure_dir, read_json, sha256_file, write_csv, write_json
from skier.paths import configs_dir, data_root, results_root
from skier.pose.crop_error import FINAL_ONLY
from skier.pose.evaluate import cut_off, inside_crop, score
from skier.pose.infer import HeatmapFn, predict_records
from skier.schema import Schema, load_schema

K_POSE_GRID = (1.0, 1.1, 1.2, 1.35, 1.5)
K_TIE = 0.5  # points
K_RULE = (f"highest mean of tracked body and ski PCK@0.1 (recipe, mean over seeds); within {K_TIE} points of the "
          "best counts as a tie, and a tie goes to the smallest k_pose")
SEGMENT_COLUMNS = ["video_id", "segment_id", "n_labeled", "n_own_frame", "n_verified", "frame_w", "frame_h",
                   "fps", "image_sizes", "usable"]
LONG_COLUMNS = ["checkpoint", "k_pose", "crop", "threshold", "group", "hits", "total", "pct", "cut", "cut_share"]


def segment_alignment(records: Sequence[PoseRecord], videos_root: Path, window: int = 5) -> tuple[list[dict], list[dict]]:
    """Per-image alignment rows and per-segment summaries (``usable`` per the rule above)."""
    from skier.data.alignment import MIN_SCORE, align_segment, locate_segment

    groups = defaultdict(list)
    for r in records:
        if not r.excluded:
            groups[(r.source_video_id, r.segment_id)].append(r)
    image_rows, seg_rows = [], []
    for (vid, seg), recs in sorted(groups.items()):
        video, start = locate_segment(videos_root, vid, seg)
        rows = align_segment(video, start, [(r.image_id, r.frame_index, r.image_path) for r in recs], window)
        cap = cv2.VideoCapture(str(video))
        fw, fh, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        sizes = {(r.image_width, r.image_height) for r in recs}
        own = [a for a in rows if a.get("offset") == 0 and (a.get("best_score") or 0) >= MIN_SCORE]
        for a in rows:
            a.update(video_id=vid, segment_id=seg)
        image_rows += rows
        seg_rows.append({"video_id": vid, "segment_id": seg, "n_labeled": len(rows), "n_own_frame": len(own),
                         "n_verified": sum(bool(a["verified"]) for a in rows), "frame_w": fw, "frame_h": fh,
                         "fps": fps, "image_sizes": ";".join(f"{w}x{h}" for w, h in sorted(sizes)),
                         "usable": len(own) == len(rows) and sizes == {(fw, fh)}})
    return image_rows, seg_rows


def split_init(records: Sequence[PoseRecord], segments) -> tuple[dict, list[PoseRecord]]:
    """(init record per segment, the later labelled records). Init = the first non-excluded label."""
    by_seg = defaultdict(list)
    wanted = {tuple(s) for s in segments}
    for r in records:
        if not r.excluded and (r.source_video_id, r.segment_id) in wanted:
            by_seg[(r.source_video_id, r.segment_id)].append(r)
    init, scored = {}, []
    for key, recs in sorted(by_seg.items()):
        recs.sort(key=lambda r: r.frame_index)
        init[key] = recs[0]
        scored += recs[1:]
    return init, scored


def iou_bin(iou: float) -> str:
    return "iou>=0.5" if iou >= 0.5 else ("0.1<=iou<0.5" if iou >= 0.1 else "iou<0.1")


def time_bin(seconds: float) -> str:
    return "<2s" if seconds < 2 else ("2-5s" if seconds < 5 else ">=5s")


def tracker_at(records: Sequence[PoseRecord], init: dict, track_dir: Path, fps: dict,
               icfg: IntegrationConfig | None = None) -> dict:
    """Tracker output at each record's frame: boxes, scores, status, IoU vs. the reference box, time since init.

    Refuses a track CSV that wasn't initialized at the segment's init label with its reference box.
    """
    from skier.integration.confidence import tracking_status
    from skier.tracking.run import read_track_csv

    rows_of, status_of = {}, {}
    for key, r0 in init.items():
        rows = read_track_csv(Path(track_dir) / f"{key[0]}__{key[1]}.csv")
        box0 = np.array([rows[0][k] for k in ("x1", "y1", "x2", "y2")])
        if (rows[0]["status"] != "init" or rows[0]["frame_index"] != r0.frame_index
                or not np.allclose(box0, r0.reference_box_xyxy, atol=1e-3)):
            raise ValueError(f"track CSV for {key[0]}/{key[1]} was not initialized at label {r0.sample_id} "
                             "with its reference box")
        rows_of[key] = {row["frame_index"]: row for row in rows}
        status_of[key] = dict(zip((row["frame_index"] for row in rows), tracking_status(rows, icfg))) if icfg else {}
    n = len(records)
    out = {"boxes": np.full((n, 4), np.nan), "scores": np.full(n, np.nan), "status": [], "iou": np.zeros(n),
           "seconds": np.zeros(n)}
    for i, r in enumerate(records):
        key = (r.source_video_id, r.segment_id)
        row = rows_of[key].get(r.frame_index)
        if row is not None:
            out["boxes"][i] = [row[k] for k in ("x1", "y1", "x2", "y2")]
            out["scores"][i] = row["score"]
            out["iou"][i] = iou_xyxy(out["boxes"][i], r.reference_box_xyxy)
        out["status"].append(status_of[key].get(r.frame_index, "missing") if icfg else "unknown")
        out["seconds"][i] = (r.frame_index - init[key].frame_index) / fps[key]
    return out


def tracker_crop_boxes(tracker_boxes: np.ndarray, k_pose: float) -> np.ndarray:
    out = np.full_like(np.asarray(tracker_boxes, dtype=np.float64), np.nan)
    for i, b in enumerate(tracker_boxes):
        if np.isfinite(b).all():
            out[i] = expand_box(b, k_pose)
    return out


def predict_on(records: Sequence[PoseRecord], heatmap_fn: HeatmapFn, boxes: np.ndarray, expand: float,
               batch_size: int = 32) -> np.ndarray:
    """``[N, K, 2]``; a record without a finite box gets NaN (a miss)."""
    finite = np.isfinite(boxes).all(axis=1)
    xy = np.full((len(records), records[0].keypoints_xy.shape[0], 2), np.nan) if records else np.zeros((0, 0, 2))
    idx = np.flatnonzero(finite)
    if len(idx):
        xy[idx] = predict_records([records[i] for i in idx], heatmap_fn, expand, batch_size, boxes=boxes[idx]).xy
    return xy


def score_crops(pred_xy: np.ndarray, records: Sequence[PoseRecord], schema: Schema, boxes: np.ndarray,
                expand: float) -> dict:
    """``evaluate.score`` for crops around ``boxes``; a missing box cuts off every label."""
    finite = np.isfinite(boxes).all(axis=1)
    safe = [b if f else r.reference_box_xyxy for b, f, r in zip(boxes, finite, records)]
    out = score(pred_xy, records, schema, safe, expand)
    inside = inside_crop(records, safe, expand)
    inside[~finite] = False
    out["cut_off"] = cut_off(inside, gt_arrays(records)[1], schema)
    return out


def _delta(a: dict, b: dict) -> float | None:
    return None if a["pct"] is None or b["pct"] is None else a["pct"] - b["pct"]


def compare_crops(records: Sequence[PoseRecord], schema: Schema, ref_pred: np.ndarray, trk_pred: np.ndarray,
                  trk_boxes: np.ndarray, expand: float, strata_keys: dict[str, Sequence[str]]) -> dict:
    """Reference vs. tracked scores, Δ = reference − tracked, and the same per stratum (extra rows only)."""
    ref = score(ref_pred, records, schema, expand=expand)
    trk = score_crops(trk_pred, records, schema, trk_boxes, expand)
    delta = {f"{t:g}": {g: _delta(ref["groups"][f"{t:g}"][g], trk["groups"][f"{t:g}"][g]) for g in schema.groups}
             for t in THRESHOLDS}
    gt, rel, diag, _ = gt_arrays(records)

    def block(mask: np.ndarray) -> dict:
        idx = np.flatnonzero(mask)
        r = group_pck(ref_pred[idx], gt[idx], rel[idx], diag[idx], schema, 0.1)
        t = group_pck(trk_pred[idx], gt[idx], rel[idx], diag[idx], schema, 0.1)
        return {"n_images": int(len(idx)), "reference": r, "tracked": t, "delta": {g: _delta(r[g], t[g]) for g in r}}

    strata = {name: {v: block(np.array([k == v for k in keys])) for v in sorted(set(keys))}
              for name, keys in strata_keys.items()}
    return {"n_samples": len(records), "reference": ref, "tracked": trk, "delta_pck": delta,
            "strata": {"threshold": 0.1, **strata}}


def choose_k_pose(table: Sequence[dict]) -> dict:
    """``table``: rows ``{"k_pose", "body", "ski"}`` (the recipe's tracked PCK@0.1, mean over seeds)."""
    scored = [(float(r["k_pose"]), (r["body"] + r["ski"]) / 2) for r in table]
    best = max(s for _, s in scored)
    pick = min(k for k, s in scored if s >= best - K_TIE)
    return {"rule": K_RULE, "pick": pick, "best_mean": best,
            "table": [{"k_pose": k, "mean_body_ski": s} for k, s in scored]}


def recipe_of(name: str) -> str:
    """``E2-s230`` → ``E2``."""
    return name.rsplit("-s", 1)[0]


def long_rows(name: str, k_pose: float | None, crop: str, block: dict) -> list[dict]:
    rows = []
    for t, groups in block["groups"].items():
        for g, c in groups.items():
            cut = block["cut_off"][g]
            rows.append({"checkpoint": name, "k_pose": k_pose, "crop": crop, "threshold": t, "group": g,
                         "hits": c["hits"], "total": c["total"], "pct": c["pct"], "cut": cut["cut"],
                         "cut_share": cut["share"]})
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Propagation evaluation (plan Step 7): pose on reference crops vs. "
                                            "tracker crops at the same labelled frames.")
    p.add_argument("--split", type=Path, required=True, help="frozen pose_split_v1.json")
    p.add_argument("--partition", default="val")
    p.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8, once)")
    p.add_argument("--check-alignment", action="store_true",
                   help="only check the partition's segments against their clips and write usable_segments.json")
    p.add_argument("--window", type=int, default=5, help="alignment search window (frames)")
    p.add_argument("--usable", type=Path, help="usable_segments.json from --check-alignment")
    p.add_argument("--track-dir", type=Path, help="CSVs from run_tracker.py --ski2dpose-partition")
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH",
                   help="e.g. E2-s230=$SKIER_RESULTS/pose/d21/E2-s230/best.pt (repeat)")
    p.add_argument("--recipe", default="E2", help="the recipe whose seed mean picks k_pose (D23)")
    p.add_argument("--k-pose", type=float, nargs="+", help=f"default: the grid {K_POSE_GRID}")
    p.add_argument("--config", type=Path, help="integration config for the status stratum; default: "
                                               "configs/integration/default.yaml")
    p.add_argument("--hrnet-config", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/integration/propagation-<partition>")
    return p


def _load(args, schema):
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                           s2d / "archives/ski2dpose_info.json", schema=schema)
    return select_partition(records, read_json(args.split), args.partition)[0], s2d


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition in FINAL_ONLY and not args.final_test:
        print(f"error: {args.partition!r} is evaluated once, in Step 8; pass --final-test then", file=sys.stderr)
        return 1
    schema = load_schema()
    manifest = read_json(args.split)
    out = ensure_dir(args.out or results_root() / "integration" / f"propagation-{args.partition}")

    if args.check_alignment:
        eligible, s2d = _load(args, schema)
        image_rows, seg_rows = segment_alignment(eligible, s2d / "videos", args.window)
        from skier.data.alignment import ALIGN_COLUMNS

        write_csv(image_rows, out / "alignment_images.csv", ALIGN_COLUMNS)
        write_csv(seg_rows, out / "alignment_segments.csv", SEGMENT_COLUMNS)
        usable = [[s["video_id"], s["segment_id"]] for s in seg_rows if s["usable"]]
        write_json({"partition": args.partition, "split_sha256": manifest["content_sha256"], "usable": usable,
                    "segments": seg_rows}, out / "usable_segments.json")
        print(f"{len(usable)}/{len(seg_rows)} segments usable → {out}")
        return 0

    if not (args.usable and args.track_dir and args.checkpoint and args.hrnet_config):
        print("error: give --usable, --track-dir, --checkpoint and --hrnet-config (or --check-alignment)",
              file=sys.stderr)
        return 1
    usable = read_json(args.usable)
    if usable["partition"] != args.partition or usable["split_sha256"] != manifest["content_sha256"]:
        print("error: --usable was written for another partition or split", file=sys.stderr)
        return 1
    checkpoints = {}
    for spec in args.checkpoint:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in checkpoints:
            print(f"error: bad or repeated --checkpoint {spec!r}", file=sys.stderr)
            return 1
        checkpoints[name] = Path(path)
    icfg = IntegrationConfig.from_yaml(args.config or configs_dir() / "integration" / "default.yaml")
    icfg = icfg if icfg.conf_signal is not None else None  # no status stratum until the confidence check is done
    ks = tuple(args.k_pose or K_POSE_GRID)

    eligible, s2d = _load(args, schema)
    init, scored = split_init(eligible, usable["usable"])
    fps = {tuple(s): next(r["fps"] for r in usable["segments"] if [r["video_id"], r["segment_id"]] == s)
           for s in usable["usable"]}
    trk = tracker_at(scored, init, args.track_dir, fps, icfg)
    strata_keys = {"tracker_iou": [iou_bin(v) for v in trk["iou"]], "time_since_init": [time_bin(s) for s in trk["seconds"]],
                   "video": [r.source_video_id for r in scored],
                   "segment": [f"{r.source_video_id}/{r.segment_id}" for r in scored]}
    if icfg is not None:
        strata_keys["status"] = ["ok" if s in ("track", "init") else s for s in trk["status"]]

    from skier.pose.model import build_hrnet, torch_heatmap_fn  # torch + mmpose from here on
    from skier.pose.train import load_checkpoint

    result = {"partition": args.partition, "split_sha256": manifest["content_sha256"], "n_segments": len(init),
              "n_scored": len(scored), "track_dir": str(args.track_dir), "k_pose_grid": list(ks),
              "integration_config": None if icfg is None else dataclasses.asdict(icfg), "checkpoints": {}}
    long, preds, by_k = [], {}, defaultdict(list)
    for name, path in checkpoints.items():
        model, _ = build_hrnet(schema.num_keypoints, args.hrnet_config, None, device=args.device)
        ck = load_checkpoint(path, model, schema=schema, map_location=args.device)
        if ck.get("split_sha256") != manifest["content_sha256"]:
            print(f"error: {name} was trained on a different split", file=sys.stderr)
            return 1
        expand = float(ck.get("config", {}).get("expand", 1.25))
        fn = torch_heatmap_fn(model, args.device)
        ref_pred = predict_records(scored, fn, expand, args.batch_size).xy
        preds[f"{name}__reference"] = ref_pred
        entry = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path), "epoch": ck.get("epoch"),
                 "expand": expand, "by_k_pose": {}}
        for k in ks:
            boxes = tracker_crop_boxes(trk["boxes"], k)
            trk_pred = predict_on(scored, fn, boxes, expand, args.batch_size)
            preds[f"{name}__k{k:g}"] = trk_pred
            block = compare_crops(scored, schema, ref_pred, trk_pred, boxes, expand, strata_keys)
            entry["by_k_pose"][f"{k:g}"] = block
            if k == ks[0]:
                long += long_rows(name, None, "reference", block["reference"])
            long += long_rows(name, k, "tracked", block["tracked"])
            g = block["tracked"]["groups"]["0.1"]
            if recipe_of(name) == args.recipe:
                by_k[k].append((g["body"]["pct"], g["ski"]["pct"]))
        result["checkpoints"][name] = entry
    if by_k:
        table = [{"k_pose": k, "body": float(np.mean([b for b, _ in v])), "ski": float(np.mean([s for _, s in v]))}
                 for k, v in sorted(by_k.items())]
        result["k_pose_rule"] = {"recipe": args.recipe, "n_seeds": len(next(iter(by_k.values()))), **choose_k_pose(table)}
    write_json(result, out / "propagation.json")
    write_csv(long, out / "propagation.csv", LONG_COLUMNS)
    np.savez(out / "predictions.npz", sample_ids=np.array([r.sample_id for r in scored]),
             tracker_boxes=trk["boxes"], tracker_iou=trk["iou"], seconds_since_init=trk["seconds"], **preds)
    for name, e in result["checkpoints"].items():
        for k, b in e["by_k_pose"].items():
            r, t = b["reference"]["groups"]["0.1"], b["tracked"]["groups"]["0.1"]
            print(f"{name} k_pose {k}: body {r['body']['pct']:.1f} → {t['body']['pct']:.1f}, "
                  f"ski {r['ski']['pct']:.1f} → {t['ski']['pct']:.1f} (reference → tracked, PCK@0.1)")
    if "k_pose_rule" in result:
        print(f"k_pose rule ({args.recipe}): pick {result['k_pose_rule']['pick']}")
    print(f"{len(scored)} scored labels in {len(init)} segments → {out}")
    return 0
