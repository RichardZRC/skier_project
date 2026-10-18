"""Evaluate a trained 24-point model (plan Step 6 run block; reused for test in Step 8).

``evaluate`` is numpy-only. The model enters as a plain ``heatmap_fn`` (as in
``skier.pose.infer``), so the scoring is tested without torch. For one partition it
reports:
- clean scores: groups at PCK@0.1/0.05, per joint, errors, per video, the E0-comparable
  body joints, and the missing-prediction rate;
- strata by discipline and by view flag (an image can have several views, so view
  strata overlap);
- for each crop-error level: the same scores on the perturbed crops, plus the **cut-off
  share**, meaning reliable points outside the crop. Those still count as misses.

Every block uses the same sample IDs and the same denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.eval.pose_metrics import THRESHOLDS, group_pck, gt_arrays, pck, pose_summary
from skier.geometry import apply_affine, crop_affine
from skier.io_utils import ensure_dir, read_json, sha256_file, write_json
from skier.paths import data_root, results_root
from skier.pose.coco_map import shared_joints
from skier.pose.crop_error import FINAL_ONLY, check_crop_error_set, perturbed_boxes
from skier.pose.infer import HeatmapFn, predict_records
from skier.pose.preprocess import INPUT_H, INPUT_W, pose_crop_box
from skier.schema import Schema, load_schema

VIEWS = ("front", "side", "back")


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(sample_ids)).encode()).hexdigest()


def inside_crop(records: Sequence[PoseRecord], boxes: Sequence, expand: float = 1.25) -> np.ndarray:
    """``[N, K]``: label inside the model-input crop around ``box`` (the rule ``encode`` uses)."""
    out = []
    for r, box in zip(records, boxes):
        xy = apply_affine(crop_affine(pose_crop_box(box, expand), INPUT_W, INPUT_H), r.keypoints_xy)
        out.append((xy[:, 0] >= 0) & (xy[:, 0] < INPUT_W) & (xy[:, 1] >= 0) & (xy[:, 1] < INPUT_H))
    return np.array(out, dtype=bool).reshape(len(records), -1)


def cut_off(inside: np.ndarray, reliable: np.ndarray, schema: Schema) -> dict:
    out = {}
    for g, joints in schema.groups.items():
        rel = reliable[:, list(joints)]
        cut = int((rel & ~inside[:, list(joints)]).sum())
        out[g] = {"cut": cut, "total": int(rel.sum()), "share": cut / int(rel.sum()) if rel.sum() else None}
    return out


def score(pred_xy: np.ndarray, records: Sequence[PoseRecord], schema: Schema, boxes: Sequence | None = None,
          expand: float = 1.25) -> dict:
    gt, rel, diag, vids = gt_arrays(records)
    out = pose_summary(pred_xy, gt, rel, diag, schema, video_ids=vids)
    shared = shared_joints(schema)
    out["body_shared"] = {f"{t:g}": pck(pred_xy, gt, rel, diag, shared, t) for t in THRESHOLDS}
    missing = np.isnan(pred_xy[..., 0]) & rel
    out["missing_rate"] = float(missing.sum() / rel.sum()) if rel.sum() else None
    boxes = [r.reference_box_xyxy for r in records] if boxes is None else boxes
    out["cut_off"] = cut_off(inside_crop(records, boxes, expand), rel, schema)
    return out


def strata(pred_xy: np.ndarray, records: Sequence[PoseRecord], schema: Schema, thr: float = 0.1) -> dict:
    gt, rel, diag, _ = gt_arrays(records)

    def block(mask: np.ndarray) -> dict:
        idx = np.flatnonzero(mask)
        return {"n_images": int(len(idx)), **group_pck(pred_xy[idx], gt[idx], rel[idx], diag[idx], schema, thr)}

    disciplines = sorted({r.discipline or "unknown" for r in records})
    views = [(r.view_angle or "").split("+") for r in records]
    return {
        "threshold": thr,
        "discipline": {d: block(np.array([(r.discipline or "unknown") == d for r in records])) for d in disciplines},
        "view": {v: block(np.array([v in vs for vs in views])) for v in VIEWS},
        "note": "view strata overlap: one image can have several views",
    }


def evaluate(records: Sequence[PoseRecord], heatmap_fn: HeatmapFn, schema: Schema, crop_error: dict | None = None,
             expand: float = 1.25, batch_size: int = 32) -> tuple[dict, dict[str, np.ndarray]]:
    """``(metrics, predictions)``; predictions are ``[N, K, 2]`` per block (``clean`` and each level)."""
    ids = [r.sample_id for r in records]
    clean = predict_records(records, heatmap_fn, expand=expand, batch_size=batch_size)
    metrics = {"n_samples": len(records), "sample_ids_sha256": sample_ids_sha256(ids),
               "clean": score(clean.xy, records, schema, expand=expand), "strata": strata(clean.xy, records, schema),
               "timing": clean.timing}
    preds = {"clean": clean.xy}
    if crop_error is not None:
        levels = {}
        for level, r in crop_error["levels"].items():
            boxes = perturbed_boxes(records, crop_error, level)
            p = predict_records(records, heatmap_fn, expand=expand, batch_size=batch_size, boxes=boxes)
            levels[level] = {"radius": r, **score(p.xy, records, schema, boxes, expand)}
            preds[level] = p.xy
        metrics["crop_error"] = {"content_sha256": crop_error["content_sha256"], "levels": levels}
    return metrics, preds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained 24-point checkpoint on a frozen split partition "
                                            "(clean, strata, crop-error set).")
    p.add_argument("--checkpoint", type=Path, required=True, help="our best.pt / last.pt")
    p.add_argument("--hrnet-config", type=Path, required=True, help="the MMPose HRNet-W32 config")
    p.add_argument("--split", type=Path, required=True, help="frozen pose_split_v1.json")
    p.add_argument("--partition", default="val")
    p.add_argument("--crop-error-set", type=Path, help="e.g. data/manifests/crop_error_set_v1.json")
    p.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8, once)")
    p.add_argument("--experiment", help="default: the checkpoint's folder name (e.g. E2)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/<experiment>/eval/<partition>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition in FINAL_ONLY and not args.final_test:
        print(f"error: {args.partition!r} is evaluated once, in Step 8; pass --final-test then", file=sys.stderr)
        return 1
    manifest = read_json(args.split)
    schema = load_schema()

    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                           s2d / "archives/ski2dpose_info.json", schema=schema)
    eligible, excluded = select_partition(records, manifest, args.partition)
    ce = None
    if args.crop_error_set is not None:
        ce = read_json(args.crop_error_set)
        check_crop_error_set(ce, manifest["content_sha256"], args.partition, [r.sample_id for r in eligible])

    from skier.pose.model import build_hrnet, torch_heatmap_fn  # torch + mmpose from here on
    from skier.pose.train import load_checkpoint

    model, _ = build_hrnet(schema.num_keypoints, args.hrnet_config, None, device=args.device)
    ck = load_checkpoint(args.checkpoint, model, schema=schema, map_location=args.device)
    if ck.get("split_sha256") != manifest["content_sha256"]:
        print("error: the checkpoint was trained on a different split", file=sys.stderr)
        return 1
    expand = float(ck.get("config", {}).get("expand", 1.25))
    metrics, preds = evaluate(eligible, torch_heatmap_fn(model, args.device), schema, ce, expand, args.batch_size)

    experiment = args.experiment or args.checkpoint.resolve().parent.name
    out = ensure_dir(args.out or results_root() / "pose" / experiment / "eval" / args.partition)
    ck_sha = sha256_file(args.checkpoint)
    metrics.update(experiment=experiment, partition=args.partition, split_sha256=manifest["content_sha256"],
                   checkpoint=str(args.checkpoint), checkpoint_sha256=ck_sha, checkpoint_epoch=ck.get("epoch"),
                   config=ck.get("config"), excluded_records=[r.sample_id for r in excluded])
    write_json(metrics, out / "metrics.json")
    np.savez(out / "predictions.npz", sample_ids=np.array([r.sample_id for r in eligible]),
             checkpoint_sha256=np.array(ck_sha), **preds)
    g = metrics["clean"]["groups"]["0.1"]
    print(f"{experiment} {args.partition}: body PCK@0.1 {g['body']['pct']:.1f}% ({g['body']['hits']}/{g['body']['total']}), "
          f"ski {g['ski']['pct']:.1f}% ({g['ski']['hits']}/{g['ski']['total']}) -> {out}")
    return 0
