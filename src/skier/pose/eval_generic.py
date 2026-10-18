"""E0: the unchanged COCO HRNet on reference crops, scored on shared joints (plan Step 3.4).

Requires a frozen split manifest, so E0 can't run before the split is frozen.
torch/mmpose are imported only when the model is actually built.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from skier.data.ski2dpose import load_records
from skier.eval.pose_metrics import gt_arrays, normalized_error, pck, per_video_pck
from skier.io_utils import ensure_dir, read_json, sha256_file, write_json
from skier.paths import data_root, results_root
from skier.pose.coco_map import approx_joints, coco17_to_ski24, shared_joints
from skier.pose.infer import Predictions, predict_records
from skier.schema import Schema, load_schema
from skier.splits import apply_split

THRESHOLDS = (0.1, 0.05)


def score_e0(pred17: np.ndarray, records, schema: Schema) -> dict:
    """Metrics for E0: shared joints, a separate approximate wrist->hand row, ski = N/A."""
    gt, rel, diag, vids = gt_arrays(records)
    pred24 = coco17_to_ski24(pred17, schema, include_approx=True)
    shared, approx = shared_joints(schema), approx_joints(schema)
    out = {
        "experiment": "E0",
        "n_samples": len(records),
        "shared_joints": [schema.names[j] for j in shared],
        "body_shared": {f"pck@{t:g}": pck(pred24, gt, rel, diag, shared, t) for t in THRESHOLDS},
        "approx_wrist_to_hand": {f"pck@{t:g}": pck(pred24, gt, rel, diag, approx, t) for t in THRESHOLDS},
        "ski": "N/A (the COCO model predicts no ski landmarks)",
        "errors_body_shared": normalized_error(pred24, gt, rel, diag, shared),
        "per_video_body_shared_pck@0.1": per_video_pck(pred24, gt, rel, diag, vids, shared, 0.1),
        "per_joint_pck@0.1": {schema.names[j]: pck(pred24, gt, rel, diag, [j], 0.1) for j in shared + approx},
    }
    return out


def select_partition(records, manifest: dict, partition: str) -> tuple[list, list]:
    """Records of one partition, split into (eligible, excluded)."""
    if partition not in manifest["partitions"]:
        raise ValueError(f"manifest has no partition {partition!r}; has {sorted(manifest['partitions'])}")
    apply_split(records, manifest)
    part = [r for r in records if r.split == partition]
    return [r for r in part if not r.excluded], [r for r in part if r.excluded]


def save_predictions(path: Path, preds: Predictions, pred24: np.ndarray, checkpoint_sha: str | None) -> None:
    np.savez(path, sample_ids=np.array(preds.sample_ids), pred17=preds.xy, scores17=preds.scores,
             pred24=pred24, checkpoint_sha256=np.array(checkpoint_sha or ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the unchanged COCO HRNet (E0) on a frozen split partition.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True, help="frozen pose_split_v1.json")
    p.add_argument("--partition", default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/E0/<partition>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition == "test":
        print("note: evaluating the TEST partition; this should happen once, in Step 8", file=sys.stderr)
    manifest = read_json(args.split)
    schema = load_schema()
    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                           s2d / "archives/ski2dpose_info.json", schema=schema)
    eligible, excluded = select_partition(records, manifest, args.partition)

    from skier.pose.model import build_hrnet, torch_heatmap_fn  # torch + mmpose from here on
    import torch

    model, load_info = build_hrnet(17, args.config, args.checkpoint, device=args.device)
    heatmap_fn = torch_heatmap_fn(model, args.device)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    warm = eligible[: args.batch_size * args.warmup_batches]
    if warm:
        predict_records(warm, heatmap_fn, batch_size=args.batch_size)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds = predict_records(eligible, heatmap_fn, batch_size=args.batch_size)
    if use_cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    out = ensure_dir(args.out or results_root() / "pose" / "E0" / args.partition)
    ckpt_sha = sha256_file(args.checkpoint)
    save_predictions(out / "predictions.npz", preds, coco17_to_ski24(preds.xy, schema, include_approx=True), ckpt_sha)
    metrics = score_e0(preds.xy, eligible, schema)
    metrics.update(
        partition=args.partition,
        split_sha256=manifest["content_sha256"],
        excluded_records=[r.sample_id for r in excluded],
        checkpoint=args.checkpoint.name,
        checkpoint_sha256=ckpt_sha,
        load_info=load_info,
        timing={**preds.timing, "wall_s": round(wall, 3), "ms_per_image_wall": round(1000 * wall / max(1, len(eligible)), 2),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else None},
    )
    write_json(metrics, out / "metrics.json")
    b = metrics["body_shared"]["pck@0.1"]
    print(f"E0 {args.partition}: body (shared joints) PCK@0.1 {b['pct']:.1f}% ({b['hits']}/{b['total']}) → {out}")
    return 0
