"""Cross-check our pose pipeline against MMPose's own inference (plan Step 3, run block).

(a) ``mmpose.apis.inference_topdown`` with the tight reference box (MMPose pads by 1.25
    itself), with **flip test disabled**: the COCO config enables it by default and our
    pipeline never flips (plan §10, 5.2), so leaving it on would compare different things.
(b) our ``predict_records`` (expand 1.25 -> fit aspect -> warp -> normalize -> PoseNet ->
    MSRA decode -> inverse affine).

The mean 17-joint distance should be under about 2 original-image pixels. A larger gap
means our crop, normalization or heatmap convention differs from the checkpoint's;
fix that before E0 or any training. Pilot-video images only.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from skier.data.ski2dpose import load_records
from skier.io_utils import ensure_dir, read_id_list, write_json
from skier.paths import data_root, repo_root, results_root
from skier.pose.infer import predict_records

MAX_MEAN_PX = 2.0


def compare_points(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-image and overall distances between two ``[N, K, 2]`` predictions (original px)."""
    d = np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)
    return {"mean_px": float(np.nanmean(d)), "max_px": float(np.nanmax(d)),
            "per_image_mean_px": [round(float(x), 3) for x in np.nanmean(d, axis=1)],
            "n_nan": int(np.isnan(d).sum())}


def pick_pilot_records(records, pilot_videos: list[str], n: int, seed: int = 230) -> list:
    pool = sorted((r for r in records if r.source_video_id in set(pilot_videos) and not r.excluded),
                  key=lambda r: r.sample_id)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(n, len(pool))), key=lambda r: r.sample_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare our pose pipeline with MMPose inference_topdown.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pilot-file", type=Path, help="default: data/manifests/pilot_videos.txt")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/crosscheck")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    from mmpose.apis import inference_topdown, init_model

    from skier.pose.model import build_hrnet, torch_heatmap_fn

    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json")
    pilots = read_id_list(args.pilot_file or repo_root() / "data/manifests/pilot_videos.txt")
    chosen = pick_pilot_records(records, pilots, args.n)

    estimator = init_model(str(args.config), str(args.checkpoint), device=args.device,
                           cfg_options={"model.test_cfg.flip_test": False})
    mm = []
    for r in chosen:
        res = inference_topdown(estimator, cv2.imread(str(r.image_path)),
                                bboxes=np.array([r.reference_box_xyxy]), bbox_format="xyxy")
        mm.append(np.asarray(res[0].pred_instances.keypoints)[0])
    model, _ = build_hrnet(17, args.config, args.checkpoint, device=args.device)
    ours = predict_records(chosen, torch_heatmap_fn(model, args.device), batch_size=len(chosen))

    report = compare_points(np.stack(mm), ours.xy)
    report.update(sample_ids=[r.sample_id for r in chosen], threshold_px=MAX_MEAN_PX,
                  passed=report["mean_px"] <= MAX_MEAN_PX)
    out = ensure_dir(args.out or results_root() / "pose" / "crosscheck")
    write_json(report, out / "crosscheck.json")
    print(f"mean {report['mean_px']:.3f} px, max {report['max_px']:.3f} px over {len(chosen)} pilot images "
          f"→ {'PASS' if report['passed'] else 'FAIL'} ({out / 'crosscheck.json'})")
    return 0 if report["passed"] else 1
