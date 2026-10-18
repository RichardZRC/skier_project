"""Pretrained-pose smoke test (guide 1H, code spec §10).

Tests the install with MMPose's own pipeline (init_model + inference_topdown); our
own pipeline is cross-checked against it in Step 3. torch and mmpose are imported
inside ``run()`` so ``--help`` works without them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from skier.io_utils import ensure_dir, sha256_file, write_json
from skier.paths import results_root

# COCO-17 skeleton (0-based). Never draw the 24-point schema on generic predictions.
COCO_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                 (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def draw_coco(img: np.ndarray, keypoints: np.ndarray, box) -> np.ndarray:
    out = img.copy()
    r = max(3, round(min(out.shape[:2]) / 200))
    pts = [(int(round(x)), int(round(y))) for x, y in keypoints]
    for a, b in COCO_SKELETON:
        cv2.line(out, pts[a], pts[b], (0, 255, 0), max(1, r // 2), cv2.LINE_8)
    for p in pts:
        cv2.circle(out, p, r, (0, 0, 255), -1, cv2.LINE_8)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_8)
    return out


def run(args: argparse.Namespace) -> int:
    import mmpose
    import torch
    from mmpose.apis import inference_topdown, init_model

    out = ensure_dir(args.out or results_root() / "step1" / "smoke_pose")
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"error: cannot read {args.image}", file=sys.stderr)
        return 1
    bboxes = np.array([args.box], dtype=np.float32)

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)

    def infer():
        return inference_topdown(model, img, bboxes=bboxes, bbox_format="xyxy")

    inst = infer()[0].pred_instances
    keypoints = np.asarray(inst.keypoints)[0]
    scores = np.asarray(inst.keypoint_scores)[0]
    if keypoints.shape != (17, 2) or scores.shape != (17,):
        print(f"error: unexpected shapes {keypoints.shape}, {scores.shape}", file=sys.stderr)
        return 1
    if not (np.isfinite(keypoints).all() and np.isfinite(scores).all()):
        print("error: non-finite keypoints or scores", file=sys.stderr)
        return 1

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        infer()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(args.runs):
        sync()
        t0 = time.perf_counter()
        infer()
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None

    np.savez(out / "generic_pose_prediction.npz", keypoints=keypoints, scores=scores, box=np.asarray(args.box))
    write_json({
        "config": Path(args.config).name,
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "image": Path(args.image).name,
        "box_xyxy": list(args.box),
        "device": args.device,
        "warmup": args.warmup,
        "runs": args.runs,
        "ms_per_image_mean": round(float(np.mean(times_ms)), 2),
        "ms_per_image_median": round(float(np.median(times_ms)), 2),
        "peak_mem_mb": round(peak_mb, 1) if peak_mb is not None else None,
        "torch": torch.__version__,
        "mmpose": mmpose.__version__,
    }, out / "smoke_pose.json")
    cv2.imwrite(str(out / "smoke_pose_overlay.png"), draw_coco(img, keypoints, args.box))
    print(f"17 finite keypoints; {np.median(times_ms):.1f} ms/image (median) → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the COCO HRNet once through MMPose to test the install.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--box", type=float, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"),
                   help="tight skier box in pixels (MMPose pads it by 1.25 itself)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/smoke_pose")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
