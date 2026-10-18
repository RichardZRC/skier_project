"""Preview grids of augmented training samples (plan Step 6 run block: "check they look
plausible before training").

Pilot videos only by default, so it can run before the split freeze. Every config shows
the same sample IDs with the same seed, so the sheets are paired: E1–E4 share the
flip/rotation/scale of each tile, and differ only in the augmentation under study.
Points: filled = supervised (weight 1); hollow = masked (unreliable, or outside the crop).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from skier.data.records import PoseRecord
from skier.io_utils import ensure_dir, read_id_list, write_json
from skier.paths import configs_dir, data_root, repo_root, results_root
from skier.pose.augment import AugParams
from skier.pose.dataset import PoseDataset, make_sample, read_image
from skier.pose.preprocess import INPUT_H, INPUT_W, MEAN, STD
from skier.pose.train import TrainConfig
from skier.schema import Schema, load_schema
from skier.viz import draw_edges, draw_header, draw_points

UPSCALE = 2


def describe(p: AugParams) -> list[str]:
    lines = [f"{'flip ' if p.flip else ''}rot {p.rot_deg:+.0f} scale {p.scale:.2f}"]
    if p.shift != (0.0, 0.0) or p.jitter_scale != 1.0:
        lines.append(f"jitter {p.shift[0]:+.2f},{p.shift[1]:+.2f} x{p.jitter_scale:.2f}")
    extra = []
    if p.blur_length:
        extra.append(f"blur {p.blur_length}px @{p.blur_angle:.0f}")
    if p.occlusion is not None:
        extra.append(f"occl {100 * p.occlusion[2] * p.occlusion[3] / (INPUT_W * INPUT_H):.0f}%")
    if extra:
        lines.append(" ".join(extra))
    return lines


def render_sample(record: PoseRecord, image_bgr: np.ndarray, params: AugParams, schema: Schema,
                  cfg: TrainConfig) -> np.ndarray:
    s = make_sample(record, image_bgr, params, schema, expand=cfg.expand, sigma=cfg.sigma)
    rgb = s.x.transpose(1, 2, 0) * STD + MEAN
    crop = np.clip(np.rint(rgb[..., ::-1]), 0, 255).astype(np.uint8)
    big = cv2.resize(crop, (INPUT_W * UPSCALE, INPUT_H * UPSCALE), interpolation=cv2.INTER_LINEAR)
    xy = s.keypoints_input * UPSCALE + (UPSCALE - 1) / 2  # pixel centers after resizing
    supervised = s.weight > 0
    draw_edges(big, xy, supervised, schema)
    draw_points(big, xy, supervised, schema, label_indices=False)
    return draw_header(big, describe(params))


def tile(images: Sequence[np.ndarray], cols: int = 6) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = -(-len(images) // cols)
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return sheet


def preview(records: Sequence[PoseRecord], cfg: TrainConfig, schema: Schema, indices: Sequence[int],
            image_loader: Callable = read_image) -> tuple[list[np.ndarray], list[dict]]:
    """Tiles and parameters for ``records[i]``, epoch 0, with the config's own seed."""
    ds = PoseDataset(records, schema, train=True, aug_cfg=cfg.aug, seed=cfg.seed, expand=cfg.expand, sigma=cfg.sigma)
    tiles, params = [], []
    for i in indices:
        p = ds.params_for(i)
        tiles.append(render_sample(records[i], image_loader(records[i].image_path), p, schema, cfg))
        params.append({"sample_id": records[i].sample_id, **dataclasses.asdict(p)})
    return tiles, params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contact sheets of augmented training samples (pilot videos only).")
    p.add_argument("--config", type=Path, nargs="+", help="default: configs/pose/e1.yaml ... e4.yaml")
    p.add_argument("--videos-file", type=Path, help="default: data/manifests/pilot_videos.txt")
    p.add_argument("--n", type=int, default=12, help="samples per sheet")
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--seed", type=int, default=230, help="which samples are picked")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/aug_preview")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = args.config or [configs_dir() / "pose" / f"e{n}.yaml" for n in (1, 2, 3, 4)]
    videos = set(read_id_list(args.videos_file or repo_root() / "data" / "manifests" / "pilot_videos.txt"))
    schema = load_schema()

    from skier.data.ski2dpose import load_records

    s2d = data_root() / "ski2dpose"
    records = [r for r in load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                                       s2d / "archives/ski2dpose_info.json", schema=schema)
               if r.source_video_id in videos and not r.excluded]
    if not records:
        print("error: no eligible records in the listed videos", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(records), size=min(args.n, len(records)), replace=False).tolist())
    out = ensure_dir(args.out or results_root() / "pose" / "aug_preview")
    all_params = {}
    for path in configs:
        cfg = TrainConfig.from_yaml(path)
        tiles, params = preview(records, cfg, schema, indices)
        cv2.imwrite(str(out / f"{Path(path).stem}.png"), tile(tiles, args.cols))
        all_params[Path(path).stem] = params
    write_json(all_params, out / "params.json")
    print(f"{len(indices)} samples from {len(videos)} videos x {len(configs)} configs -> {out}")
    return 0
