"""Are E1-E4 trained on the same targets? CPU only, training images (no val/test).

Builds the training samples of one epoch under each config (same seed, epoch, index) and compares, per joint:
the share of samples where the joint is supervised (weight > 0), and for E3 vs E1 (pixel-only augmentation)
whether targets and weights are identical. Also records where the tail target peak sits for E1 vs E2.
"""
import os
import sys
from pathlib import Path

import numpy as np

from skier.data.ski2dpose import load_records
from skier.io_utils import read_json
from skier.pose.dataset import PoseDataset
from skier.pose.eval_generic import select_partition
from skier.pose.train import TrainConfig
from skier.schema import load_schema

repo = Path.home() / "skier_project"
s2d = Path(os.environ["SKIER_DATA"]) / "ski2dpose"
schema = load_schema()
records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                       schema=schema)
train, _ = select_partition(records, read_json(repo / "data/manifests/pose_split_v1.json"), "train")
epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 5
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
idx = np.linspace(0, len(train) - 1, n).astype(int)

cfgs = {e: TrainConfig.from_yaml(repo / f"configs/pose/e{e}.yaml") for e in (1, 2, 3, 4)}
data = {}
for e, cfg in cfgs.items():
    ds = PoseDataset(train, schema, train=True, aug_cfg=cfg.aug, seed=cfg.seed, expand=cfg.expand, sigma=cfg.sigma)
    ds.set_epoch(epoch)
    samples = [ds[i] for i in idx]
    data[e] = {"w": np.stack([s["weight"] for s in samples]), "t": np.stack([s["target"] for s in samples]),
               "M": np.stack([s["M"] for s in samples])}
    print(f"E{e}: seed {cfg.seed}, sigma {cfg.sigma}, expand {cfg.expand}, built {len(samples)} samples (epoch {epoch})")

ski = list(schema.groups["ski"])
reliable = np.stack([train[i].reliable for i in idx])
print(f"\nshare supervised (weight > 0) among reliable labels, epoch {epoch}, {n} training images")
print(f"{'joint':<16}" + "".join(f"{'E' + str(e):>8}" for e in (1, 2, 3, 4)))
for j in ski + list(schema.groups["pole"]):
    rel = reliable[:, j]
    print(f"{schema.names[j]:<16}" + "".join(f"{(data[e]['w'][rel, j] > 0).mean():8.1%}" for e in (1, 2, 3, 4)))

print("\nE3 vs E1 (pixel-only augmentation; targets should be identical):")
print("  crop matrices identical:", np.allclose(data[3]["M"], data[1]["M"]))
print("  weights identical:      ", np.array_equal(data[3]["w"], data[1]["w"]))
print("  max |target diff|:      ", float(np.abs(data[3]["t"] - data[1]["t"]).max()))
print("E4 vs E2 (E4 = E2 + pixel augmentation):")
print("  crop matrices identical:", np.allclose(data[4]["M"], data[2]["M"]))
print("  weights identical:      ", np.array_equal(data[4]["w"], data[2]["w"]))
