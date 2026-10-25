"""Diagnostic: per-joint PCK@0.1 of E1-E4's best.pt on (unaugmented) TRAINING images vs val.

Underfitting (tails also wrong on train images) vs. a generalization gap (right on train, wrong on val).
Training images only plus the already-scored val numbers; no test.
"""
import csv
import os
from pathlib import Path

import numpy as np

from skier.data.ski2dpose import load_records
from skier.eval.pose_metrics import gt_arrays
from skier.io_utils import read_json
from skier.pose.eval_generic import select_partition
from skier.pose.infer import predict_records
from skier.pose.model import build_hrnet, torch_heatmap_fn
from skier.pose.train import load_checkpoint
from skier.schema import load_schema

repo = Path.home() / "skier_project"
s2d = Path(os.environ["SKIER_DATA"]) / "ski2dpose"
R = Path(os.environ["SKIER_RESULTS"]) / "pose"
cfg_py = Path(os.environ["SKIER_DATA"]) / "pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
schema = load_schema()
records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                       schema=schema)
train, _ = select_partition(records, read_json(repo / "data/manifests/pose_split_v1.json"), "train")
gt, rel, diag, _ = gt_arrays(train)
val_pj = {r["joint"]: r for r in csv.DictReader(open(R / "compare/val/per_joint.csv"))}

joints = [19, 23, 16, 20, 17, 18]
res = {}
for n in (1, 2, 3, 4):
    model, _ = build_hrnet(24, cfg_py, None, device="cuda:0")
    load_checkpoint(R / f"E{n}/best.pt", model, schema=schema, map_location="cuda:0")
    p = predict_records(train, torch_heatmap_fn(model, "cuda:0"), batch_size=32).xy
    d = np.linalg.norm(p - gt, axis=-1) / diag[:, None]
    hit = (d <= 0.1) & rel & np.isfinite(d)
    res[n] = {j: hit[:, j].sum() / rel[:, j].sum() for j in range(24)}
    ski = list(schema.groups["ski"])
    res[n]["ski"] = hit[:, ski].sum() / rel[:, ski].sum()

print(f"PCK@0.1 on {len(train)} training images (no augmentation)  |  val, for comparison")
print(f"{'joint':<15}" + "".join(f"{'E' + str(n) + ' tr':>8}" for n in (1, 2, 3, 4)) + " |"
      + "".join(f"{'E' + str(n) + ' val':>8}" for n in (1, 2, 3, 4)))
for j in joints:
    nm = schema.names[j]
    print(f"{nm:<15}" + "".join(f"{res[n][j]:8.1%}" for n in (1, 2, 3, 4)) + " |"
          + "".join(f"{float(val_pj[nm][f'E{n}_pct']) / 100:8.1%}" for n in (1, 2, 3, 4)))
print(f"{'all ski':<15}" + "".join(f"{res[n]['ski']:8.1%}" for n in (1, 2, 3, 4)))
