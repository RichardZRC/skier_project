"""D20 readout: per-joint PCK@0.1 of each run's best.pt on training images (no augmentation) and clean val.

Usage: python trial_eval.py E1 E1-seed231 E3 E3-ft150   (folders under $SKIER_RESULTS/pose/)
Training and val images only; never test. A run "learns the tails" if both training-image tail PCKs are >= 60% (D20).
"""
import json
import os
import sys
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

TAIL_THRESHOLD = 0.60
repo = Path.home() / "skier_project"
s2d = Path(os.environ["SKIER_DATA"]) / "ski2dpose"
R = Path(os.environ["SKIER_RESULTS"]) / "pose"
cfg_py = Path(os.environ["SKIER_DATA"]) / "pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
schema = load_schema()
records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                       schema=schema)
manifest = read_json(repo / "data/manifests/pose_split_v1.json")
parts = {p: select_partition(records, manifest, p)[0] for p in ("train", "val")}
ski, body = list(schema.groups["ski"]), list(schema.groups["body"])
joints = [19, 23, 16, 20, 17, 18, 21, 22]


def pck(run: str, part: str) -> dict:
    model, _ = build_hrnet(24, cfg_py, None, device="cuda:0")
    ck = load_checkpoint(R / run / "best.pt", model, schema=schema, map_location="cuda:0")
    recs = parts[part]
    gt, rel, diag, _ = gt_arrays(recs)
    xy = predict_records(recs, torch_heatmap_fn(model, "cuda:0"), batch_size=32).xy
    d = np.linalg.norm(xy - gt, axis=-1) / diag[:, None]
    hit = (d <= 0.1) & rel & np.isfinite(d)
    out = {schema.names[j]: float(hit[:, j].sum() / rel[:, j].sum()) for j in range(24)}
    out["ski"] = float(hit[:, ski].sum() / rel[:, ski].sum())
    out["body"] = float(hit[:, body].sum() / rel[:, body].sum())
    out["best_epoch"] = int(ck["epoch"])
    return out


runs = sys.argv[1:]
res = {run: {p: pck(run, p) for p in ("train", "val")} for run in runs}
w = 9
print(f"{'joint':<16}" + "".join(f"{r[:w] + ' tr':>{w + 4}}{r[:w] + ' val':>{w + 5}}" for r in runs))
for name in [schema.names[j] for j in joints] + ["ski", "body"]:
    print(f"{name:<16}" + "".join(f"{res[r]['train'][name]:{w + 4}.1%}{res[r]['val'][name]:{w + 5}.1%}" for r in runs))
print(f"{'best epoch':<16}" + "".join(f"{res[r]['train']['best_epoch']:>{2 * w + 9}d}" for r in runs))
print()
for r in runs:
    t = res[r]["train"]
    learns = t["ski_tail_right"] >= TAIL_THRESHOLD and t["ski_tail_left"] >= TAIL_THRESHOLD
    print(f"{r}: training-image tails {t['ski_tail_right']:.1%} / {t['ski_tail_left']:.1%} -> "
          f"{'LEARNS the tails' if learns else 'FAILS on the tails'} (D20 threshold {TAIL_THRESHOLD:.0%})")
json.dump(res, open(R / "trial_d20.json", "w"), indent=2)
print(f"wrote {R / 'trial_d20.json'}")
