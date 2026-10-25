"""Where do the ski-tail (and tip) predictions land? Val only, clean crops.

For each reliable ski-tail label: which labelled point of the same skier is the prediction nearest to
(own tail, own tip, toes, heel, the other ski's points...)? Distances in units of the reference-box diagonal.
"""
import os
from pathlib import Path

import numpy as np

from skier.data.ski2dpose import load_records
from skier.geometry import box_diagonal
from skier.schema import load_schema

R = Path(os.environ["SKIER_RESULTS"]) / "pose"
s2d = Path(os.environ["SKIER_DATA"]) / "ski2dpose"
schema = load_schema()
names = schema.names
recs = {r.sample_id: r for r in load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                                              s2d / "archives/ski2dpose_info.json", schema=schema)}
ski = list(schema.groups["ski"])
short = {16: "tipR", 17: "toeR", 18: "heelR", 19: "tailR", 20: "tipL", 21: "toeL", 22: "heelL", 23: "tailL"}

for n in (1, 2, 3, 4):
    d = np.load(R / f"E{n}/eval/val/predictions.npz", allow_pickle=True)
    if n == 1:
        print("npz keys:", d.files)
    ids, xy = d["sample_ids"], d["clean"]
    print(f"\n=== E{n}")
    for j in (19, 23, 16, 20):
        counts, hit, med = {}, 0, []
        for sid, p in zip(ids, xy):
            r = recs[str(sid)]
            if not r.reliable[j] or not np.isfinite(p[j]).all():
                continue
            diag = box_diagonal(r.reference_box_xyxy)
            dist = {k: np.linalg.norm(p[j] - r.keypoints_xy[k]) / diag for k in ski}
            nearest = min(dist, key=dist.get)
            counts[short[nearest]] = counts.get(short[nearest], 0) + 1
            hit += dist[j] <= 0.1
            med.append(dist[j])
        tot = sum(counts.values())
        top = ", ".join(f"{k} {v / tot:.0%}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {names[j]:<15} n={tot:3d} PCK {hit / tot:5.1%}  median err {np.median(med):.3f} diag | nearest label: {top}")
