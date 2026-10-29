"""Step 8 cheap trial: the run_final.sh val outputs must reproduce the stored val results exactly.

Compares $SKIER_RESULTS/final/val/ with the Step 3, 6 and 7 outputs:
- pose counts (D21 evals, E0);
- the SkiTB tracking summary;
- the usable segments;
- propagation counts at the chosen k_pose;
- the replay numbers from the D24 confidence check.
Exits 1 on any difference. Usage: python check_val_dryrun.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ["SKIER_RESULTS"])
F = R / "final" / "val"
problems, checks = [], 0


def same(what, a, b, tol=0.0):
    global checks
    checks += 1
    ok = (abs(a - b) <= tol) if isinstance(a, float) or isinstance(b, float) else a == b
    if not ok:
        problems.append(f"{what}: {a!r} vs {b!r}")


def load(p):
    return json.loads(Path(p).read_text())


for run in ("E1-s230", "E1-s231", "E2-s230", "E2-s231"):
    new, old = load(F / "pose" / run / "metrics.json"), load(R / "pose" / "d21" / run / "eval" / "val" / "metrics.json")
    same(f"pose {run} sample ids", new["sample_ids_sha256"], old["sample_ids_sha256"])
    for t, groups in old["clean"]["groups"].items():
        for g, c in groups.items():
            same(f"pose {run} {g}@{t}", (new["clean"]["groups"][t][g]["hits"], new["clean"]["groups"][t][g]["total"]),
                 (c["hits"], c["total"]))
e0n, e0o = load(F / "pose" / "E0" / "metrics.json"), load(R / "pose" / "E0" / "val" / "metrics.json")
for t in ("pck@0.1", "pck@0.05"):
    same(f"E0 body_shared {t}", e0n["body_shared"][t], e0o["body_shared"][t])

tn, to = load(F / "tracking" / "skitb" / "summary.json"), load(R / "tracking" / "skitb-val" / "summary.json")
for k in ("pooled_mean_iou", "per_clip_mean_iou", "n_frames", "n_clips", "n_failure_episodes"):
    same(f"tracking {k}", tn[k], to[k])
same("tracking auc", tn["success"]["auc"], to["success"]["auc"])

pn = R / "final" / "val" / "propagation"
po = R / "integration" / "propagation-val"
same("usable segments", load(pn / "usable_segments.json")["usable"], load(po / "usable_segments.json")["usable"])
k = f"{load(pn / 'propagation.json')['k_pose_grid'][0]:g}"
new_p, old_p = load(pn / "propagation.json"), load(po / "propagation.json")
for name, e in new_p["checkpoints"].items():
    for crop in ("reference", "tracked"):
        for t, groups in e["by_k_pose"][k][crop]["groups"].items():
            for g, c in groups.items():
                o = old_p["checkpoints"][name]["by_k_pose"][k][crop]["groups"][t][g]
                same(f"propagation {name} k{k} {crop} {g}@{t}", (c["hits"], c["total"]), (o["hits"], o["total"]))

conf = load(R / "integration" / "confidence-val" / "confidence.json")["replay"]["status_drives_replay"]
rows = [r for r in csv.DictReader(open(F / "replay" / "smoothed.csv")) if r["source"] == "skitb"]
for col, key in (("box_containment_mean", "box_containment_mean"), ("frac_fully_inside", "frac_fully_inside"),
                 ("acceleration_mean", "acceleration_mean"), ("area_fraction_mean", "area_fraction_mean")):
    vals = [float(r[col]) for r in rows if r[col] != ""]
    same(f"replay {col}", float(np.mean(vals)), conf[key], tol=1e-9)

print(f"{checks} checks, {len(problems)} differences")
for p in problems:
    print("  DIFF", p)
sys.exit(1 if problems else 0)
