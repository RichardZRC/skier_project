"""Step 8: our port of the SkiTB toolkit metrics vs. the real toolkit report(), on our SiamFC tracks of one partition.

The track CSVs are written in the toolkit's result layout (xywh %.3f; confidences %.8f, init 1.0; zero times).
Then the toolkit's own ExperimentSkiTB.report() runs. Two workarounds apply here only:
- np.loadtxt drops delimiter='\\n', which numpy 1.26 refuses;
- clips are named SEQ_idx, which dodges the toolkit's ST/ bug.
The per-clip and overall F, precision, recall, threshold and GSR are compared with skier.eval.skitb_official.

Usage: python toolkit_crosscheck.py PARTITION TRACK_DIR OUT_JSON
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from skier.data.skitb import load_skitb_clips
from skier.eval import skitb_official as off
from skier.tracking.run import read_track_csv

part, track_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
D = Path(os.environ["SKIER_DATA"])
skitb, split = D / "skitb", D / "skitb" / "AL" / "AL_train_val_test_athlete_60-40.json"
tmp = Path(tempfile.mkdtemp())
names, ours = [], {}
for c in load_skitb_clips(skitb, split, part):
    rows = read_track_csv(track_dir / f"{c.sequence_id}__{c.clip_id}.csv")
    boxes, conf, anno, vis = off.skitb_clip_inputs(c, rows)
    name = f"{c.sequence_id}_{c.clip_id}"
    rec = tmp / "SkiTB" / "siamfc" / "SC" / "AL" / "ope" / name
    rec.mkdir(parents=True)
    np.savetxt(rec / "boxes.txt", boxes, fmt="%.3f", delimiter=",")
    np.savetxt(rec / "confidences.txt", conf, fmt="%.8f")
    np.savetxt(rec / "times.txt", np.zeros(len(conf)), fmt="%.8f")
    names.append(name)
    ours[name] = off.clip_metrics(boxes, conf, anno, vis)

real = np.loadtxt
np.loadtxt = lambda *a, **kw: real(*a, **{k: v for k, v in kw.items() if not (k == "delimiter" and v == "\n")})
sys.path.insert(0, str(D / "vendor" / "SkiTB-toolkit"))
from toolkit.experiments import ExperimentSkiTB  # noqa: E402

exp = ExperimentSkiTB(str(skitb), discipline="AL", mode="SC", test_videos=names, result_dir=str(tmp),
                      report_dir=str(tmp / "reports"))
perf = exp.report(["siamfc"], fn_note=part)["siamfc"]
keys = ("f_score", "precision", "recall", "f_score_thresh")
clip_diff = max(abs(ours[n][k] - perf["seq_wise"][n][k]) for n in names for k in keys)
gsr_diff = max(abs(ours[n]["gsr_score"] - perf["seq_wise"][n]["generalized_success_robustness_score"]) for n in names)
mine = off.overall(ours, names)
overall_diff = max(abs(mine[k] - perf["overall"][k]) for k in keys)
overall_diff = max(overall_diff, abs(mine["gsr_score"] - perf["overall"]["generalized_success_robustness_score"]))
record = {"partition": part, "n_clips": len(names), "max_abs_diff_per_clip": clip_diff, "max_abs_diff_gsr": gsr_diff,
          "max_abs_diff_overall": overall_diff, "ours_overall": mine,
          "toolkit_overall": {k: perf["overall"][k] for k in keys} | {"gsr_score": perf["overall"]["generalized_success_robustness_score"]},
          "toolkit": "vendor SkiTB-toolkit, report() with the two workarounds above"}
out.write_text(json.dumps(record, indent=1))
print(json.dumps({k: v for k, v in record.items() if k.startswith(("n_", "max"))}, indent=1))
sys.exit(0 if max(clip_diff, gsr_diff, overall_diff) < 1e-9 else 1)
