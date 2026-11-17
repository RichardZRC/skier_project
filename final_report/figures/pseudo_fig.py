"""Final report figure: what the pseudo-label student trained on (D29), as fixed in D30.

Six kept pseudo-label frames, one from each of six SkiTB training races; races and frames are picked at random with
seed 230, not by eye. Each panel crops around the SkiTB box. Filled points and lines: joints that passed the keep rule
(the training targets). Hollow points: the teacher's other body and ski points, dropped by the rule. The output shows
SkiTB frames (CC BY-NC 4.0), so it is never committed (.gitignore); it goes to Overleaf.

Usage (laptop venv): python final_report/figures/pseudo_fig.py final_report/figures/pseudo_examples.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skier.io_utils import read_json
from skier.paths import data_root, results_root
from skier.schema import load_schema

SEED, N_PANELS, PANEL_H = 230, 6, 360
BODY, SKI = (200, 110, 30), (30, 140, 245)  # BGR: blue, orange

out = Path(sys.argv[1])
schema = load_schema()
body, ski = set(schema.groups["body"]), set(schema.groups["ski"])
lab = results_root() / "pseudo/labels"
frames = read_json(lab / "pseudo_labels.json")["frames"]
with np.load(lab / "raw_predictions.npz") as z:
    row = {str(s): i for i, s in enumerate(z["sample_ids"])}
    mean_xy = z["xy"].mean(axis=0)  # the two teachers' final points, averaged (as the pseudo point)

# the selection rule (module docstring)
rng = np.random.default_rng(SEED)
races = sorted({f["sequence"] for f in frames})
picks = []
for race in sorted(rng.choice(races, size=N_PANELS, replace=False)):
    in_race = sorted((f for f in frames if f["sequence"] == race), key=lambda f: f["sample_id"])
    picks.append(in_race[int(rng.integers(len(in_race)))])


def pt(xy):
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def color(j):
    return SKI if j in ski else BODY


panels, titles = [], []
for f in picks:
    img = cv2.imread(str(data_root() / f["image"]))
    kept = np.array(f["kept"], dtype=bool)
    xy = mean_xy[row[f["sample_id"]]]
    x0, y0, x1, y1 = f["gt_box"]
    side = 1.4 * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = img.shape[:2]
    c0, r0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c1, r1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    s = max(1, round(3 * (r1 - r0) / PANEL_H))
    for e in schema.edges:
        if e.group != "pole" and kept[e.a] and kept[e.b]:
            cv2.line(img, pt(xy[e.a]), pt(xy[e.b]), color(e.b), s, cv2.LINE_AA)
    for j in sorted(body | ski):
        if not np.isfinite(xy[j]).all():
            continue
        if kept[j]:
            cv2.circle(img, pt(xy[j]), s + 2, color(j), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, pt(xy[j]), s + 2, color(j), max(1, s // 2), cv2.LINE_AA)
    im = img[r0:r1, c0:c1]
    im = cv2.resize(im, (max(1, round(im.shape[1] * PANEL_H / im.shape[0])), PANEL_H), interpolation=cv2.INTER_AREA)
    panels.append(im[..., ::-1])
    n_body, n_ski = int(kept[sorted(body)].sum()), int(kept[sorted(ski)].sum())
    titles.append(f"{f['sequence']} cam {f['clip']}\nkept: body {n_body}/14, ski {n_ski}/8")
    print(f["sample_id"], f"body {n_body}/14 ski {n_ski}/8")

cols = 3
fig, axes = plt.subplots(2, cols, figsize=(7.2, 5.4), gridspec_kw={"wspace": 0.04, "hspace": 0.18})
for ax, p, t in zip(axes.ravel(), panels, titles):
    ax.imshow(p)
    ax.set_title(t, fontsize=7, pad=2)
    ax.axis("off")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(out)
