"""Final report figure: tracker crop vs refined crop on val frames (E2, seed 230), with the predicted skeleton.

Frames are picked by a fixed rule, not by eye: the two frames with tracker IoU 0.1-0.5 whose ski hits@0.1 gain most
from refinement (from different videos), and the frame with IoU < 0.1 that has the most labeled ski points. Val only.
The output shows dataset images, so it is never committed (.gitignore).

Usage (laptop venv): python final_report/figures/qualitative_fig.py final_report/figures/qualitative_val.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skier.data.ski2dpose import load_records
from skier.eval.pose_metrics import gt_arrays, sample_hits
from skier.integration.propagation import tracker_crop_boxes
from skier.io_utils import read_json
from skier.paths import data_root, repo_root, results_root
from skier.pose.eval_generic import select_partition
from skier.pose.preprocess import pose_crop_box
from skier.schema import load_schema

RUN, K_POSE, SETTING = "E2-s230", "1.1", "s0.3_k1.3"
TRACKER, REFINED = (43, 130, 217), (10, 81, 140)  # BGR of the strata figure's orange and brown
PANEL_H = 420

out = Path(sys.argv[1])
schema = load_schema()
pole = set(schema.groups["pole"])
ski = list(schema.groups["ski"])
z = np.load(results_root() / "final/val/propagation/predictions.npz")
ids = [str(s) for s in z["sample_ids"]]
s2d = data_root() / "ski2dpose"
records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                       schema=schema)
by_id = {r.sample_id: r for r in select_partition(records, read_json(repo_root() / "data/manifests/pose_split_v1.json"),
                                                  "val")[0]}
recs = [by_id[i] for i in ids]
gt, rel, diag, videos = gt_arrays(recs)
tracked = z[f"{RUN}__k{K_POSE}"]
refined = z[f"{RUN}__k{K_POSE}__{SETTING}"]
tracker_crops = tracker_crop_boxes(z["tracker_boxes"], float(K_POSE))
refined_crops = z[f"{RUN}__k{K_POSE}__{SETTING}__crop_boxes"]
iou = z["tracker_iou"]
hit_t, n_ski = sample_hits(tracked, gt, rel, diag, ski, 0.1)
hit_r, _ = sample_hits(refined, gt, rel, diag, ski, 0.1)

# the selection rule (module docstring); ties keep the file order
picks, seen = [], set()
for i in np.argsort(-(hit_r - hit_t), kind="stable"):
    if 0.1 <= iou[i] < 0.5 and videos[i] not in seen:
        picks.append(int(i))
        seen.add(videos[i])
    if len(picks) == 2:
        break
picks.append(max((i for i in range(len(ids)) if iou[i] < 0.1), key=lambda i: (n_ski[i], -i)))


def pt(xy):
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def draw(img, crop, xy, color, rec, s):
    x0, y0, x1, y1 = crop
    cv2.rectangle(img, pt((x0, y0)), pt((x1, y1)), color, s + 1, cv2.LINE_AA)
    for e in schema.edges:
        if e.group != "pole" and np.isfinite(xy[e.a]).all() and np.isfinite(xy[e.b]).all():
            cv2.line(img, pt(xy[e.a]), pt(xy[e.b]), color, s, cv2.LINE_AA)
    for j in range(len(xy)):
        if j not in pole and np.isfinite(xy[j]).all():
            cv2.circle(img, pt(xy[j]), s + 1, color, -1, cv2.LINE_AA)
    for j in ski:  # labeled ski points, to judge the skis by eye
        if rec.reliable[j]:
            cv2.circle(img, pt(rec.keypoints_xy[j]), s + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt(rec.keypoints_xy[j]), s + 2, (0, 0, 0), 1, cv2.LINE_AA)


panels, titles = [], []
for tag, i in zip("abc", picks):
    rec = recs[i]
    img = cv2.imread(str(rec.image_path))
    t_crop = pose_crop_box(tracker_crops[i], 1.25)
    r_crop = pose_crop_box(refined_crops[i], 1.25)
    g_crop = pose_crop_box(rec.reference_box_xyxy, 1.25)
    boxes = np.stack([t_crop, r_crop, g_crop])
    x0, y0 = np.maximum(boxes[:, :2].min(axis=0) - 10, 0)
    x1, y1 = np.minimum(boxes[:, 2:].max(axis=0) + 10, [rec.image_width - 1, rec.image_height - 1])
    s = max(2, round(4 * (y1 - y0) / PANEL_H))  # about 4 px per line after the resize to PANEL_H
    for crop, xy, color, name, hits in ((t_crop, tracked[i], TRACKER, "tracker", hit_t[i]),
                                        (r_crop, refined[i], REFINED, "refined", hit_r[i])):
        im = img.copy()
        draw(im, crop, xy, color, rec, s)
        im = im[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        im = cv2.resize(im, (max(1, round(im.shape[1] * PANEL_H / im.shape[0])), PANEL_H), interpolation=cv2.INTER_AREA)
        panels.append(im[..., ::-1])
        titles.append(f"({tag}) {name}\nski {hits}/{n_ski[i]}")
    print(f"({tag}) {ids[i]}  IoU {iou[i]:.2f}  {z['seconds_since_init'][i]:.1f} s  ski hits {hit_t[i]} -> {hit_r[i]} "
          f"of {n_ski[i]}")

widths = [p.shape[1] for p in panels]
fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 7.2 * PANEL_H / sum(widths) + 0.35),
                         gridspec_kw={"width_ratios": widths, "wspace": 0.04})
for ax, p, t in zip(axes, panels, titles):
    ax.imshow(p)
    ax.set_title(t, fontsize=7, pad=2)
    ax.axis("off")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(out)
