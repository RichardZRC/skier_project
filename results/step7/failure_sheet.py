"""Step 7 R7: review sheets for the val labels where the tracker crop costs the most (plan §12 review block).

Usage: python failure_sheet.py [CHECKPOINT_NAME] [K_POSE] [N]   (defaults: E2-s230 1.1 24)
Reads $SKIER_RESULTS/integration/propagation-val/{predictions.npz, usable_segments.json} and the track CSVs.
Per label: loss = reference-crop hits − tracker-crop hits (body + ski, PCK@0.1, reliable labels). The N largest,
ordered by tracker-IoU stratum, become rows of three panels:
  1. labels (white crosses), reference box (yellow), tracker box (green), tracker pose crop (green, dashed);
  2. the prediction from the reference crop; 3. the prediction from the tracker crop.
Writes sheet_*.jpg and failure_review.csv (with an automatic first-pass hint and an empty category column for
Ruichao: plan §12 failure categories). Val images only.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from skier.data.ski2dpose import load_records
from skier.eval.pose_metrics import gt_arrays
from skier.integration import propagation as prop
from skier.integration.config import IntegrationConfig
from skier.io_utils import ensure_dir, read_json, write_csv
from skier.pose.eval_generic import select_partition
from skier.pose.evaluate import inside_crop
from skier.pose.preprocess import pose_crop_box
from skier.schema import load_schema
from skier.viz import REF_BOX_COLOR, draw_box, draw_header, prediction_overlay

name, k, n_show = (sys.argv[1:] + ["E2-s230", "1.1", "24"][len(sys.argv) - 1:])[:3]
k, n_show = float(k), int(n_show)
repo = Path.home() / "skier_project"
R, s2d = Path(os.environ["SKIER_RESULTS"]), Path(os.environ["SKIER_DATA"]) / "ski2dpose"
P = R / "integration" / "propagation-val"
out = ensure_dir(R / "integration" / "failure_review")
schema = load_schema()
records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                       schema=schema)
eligible, _ = select_partition(records, read_json(repo / "data/manifests/pose_split_v1.json"), "val")
usable = read_json(P / "usable_segments.json")
init, scored = prop.split_init(eligible, usable["usable"])
fps = {(s["video_id"], s["segment_id"]): s["fps"] for s in usable["segments"]}
icfg = IntegrationConfig.from_yaml(repo / "configs/integration/default.yaml")
trk = prop.tracker_at(scored, init, R / "tracking" / "ski2dpose-val", fps, icfg)
pred = np.load(P / "predictions.npz")
assert list(pred["sample_ids"]) == [r.sample_id for r in scored], "propagation predictions don't match the records"
ref_xy, trk_xy = pred[f"{name}__reference"], pred[f"{name}__k{k:g}"]
gt, rel, diag, _ = gt_arrays(scored)
groups = list(schema.groups["body"]) + list(schema.groups["ski"])


def hits(xy):
    d = np.linalg.norm(xy - gt, axis=-1) / diag[:, None]
    return ((d <= 0.1) & rel & np.isfinite(d))[:, groups].sum(axis=1)


h_ref, h_trk, total = hits(ref_xy), hits(trk_xy), rel[:, groups].sum(axis=1)
crop_boxes = prop.tracker_crop_boxes(trk["boxes"], k)
finite = np.isfinite(crop_boxes).all(axis=1)
inside = inside_crop(scored, [b if f else r.reference_box_xyxy for b, f, r in zip(crop_boxes, finite, scored)])
inside[~finite] = False
cut = (rel & ~inside)[:, groups].sum(axis=1)
order = sorted(range(len(scored)), key=lambda i: (-(h_ref[i] - h_trk[i]), scored[i].sample_id))[:n_show]
rank = {"iou<0.1": 0, "0.1<=iou<0.5": 1, "iou>=0.5": 2}
order.sort(key=lambda i: (rank[prop.iou_bin(trk["iou"][i])], -(h_ref[i] - h_trk[i]), scored[i].sample_id))


def area(b):
    return float((b[2] - b[0]) * (b[3] - b[1]))


area_ratio = np.array([area(t) / area(r.reference_box_xyxy) if np.isfinite(t).all() else np.nan
                       for t, r in zip(trk["boxes"], scored)])


def hint(i):
    if cut[i] == 0 and area_ratio[i] > 4:
        return f"scale drift: tracker box {area_ratio[i]:.0f}x the skier's, skier small in the crop"
    if not np.isfinite(trk["boxes"][i]).all() or trk["iou"][i] < 0.1:
        return "tracker drift: crop off the skier"
    if cut[i] > 0:
        return f"tracker crop cuts off {cut[i]} labels"
    return "crop holds the labels; pose differs"


def letterbox(img, size=360):
    s = size / max(img.shape[:2])
    img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


rows, panels = [], []
for rank_i, i in enumerate(order, 1):
    r = scored[i]
    img = cv2.imread(str(r.image_path))
    ref_crop = pose_crop_box(r.reference_box_xyxy)
    boxes = [ref_crop] + ([pose_crop_box(crop_boxes[i])] if np.isfinite(crop_boxes[i]).all() else [])
    x1, y1 = np.min([b[:2] for b in boxes], axis=0)
    x2, y2 = np.max([b[2:] for b in boxes], axis=0)
    pad = 0.05 * max(x2 - x1, y2 - y1)
    X1, Y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    X2, Y2 = int(min(img.shape[1], x2 + pad)), int(min(img.shape[0], y2 + pad))
    a = prediction_overlay(img, np.full_like(gt[i], np.nan), gt[i], rel[i], schema)
    draw_box(a, r.reference_box_xyxy, REF_BOX_COLOR)
    if np.isfinite(trk["boxes"][i]).all():
        draw_box(a, trk["boxes"][i], (0, 255, 0))
        draw_box(a, pose_crop_box(crop_boxes[i]), (0, 255, 0), dashed=True)
    b = prediction_overlay(img, ref_xy[i], gt[i], rel[i], schema)
    c = prediction_overlay(img, trk_xy[i], gt[i], rel[i], schema)
    if np.isfinite(trk["boxes"][i]).all():
        draw_box(c, pose_crop_box(crop_boxes[i]), (0, 255, 0), dashed=True)
    tiles = [letterbox(p[Y1:Y2, X1:X2]) for p in (a, b, c)]
    draw_header(tiles[0], [f"#{rank_i} {r.sample_id}", f"IoU {trk['iou'][i]:.2f}, {trk['status'][i]}, {trk['seconds'][i]:.1f} s"])
    draw_header(tiles[1], [f"reference crop: {h_ref[i]}/{total[i]}"])
    draw_header(tiles[2], [f"tracker crop: {h_trk[i]}/{total[i]}"])
    panels.append(np.hstack(tiles))
    rows.append({"rank": rank_i, "sample_id": r.sample_id, "tracker_iou": round(float(trk["iou"][i]), 3),
                 "status": trk["status"][i], "seconds_since_init": round(float(trk["seconds"][i]), 2),
                 "reference_hits": int(h_ref[i]), "tracker_hits": int(h_trk[i]), "labels": int(total[i]),
                 "cut_off": int(cut[i]), "box_area_ratio": round(float(area_ratio[i]), 2), "auto_hint": hint(i),
                 "category": "", "note": ""})
per_page = 6
for p in range(0, len(panels), per_page):
    cv2.imwrite(str(out / f"sheet_{p // per_page + 1}.jpg"), np.vstack(panels[p:p + per_page]), [cv2.IMWRITE_JPEG_QUALITY, 90])
write_csv(rows, out / "failure_review.csv", list(rows[0]))
print(f"{name} k_pose {k:g}: {len(rows)} labels, {len(panels) // per_page + bool(len(panels) % per_page)} sheets → {out}")
for row in rows:
    print(row["rank"], row["sample_id"], row["tracker_iou"], row["status"], f"{row['reference_hits']}->{row['tracker_hits']}/{row['labels']}",
          row["auto_hint"])
