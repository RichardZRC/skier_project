"""Why follow-cam is weak on skis: camera viewpoint per partition, and per-joint ski PCK on follow-cam (post hoc).

The viewpoint comes from the Ski 2DPose info JSON's per-segment flags (front_view / side_view / back_view, several can
be true), counted over labeled images. The per-joint numbers are read from the frozen tables
(results/step8/{followcam,val}/pose_per_joint.csv). Nothing is re-evaluated.

Usage (GPU machine, inside Ubuntu): python results/step8/followcam_views.py [--out results/step8/followcam_views.txt]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.io_utils import read_json  # noqa: E402
from skier.paths import data_root, repo_root  # noqa: E402

FLAGS = ("front_view", "side_view", "back_view")
SKI_JOINTS = ("ski_tip_right", "toes_right", "heel_right", "ski_tail_right",
              "ski_tip_left", "toes_left", "heel_left", "ski_tail_left")


def view_shares(labels: dict, info: dict, part: dict) -> dict:
    counts: dict = {p: Counter() for p in sorted(set(part.values()))}
    for vid, segs in labels.items():
        for seg, imgs in segs.items():
            flags = ((info.get(vid) or {}).get("splits_info") or {}).get(seg) or {}
            view = "+".join(f.removesuffix("_view") for f in FLAGS if flags.get(f)) or "none"
            counts[part[vid]][view] += len(imgs)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=repo_root() / "results/step8/followcam_views.txt")
    a = p.parse_args(argv)
    s2d = data_root() / "ski2dpose"
    manifest = read_json(repo_root() / "data/manifests/pose_split_v1.json")
    part = {v: q for q, d in manifest["partitions"].items() for v in d["videos"]}
    counts = view_shares(read_json(s2d / "labels/ski2dpose_labels.json"),
                         read_json(s2d / "archives/ski2dpose_info.json"), part)
    lines = ["Camera viewpoint by partition (share of labeled images; flags from the info JSON, several can be true)", ""]
    for q in ("train", "val", "test", "followcam"):
        c = counts[q]
        tot = sum(c.values())
        back_any = sum(v for k, v in c.items() if "back" in k)
        lines.append(f"{q:<10} {tot:>5} images | any back view {100 * back_any / tot:5.1f}% | back only "
                     f"{100 * c.get('back', 0) / tot:5.1f}% | " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in c.most_common()))
    lines += ["", "Ski PCK@0.1 per joint (frozen tables, seeds pooled): follow-cam E1 / E2 vs. val E2", ""]
    rows = {q: {r["joint"]: r for r in csv.DictReader(open(repo_root() / f"results/step8/{q}/pose_per_joint.csv"))}
            for q in ("followcam", "val")}
    for j in SKI_JOINTS:
        f, v = rows["followcam"][j], rows["val"][j]
        lines.append(f"{j:<15} follow-cam {float(f['E1']):5.1f} / {float(f['E2']):5.1f}   val {float(v['E2']):5.1f}")
    text = "\n".join(lines) + "\n"
    a.out.write_text(text, encoding="utf-8")
    print(text + f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
