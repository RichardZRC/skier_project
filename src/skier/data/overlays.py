"""Ground-truth overlays and contact sheet for the manual label review (guide 1F, code spec §7)."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from skier.data.ski2dpose_raw import annotation_array, iter_raw_labels, safe_name
from skier.io_utils import ensure_dir, read_csv, read_id_list, read_json
from skier.paths import results_root
from skier.schema import load_schema
from skier.viz import contact_sheet, render_gt_overlay


def select_overlay_samples(rows: list[dict], pilot_videos: list[str], n: int, seed: int) -> list[str]:
    """Pick ``n`` pilot-video samples, spread round-robin across (view, video, segment) groups."""
    pilot = set(pilot_videos)
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        if row["video_id"] in pilot and row.get("width") and row.get("n_reliable_body"):
            key = (row.get("view_angle") or "", row["video_id"], row["segment_id"])
            groups[key].append(row["sample_id"])
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected: list[str] = []
    while len(selected) < n and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop(0))
    if len(selected) < n:
        print(f"warning: only {len(selected)} eligible pilot images, wanted {n}", file=sys.stderr)
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Draw ground-truth overlays for the label review (guide 1F).")
    p.add_argument("--inventory", type=Path, required=True, help="pose_inventory.csv from the audit")
    p.add_argument("--labels", type=Path, required=True, metavar="LABELS_JSON")
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--videos-file", type=Path, required=True, help="pilot video IDs, one per line")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--schema", type=Path, help="default: configs/ski24_schema.json")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/overlays")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = ensure_dir(args.out or results_root() / "step1" / "overlays")
    schema = load_schema(args.schema)
    rows = read_csv(args.inventory)
    row_by_id = {r["sample_id"]: r for r in rows}
    raw_by_id = {raw.sample_id: raw for raw in iter_raw_labels(read_json(args.labels))}

    ids = select_overlay_samples(rows, read_id_list(args.videos_file), args.n, args.seed)
    drawn, thumbs = [], []
    for sid in ids:
        row = row_by_id[sid]
        img = cv2.imread(str(args.images_root / row["image_path"]))
        ann = annotation_array(raw_by_id[sid]) if sid in raw_by_id else None
        if img is None or ann is None:
            print(f"warning: skipping {sid} (image or annotation unreadable)", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        header = [
            sid,
            f"{w}x{h}  discipline: {row.get('discipline') or '?'}  view: {row.get('view_angle') or '?'}",
        ]
        overlay = render_gt_overlay(img, ann, schema, header)
        cv2.imwrite(str(out / f"{safe_name(sid)}.png"), overlay)
        thumbs.append(overlay)
        drawn.append(sid)

    cv2.imwrite(str(out / "contact_sheet.png"), contact_sheet(thumbs))
    (out / "overlay_ids.txt").write_text("".join(f"{sid}\n" for sid in drawn), encoding="utf-8")
    print(f"drew {len(drawn)} overlays in {out}")
    return 0
