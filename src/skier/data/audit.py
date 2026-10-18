"""Label audit and image inventory for Ski 2DPose (guide 1E, code spec §6).

Reports problems; never clamps, fixes or drops anything.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from skier.data import ski2dpose_raw as raw_mod
from skier.data.ski2dpose_raw import (
    NUM_KEYPOINTS,
    annotation_array,
    image_path_for,
    info_for,
    info_preview,
    iter_raw_labels,
)
from skier.io_utils import ensure_dir, read_json, read_json_with_duplicates, write_csv, write_json
from skier.paths import repo_root, results_root
from skier.schema import Schema, load_schema

EXPECTED = {"videos": 16, "segments": 158, "images": 1982}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROBLEM_COLUMNS = ["sample_id", "problem", "detail"]
INVENTORY_COLUMNS = [
    "sample_id", "video_id", "segment_id", "image_id", "frame_idx", "image_path",
    "width", "height", "n_reliable_body", "n_reliable_ski", "n_reliable_pole",
    "discipline", "view_angle",
]


@dataclass
class AuditResult:
    summary: dict
    problems: list[dict]
    inventory: list[dict]


def _joints(mask: np.ndarray) -> str:
    return "joints " + ",".join(str(i) for i in np.flatnonzero(mask))


def _shape_detail(annotation) -> str:
    try:
        shape = np.asarray(annotation, dtype=np.float64).shape
    except (TypeError, ValueError):
        return "not a rectangular numeric array"
    return f"shape {shape}, expected ({NUM_KEYPOINTS}, 3)"


def _value_key(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def audit_labels(
    labels_path: str | Path,
    info_path: str | Path | None = None,
    images_root: str | Path | None = None,
    path_template: str | None = None,
    schema: Schema | None = None,
) -> AuditResult:
    """Run pass 1 (structure) and, with ``images_root``, pass 2 (images).

    Raises ValueError for a broken nesting and NotImplementedError when pass 2 is
    requested but no image path template is known.
    """
    schema = schema or load_schema()
    labels, duplicate_keys = read_json_with_duplicates(labels_path)
    records = list(iter_raw_labels(labels))
    info = read_json(info_path) if info_path else None
    if images_root is not None and path_template is None and raw_mod.IMAGE_PATH_TEMPLATE is None:
        raise NotImplementedError("image path template not set: fill in at 1E (code spec §13)")

    problems: list[dict] = [
        {"sample_id": "", "problem": "duplicate_key", "detail": f"duplicate key {k!r}"}
        for k in duplicate_keys
    ]

    def report(sample_id: str, problem: str, detail: str = "") -> None:
        problems.append({"sample_id": sample_id, "problem": problem, "detail": detail})

    groups = {g: list(schema.groups[g]) for g in ("body", "ski", "pole")}
    visibility_values: Counter[str] = Counter()
    reliable_per_joint = np.zeros(NUM_KEYPOINTS, dtype=int)
    frame_owner: dict[tuple[str, str, str], str] = {}
    inventory: list[dict] = []

    # Pass 1: structure only.
    for raw in records:
        sid = raw.sample_id
        row = {
            "sample_id": sid, "video_id": raw.video_id, "segment_id": raw.segment_id,
            "image_id": raw.image_id, "frame_idx": raw.frame_idx,
        }
        if info is not None:
            meta = info_for(info, raw.video_id, raw.segment_id)
            row["discipline"] = meta.get("discipline")
            row["view_angle"] = meta.get("view_angle")
        inventory.append(row)

        if raw.annotation is None:
            report(sid, "missing_annotation", "entry is not an object or has no 'annotation'")
            continue

        ann = annotation_array(raw)
        if ann is None:
            report(sid, "bad_shape", _shape_detail(raw.annotation))
        else:
            finite = np.isfinite(ann)
            if not finite.all():
                report(sid, "non_finite", _joints(~finite.all(axis=1)))
            xy, vis = ann[:, :2], ann[:, 2]
            out = finite[:, :2] & ((xy < 0) | (xy > 1))
            if out.any():
                report(sid, "coord_out_of_range", _joints(out.any(axis=1)))
            vis_ok = finite[:, 2]
            bad_vis = vis_ok & ~np.isin(vis, (0.0, 1.0))
            if bad_vis.any():
                values = sorted({_value_key(v) for v in vis[bad_vis]})
                report(sid, "bad_visibility", "values " + ",".join(values))
            visibility_values.update(_value_key(v) for v in vis[vis_ok])
            reliable = vis == 1
            reliable_per_joint += reliable
            for g, members in groups.items():
                row[f"n_reliable_{g}"] = int(reliable[members].sum())

        if raw.frame_idx is None:
            report(sid, "missing_frame_idx")
        else:
            key = (raw.video_id, raw.segment_id, str(raw.frame_idx))
            if key in frame_owner:
                report(sid, "duplicate_frame_idx", f"same frame_idx as {frame_owner[key]}")
            else:
                frame_owner[key] = sid

    # Pass 2: images.
    if images_root is not None:
        images_root = Path(images_root)
        claimed: dict[str, str] = {}
        for raw, row in zip(records, inventory):
            sid = raw.sample_id
            path = image_path_for(images_root, raw.video_id, raw.segment_id, raw.image_id, path_template)
            rel = path.relative_to(images_root).as_posix()
            row["image_path"] = rel
            if rel in claimed:
                report(sid, "duplicate_image_path", f"same file as {claimed[rel]}")
            else:
                claimed[rel] = sid
            if not path.is_file():
                report(sid, "label_without_image", rel)
                continue
            try:
                with Image.open(path) as im:  # reads the header only
                    row["width"], row["height"] = im.size
            except Exception as exc:  # PIL raises several types for bad files
                report(sid, "unreadable_image", f"{type(exc).__name__}")
        for f in sorted(images_root.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rel = f.relative_to(images_root).as_posix()
                if rel not in claimed:
                    report(rel, "image_without_label")

    problems.sort(key=lambda p: (p["problem"], p["sample_id"]))
    inventory.sort(key=lambda r: r["sample_id"])

    n_videos = len(labels)
    n_segments = sum(len(segs) for segs in labels.values())
    summary = {
        "n_videos": n_videos,
        "n_segments": n_segments,
        "n_images": len(records),
        "images_per_video": {v: sum(len(imgs) for imgs in segs.values()) for v, segs in labels.items()},
        "segments_per_video": {v: len(segs) for v, segs in labels.items()},
        "visibility_values": dict(visibility_values),
        "reliable_per_joint": reliable_per_joint.tolist(),
        "problem_counts": dict(Counter(p["problem"] for p in problems)),
        "expected": EXPECTED,
        "matches_expected": (n_videos, n_segments, len(records))
        == (EXPECTED["videos"], EXPECTED["segments"], EXPECTED["images"]),
        "images_checked": images_root is not None,
        "args": {
            "labels": str(labels_path),
            "info": str(info_path) if info_path else None,
            "images_root": str(images_root) if images_root is not None else None,
            "path_template": path_template or raw_mod.IMAGE_PATH_TEMPLATE,
        },
    }
    if info is not None:
        summary["info_preview"] = info_preview(info)
    return AuditResult(summary, problems, inventory)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit Ski 2DPose labels and build the image inventory (guide 1E).",
    )
    p.add_argument("labels", type=Path, metavar="LABELS_JSON")
    p.add_argument("--info", type=Path, help="ski2dpose_info.json (adds info_preview, discipline, view)")
    p.add_argument("--images-root", type=Path, help="run pass 2 against the extracted images")
    p.add_argument("--path-template", help="override IMAGE_PATH_TEMPLATE, e.g. '{video_id}/{segment_id}/{image_id}.jpg'")
    p.add_argument("--inventory", type=Path, help="default: <repo>/data/manifests/pose_inventory.csv")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/audit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = ensure_dir(args.out or results_root() / "step1" / "audit")
    inventory_path = args.inventory or repo_root() / "data" / "manifests" / "pose_inventory.csv"
    try:
        result = audit_labels(args.labels, args.info, args.images_root, args.path_template)
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(result.summary, out / "label_summary.json")
    write_csv(result.problems, out / "problems.csv", PROBLEM_COLUMNS)
    write_csv(result.inventory, inventory_path, INVENTORY_COLUMNS)

    s = result.summary
    print(f"videos {s['n_videos']}  segments {s['n_segments']}  images {s['n_images']}"
          f"  (matches expected: {s['matches_expected']})")
    for problem, count in sorted(s["problem_counts"].items()):
        print(f"  {problem}: {count}")
    print(f"wrote {out / 'label_summary.json'}, {out / 'problems.csv'}, {inventory_path}")
    return 0
