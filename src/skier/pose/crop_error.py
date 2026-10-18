"""The fixed crop-error set (plan Step 6, D16): perturbed crops standing in for tracker
localization error.

For each sample and severity level ``r``, the reference box's center moves by exactly
``r`` (in units of the box's width and height) in a random direction, and its size by a
factor in ``[1/(1+r), 1+r]``. Evaluation then crops around that box exactly as it crops
around the reference box (pad 1.25, fit the aspect).

Only the perturbation parameters are stored, never coordinates, so the file holds no
label data; boxes are rebuilt from the records with ``jitter_box``. Each sample's draws
depend only on ``(seed, sample_id)``. The file is generated once and frozen like the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.io_utils import read_json, write_json
from skier.paths import repo_root
from skier.pose.augment import jitter_box

LEVELS = {"mild": 0.05, "moderate": 0.10, "severe": 0.20}
FINAL_ONLY = ("test", "followcam")  # evaluated once, in Step 8 (D11)


def _sample_rng(seed: int, sample_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, int(hashlib.sha256(sample_id.encode()).hexdigest()[:15], 16)])


def _content_sha256(ce: dict) -> str:
    content = {k: ce[k] for k in ("partition", "split_sha256", "seed", "levels", "items")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def make_crop_error_set(sample_ids: Sequence[str], split_sha256: str, partition: str, seed: int = 230,
                        levels: dict[str, float] | None = None) -> dict:
    levels = dict(levels or LEVELS)
    ids = sorted(sample_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    items = []
    for sid in ids:
        rng = _sample_rng(seed, sid)
        for level, r in levels.items():
            phi, u = rng.random(2)
            items.append({"sample_id": sid, "level": level, "dx": r * math.cos(2 * math.pi * phi),
                          "dy": r * math.sin(2 * math.pi * phi), "scale": (1 + r) ** (2 * u - 1)})
    ce = {"version": 1, "partition": partition, "split_sha256": split_sha256, "seed": seed, "levels": levels,
          "n_samples": len(ids), "items": items,
          "note": "shift (dx, dy) in reference-box widths/heights, |shift| = level; size x scale (plan Step 6, D16)"}
    ce["content_sha256"] = _content_sha256(ce)
    return ce


def check_crop_error_set(ce: dict, split_sha256: str, partition: str, sample_ids: Sequence[str]) -> None:
    """Refuse a set made for another split or partition, edited by hand, or missing samples."""
    if ce.get("content_sha256") != _content_sha256(ce):
        raise ValueError("crop-error set content doesn't match its content_sha256 (edited?)")
    if ce["split_sha256"] != split_sha256:
        raise ValueError("crop-error set was made for a different split")
    if ce["partition"] != partition:
        raise ValueError(f"crop-error set is for partition {ce['partition']!r}, not {partition!r}")
    have = {(it["sample_id"], it["level"]) for it in ce["items"]}
    missing = [(s, lv) for s in sample_ids for lv in ce["levels"] if (s, lv) not in have]
    if missing:
        raise ValueError(f"{len(missing)} (sample, level) pairs missing from the crop-error set, e.g. {missing[0]}")


def perturbed_boxes(records: Sequence[PoseRecord], ce: dict, level: str) -> list[np.ndarray]:
    if level not in ce["levels"]:
        raise KeyError(f"unknown level {level!r}; have {sorted(ce['levels'])}")
    by_id = {it["sample_id"]: it for it in ce["items"] if it["level"] == level}
    return [jitter_box(r.reference_box_xyxy, by_id[r.sample_id]["dx"], by_id[r.sample_id]["dy"],
                       by_id[r.sample_id]["scale"]) for r in records]


def default_out(partition: str) -> Path:
    name = "crop_error_set_v1.json" if partition == "val" else f"crop_error_set_{partition}_v1.json"
    return repo_root() / "data" / "manifests" / name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate the frozen crop-error set (plan Step 6) from the split manifest.")
    p.add_argument("--split", type=Path, help="frozen split; default: data/manifests/pose_split_v1.json")
    p.add_argument("--partition", default="val")
    p.add_argument("--final-test", action="store_true", help="allow test/followcam (Step 8 only)")
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--out", type=Path, help="default: data/manifests/crop_error_set_v1.json (val)")
    p.add_argument("--force", action="store_true", help="overwrite an existing (frozen) set")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition in FINAL_ONLY and not args.final_test:
        print(f"error: {args.partition!r} is evaluated once, in Step 8; pass --final-test then", file=sys.stderr)
        return 1
    out = args.out or default_out(args.partition)
    if out.exists() and not args.force:
        print(f"error: {out} exists and is frozen; pass --force only for a documented reason "
              f"(record it in docs/decisions.md)", file=sys.stderr)
        return 1
    manifest = read_json(args.split or repo_root() / "data" / "manifests" / "pose_split_v1.json")
    if args.partition not in manifest["partitions"]:
        print(f"error: the split has no partition {args.partition!r}", file=sys.stderr)
        return 1
    ids = manifest["partitions"][args.partition]["sample_ids"]
    ce = make_crop_error_set(ids, manifest["content_sha256"], args.partition, seed=args.seed)
    write_json(ce, out)
    print(f"{len(ids)} {args.partition} samples x {len(ce['levels'])} levels {ce['levels']} -> {out}")
    return 0
