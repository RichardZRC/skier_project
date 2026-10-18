"""Video-grouped pose split (plan Step 2.3, decisions D4 and D11).

Whole source videos stay in one partition. The search is deterministic and
exhaustive: every test set of 2-4 videos, then every disjoint val set of 2-4
videos, is checked against the hard rules and scored; the lowest score wins and
ties are broken by a seeded shuffle. Videos can also be set aside whole into named
extra partitions (e.g. ``followcam``) that sit outside the 70/15/15 split.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.io_utils import read_id_list, write_json
from skier.paths import data_root, repo_root
from skier.schema import Schema, load_schema

FOLLOW_CAM = "Follow Cam"
MAIN_PARTITIONS = ("train", "val", "test")
TIE_TOL = 1e-12


@dataclass(frozen=True)
class VideoStats:
    video_id: str
    n_images: int
    n_ski_points: int  # reliable ski landmarks over all eligible images
    disciplines: frozenset[str]
    followcam: bool


@dataclass
class Split:
    partitions: dict[str, list[str]]  # partition -> sorted video IDs
    score: float
    n_feasible: int  # (test, val) combinations that passed the hard rules
    n_ties: int
    args: dict = field(default_factory=dict)


def video_stats_from_records(records: Iterable[PoseRecord], schema: Schema | None = None) -> list[VideoStats]:
    """One row per video, counting eligible (non-excluded) records only."""
    schema = schema or load_schema()
    ski = list(schema.groups["ski"])
    acc: dict[str, dict] = defaultdict(lambda: {"img": 0, "ski": 0, "disc": set()})
    for r in records:
        if r.excluded:
            continue
        a = acc[r.source_video_id]
        a["img"] += 1
        a["ski"] += int(r.reliable[ski].sum())
        if r.discipline:
            a["disc"].add(r.discipline)
    return [
        VideoStats(v, a["img"], a["ski"], frozenset(a["disc"]), FOLLOW_CAM in a["disc"])
        for v, a in sorted(acc.items())
    ]


def _validate(stats, forced_train, forced_test, set_aside, max_videos, targets):
    ids = [s.video_id for s in stats]
    if len(set(ids)) != len(ids):
        raise ValueError("video_stats has duplicate video IDs")
    if len(targets) != 3 or not np.isclose(sum(targets), 1.0):
        raise ValueError(f"targets must be 3 fractions summing to 1, got {targets}")
    groups = {"forced_train": list(forced_train), "forced_test": list(forced_test)}
    for name, vids in set_aside.items():
        if name in MAIN_PARTITIONS:
            raise ValueError(f"set-aside partition can't be named {name!r}")
        groups[f"set_aside[{name}]"] = list(vids)
    known = set(ids)
    seen: dict[str, str] = {}
    for name, vids in groups.items():
        for v in vids:
            if v not in known:
                raise ValueError(f"{name}: unknown video ID {v!r}")
            if v in seen:
                raise ValueError(f"video {v!r} is in both {seen[v]} and {name}")
            seen[v] = name
    if len(forced_test) > max_videos:
        raise ValueError(f"forced_test has {len(forced_test)} videos, more than max_videos={max_videos}")


def make_pose_split(
    video_stats: Sequence[VideoStats],
    targets: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 230,
    forced_test: Sequence[str] = (),
    forced_train: Sequence[str] = (),
    set_aside: Mapping[str, Sequence[str]] | None = None,
    min_ski_points: int = 400,
    min_videos: int = 2,
    max_videos: int = 4,
    discipline_weight: float = 0.05,
) -> Split:
    set_aside = {k: sorted(v) for k, v in (set_aside or {}).items()}
    _validate(video_stats, forced_train, forced_test, set_aside, max_videos, targets)
    args = {
        "targets": list(targets), "seed": seed, "forced_test": sorted(forced_test),
        "forced_train": sorted(forced_train), "set_aside": set_aside, "min_ski_points": min_ski_points,
        "min_videos": min_videos, "max_videos": max_videos, "discipline_weight": discipline_weight,
    }
    aside = {v for vids in set_aside.values() for v in vids}
    main = sorted((s for s in video_stats if s.video_id not in aside), key=lambda s: s.video_id)
    total = sum(s.n_images for s in main)
    if total == 0:
        raise ValueError("no images in the main (not set-aside) videos")
    race_types = sorted(set().union(*(s.disciplines for s in main)) - {FOLLOW_CAM})
    type_bit = {t: 1 << i for i, t in enumerate(race_types)}

    cand = [s for s in main if s.video_id not in set(forced_train)]
    cand_bit = {s.video_id: 1 << i for i, s in enumerate(cand)}
    forced_mask = sum(cand_bit[v] for v in forced_test)

    # Every candidate subset of allowed size, as parallel arrays.
    subsets, masks, imgs, skis, ntypes, sizes = [], [], [], [], [], []
    for k in range(min_videos, max_videos + 1):
        for combo in itertools.combinations(range(len(cand)), k):
            members = [cand[i] for i in combo]
            subsets.append(combo)
            masks.append(sum(1 << i for i in combo))
            imgs.append(sum(s.n_images for s in members))
            skis.append(sum(s.n_ski_points for s in members))
            tmask = 0
            for s in members:
                for d in s.disciplines:
                    tmask |= type_bit.get(d, 0)
            ntypes.append(bin(tmask).count("1"))
            sizes.append(k)
    if not subsets:
        raise ValueError(f"only {len(cand)} candidate videos for val/test; need at least {2 * min_videos}")
    masks = np.array(masks, dtype=np.int64)
    frac = np.array(imgs, dtype=np.float64) / total
    skis = np.array(skis)
    miss = len(race_types) - np.array(ntypes)
    sizes = np.array(sizes)

    test_ok = ((masks & forced_mask) == forced_mask) & (skis >= min_ski_points)
    val_ok = ((masks & forced_mask) == 0) & (skis >= min_ski_points)
    val_part = np.abs(frac - targets[1]) + discipline_weight * miss

    best, ties, n_feasible = np.inf, [], 0
    for t in np.flatnonzero(test_ok):
        ok = val_ok & ((masks & masks[t]) == 0) & (len(main) - sizes - sizes[t] >= 1)  # train keeps a video
        if not ok.any():
            continue
        n_feasible += int(ok.sum())
        scores = np.where(ok, val_part + abs(frac[t] - targets[2]) + discipline_weight * miss[t], np.inf)
        local = scores.min()
        if local < best - TIE_TOL:
            best, ties = local, []
        if local <= best + TIE_TOL:
            ties.extend((int(t), int(v)) for v in np.flatnonzero(scores <= best + TIE_TOL))
    if not ties:
        raise ValueError(
            f"no feasible split: {int(test_ok.sum())} of {len(subsets)} candidate test sets meet forced_test and "
            f"min_ski_points={min_ski_points}, {int(val_ok.sum())} candidate val sets meet min_ski_points, "
            f"and none pair up disjointly while leaving a training video"
        )
    t, v = ties[int(np.random.default_rng(seed).permutation(len(ties))[0])]
    test = sorted(cand[i].video_id for i in subsets[t])
    val = sorted(cand[i].video_id for i in subsets[v])
    train = sorted(s.video_id for s in main if s.video_id not in set(test) | set(val))
    partitions = {"train": train, "val": val, "test": test, **set_aside}
    return Split(partitions, float(best), n_feasible, len(ties), args)


def split_manifest(split: Split, records: Sequence[PoseRecord], schema: Schema | None = None) -> dict:
    """The frozen split file: videos and sample IDs per partition, counts, exclusions, hash."""
    schema = schema or load_schema()
    body, ski = list(schema.groups["body"]), list(schema.groups["ski"])
    part_of = {v: p for p, vids in split.partitions.items() for v in vids}
    unplaced = sorted({r.source_video_id for r in records if not r.excluded and r.source_video_id not in part_of})
    if unplaced:
        raise ValueError(f"videos with eligible records but no partition: {unplaced}")
    main_images = sum(1 for r in records if not r.excluded and part_of[r.source_video_id] in MAIN_PARTITIONS)

    partitions = {}
    for p, vids in split.partitions.items():
        recs = [r for r in records if not r.excluded and part_of[r.source_video_id] == p]
        partitions[p] = {
            "videos": vids,
            "sample_ids": sorted(r.sample_id for r in recs),
            "counts": {
                "images": len(recs),
                "segments": len({(r.source_video_id, r.segment_id) for r in recs}),
                "reliable_body_points": int(sum(r.reliable[body].sum() for r in recs)),
                "reliable_ski_points": int(sum(r.reliable[ski].sum() for r in recs)),
                "images_by_discipline": dict(sorted(Counter(r.discipline or "unknown" for r in recs).items())),
                "followcam_segments": len({(r.source_video_id, r.segment_id) for r in recs if r.discipline == FOLLOW_CAM}),
                "fraction_of_main": round(len(recs) / main_images, 4) if p in MAIN_PARTITIONS else None,
            },
        }
    content = {p: {"videos": d["videos"], "sample_ids": d["sample_ids"]} for p, d in partitions.items()}
    return {
        "version": 1,
        "dataset": "ski2dpose",
        "generator": {"function": "skier.splits.make_pose_split", **split.args},
        "score": round(split.score, 6),
        "n_feasible": split.n_feasible,
        "n_ties": split.n_ties,
        "partitions": partitions,
        "excluded": [
            {"sample_id": r.sample_id, "reason": r.excluded_reason, "video_partition": part_of.get(r.source_video_id)}
            for r in records if r.excluded
        ],
        "content_sha256": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
    }


def apply_split(records: Iterable[PoseRecord], manifest: dict) -> None:
    """Set ``record.split`` from a manifest (by video). Excluded records keep their video's partition."""
    part_of = {v: p for p, d in manifest["partitions"].items() for v in d["videos"]}
    for r in records:
        r.split = part_of.get(r.source_video_id)


def _parse_set_aside(values: list[str], stats: Sequence[VideoStats]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set-aside expects NAME=ID,ID or followcam=auto, got {item!r}")
        name, ids = item.split("=", 1)
        if ids == "auto":
            if name != "followcam":
                raise ValueError("'auto' is only defined for followcam")
            out[name] = [s.video_id for s in stats if s.followcam]
        else:
            out[name] = [v for v in ids.split(",") if v]
    return out


def summary_table(manifest: dict) -> str:
    lines = [f"{'partition':<10} {'videos':>6} {'images':>6} {'of main':>7} {'body pts':>8} {'ski pts':>8}  disciplines"]
    for p, d in manifest["partitions"].items():
        c = d["counts"]
        frac = f"{c['fraction_of_main']:.0%}" if c["fraction_of_main"] is not None else "-"
        disc = ", ".join(f"{k} {n}" for k, n in c["images_by_discipline"].items())
        lines.append(f"{p:<10} {len(d['videos']):>6} {c['images']:>6} {frac:>7} "
                     f"{c['reliable_body_points']:>8} {c['reliable_ski_points']:>8}  {disc}")
    lines.append(f"excluded records: {len(manifest['excluded'])}   score {manifest['score']}   "
                 f"ties {manifest['n_ties']}   sha256 {manifest['content_sha256'][:12]}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate the frozen, video-grouped pose split (plan Step 2.3).")
    p.add_argument("--labels", type=Path, help="default: $SKIER_DATA/ski2dpose/labels/ski2dpose_labels.json")
    p.add_argument("--images-root", type=Path, help="default: $SKIER_DATA/ski2dpose/images")
    p.add_argument("--info", type=Path, help="default: $SKIER_DATA/ski2dpose/archives/ski2dpose_info.json")
    p.add_argument("--pilot-file", type=Path, help="forced into train; default: data/manifests/pilot_videos.txt")
    p.add_argument("--forced-test", nargs="*", default=[], metavar="VIDEO_ID")
    p.add_argument("--set-aside", action="append", default=[], metavar="NAME=ID,ID",
                   help="whole videos into an extra partition; 'followcam=auto' takes every Follow Cam video")
    p.add_argument("--min-ski-points", type=int, default=400)
    p.add_argument("--seed", type=int, default=230)
    p.add_argument("--path-template", help="override IMAGE_PATH_TEMPLATE")
    p.add_argument("--out", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--force", action="store_true", help="overwrite an existing (frozen) manifest")
    return p


def main(argv: list[str] | None = None) -> int:
    from skier.data.ski2dpose import load_records

    args = build_parser().parse_args(argv)
    s2d = data_root() / "ski2dpose"
    out = args.out or repo_root() / "data" / "manifests" / "pose_split_v1.json"
    if out.exists() and not args.force:
        print(f"error: {out} exists and is frozen; pass --force only for a documented reason "
              f"(record it in docs/decisions.md)", file=sys.stderr)
        return 1
    pilot_file = args.pilot_file or repo_root() / "data" / "manifests" / "pilot_videos.txt"
    if not pilot_file.is_file():
        print(f"error: pilot file {pilot_file} not found; choose the pilot videos first", file=sys.stderr)
        return 1

    schema = load_schema()
    records = load_records(
        args.labels or s2d / "labels" / "ski2dpose_labels.json",
        args.images_root or s2d / "images",
        args.info or s2d / "archives" / "ski2dpose_info.json",
        schema=schema,
        path_template=args.path_template,
    )
    stats = video_stats_from_records(records, schema)
    try:
        split = make_pose_split(
            stats,
            seed=args.seed,
            forced_test=args.forced_test,
            forced_train=read_id_list(pilot_file),
            set_aside=_parse_set_aside(args.set_aside, stats),
            min_ski_points=args.min_ski_points,
        )
        manifest = split_manifest(split, records, schema)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(manifest, out)
    print(summary_table(manifest))
    print(f"wrote {out}")
    return 0
