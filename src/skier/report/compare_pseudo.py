"""The pseudo-label student PL vs. E2 (Step 6b, D29), paired, seeds pooled, D25 intervals, per event group.

Pose: reference crops (``eval_pose`` outputs), clean and, on val, the moderate crop-error level. System: the
propagation run's reference, tracker and refined crops on the frozen SiamFC tracks (``eval_propagation``
outputs). Test splits into same event as training / other events (pre-registered in D29; the key number is
other-event ski); follow-cam uses video intervals. Every count is checked against the stored one, and both
models must have been scored on the same samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from skier.eval.bootstrap import N_BOOT
from skier.io_utils import ensure_dir, read_json, write_json
from skier.paths import data_root, repo_root, results_root
from skier.report.events import group_markdown, grouped, partition_groups, seed_arrays, video_events
from skier.report.tables import propagation_arrays
from skier.schema import load_schema

POSE_ROWS = (("body", 0.1), ("ski", 0.1), ("ski", 0.05))
PROP_ROWS = (("body", 0.1), ("ski", 0.1))


def pose_compare(e2_dirs: Sequence[Path], pl_dirs: Sequence[Path], by_id: dict, schema, groups: dict, by_video: bool,
                 key: str, n_boot: int) -> dict:
    recs, a = seed_arrays(e2_dirs, by_id, schema, key)
    recs_b, b = seed_arrays(pl_dirs, by_id, schema, key)
    if [r.sample_id for r in recs] != [r.sample_id for r in recs_b]:
        raise ValueError("E2 and PL were scored on different samples")
    return grouped(recs, {"E2": a, "PL": b}, POSE_ROWS, groups, n_boot, pairs=(("PL", "E2"),), by_video=by_video)


def propagation_compare(e2_dir: Path, pl_dir: Path, by_id: dict, schema, k_pose: float, groups: dict,
                        n_boot: int) -> dict:
    pa = propagation_arrays(e2_dir, by_id, schema, k_pose, ("E2",))
    pb = propagation_arrays(pl_dir, by_id, schema, k_pose, ("PL",))
    if [str(s) for s in pa["npz"]["sample_ids"]] != [str(s) for s in pb["npz"]["sample_ids"]]:
        raise ValueError("E2 and PL propagation runs scored different labels")
    if pa["rkey"] != pb["rkey"]:
        raise ValueError(f"different refinement settings: {pa['rkey']} vs {pb['rkey']}")
    crops = [c for c in pa["crops"] if c in pb["crops"]]
    arrays = {}
    for c in crops:
        arrays[f"E2 {c}"], arrays[f"PL {c}"] = pa["arrays"][("E2", c)], pb["arrays"][("PL", c)]
    return grouped(pa["recs"], arrays, PROP_ROWS, groups, n_boot, pairs=[(f"PL {c}", f"E2 {c}") for c in crops])


def to_markdown(res: dict) -> str:
    L = [f"# PL (pseudo-label student, D29) vs. E2: {res['partition']} (post-freeze extra)", "",
         "PCK (%) seed-pooled, paired 95% intervals (D25: segments within videos; follow-cam: videos).", ""]
    for section, g_all in res["sections"].items():
        L += [f"## {section}", ""]
        for gname, g in g_all.items():
            L += group_markdown(gname, g)
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PL (pseudo-label student, D29) vs. E2, paired, per event group.")
    p.add_argument("--partition", required=True, choices=("val", "test", "followcam"))
    p.add_argument("--e2", type=Path, action="append", required=True, help="E2 eval_pose folder; once per seed")
    p.add_argument("--pl", type=Path, action="append", required=True, help="PL eval_pose folder; once per seed")
    p.add_argument("--levels", nargs="+", default=["clean"], help="clean and/or crop-error levels (val: moderate)")
    p.add_argument("--prop-e2", type=Path, help="E2's eval_propagation folder (frozen run)")
    p.add_argument("--prop-pl", type=Path, help="PL's eval_propagation folder")
    p.add_argument("--k-pose", type=float, default=1.1)
    p.add_argument("--split", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pseudo/eval/<partition>/compare")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.e2) != len(args.pl):
        print("error: give one --e2 and one --pl folder per seed", file=sys.stderr)
        return 1
    if (args.prop_e2 is None) != (args.prop_pl is None):
        print("error: give both --prop-e2 and --prop-pl, or neither", file=sys.stderr)
        return 1
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    schema = load_schema()
    s2d = data_root() / "ski2dpose"
    manifest = read_json(args.split or repo_root() / "data/manifests/pose_split_v1.json")
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    by_id = {r.sample_id: r for r in select_partition(records, manifest, args.partition)[0]}
    loc, train_locs = video_events(read_json(s2d / "archives/ski2dpose_info.json"), manifest)
    try:
        groups, by_video = partition_groups(args.partition, manifest, loc, train_locs)
        sections = {f"Pose model, reference crops ({lv})": pose_compare(args.e2, args.pl, by_id, schema, groups,
                                                                       by_video, lv, args.n_boot)
                    for lv in args.levels}
        if args.prop_e2 is not None:
            sections["System, SiamFC tracks (k_pose, refined crop as frozen)"] = propagation_compare(
                args.prop_e2, args.prop_pl, by_id, schema, args.k_pose, groups, args.n_boot)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    res = {"partition": args.partition, "note": "D29 post-freeze extra; D27 stays the main result",
           "split_sha256": manifest["content_sha256"], "training_locations": sorted(train_locs),
           "e2": [str(p) for p in args.e2], "pl": [str(p) for p in args.pl],
           "prop_e2": str(args.prop_e2) if args.prop_e2 else None, "prop_pl": str(args.prop_pl) if args.prop_pl else None,
           "sections": sections}
    out = ensure_dir(args.out or results_root() / "pseudo" / "eval" / args.partition / "compare")
    write_json(res, out / "compare.json")
    (out / "compare.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print("wrote", out)
    return 0
