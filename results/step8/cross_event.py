"""Cross-event breakdown of the frozen results (post hoc; no new evaluation of test).

All 4 training videos come from one event (Almaty 2017 Universiade), and one of the 3 test videos is from that event
too. This splits the frozen D27 test predictions into "same event as training" and "other events", and puts val (all
other events) and the follow-cam set (another camera setup) next to them. A video's event is its ``video_info.location``
in the Ski 2DPose info JSON; "same event" means a location shared with a training video.

Everything is rebuilt from the saved ``predictions.npz`` files with the report-table code (skier.report.tables), the
full-set counts are checked against the stored ones, and the group counts must add up to them. Intervals follow D25:
segments resampled within videos (follow-cam: whole videos, D27).

Usage (GPU machine, inside Ubuntu): python results/step8/cross_event.py [--out results/step8/cross_event]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.eval.bootstrap import N_BOOT, ratio_ci  # noqa: E402
from skier.io_utils import ensure_dir, read_json, write_json  # noqa: E402
from skier.paths import data_root, repo_root, results_root  # noqa: E402
from skier.report.tables import (SEEDS, align, check_counts, clusters_of, diff_block, hit_arrays,  # noqa: E402
                                 load_eval, pooled, propagation_arrays, stored_pose_counts)
from skier.schema import load_schema  # noqa: E402

POSE_ROWS = (("body", 0.1), ("ski", 0.1), ("ski", 0.05))
PROP_ROWS = (("body", 0.1), ("ski", 0.1))


def video_events(info: dict, manifest: dict) -> tuple[dict, set]:
    """``({video: location}, training locations)``; refuses a split video without a location."""
    part = {v: p for p, d in manifest["partitions"].items() for v in d["videos"]}
    loc = {}
    for v in part:
        location = ((info.get(v) or {}).get("video_info") or {}).get("location")
        if not location:
            raise ValueError(f"video {v}: no video_info.location in the info JSON")
        loc[v] = location
    return loc, {loc[v] for v, p in part.items() if p == "train"}


def cell(h, n, clusters, strata, n_boot) -> dict:
    ci = ratio_ci(h, n, clusters, strata, n_boot=n_boot)
    return {"hits": int(h.sum()), "total": int(n.sum()), "pct": ci["estimate"], "ci95": ci["ci"],
            "n_clusters": ci["n_clusters"]}


def pose_arrays(pose_dir: Path, by_id: dict, schema) -> tuple[list, dict]:
    """Seed-pooled per-sample hit arrays for E1 and E2 (reference crops), checked against the stored counts."""
    per, order = {}, None
    for r in ("E1", "E2"):
        seeds = []
        for s in SEEDS:
            m, z = load_eval(pose_dir / f"{r}-s{s}")
            ids = [str(i) for i in z["sample_ids"]]
            order = order or ids
            if ids != order:
                raise ValueError(f"{r}-s{s}: sample order differs")
            a = hit_arrays(z["clean"], align(by_id, ids), schema, ("body", "ski"))
            check_counts(a, {k: v for k, v in stored_pose_counts(m).items() if k[0] in ("body", "ski")}, f"{r}-s{s}")
            seeds.append(a)
        per[r] = pooled(seeds)
    return align(by_id, order), per


def grouped(recs, arrays: dict, rows, groups: dict, n_boot: int, pairs=(), by_video: bool = False) -> dict:
    """Per group: each run's cells and the paired Δ a − b for each ``(a, b)`` in ``pairs``.

    Groups whose name ends in ", all" are the union of the others and are left out of the add-up check."""
    segments, videos = clusters_of(recs)
    clusters, strata = (videos, None) if by_video else (segments, videos)
    clusters, strata = np.asarray(clusters), None if strata is None else np.asarray(strata)
    vids = np.asarray(videos)
    out = {}
    for gname, members in groups.items():
        mask = np.isin(vids, list(members))
        if not mask.any():
            continue
        c, s = clusters[mask], None if strata is None else strata[mask]
        g = {"videos": sorted(set(vids[mask])), "n_samples": int(mask.sum()), "runs": {}, "deltas": []}
        for run, a in arrays.items():
            g["runs"][run] = {f"{grp}@{t:g}": cell(a[(grp, t)][0][mask], a[(grp, t)][1][mask], c, s, n_boot)
                              for grp, t in rows}
        for ra, rb in pairs:
            for grp, t in rows:
                g["deltas"].append({"a": ra, "b": rb, "group": f"{grp}@{t:g}",
                                    **diff_block(tuple(x[mask] for x in arrays[ra][(grp, t)]),
                                                 tuple(x[mask] for x in arrays[rb][(grp, t)]), c, s, n_boot)})
        out[gname] = g
    # the groups (other than the ", all" union) must partition the full set
    parts = [k for k in out if not k.endswith(", all")]
    for run, a in arrays.items():
        for grp, t in rows:
            tot = sum(out[k]["runs"][run][f"{grp}@{t:g}"]["total"] for k in parts)
            if tot != int(a[(grp, t)][1].sum()):
                raise ValueError(f"groups don't add up: {run} {grp}@{t:g} {tot} vs {int(a[(grp, t)][1].sum())}")
    return out


def to_markdown(res: dict) -> str:
    L = ["# Cross-event breakdown (post hoc; frozen predictions, no new evaluation)", "",
         f"Training event(s): {', '.join(res['training_locations'])}. Event = the info JSON's video location.",
         "PCK (%) seed-pooled, 95% interval (D25: segments within videos; follow-cam: videos).", ""]

    def fmt(c):
        return f"{c['pct']:.1f} [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}] ({c['hits']}/{c['total']})"

    for section, title in (("pose", "Pose model, reference crops"),
                           ("propagation_siamfc", "System, SiamFC tracks (main result), test"),
                           ("propagation_stark", "System, STARK tracks (D28 extra), test")):
        if section not in res:
            continue
        L += [f"## {title}", ""]
        for gname, g in res[section].items():
            L += [f"**{gname}** ({', '.join(g['videos'])}; {g['n_samples']} samples)", "",
                  "| Run | " + " | ".join(next(iter(g["runs"].values())).keys()) + " |",
                  "| --- |" + " --- |" * len(next(iter(g["runs"].values())))]
            for run, cells in g["runs"].items():
                L.append(f"| {run} | " + " | ".join(fmt(c) for c in cells.values()) + " |")
            for d in g["deltas"]:
                L.append(f"| Δ {d['a']} − {d['b']} {d['group']} | {d['delta']:+.1f} "
                         f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] | | |")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-pose", type=float, default=1.1)
    p.add_argument("--out", type=Path, default=repo_root() / "results/step8/cross_event")
    a = p.parse_args(argv)
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    schema = load_schema()
    s2d = data_root() / "ski2dpose"
    info = read_json(s2d / "archives/ski2dpose_info.json")
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    manifest = read_json(repo_root() / "data/manifests/pose_split_v1.json")
    loc, train_locs = video_events(info, manifest)
    R = results_root()
    res = {"note": "post-hoc breakdown of frozen predictions (D27 test, val, follow-cam); no new evaluation",
           "split_sha256": manifest["content_sha256"], "training_locations": sorted(train_locs),
           "video_location": {v: loc[v] for v in sorted(loc)}, "pose": {}}
    by_part = {part: {r.sample_id: r for r in select_partition(records, manifest, part)[0]}
               for part in ("test", "val", "followcam")}
    test_vids = manifest["partitions"]["test"]["videos"]
    same = {v for v in test_vids if loc[v] in train_locs}
    test_groups = {"test, same event as training": same, "test, other events": set(test_vids) - same,
                   "test, all": set(test_vids)}
    e21 = (("E2", "E1"),)
    recs, arr = pose_arrays(R / "final/test/pose", by_part["test"], schema)
    res["pose"].update(grouped(recs, arr, POSE_ROWS, test_groups, a.n_boot, e21))
    recs, arr = pose_arrays(R / "final/val/pose", by_part["val"], schema)
    if any(loc[v] in train_locs for v in manifest["partitions"]["val"]["videos"]):
        raise ValueError("a val video shares the training event; the 'val, other events' label would be wrong")
    res["pose"].update(grouped(recs, arr, POSE_ROWS, {"val (all other events)": set(manifest["partitions"]["val"]["videos"])},
                               a.n_boot, e21))
    recs, arr = pose_arrays(R / "final/followcam/pose", by_part["followcam"], schema)
    res["pose"].update(grouped(recs, arr, POSE_ROWS,
                               {"follow-cam (other camera setup)": set(manifest["partitions"]["followcam"]["videos"])},
                               a.n_boot, e21, by_video=True))
    for name, root in (("propagation_siamfc", R / "final/test/propagation"),
                       ("propagation_stark", R / "final_stark/test/propagation")):
        if not (root / "propagation.json").is_file():
            continue
        pa = propagation_arrays(root, by_part["test"], schema, a.k_pose, ("E2",))
        arrays = {f"E2 {crop}": pa["arrays"][("E2", crop)] for crop in pa["crops"]}
        pairs = [("E2 reference", "E2 tracked")] + ([("E2 refined", "E2 tracked")] if "refined" in pa["crops"] else [])
        res[name] = grouped(pa["recs"], arrays, PROP_ROWS, test_groups, a.n_boot, pairs)
    out = ensure_dir(a.out)
    write_json(res, out / "cross_event.json")
    (out / "cross_event.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
