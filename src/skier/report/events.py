"""Event groups: same event as training vs. other events (the cross-event breakdown, and D29).

All 4 training videos come from one event (Almaty 2017). A video's event is its ``video_info.location`` in the
Ski 2DPose info JSON; "same event" means a location shared with a training video. ``grouped`` gives each run's
cells and paired differences per group, with D25 intervals (segments within videos; follow-cam: videos).
Used by ``results/step8/cross_event.py`` (post hoc, frozen predictions) and ``compare_pseudo`` (D29,
pre-registered).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from skier.eval.bootstrap import ratio_ci
from skier.report.tables import THRS, align, check_counts, clusters_of, diff_block, hit_arrays, load_eval, pooled

SAME, OTHER, ALL = "test, same event as training", "test, other events", "test, all"
VAL = "val (all other events)"
FOLLOWCAM = "follow-cam (other camera setup)"


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


def partition_groups(partition: str, manifest: dict, loc: dict, train_locs: set) -> tuple[dict, bool]:
    """``(groups, by_video)`` for one partition: test splits into same event / other events (+ their union);
    val must be entirely other events; follow-cam is one group with video intervals (D27)."""
    vids = manifest["partitions"][partition]["videos"]
    if partition == "test":
        same = {v for v in vids if loc[v] in train_locs}
        return {SAME: same, OTHER: set(vids) - same, ALL: set(vids)}, False
    if partition == "val":
        if any(loc[v] in train_locs for v in vids):
            raise ValueError("a val video shares the training event; the 'val, other events' label would be wrong")
        return {VAL: set(vids)}, False
    if partition == "followcam":
        return {FOLLOWCAM: set(vids)}, True
    raise ValueError(f"no event groups for partition {partition!r}")


def cell(h, n, clusters, strata, n_boot) -> dict:
    ci = ratio_ci(h, n, clusters, strata, n_boot=n_boot)
    return {"hits": int(h.sum()), "total": int(n.sum()), "pct": ci["estimate"], "ci95": ci["ci"],
            "n_clusters": ci["n_clusters"]}


def seed_arrays(run_dirs: Sequence[Path], by_id: dict, schema, key: str = "clean",
                groups: Sequence[str] = ("body", "ski")) -> tuple[list, dict]:
    """``(records, seed-pooled per-sample hit arrays)`` from eval runs (``predictions.npz`` + ``metrics.json``);
    ``key`` is ``clean`` or a crop-error level. Counts are checked against the stored ones."""
    seeds, order = [], None
    for d in run_dirs:
        m, z = load_eval(Path(d))
        ids = [str(i) for i in z["sample_ids"]]
        order = order or ids
        if ids != order:
            raise ValueError(f"{d}: sample order differs from the other runs")
        a = hit_arrays(z[key], align(by_id, ids), schema, tuple(groups))
        stored = (m["clean"] if key == "clean" else m["crop_error"]["levels"][key])["groups"]
        check_counts(a, {(g, t): stored[f"{t:g}"][g] for g in groups for t in THRS}, f"{d} {key}")
        seeds.append(a)
    return align(by_id, order), pooled(seeds)


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


def fmt_cell(c: dict) -> str:
    return f"{c['pct']:.1f} [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}] ({c['hits']}/{c['total']})"


def group_markdown(gname: str, g: dict) -> list[str]:
    cols = list(next(iter(g["runs"].values())).keys())
    L = [f"**{gname}** ({', '.join(g['videos'])}; {g['n_samples']} samples)", "",
         "| Run | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    for run, cells in g["runs"].items():
        L.append(f"| {run} | " + " | ".join(fmt_cell(c) for c in cells.values()) + " |")
    for d in g["deltas"]:
        L.append(f"| Δ {d['a']} − {d['b']} {d['group']} | {d['delta']:+.1f} "
                 f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] | | |")
    return L + [""]
