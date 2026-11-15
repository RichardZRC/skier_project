"""Paired tracker comparison (Step 7b, D28): run B − run A on the same clips and labels, with D25 intervals.

A and B are two evaluation roots of one partition (``$SKIER_RESULTS/final/<part>`` for SiamFC,
``$SKIER_RESULTS/final_stark/<part>`` for STARK). Every count is rebuilt with the code that ``make_tables`` uses
(``tracking_arrays``, ``propagation_arrays``), and the two roots must cover the same SkiTB clips (same frames) and the
same scored Ski 2DPose labels. The pose model and its reference crops don't depend on the tracker, so reference-crop
hits must agree exactly; that is checked, not assumed. Intervals are paired:
- SkiTB tracking and SkiTB replay: sequences resampled;
- propagation and Ski 2DPose replay: segments resampled within videos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from skier.eval.bootstrap import N_BOOT, cluster_bootstrap, ratio_diff_ci
from skier.io_utils import ensure_dir, read_csv, read_json, write_json
from skier.paths import configs_dir, data_root, repo_root
from skier.report.tables import (THRS, _ci, _f, _num, _per_video, check_root_tracker, diff_block, pct,
                                 propagation_arrays, tracking_arrays)
from skier.schema import Schema, load_schema

PROP_ROWS = (("body", 0.1), ("ski", 0.1), ("ski", 0.05))
IOU_BINS = ("iou>=0.5", "0.1<=iou<0.5", "iou<0.1")


def _paired(est_a: float, est_b: float, ci: dict) -> dict:
    return {"a": est_a, "b": est_b, "delta": ci["estimate"], "ci95": ci["ci"], "n_clusters": ci["n_clusters"],
            "n_boot": ci["n_boot"]}


def compare_tracking(dir_a: Path, dir_b: Path, skitb_root: Path, split_json: Path, partition: str, fps_table: Path,
                     n_boot: int = N_BOOT) -> dict:
    """Pooled mean IoU, success AUC and toolkit F-score, B − A over the same SkiTB clips (sequence bootstrap)."""
    from skier.eval.track_metrics import SUCCESS_THRESHOLDS

    a = tracking_arrays(dir_a, skitb_root, split_json, partition, fps_table)
    b = tracking_arrays(dir_b, skitb_root, split_json, partition, fps_table)
    if a["names"] != b["names"] or not np.array_equal(a["n"], b["n"]):
        raise ValueError(f"tracking: {dir_a} and {dir_b} don't cover the same clips and frames")
    seqs, n = a["seqs"], a["n"]

    def iou(x, i):
        return x["iou_sum"][i].sum() / n[i].sum()

    def auc(x, i):
        return (x["succ"][i].sum(axis=0) / n[i].sum()).mean()

    def f(x, i):
        return x["f_curves"][i].mean(axis=0).max()

    every = np.arange(len(n))
    rows = []
    for metric, fn in (("pooled_mean_iou", iou), ("auc", auc), ("official_f_score", f)):
        ci = cluster_bootstrap(lambda i, fn=fn: fn(b, i) - fn(a, i), seqs, n_boot=n_boot)
        rows.append({"metric": metric, **_paired(float(fn(a, every)), float(fn(b, every)), ci)})
    failures = {"episodes": {k: sum(c["n_failures"] for c in x["per_clip"]) for k, x in (("a", a), ("b", b))},
                "clips_with_failures": {k: sum(c["n_failures"] > 0 for c in x["per_clip"]) for k, x in (("a", a), ("b", b))}}
    return {"n_clips": len(n), "n_sequences": len(set(seqs)), "n_frames": int(n.sum()), "unit": "SkiTB sequence (D25)",
            "rows": rows, "failures": failures,
            "success_curves": {"thresholds": [float(t) for t in SUCCESS_THRESHOLDS],
                               "a": (a["succ"].sum(axis=0) / n.sum()).tolist(),
                               "b": (b["succ"].sum(axis=0) / n.sum()).tolist()}}


def compare_propagation(dir_a: Path, dir_b: Path, records_by_id: dict, schema: Schema, k_a: float, k_b: float,
                        n_boot: int = N_BOOT, recipes=("E1", "E2")) -> dict:
    """Pose PCK under each tracker's crops, B − A on the same labels (segments within videos)."""
    from skier.integration.propagation import iou_bin

    a = propagation_arrays(dir_a, records_by_id, schema, k_a, recipes)
    b = propagation_arrays(dir_b, records_by_id, schema, k_b, recipes)
    ids_a, ids_b = ([str(s) for s in x["npz"]["sample_ids"]] for x in (a, b))
    if ids_a != ids_b:
        raise ValueError(f"propagation: {dir_a} and {dir_b} scored different labels")
    for r in recipes:  # the reference crop doesn't depend on the tracker
        ra, rb = a["arrays"][(r, "reference")], b["arrays"][(r, "reference")]
        if any(not (np.array_equal(ra[key][0], rb[key][0]) and np.array_equal(ra[key][1], rb[key][1])) for key in ra):
            raise ValueError(f"propagation: {r} reference-crop hits differ between {dir_a} and {dir_b}")
    clusters, strata = a["clusters"], a["strata"]
    crops = [c for c in ("tracked", "refined") if c in a["crops"] and c in b["crops"]]
    rows, per_video = [], []
    for r in recipes:
        for crop in crops:
            for g, t in PROP_ROWS:
                ha, hb = a["arrays"][(r, crop)][(g, t)], b["arrays"][(r, crop)][(g, t)]
                rows.append({"run": r, "crop": crop, "group": g, "threshold": t, "a": pct(*ha), "b": pct(*hb),
                             **diff_block(hb, ha, clusters, strata, n_boot)})
                if t == 0.1:
                    per_video.append({"run": r, "crop": crop, "group": g, "a": _per_video(*ha, strata),
                                      "b": _per_video(*hb, strata)})
    bins = {k: {v: int(sum(iou_bin(x) == v for x in m["npz"]["tracker_iou"])) for v in IOU_BINS}
            for k, m in (("a", a), ("b", b))}
    reference = {r: {g: pct(*a["arrays"][(r, "reference")][(g, 0.1)]) for g in ("body", "ski")} for r in recipes}
    return {"n_scored": len(ids_a), "n_segments": len(set(clusters)), "n_videos": len(set(strata)),
            "unit": "segment within video (D25)", "reference_identical": True, "reference": reference,
            "refine": {"a": a["rkey"], "b": b["rkey"]}, "rows": rows, "per_video": per_video, "tracker_iou_bins": bins}


def compare_replay(dir_a: Path, dir_b: Path, n_boot: int = N_BOOT) -> dict:
    """Smoothed replay, B − A: SkiTB GT-box containment and crop area (sequences), Ski 2DPose ski-end and body+ski
    landmark containment (segments within videos)."""
    ra, rb = (read_csv(d / "smoothed.csv") for d in (dir_a, dir_b))
    out: dict = {}
    sk_a, sk_b = ({r["clip"]: r for r in rows if r["source"] == "skitb"} for rows in (ra, rb))
    if sk_a or sk_b:
        if sorted(sk_a) != sorted(sk_b):
            raise ValueError("replay: the two runs cover different SkiTB clips")
        names = sorted(sk_a)
        seqs = np.array([sk_a[c]["sequence"] for c in names])
        rows = []
        for metric in ("box_containment_mean", "frac_fully_inside", "area_fraction_mean", "acceleration_mean"):
            va = np.array([_num(sk_a[c][metric]) for c in names])
            vb = np.array([_num(sk_b[c][metric]) for c in names])
            ok = np.isfinite(va) & np.isfinite(vb)
            ones = np.ones(ok.sum())
            ci = ratio_diff_ci(vb[ok], ones, va[ok], ones, seqs[ok], scale=1.0, n_boot=n_boot)
            rows.append({"metric": metric, "n_clips": int(ok.sum()),
                         **_paired(float(va[ok].mean()), float(vb[ok].mean()), ci)})
        out["skitb"] = {"n_clips": len(names), "unit": "SkiTB sequence (D25); mean over clips", "rows": rows}
    sp_a, sp_b = ({r["sequence"]: r for r in rows if r["source"] == "ski2dpose"} for rows in (ra, rb))
    if sp_a or sp_b:
        if sorted(sp_a) != sorted(sp_b):
            raise ValueError("replay: the two runs cover different Ski 2DPose segments")
        segs = sorted(sp_a)
        vids = np.array([s.split("/")[0] for s in segs])
        rows = []
        for group, pre in (("ski_endpoints", "ski_end"), ("body_ski", "body_ski")):
            ins_a, ins_b, tot_a, tot_b = (np.array([_num(x[s][f"{pre}_{k}"]) for s in segs])
                                          for x, k in ((sp_a, "inside"), (sp_b, "inside"), (sp_a, "total"), (sp_b, "total")))
            if not np.array_equal(tot_a, tot_b, equal_nan=True):
                raise ValueError(f"replay: {group} label totals differ between the runs")
            keep = np.isfinite(tot_a) & (tot_a > 0)
            ci = ratio_diff_ci(ins_b[keep], tot_b[keep], ins_a[keep], tot_a[keep], np.array(segs)[keep], vids[keep],
                               n_boot=n_boot)
            rows.append({"group": group, "total": int(tot_a[keep].sum()),
                         **_paired(pct(ins_a[keep], tot_a[keep]), pct(ins_b[keep], tot_b[keep]), ci)})
        out["ski2dpose"] = {"n_segments": len(segs), "unit": "segment within video (D25)", "rows": rows}
    return out


def to_markdown(c: dict) -> str:
    a, b = c["a"]["tracker"], c["b"]["tracker"]
    out = [f"# Tracker comparison: {c['partition']} ({b} − {a}; D28 post-freeze extra)", "",
           f"A = {a} (`{c['a']['root']}`), B = {b} (`{c['b']['root']}`). Paired 95% intervals (D25), "
           f"{c['n_boot']:,} resamples, seed 230.", ""]
    if "tracking" in c:
        t = c["tracking"]
        out += [f"## SkiTB tracking ({t['n_clips']} clips, {t['n_sequences']} sequences, {t['n_frames']:,} frames)", "",
                f"| Metric | {a} | {b} | Δ | 95% interval |", "| --- | --- | --- | --- | --- |"]
        for r in t["rows"]:
            out.append(f"| {r['metric']} | {r['a']:.3f} | {r['b']:.3f} | {r['delta']:+.3f} | {_ci(r['ci95'], 3)} |")
        fe = t["failures"]
        out += ["", f"Failure episodes (IoU < 0.1 for ≥ 0.5 s): {a} {fe['episodes']['a']} in "
                    f"{fe['clips_with_failures']['a']} clips; {b} {fe['episodes']['b']} in "
                    f"{fe['clips_with_failures']['b']} clips.", ""]
    if "propagation" in c:
        p = c["propagation"]
        out += [f"## Pose under tracker crops ({p['n_scored']} labels, {p['n_segments']} segments, "
                f"{p['n_videos']} videos; seeds pooled)", "",
                "Reference crops are identical for both trackers: " +
                ", ".join(f"{r} body {_f(v['body'])} / ski {_f(v['ski'])}" for r, v in p["reference"].items()) + ".",
                "", f"| Run | Crop | Group | {a} | {b} | Δ (points) | 95% interval |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in p["rows"]:
            out.append(f"| {r['run']} | {r['crop']} | {r['group']} @{r['threshold']:g} | {_f(r['a'])}% | {_f(r['b'])}% | "
                       f"{r['delta']:+.1f} | {_ci(r['ci95'])} |")
        bins = p["tracker_iou_bins"]
        out += ["", "Labels by tracker IoU vs. the reference box: " +
                "; ".join(f"{k} {bins['a'][k]} → {bins['b'][k]}" for k in IOU_BINS) + f" ({a} → {b}).", ""]
    if "replay" in c:
        r = c["replay"]
        out += ["## Replay (smoothed)", ""]
        if "skitb" in r:
            out += [f"SkiTB, {r['skitb']['n_clips']} clips (mean over clips):", "",
                    f"| Metric | {a} | {b} | Δ | 95% interval |", "| --- | --- | --- | --- | --- |"]
            for x in r["skitb"]["rows"]:
                out.append(f"| {x['metric']} | {x['a']:.3f} | {x['b']:.3f} | {x['delta']:+.3f} | {_ci(x['ci95'], 3)} |")
            out.append("")
        if "ski2dpose" in r:
            out += [f"Ski 2DPose, {r['ski2dpose']['n_segments']} segments (landmarks inside the replay crop):", "",
                    f"| Landmarks | Labels | {a} | {b} | Δ (points) | 95% interval |", "| --- | --- | --- | --- | --- | --- |"]
            for x in r["ski2dpose"]["rows"]:
                out.append(f"| {x['group']} | {x['total']} | {_f(x['a'])}% | {_f(x['b'])}% | {x['delta']:+.1f} | "
                           f"{_ci(x['ci95'])} |")
            out.append("")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paired tracker comparison (Step 7b, D28): root B − root A on one "
                                            "partition, with D25 intervals.")
    p.add_argument("--a", type=Path, required=True, help="baseline root, e.g. $SKIER_RESULTS/final/val (SiamFC)")
    p.add_argument("--b", type=Path, required=True, help="compared root, e.g. $SKIER_RESULTS/final_stark/val")
    p.add_argument("--config-a", type=Path, help="A's integration config; default: configs/integration/default.yaml")
    p.add_argument("--config-b", type=Path, required=True, help="B's integration config, e.g. "
                                                                "configs/integration/stark.yaml")
    p.add_argument("--partition", required=True)
    p.add_argument("--split", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--skitb-root", type=Path)
    p.add_argument("--skitb-split", type=Path)
    p.add_argument("--fps-table", type=Path, help="default: data/manifests/skitb_fps_v1.csv")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", type=Path, help="default: <b>/compare")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from skier.integration.config import IntegrationConfig

    try:
        ca = IntegrationConfig.from_yaml(args.config_a or configs_dir() / "integration" / "default.yaml")
        cb = IntegrationConfig.from_yaml(args.config_b)
        for root, cfg in ((args.a, ca), (args.b, cb)):
            check_root_tracker(root, cfg.tracker)
        c: dict = {"partition": args.partition, "a": {"root": str(args.a), "tracker": ca.tracker},
                   "b": {"root": str(args.b), "tracker": cb.tracker}, "delta": "b − a", "n_boot": args.n_boot}
        if args.skitb_root and args.skitb_split:
            from skier.data.skitb import FPS_TABLE

            c["tracking"] = compare_tracking(args.a / "tracking" / "skitb", args.b / "tracking" / "skitb",
                                             args.skitb_root, args.skitb_split, args.partition,
                                             args.fps_table or repo_root() / FPS_TABLE, args.n_boot)
        if all((r / "propagation" / "propagation.json").is_file() for r in (args.a, args.b)):
            from skier.data.ski2dpose import load_records
            from skier.pose.eval_generic import select_partition

            schema = load_schema()
            s2d = data_root() / "ski2dpose"
            records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                                   s2d / "archives/ski2dpose_info.json", schema=schema)
            manifest = read_json(args.split or repo_root() / "data/manifests/pose_split_v1.json")
            by_id = {r.sample_id: r for r in select_partition(records, manifest, args.partition)[0]}
            c["propagation"] = compare_propagation(args.a / "propagation", args.b / "propagation", by_id, schema,
                                                   ca.require("k_pose").k_pose, cb.require("k_pose").k_pose,
                                                   args.n_boot)
        if all((r / "replay" / "smoothed.csv").is_file() for r in (args.a, args.b)):
            c["replay"] = compare_replay(args.a / "replay", args.b / "replay", args.n_boot)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = ensure_dir(args.out or args.b / "compare")
    write_json(c, out / "compare.json")
    (out / "compare.md").write_text(to_markdown(c), encoding="utf-8")
    print(f"compare ({', '.join(k for k in ('tracking', 'propagation', 'replay') if k in c)}) → {out}")
    return 0
