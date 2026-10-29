"""Report tables (plan Step 8): pose, augmentation, tracking, replay and propagation.

Everything is rebuilt from saved outputs (``predictions.npz``, track CSVs, replay CSVs)
plus the labels. Each rebuilt count is checked against the count the tool stored when it
ran; a mismatch raises. Intervals follow D25:
- pose, augmentation and propagation: segments resampled within videos;
- SkiTB tracking and replay: sequences resampled.
E1 and E2 are pooled over their seeds (D21); per-seed values sit next to the pooled ones.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from skier.eval.bootstrap import N_BOOT, ratio_ci, ratio_diff_ci, cluster_bootstrap
from skier.eval.pose_metrics import gt_arrays, sample_hits
from skier.io_utils import ensure_dir, read_csv, read_json, write_csv, write_json
from skier.paths import configs_dir, data_root, repo_root, results_root
from skier.pose.coco_map import shared_joints
from skier.schema import Schema, load_schema

THRS = (0.1, 0.05)
SEEDS = (230, 231)
POSE_GROUPS = ("body", "ski", "pole", "body_shared")


def joint_sets(schema: Schema) -> dict[str, list[int]]:
    return {**{g: list(j) for g, j in schema.groups.items()}, "body_shared": shared_joints(schema)}


def hit_arrays(pred_xy, records, schema: Schema, groups=POSE_GROUPS) -> dict:
    """``{(group, thr): (hits[N], totals[N])}`` per sample."""
    gt, rel, diag, _ = gt_arrays(records)
    js = joint_sets(schema)
    return {(g, t): sample_hits(pred_xy, gt, rel, diag, js[g], t) for g in groups for t in THRS}


def check_counts(arrays: dict, stored: dict, what: str) -> None:
    """``stored``: ``{(group, thr): {"hits", "total"}}`` as the tool wrote them."""
    for key, ref in stored.items():
        h, n = arrays[key]
        if (int(h.sum()), int(n.sum())) != (ref["hits"], ref["total"]):
            raise ValueError(f"{what} {key}: rebuilt {int(h.sum())}/{int(n.sum())}, stored {ref['hits']}/{ref['total']}")


def stored_pose_counts(metrics: dict, e0: bool = False) -> dict:
    if e0:
        return {("body_shared", t): metrics["body_shared"][f"pck@{t:g}"] for t in THRS}
    out = {(g, t): metrics["clean"]["groups"][f"{t:g}"][g] for g in ("body", "ski", "pole") for t in THRS}
    out.update({("body_shared", t): metrics["clean"]["body_shared"][f"{t:g}"] for t in THRS})
    return out


def pct(h, n) -> float | None:
    h, n = float(np.sum(h)), float(np.sum(n))
    return 100.0 * h / n if n else None


def pooled(arrays_by_seed: Sequence[dict]) -> dict:
    """Sum per-sample hits and totals over seeds (D21)."""
    return {k: (sum(a[k][0] for a in arrays_by_seed), sum(a[k][1] for a in arrays_by_seed)) for k in arrays_by_seed[0]}


def clusters_of(records) -> tuple[list[str], list[str]]:
    """D25: (segment cluster, video stratum) per record."""
    return [f"{r.source_video_id}/{r.segment_id}" for r in records], [r.source_video_id for r in records]


def _cell(h, n) -> dict:
    return {"hits": int(np.sum(h)), "total": int(np.sum(n)), "pct": pct(h, n)}


def _per_video(h, n, videos) -> dict:
    v = np.asarray(videos)
    return {vid: _cell(h[v == vid], n[v == vid]) for vid in sorted(set(videos))}


def diff_block(a: tuple, b: tuple, clusters, strata, n_boot: int) -> dict:
    """Paired Δ = a − b (percentage points) with its D25 interval."""
    ci = ratio_diff_ci(a[0], a[1], b[0], b[1], clusters, strata, n_boot=n_boot)
    return {"delta": ci["estimate"], "ci95": ci["ci"], "n_clusters": ci["n_clusters"], "n_boot": ci["n_boot"]}


def load_eval(run_dir: Path) -> tuple[dict, dict]:
    with np.load(run_dir / "predictions.npz") as z:
        return read_json(run_dir / "metrics.json"), {k: z[k] for k in z.files}


def align(records_by_id: dict, sample_ids) -> list:
    missing = [s for s in sample_ids if s not in records_by_id]
    if missing:
        raise ValueError(f"{len(missing)} predicted samples are not in the partition (e.g. {missing[0]})")
    return [records_by_id[str(s)] for s in sample_ids]


# --- pose ------------------------------------------------------------------------------------

def pose_tables(pose_dir: Path, records_by_id: dict, schema: Schema, n_boot: int = N_BOOT,
                recipes=("E1", "E2"), key: str = "clean") -> dict:
    """E0 (shared joints), then each recipe pooled over its seeds; Δ E2 − E1 and recipe − E0 with intervals."""
    arrays, order, per_seed, errors = {}, None, {}, {}
    for name in ["E0"] + [f"{r}-s{s}" for r in recipes for s in SEEDS]:
        if not (pose_dir / name).is_dir():
            continue
        m, z = load_eval(pose_dir / name)
        ids = [str(s) for s in z["sample_ids"]]
        order = order or ids
        if ids != order:
            raise ValueError(f"{name}: sample order differs from the other runs")
        recs = align(records_by_id, ids)
        e0 = name == "E0"
        a = hit_arrays(z["pred24"] if e0 else z[key], recs, schema, ("body_shared",) if e0 else POSE_GROUPS)
        if key == "clean":
            check_counts(a, stored_pose_counts(m, e0), name)
        per_seed[name] = a
        if not e0:
            errors[name] = m["clean"]["errors"] if key == "clean" else m["crop_error"]["levels"][key]["errors"]
    if order is None:
        raise FileNotFoundError(f"no pose runs under {pose_dir}")
    recs = align(records_by_id, order)
    clusters, strata = clusters_of(recs)
    if "E0" in per_seed:
        arrays["E0"] = per_seed["E0"]
    for r in recipes:
        seeds = [per_seed[f"{r}-s{s}"] for s in SEEDS if f"{r}-s{s}" in per_seed]
        if len(seeds) != len(SEEDS):
            raise FileNotFoundError(f"{r}: need both seeds {SEEDS} under {pose_dir}")
        arrays[r] = pooled(seeds)
    rows = []
    for name, a in arrays.items():
        for (g, t), (h, n) in a.items():
            row = {"run": name, "group": g, "threshold": t, **_cell(h, n), "per_video": _per_video(h, n, strata)}
            if name != "E0":
                row["per_seed"] = {s: pct(*per_seed[f"{name}-s{s}"][(g, t)]) for s in SEEDS}
            rows.append(row)
    deltas = []
    if "E1" in arrays and "E2" in arrays:
        for g, t in (("body", 0.1), ("ski", 0.1), ("ski", 0.05), ("body_shared", 0.1), ("pole", 0.1)):
            deltas.append({"a": "E2", "b": "E1", "group": g, "threshold": t,
                           **diff_block(arrays["E2"][(g, t)], arrays["E1"][(g, t)], clusters, strata, n_boot)})
    if "E0" in arrays:
        for r in recipes:
            for t in THRS:
                deltas.append({"a": r, "b": "E0", "group": "body_shared", "threshold": t,
                               **diff_block(arrays[r][("body_shared", t)], arrays["E0"][("body_shared", t)],
                                            clusters, strata, n_boot)})
    headline = {f"{g}@{t:g}": ratio_ci(*arrays[recipes[-1]][(g, t)], clusters, strata, n_boot=n_boot)
                for g, t in (("body", 0.1), ("ski", 0.1))}
    gt, rel, diag, _ = gt_arrays(recs)
    preds = {r: list(_preds(pose_dir, r, key)) for r in recipes}
    per_joint = []
    for j, jname in enumerate(schema.names):
        row = {"joint": jname}
        for r in recipes:
            h = sum(sample_hits(z_pred, gt, rel, diag, [j], 0.1)[0] for z_pred in preds[r])
            n = len(SEEDS) * rel[:, j].sum()
            row[r] = 100.0 * h.sum() / n if n else None
        row["delta"] = None if row[recipes[0]] is None else row[recipes[-1]] - row[recipes[0]]
        row["total_per_seed"] = int(rel[:, j].sum())
        per_joint.append(row)
    err = {r: {g: float(np.mean([errors[f"{r}-s{s}"][g]["mean_norm_error"] for s in SEEDS])) for g in ("body", "ski")}
           for r in recipes}
    return {"n_samples": len(recs), "n_segments": len(set(clusters)), "n_videos": len(set(strata)), "rows": rows,
            "deltas": deltas, "headline_ci": {recipes[-1]: headline}, "per_joint": per_joint,
            "mean_norm_error": err}


def _preds(pose_dir: Path, recipe: str, key: str):
    for s in SEEDS:
        with np.load(pose_dir / f"{recipe}-s{s}" / "predictions.npz") as z:
            yield z[key]


def augmentation_table(d21_dir: Path, records_by_id: dict, schema: Schema, n_boot: int = N_BOOT) -> dict:
    """The val augmentation study (D16, D21): E1–E4 seed-pooled, Δ vs. E1 with intervals, clean and moderate crops."""
    out = {"rows": [], "deltas": []}
    arrays, order = {}, None
    for key in ("clean", "moderate"):
        for r in ("E1", "E2", "E3", "E4"):
            seeds = []
            for s in SEEDS:
                run = d21_dir / f"{r}-s{s}" / "eval" / "val"
                m, z = load_eval(run)
                ids = [str(i) for i in z["sample_ids"]]
                order = order or ids
                if ids != order:
                    raise ValueError(f"{r}-s{s}: sample order differs from the other runs")
                recs = align(records_by_id, ids)
                a = hit_arrays(z[key], recs, schema, ("body", "ski"))
                stored = (m["clean"] if key == "clean" else m["crop_error"]["levels"][key])["groups"]
                check_counts(a, {(g, t): stored[f"{t:g}"][g] for g in ("body", "ski") for t in THRS}, f"{r}-s{s} {key}")
                seeds.append(a)
            arrays[(r, key)] = (pooled(seeds), [pct(*a[("ski", 0.1)]) for a in seeds], [pct(*a[("body", 0.1)]) for a in seeds])
    clusters, strata = clusters_of(recs)
    for (r, key), (a, ski_seeds, body_seeds) in arrays.items():
        out["rows"].append({"run": r, "crops": key, "body@0.1": _cell(*a[("body", 0.1)]), "ski@0.1": _cell(*a[("ski", 0.1)]),
                            "ski@0.05": _cell(*a[("ski", 0.05)]), "per_seed_ski@0.1": ski_seeds,
                            "per_seed_body@0.1": body_seeds})
        if r != "E1":
            for g in ("body", "ski"):
                out["deltas"].append({"a": r, "b": "E1", "crops": key, "group": g, "threshold": 0.1,
                                      **diff_block(a[(g, 0.1)], arrays[("E1", key)][0][(g, 0.1)], clusters, strata, n_boot)})
    return out


# --- tracking --------------------------------------------------------------------------------

def tracking_tables(track_dir: Path, skitb_root: Path, split_json: Path, partition: str, fps_table: Path,
                    n_boot: int = N_BOOT) -> dict:
    """Our metrics and the toolkit's official ones over every SC clip of the partition; intervals over sequences."""
    from skier.data.skitb import fps_for, load_fps_table, load_skitb_clips
    from skier.eval.skitb_official import clip_metrics, overall, skitb_clip_inputs
    from skier.eval.track_metrics import SUCCESS_THRESHOLDS
    from skier.tracking.run import evaluate_clip, read_track_csv

    fps = load_fps_table(fps_table)
    per_clip, official = [], {}
    for c in load_skitb_clips(skitb_root, split_json, partition):
        name = f"{c.sequence_id}__{c.clip_id}"
        csv = track_dir / f"{name}.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"no track CSV for {name} in {track_dir}")
        rows = read_track_csv(csv)
        st = evaluate_clip(rows, c.boxes_xyxy, c.visible, rows[0]["frame_index"], fps_for(fps, c.sequence_id))
        ious = np.asarray(st["ious"], dtype=np.float64)
        official[name] = clip_metrics(*skitb_clip_inputs(c, rows))
        per_clip.append({"clip": name, "sequence": c.sequence_id, "n_frames": len(ious), "iou_sum": float(ious.sum()),
                         "mean_iou": st["mean_iou"], "n_failures": len(st["failure_episodes"]),
                         "success_counts": [int((ious > t).sum()) for t in SUCCESS_THRESHOLDS],
                         "official_f": official[name]["f_score"]})
    summary_path = track_dir / "summary.json"
    s = read_json(summary_path) if summary_path.is_file() else None
    n = np.array([c["n_frames"] for c in per_clip], dtype=np.float64)
    iou_sum = np.array([c["iou_sum"] for c in per_clip])
    succ = np.array([c["success_counts"] for c in per_clip], dtype=np.float64)
    f_curves = np.array([official[c["clip"]]["f_curve"] for c in per_clip])
    seqs = [c["sequence"] for c in per_clip]
    pooled_iou = iou_sum.sum() / n.sum()
    curve = succ.sum(axis=0) / n.sum()
    # summary.json comes from the tracker's in-memory boxes; the CSVs keep 3 decimals, which moves IoU by ~1e-8
    if s is not None and (abs(s["pooled_mean_iou"] - pooled_iou) > 1e-6 or abs(s["success"]["auc"] - curve.mean()) > 1e-6
                          or s["n_failure_episodes"] != sum(c["n_failures"] for c in per_clip)):
        raise ValueError(f"tracking: rebuilt numbers differ from {summary_path}")
    names = [c["clip"] for c in per_clip]
    return {
        "partition": partition, "n_clips": len(per_clip), "n_sequences": len(set(seqs)), "n_frames": int(n.sum()),
        "ours": {"pooled_mean_iou": float(pooled_iou),
                 "per_clip_mean_iou": float(np.mean([c["mean_iou"] for c in per_clip if c["mean_iou"] is not None])),
                 "success_thresholds": [float(t) for t in SUCCESS_THRESHOLDS], "success_curve": curve.tolist(),
                 "auc": float(curve.mean()), "success@0.5": float(curve[list(np.round(SUCCESS_THRESHOLDS, 2)).index(0.5)]),
                 "n_failure_episodes": sum(c["n_failures"] for c in per_clip),
                 "clips_with_failures": sum(c["n_failures"] > 0 for c in per_clip)},
        "official_toolkit": overall(official, names),
        "ci95": {
            "pooled_mean_iou": ratio_ci(iou_sum, n, seqs, scale=1.0, n_boot=n_boot)["ci"],
            "auc": cluster_bootstrap(lambda i: (succ[i].sum(axis=0) / n[i].sum()).mean(), seqs, n_boot=n_boot)["ci"],
            "official_f_score": cluster_bootstrap(lambda i: f_curves[i].mean(axis=0).max(), seqs, n_boot=n_boot)["ci"],
            "unit": "SkiTB sequence (D25)"},
        "per_sequence": [{"sequence": q, "n_clips": seqs.count(q),
                          "pooled_mean_iou": float(iou_sum[np.array(seqs) == q].sum() / n[np.array(seqs) == q].sum()),
                          "n_failure_episodes": sum(c["n_failures"] for c in per_clip if c["sequence"] == q)}
                         for q in sorted(set(seqs))],
        "per_clip": per_clip,
    }


# --- replay ----------------------------------------------------------------------------------

REPLAY_MEANS = ("box_containment_mean", "frac_fully_inside", "centering_error_mean", "centering_error_p95",
                "acceleration_mean", "acceleration_p95", "area_fraction_mean", "log_zoom_rate_mean")


def _num(v) -> float:
    return np.nan if v in (None, "") else float(v)


def replay_tables(replay_dir: Path, n_boot: int = N_BOOT, target_pct: float = 95.0) -> dict:
    """SkiTB clips: means over clips of each replay metric (GT-box target); Ski 2DPose segments: pooled landmark
    containment, ski ends separately (provisional target ≥ 95%). Smoothed and the τ = 0 baseline."""
    out = {}
    for variant in ("smoothed", "unsmoothed_baseline"):
        rows = read_csv(replay_dir / f"{variant}.csv")
        sk = [r for r in rows if r["source"] == "skitb"]
        sp = [r for r in rows if r["source"] == "ski2dpose"]
        block = {"skitb_clips": len(sk), "ski2dpose_segments": len(sp)}
        if sk:
            block["skitb"] = {k: float(np.nanmean([_num(r[k]) for r in sk])) for k in REPLAY_MEANS}
            block["skitb"]["box_containment_min"] = float(np.nanmin([_num(r["box_containment_min"]) for r in sk]))
            cont = np.array([_num(r["box_containment_mean"]) for r in sk])
            ok = np.isfinite(cont)
            block["skitb"]["box_containment_ci95"] = ratio_ci(cont[ok], np.ones(ok.sum()), [r["sequence"] for r, o in zip(sk, ok) if o],
                                                              scale=1.0, n_boot=n_boot)["ci"]
        if sp:
            seg = [r["sequence"] for r in sp]
            vid = [s.split("/")[0] for s in seg]
            lm = {}
            for g, pre in (("body_ski", "body_ski"), ("ski_endpoints", "ski_end")):
                ins = np.array([_num(r[f"{pre}_inside"]) for r in sp])
                tot = np.array([_num(r[f"{pre}_total"]) for r in sp])
                keep = np.isfinite(tot) & (tot > 0)
                lm[g] = {"inside": int(ins[keep].sum()), "total": int(tot[keep].sum()), "pct": pct(ins[keep], tot[keep]),
                         "ci95": ratio_ci(ins[keep], tot[keep], np.array(seg)[keep], np.array(vid)[keep],
                                          n_boot=n_boot)["ci"]}
            lm["ski_endpoints"]["meets_target"] = lm["ski_endpoints"]["pct"] is not None and lm["ski_endpoints"]["pct"] >= target_pct
            block["ski2dpose_landmarks"] = lm
            block["ski2dpose"] = {k: float(np.nanmean([_num(r[k]) for r in sp])) for k in
                                  ("acceleration_mean", "area_fraction_mean")}
        out[variant] = block
    return out


# --- propagation -----------------------------------------------------------------------------

def propagation_tables(prop_dir: Path, records_by_id: dict, schema: Schema, k_pose: float, n_boot: int = N_BOOT,
                       recipes=("E1", "E2")) -> dict:
    """Reference vs. tracker crop on the same labels (Step 7), seed-pooled per recipe, with intervals."""
    p = read_json(prop_dir / "propagation.json")
    with np.load(prop_dir / "predictions.npz") as z:
        npz = {k: z[k] for k in z.files}
    recs = align(records_by_id, [str(s) for s in npz["sample_ids"]])
    clusters, strata = clusters_of(recs)
    k = f"{k_pose:g}"
    arrays, per_seed = {}, {}
    for r in recipes:
        for crop in ("reference", "tracked"):
            seeds = []
            for s in SEEDS:
                name = f"{r}-s{s}"
                pred = npz[f"{name}__reference"] if crop == "reference" else npz[f"{name}__k{k}"]
                a = hit_arrays(pred, recs, schema, ("body", "ski", "pole"))
                stored = p["checkpoints"][name]["by_k_pose"][k][crop]["groups"]
                check_counts(a, {(g, t): stored[f"{t:g}"][g] for g in ("body", "ski", "pole") for t in THRS},
                             f"propagation {name} {crop}")
                seeds.append(a)
            arrays[(r, crop)] = pooled(seeds)
            per_seed[(r, crop)] = seeds
    rows = [{"run": r, "crop": crop, "group": g, "threshold": t, **_cell(*arrays[(r, crop)][(g, t)]),
             "per_seed": [pct(*a[(g, t)]) for a in per_seed[(r, crop)]],
             "per_video": _per_video(*arrays[(r, crop)][(g, t)], strata)}
            for (r, crop) in arrays for g in ("body", "ski", "pole") for t in THRS]
    deltas = [{"a": f"{r} reference", "b": f"{r} tracked", "group": g, "threshold": t,
               **diff_block(arrays[(r, "reference")][(g, t)], arrays[(r, "tracked")][(g, t)], clusters, strata, n_boot)}
              for r in recipes for g, t in (("body", 0.1), ("ski", 0.1), ("ski", 0.05))]
    deltas += [{"a": f"{recipes[-1]} tracked", "b": f"{recipes[0]} tracked", "group": g, "threshold": 0.1,
                **diff_block(arrays[(recipes[-1], "tracked")][(g, 0.1)], arrays[(recipes[0], "tracked")][(g, 0.1)],
                             clusters, strata, n_boot)} for g in ("body", "ski")]
    first = p["checkpoints"][f"{recipes[-1]}-s{SEEDS[0]}"]["by_k_pose"][k]
    strata_rows = []
    for name in ("tracker_iou", "time_since_init", "status"):
        for value in first["strata"].get(name, {}):
            for r in recipes:
                blocks = [p["checkpoints"][f"{r}-s{s}"]["by_k_pose"][k]["strata"][name][value] for s in SEEDS]
                row = {"stratum": name, "value": value, "run": r, "n_images": blocks[0]["n_images"]}
                for crop in ("reference", "tracked"):
                    for g in ("body", "ski"):
                        h = sum(b[crop][g]["hits"] for b in blocks)
                        n = sum(b[crop][g]["total"] for b in blocks)
                        row[f"{crop}_{g}"] = 100.0 * h / n if n else None
                strata_rows.append(row)
    return {"k_pose": k_pose, "n_scored": len(recs), "n_segments": len(set(clusters)), "rows": rows, "deltas": deltas,
            "cut_off_share": {g: first["tracked"]["cut_off"][g]["share"] for g in ("body", "ski")}, "strata": strata_rows}


# --- markdown --------------------------------------------------------------------------------

def _f(x, nd=1) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(c, nd=1) -> str:
    return f"[{_f(c[0], nd)}, {_f(c[1], nd)}]"


def to_markdown(t: dict) -> str:
    out = [f"# Report tables: {t['partition']}", "",
           "Intervals: 95% cluster bootstrap (D25): segments within videos for pose and propagation, sequences for "
           "SkiTB; 10,000 resamples, seed 230. E1 and E2 are pooled over seeds 230 and 231 (per-seed values in "
           "brackets).", ""]
    if "pose" in t:
        p = t["pose"]
        out += [f"## Pose ({p['n_samples']} images, {p['n_segments']} segments, {p['n_videos']} videos)", "",
                "| Run | Body @0.1 | Ski @0.1 | Ski @0.05 | Body, shared joints @0.1 | Poles @0.1 (suppl.) |",
                "| --- | --- | --- | --- | --- | --- |"]
        cell = {(r["run"], r["group"], r["threshold"]): r for r in p["rows"]}
        for run in [x for x in ("E0", "E1", "E2") if any(k[0] == x for k in cell)]:
            def c(g, th):
                r = cell.get((run, g, th))
                if r is None:
                    return "N/A"
                seeds = r.get("per_seed")
                extra = f" ({' / '.join(_f(v) for v in seeds.values())})" if seeds else ""
                return f"{_f(r['pct'])}% ({r['hits']}/{r['total']}){extra}"
            out.append(f"| {run} | {c('body', 0.1)} | {c('ski', 0.1)} | {c('ski', 0.05)} | {c('body_shared', 0.1)} | "
                       f"{c('pole', 0.1)} |")
        out += ["", "| Δ | Group | Δ (points) | 95% interval |", "| --- | --- | --- | --- |"]
        for d in p["deltas"]:
            out.append(f"| {d['a']} − {d['b']} | {d['group']} @{d['threshold']:g} | {d['delta']:+.1f} | {_ci(d['ci95'])} |")
        vids = sorted({v for r in p["rows"] for v in r["per_video"]})
        out += ["", "Per video, PCK@0.1:", "", "| Run | Group | " + " | ".join(vids) + " |",
                "| --- | --- | " + " | ".join("---" for _ in vids) + " |"]
        for r in p["rows"]:
            if r["threshold"] == 0.1 and r["group"] in ("body", "ski", "body_shared"):
                out.append(f"| {r['run']} | {r['group']} | " + " | ".join(
                    f"{_f(r['per_video'][v]['pct'])}% ({r['per_video'][v]['hits']}/{r['per_video'][v]['total']})"
                    if v in r["per_video"] else "—" for v in vids) + " |")
        out.append("")
    if "augmentation" in t:
        a = t["augmentation"]
        out += ["## Augmentation study (val, D16/D21)", "", "| Run | Crops | Body @0.1 | Ski @0.1 | Ski @0.05 |",
                "| --- | --- | --- | --- | --- |"]
        for r in a["rows"]:
            out.append(f"| {r['run']} | {r['crops']} | {_f(r['body@0.1']['pct'])}% | {_f(r['ski@0.1']['pct'])}% "
                       f"({' / '.join(_f(v) for v in r['per_seed_ski@0.1'])}) | {_f(r['ski@0.05']['pct'])}% |")
        out += ["", "| Δ vs. E1 | Crops | Group | Δ (points) | 95% interval |", "| --- | --- | --- | --- | --- |"]
        for d in a["deltas"]:
            out.append(f"| {d['a']} | {d['crops']} | {d['group']} | {d['delta']:+.1f} | {_ci(d['ci95'])} |")
        out.append("")
    if "tracking" in t:
        k = t["tracking"]
        o, f, ci = k["ours"], k["official_toolkit"], k["ci95"]
        out += [f"## Tracking (SkiTB {k['partition']}: {k['n_sequences']} sequences, {k['n_clips']} SC clips, "
                f"{k['n_frames']} scored frames)", "",
                "| Metric | Value | 95% interval |", "| --- | --- | --- |",
                f"| Pooled mean IoU (ours) | {_f(o['pooled_mean_iou'], 3)} | {_ci(ci['pooled_mean_iou'], 3)} |",
                f"| Per-clip mean IoU (ours) | {_f(o['per_clip_mean_iou'], 3)} | |",
                f"| Success AUC (ours) | {_f(o['auc'], 3)} | {_ci(ci['auc'], 3)} |",
                f"| Success @ IoU 0.5 (ours) | {_f(o['success@0.5'], 3)} | |",
                f"| Failure episodes (ours; IoU < 0.1 for ≥ 0.5 s) | {o['n_failure_episodes']} in {o['clips_with_failures']} clips | |",
                f"| **Toolkit** F-score (SC) | {_f(f['f_score'], 3)} | {_ci(ci['official_f_score'], 3)} |",
                f"| **Toolkit** precision / recall | {_f(f['precision'], 3)} / {_f(f['recall'], 3)} | |",
                f"| **Toolkit** GSR | {_f(f['gsr_score'], 3)} | |", ""]
    if "replay" in t:
        out += ["## Replay (τ 0.2 s, k 2.5; low-confidence frames drive the replay, D23/D24)", "",
                "| Variant | GT-box containment | Fully inside | Centering (mean / p95) | Acceleration (mean / p95) | "
                "Crop area | Ski ends inside (Ski 2DPose) |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for v, b in t["replay"].items():
            s = b.get("skitb", {})
            lm = b.get("ski2dpose_landmarks", {}).get("ski_endpoints")
            ski = f"{_f(lm['pct'])}% ({lm['inside']}/{lm['total']}) {_ci(lm['ci95'])}" if lm else "—"
            out.append(f"| {v} | {_f(s.get('box_containment_mean'), 3)} {_ci(s.get('box_containment_ci95', [None, None]), 3)} | "
                       f"{_f(s.get('frac_fully_inside'), 3)} | {_f(s.get('centering_error_mean'), 3)} / "
                       f"{_f(s.get('centering_error_p95'), 3)} | {_f(s.get('acceleration_mean'), 2)} / "
                       f"{_f(s.get('acceleration_p95'), 2)} | {_f(s.get('area_fraction_mean'), 2)} | {ski} |")
        out.append("")
    if "propagation" in t:
        pr = t["propagation"]
        out += [f"## Propagation: reference vs. tracker crop (k_pose {pr['k_pose']:g}; {pr['n_scored']} labels in "
                f"{pr['n_segments']} segments)", "",
                "| Run | Crop | Body @0.1 | Ski @0.1 | Ski @0.05 |", "| --- | --- | --- | --- | --- |"]
        cell = {(r["run"], r["crop"], r["group"], r["threshold"]): r for r in pr["rows"]}
        for run, crop in sorted({(r["run"], r["crop"]) for r in pr["rows"]}):
            out.append(f"| {run} | {crop} | " + " | ".join(
                f"{_f(cell[(run, crop, g, th)]['pct'])}% ({cell[(run, crop, g, th)]['hits']}/{cell[(run, crop, g, th)]['total']})"
                for g, th in (("body", 0.1), ("ski", 0.1), ("ski", 0.05))) + " |")
        out += ["", f"Tracker-crop cut-off share: body {_f(100 * pr['cut_off_share']['body'])}%, "
                f"ski {_f(100 * pr['cut_off_share']['ski'])}%.", "",
                "| Δ | Group | Δ (points) | 95% interval |", "| --- | --- | --- | --- |"]
        for d in pr["deltas"]:
            out.append(f"| {d['a']} − {d['b']} | {d['group']} @{d['threshold']:g} | {d['delta']:+.1f} | {_ci(d['ci95'])} |")
        out += ["", "| Stratum | Value | Run | Labels | Body ref → tracked | Ski ref → tracked |",
                "| --- | --- | --- | --- | --- | --- |"]
        for s in pr["strata"]:
            out.append(f"| {s['stratum']} | {s['value']} | {s['run']} | {s['n_images']} | {_f(s['reference_body'])} → "
                       f"{_f(s['tracked_body'])} | {_f(s['reference_ski'])} → {_f(s['tracked_ski'])} |")
        out.append("")
    return "\n".join(out) + "\n"


# --- CLI -------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report tables (plan Step 8) from one run_final.sh output folder.")
    p.add_argument("--root", type=Path, required=True, help="$SKIER_RESULTS/final/<partition> (pose/, tracking/, ...)")
    p.add_argument("--partition", required=True)
    p.add_argument("--split", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--skitb-root", type=Path, help="for the tracking table")
    p.add_argument("--skitb-split", type=Path, help="SkiTB split JSON (with --skitb-root)")
    p.add_argument("--fps-table", type=Path, help="default: data/manifests/skitb_fps_v1.csv")
    p.add_argument("--d21-dir", type=Path, help="val only: $SKIER_RESULTS/pose/d21 for the augmentation table")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", type=Path, help="default: <root>/tables")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from skier.data.ski2dpose import load_records
    from skier.integration.config import IntegrationConfig
    from skier.pose.eval_generic import select_partition

    schema = load_schema()
    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    manifest = read_json(args.split or repo_root() / "data/manifests/pose_split_v1.json")
    by_id = {r.sample_id: r for r in select_partition(records, manifest, args.partition)[0]}
    t: dict = {"partition": args.partition, "split_sha256": manifest["content_sha256"]}
    try:
        if (args.root / "pose").is_dir():
            t["pose"] = pose_tables(args.root / "pose", by_id, schema, args.n_boot)
        if args.d21_dir is not None:
            if args.partition != "val":
                raise ValueError("the augmentation table is val only (D16)")
            t["augmentation"] = augmentation_table(args.d21_dir, by_id, schema, args.n_boot)
        if (args.root / "tracking" / "skitb").is_dir() and args.skitb_root and args.skitb_split:
            from skier.data.skitb import FPS_TABLE

            t["tracking"] = tracking_tables(args.root / "tracking" / "skitb", args.skitb_root, args.skitb_split,
                                            args.partition, args.fps_table or repo_root() / FPS_TABLE, args.n_boot)
        if (args.root / "replay" / "smoothed.csv").is_file():
            t["replay"] = replay_tables(args.root / "replay", args.n_boot)
        if (args.root / "propagation" / "propagation.json").is_file():
            k = IntegrationConfig.from_yaml(configs_dir() / "integration" / "default.yaml").require("k_pose").k_pose
            t["propagation"] = propagation_tables(args.root / "propagation", by_id, schema, k, args.n_boot)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = ensure_dir(args.out or args.root / "tables")
    write_json(t, out / "tables.json")
    (out / "tables.md").write_text(to_markdown(t), encoding="utf-8")
    if "tracking" in t:
        write_csv(t["tracking"]["per_sequence"], out / "tracking_per_sequence.csv",
                  ["sequence", "n_clips", "pooled_mean_iou", "n_failure_episodes"])
    if "pose" in t:
        write_csv(t["pose"]["per_joint"], out / "pose_per_joint.csv", list(t["pose"]["per_joint"][0]))
    print(f"tables ({', '.join(k for k in t if k not in ('partition', 'split_sha256'))}) → {out}")
    return 0
