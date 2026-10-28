"""Compare experiments on one partition (plan Step 6): a table, per-joint deltas, and the
pre-registered recipe-selection rule (D16).

Inputs are ``metrics.json`` files from ``eval_pose.py``. E0's file from
``eval_generic_pose.py`` is accepted for the body-on-shared-joints column only.
Comparisons need the same split, partition and sample IDs (and the same crop-error set),
so the denominators match; anything else is refused.

The rule, written down before any E2–E4 results existed:
1. Disqualify a candidate whose clean body PCK@0.1 is more than 2 points below E1's.
2. Among the rest, take the highest clean ski PCK@0.1. If the top two are within 1 point,
   take the one with the higher ski PCK@0.1 on the **moderate** crop-error level.
3. If no candidate beats E1's clean ski PCK@0.1, the result is negative: keep E1.

The rule's pick is a suggestion; Ruichao makes the final choice and records it. Paired
bootstrap intervals come in Step 8.

Several metrics files per run (one per seed) are pooled first (D21): the hits and totals of
every PCK block are summed, so with equal per-seed totals each percentage is the seed mean,
and the rule applies to those means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skier.io_utils import ensure_dir, read_json, write_csv, write_json
from skier.paths import results_root

BASELINE = "E1"
CANDIDATES = ("E2", "E3", "E4")
BODY_DROP = 2.0  # points
TIE = 1.0  # points
TIE_LEVEL = "moderate"
RULE = ("D16: drop candidates whose clean body PCK@0.1 falls > 2 points below E1; take the highest clean ski "
        "PCK@0.1; if the top two are within 1 point, take the higher moderate crop-error ski PCK@0.1; if none "
        "beats E1, keep E1 (negative result)")


def is_e0(m: dict) -> bool:
    return "clean" not in m and "body_shared" in m


def row_for(name: str, m: dict) -> dict:
    if is_e0(m):
        return {"name": name, "body": None, "body_shared": m["body_shared"]["pck@0.1"], "ski": None,
                "ski_005": None, "crop": {}}
    c = m["clean"]
    crop = {lv: {"ski": d["groups"]["0.1"]["ski"], "ski_cut_share": d["cut_off"]["ski"]["share"]}
            for lv, d in m.get("crop_error", {}).get("levels", {}).items()}
    return {"name": name, "body": c["groups"]["0.1"]["body"], "body_shared": c["body_shared"]["0.1"],
            "ski": c["groups"]["0.1"]["ski"], "ski_005": c["groups"]["0.05"]["ski"], "crop": crop,
            "seeds": m.get("seeds")}


def check_comparable(metrics: dict[str, dict]) -> None:
    def same(key, items):
        values = {n: m.get(key) for n, m in items}
        if len(set(values.values())) > 1:
            raise ValueError(f"runs differ in {key}: {values}")

    items = list(metrics.items())
    for key in ("split_sha256", "partition", "n_samples"):
        same(key, items)
    ours = [(n, m) for n, m in items if not is_e0(m)]
    same("sample_ids_sha256", ours)
    with_ce = [(n, {"ce": m["crop_error"]["content_sha256"]}) for n, m in ours if "crop_error" in m]
    same("ce", with_ce)
    same("pooled seeds", [(n, {"pooled seeds": tuple(s["seed"] for s in m.get("seeds") or ())}) for n, m in ours])


POOLED_CLEAN = ("groups", "body_shared", "per_joint", "cut_off")
POOLED_LEVEL = ("groups", "body_shared", "cut_off")


def _pool(nodes: list, where: str):
    """Sum one node over seeds: PCK blocks (hits/total) and cut-off counts (cut/total), recursively."""
    first = nodes[0]
    if not isinstance(first, dict) or any(not isinstance(n, dict) or set(n) != set(first) for n in nodes):
        raise ValueError(f"cannot pool {where}: not the same block in every seed")
    if "total" not in first:
        return {k: _pool([n[k] for n in nodes], f"{where}.{k}") for k in first}
    totals = {n["total"] for n in nodes}
    if len(totals) > 1:
        raise ValueError(f"seeds differ in the total of {where}: {sorted(totals)}")
    total = first["total"] * len(nodes)
    if "hits" in first:
        hits = sum(n["hits"] for n in nodes)
        return {"hits": hits, "total": total, "pct": 100.0 * hits / total if total else None}
    cut = sum(n["cut"] for n in nodes)
    return {"cut": cut, "total": total, "share": cut / total if total else None}


def pool_seeds(runs: list[dict]) -> dict:
    """One recipe's metrics pooled over seeds (D21), in the layout ``compare`` reads.

    All seeds are scored on the same samples, so every block's total must match across seeds,
    and each pooled percentage then equals the mean of the per-seed percentages. ``seeds``
    keeps each run's own headline numbers.
    """
    if len(runs) < 2:
        raise ValueError("pooling needs at least 2 runs")
    check_comparable({str(i): m for i, m in enumerate(runs)})
    seeds = [(m.get("config") or {}).get("seed") for m in runs]
    if None in seeds or len(set(seeds)) < len(seeds):
        raise ValueError(f"pooled runs need distinct recorded seeds, got {seeds}")
    if len({"crop_error" in m for m in runs}) > 1:
        raise ValueError("only some seeds have crop-error results")
    first = runs[0]
    out = {k: first[k] for k in ("split_sha256", "partition", "n_samples", "sample_ids_sha256")}
    out["clean"] = _pool([{k: m["clean"][k] for k in POOLED_CLEAN if k in m["clean"]} for m in runs], "clean")
    if "crop_error" in first:
        levels = _pool([{lv: {k: d[k] for k in POOLED_LEVEL if k in d} for lv, d in m["crop_error"]["levels"].items()}
                        for m in runs], "crop_error")
        for lv, d in levels.items():
            radii = {m["crop_error"]["levels"][lv]["radius"] for m in runs}
            if len(radii) > 1:
                raise ValueError(f"seeds differ in the {lv} crop-error radius: {sorted(radii)}")
            d["radius"] = radii.pop()
        out["crop_error"] = {"content_sha256": first["crop_error"]["content_sha256"], "levels": levels}
    out["seeds"] = [{"seed": s, "experiment": m.get("experiment"),
                     **{k: pct(b) for k, b in metric_blocks(row_for("", m)).items()}} for s, m in zip(seeds, runs)]
    return out


def pct(block: dict | None) -> float | None:
    return None if block is None else block["pct"]


def delta(a: dict | None, b: dict | None) -> float | None:
    """``a - b`` in points; None if either is missing."""
    pa, pb = pct(a), pct(b)
    return None if pa is None or pb is None else pa - pb


def select_recipe(rows: dict[str, dict], baseline: str = BASELINE, candidates=CANDIDATES) -> dict:
    out = {"rule": RULE, "baseline": baseline}
    if baseline not in rows or rows[baseline]["ski"] is None:
        return {**out, "pick": None, "reason": f"{baseline} missing"}
    b = rows[baseline]
    present = [c for c in candidates if c in rows and rows[c]["ski"] is not None]
    disqualified = [c for c in present if delta(rows[c]["body"], b["body"]) < -BODY_DROP]
    better = sorted((c for c in present if c not in disqualified and delta(rows[c]["ski"], b["ski"]) > 0),
                    key=lambda c: pct(rows[c]["ski"]), reverse=True)
    out.update(candidates=present, disqualified_body_drop=disqualified, beat_baseline=better)
    if not better:
        return {**out, "pick": baseline, "reason": f"no candidate beats {baseline} on clean ski PCK@0.1: "
                                                   f"negative result, keep {baseline}"}
    pick, reason = better[0], "highest clean ski PCK@0.1"
    if len(better) > 1 and delta(rows[better[0]]["ski"], rows[better[1]]["ski"]) <= TIE:
        first, second = better[:2]
        c1, c2 = (pct(rows[c]["crop"].get(TIE_LEVEL, {}).get("ski")) for c in (first, second))
        if c1 is None or c2 is None:
            reason += f"; the top two are within {TIE:g} point, but there are no {TIE_LEVEL} crop-error results to break the tie"
        else:
            pick = first if c1 >= c2 else second
            reason = f"top two within {TIE:g} point on clean ski PCK@0.1; higher {TIE_LEVEL} crop-error ski PCK@0.1"
    return {**out, "pick": pick, "reason": reason}


def _cell(block: dict | None) -> str:
    if block is None or block["pct"] is None:
        return "—"
    return f"{block['pct']:.1f}% ({block['hits']}/{block['total']})"


def _d(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1f}"


def pooled_seeds(rows: dict[str, dict]) -> list | None:
    """The seeds the runs are pooled over (the same for every pooled run), or None."""
    return next(([s["seed"] for s in r["seeds"]] for r in rows.values() if r.get("seeds")), None)


def markdown(rows: dict[str, dict], levels: list[str], selection: dict, baseline: str, partition: str) -> str:
    seeds = pooled_seeds(rows)
    head = ["Run", "Body @0.1", "Body (shared joints) @0.1", "Ski @0.1"]
    head += [f"Ski @0.1 per seed ({' / '.join(map(str, seeds))})" if seeds else None, "Ski @0.05"]
    head += [f"Ski @0.1, {lv} crops" for lv in levels]
    head += [f"Ski cut off, {levels[-1]}" if levels else None, f"Δ body vs {baseline}", f"Δ ski vs {baseline}"]
    head = [h for h in head if h]
    title = f"# Experiment comparison: {partition}"
    if seeds:
        title += f", means over seeds {' and '.join(map(str, seeds))} (D21)"
    lines = [title, "", "| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    b = rows.get(baseline)
    for name, r in rows.items():
        d_body = delta(r["body"], b["body"]) if b else None
        cells = [name, _cell(r["body"]), _cell(r["body_shared"]), _cell(r["ski"])]
        if seeds:
            cells.append(" / ".join("—" if s["ski@0.1"] is None else f"{s['ski@0.1']:.1f}%" for s in r["seeds"])
                         if r.get("seeds") else "—")
        cells.append(_cell(r["ski_005"]))
        cells += [f"{pct(r['crop'][lv]['ski']):.1f}%" if lv in r["crop"] and pct(r["crop"][lv]["ski"]) is not None
                  else "—" for lv in levels]
        if levels:
            share = r["crop"].get(levels[-1], {}).get("ski_cut_share")
            cells.append("—" if share is None else f"{100 * share:.1f}%")
        flag = " ⚠" if d_body is not None and d_body < -BODY_DROP else ""
        cells += [_d(d_body) + flag, _d(delta(r["ski"], b["ski"]) if b else None)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"⚠ = body PCK@0.1 more than {BODY_DROP:g} points below {baseline}."]
    if seeds:
        lines.append("Each pooled run sums the hits and totals of its seeds, so every percentage is the seed mean; "
                     "the rule applies to these means (D21).")
    lines += ["", f"**Rule's suggestion: {selection.get('pick')}** ({selection.get('reason')}).",
              f"Rule: {RULE}. Ruichao makes the final choice; record it in docs/decisions.md.",
              "Paired bootstrap intervals over source videos come in Step 8."]
    return "\n".join(lines) + "\n"


def per_joint_rows(metrics: dict[str, dict], baseline: str) -> list[dict]:
    ours = {n: m for n, m in metrics.items() if not is_e0(m)}
    if not ours:
        return []
    base = ours.get(baseline)
    joints = list(next(iter(ours.values()))["clean"]["per_joint"])
    out = []
    for j in joints:
        row = {"joint": j}
        for n, m in ours.items():
            block = m["clean"]["per_joint"][j]
            row[f"{n}_pct"] = block["pct"]
            row[f"{n}_total"] = block["total"]
            if base is not None and n != baseline:
                row[f"{n}_delta"] = delta(block, base["clean"]["per_joint"][j])
        out.append(row)
    return out


def metric_blocks(r: dict) -> dict:
    return {"body@0.1": r["body"], "body_shared@0.1": r["body_shared"], "ski@0.1": r["ski"], "ski@0.05": r["ski_005"],
            **{f"ski@0.1_{lv}": c["ski"] for lv, c in r["crop"].items()}}


def long_rows(rows: dict[str, dict], baseline: str) -> list[dict]:
    base = metric_blocks(rows[baseline]) if baseline in rows else {}
    return [{"run": name, "metric": key, "hits": block["hits"], "total": block["total"], "pct": block["pct"],
             f"delta_vs_{baseline}": delta(block, base.get(key))}
            for name, r in rows.items() for key, block in metric_blocks(r).items() if block is not None]


def compare(metrics: dict[str, dict], baseline: str = BASELINE) -> dict:
    check_comparable(metrics)
    rows = {n: row_for(n, m) for n, m in metrics.items()}
    levels = sorted({lv for r in rows.values() for lv in r["crop"]},
                    key=lambda lv: next(m["crop_error"]["levels"][lv]["radius"] for m in metrics.values()
                                        if lv in m.get("crop_error", {}).get("levels", {})))
    selection = select_recipe(rows, baseline)
    if pooled_seeds(rows):
        selection["seed_means_over"] = pooled_seeds(rows)
        selection["per_seed"] = {n: r["seeds"] for n, r in rows.items() if r.get("seeds")}
    partition = next(iter(metrics.values()))["partition"]
    return {"rows": rows, "levels": levels, "selection": selection, "partition": partition,
            "markdown": markdown(rows, levels, selection, baseline, partition),
            "long": long_rows(rows, baseline), "per_joint": per_joint_rows(metrics, baseline)}


def parse_runs(values: list[str]) -> dict[str, list[Path]]:
    runs = {}
    for v in values:
        name, sep, paths = v.partition("=")
        if not sep or not name or not all(paths.split(",")):
            raise ValueError(f"expected NAME=PATH/metrics.json[,PATH/metrics.json...], got {v!r}")
        if name in runs:
            raise ValueError(f"run {name!r} given twice")
        runs[name] = [Path(p) for p in paths.split(",")]
    return runs


def load_run(paths: list[Path]) -> dict:
    """One metrics file as is; several (one per seed) pooled (D21)."""
    return read_json(paths[0]) if len(paths) == 1 else pool_seeds([read_json(p) for p in paths])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare experiments (plan Step 6): table, per-joint deltas, and the "
                                            "pre-registered recipe-selection rule (D16).")
    p.add_argument("runs", nargs="+", metavar="NAME=METRICS_JSON[,METRICS_JSON...]",
                   help="e.g. E1=$SKIER_RESULTS/pose/E1/eval/val/metrics.json; several comma-separated files "
                        "(one per seed) are pooled and compared on the seed means (D21)")
    p.add_argument("--baseline", default=BASELINE)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/compare/<partition>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = {n: load_run(ps) for n, ps in parse_runs(args.runs).items()}
        result = compare(metrics, args.baseline)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = ensure_dir(args.out or results_root() / "pose" / "compare" / result["partition"])
    (out / "comparison.md").write_text(result["markdown"])
    if result["long"]:
        write_csv(result["long"], out / "comparison.csv", list(result["long"][0]))
    if result["per_joint"]:
        write_csv(result["per_joint"], out / "per_joint.csv", list(result["per_joint"][0]))
    write_json(result["selection"], out / "selection.json")
    print(result["markdown"])
    print(f"wrote {out}")
    return 0
