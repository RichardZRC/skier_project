"""Pick the SiamFC-vs-STARK demo clips by fixed rules on the val tracking runs (D28 extra; val only, never test).

Each clip's mean IoU vs. the GT box comes from the two val chains' SkiTB summaries:
  both good:      highest min(IoU_siamfc, IoU_stark)
  both bad:       lowest  max(IoU_siamfc, IoU_stark)
  STARK better:   largest IoU_stark - IoU_siamfc
Only clips of at least --min-seconds (default 5 s) qualify, so each demo is long enough to watch.
Picks are greedy in rank order, one clip per SkiTB sequence across all categories (different races).
Usage (GPU machine): python results/step7b/demo_set.py [--n-good 2 --n-bad 1 --n-better 2] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.paths import results_root  # noqa: E402

RUNS = {"siamfc": "final/val/tracking/skitb/summary.json", "stark": "final_stark/val/tracking/skitb/summary.json"}


def clip_stats(path: Path) -> dict[str, tuple[float, float]]:
    """``clip -> (mean IoU, length in seconds)`` from a run_tracker SkiTB summary."""
    per = json.loads(path.read_text())["per_clip"]
    items = per.items() if isinstance(per, dict) else ((p.get("clip") or p.get("name"), p) for p in per)
    return {str(name): (float(v["mean_iou"]), float(v["n_frames"]) / float(v["fps"])) for name, v in items}


def pick(table: list[dict], key, n: int, used: set[str], reverse: bool) -> list[dict]:
    chosen = []
    for row in sorted(table, key=key, reverse=reverse):
        if len(chosen) == n:
            break
        if row["sequence"] in used:
            continue
        chosen.append(row)
        used.add(row["sequence"])
    return chosen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-good", type=int, default=2)
    p.add_argument("--n-bad", type=int, default=1)
    p.add_argument("--n-better", type=int, default=2)
    p.add_argument("--min-seconds", type=float, default=5.0, help="shortest clip that qualifies")
    p.add_argument("--show", type=int, default=8, help="candidates printed per category")
    p.add_argument("--out", type=Path, help="write the picks as JSON (small text evidence)")
    a = p.parse_args(argv)
    r = results_root()
    stats = {k: clip_stats(r / v) for k, v in RUNS.items()}
    if set(stats["siamfc"]) != set(stats["stark"]):
        raise SystemExit("the two val runs don't cover the same clips")
    table = []
    for c in sorted(stats["siamfc"]):
        (ia, sec), (ib, _) = stats["siamfc"][c], stats["stark"][c]
        if sec >= a.min_seconds:
            table.append({"clip": c, "sequence": c.split("__")[0], "seconds": round(sec, 1), "siamfc": round(ia, 3),
                          "stark": round(ib, 3), "diff": round(ib - ia, 3)})
    print(f"{len(table)} of {len(stats['siamfc'])} val clips are at least {a.min_seconds:g} s long")
    rules = {"both_good": (lambda t: min(t["siamfc"], t["stark"]), a.n_good, True),
             "both_bad": (lambda t: max(t["siamfc"], t["stark"]), a.n_bad, False),
             "stark_better": (lambda t: t["diff"], a.n_better, True)}
    for name, (key, _, rev) in rules.items():
        print(f"{name} candidates:", [(t["clip"], t["seconds"], t["siamfc"], t["stark"])
                                      for t in sorted(table, key=key, reverse=rev)[:a.show]])
    used: set[str] = set()
    picks = {name: pick(table, key, n, used, rev) for name, (key, n, rev) in rules.items()}
    for name, rows in picks.items():
        print(f"PICK {name}:", [(t["clip"], t["seconds"], t["siamfc"], t["stark"]) for t in rows])
    if a.out:
        a.out.write_text(json.dumps({"rules": {k: v.split(": ")[-1] for k, v in (
            ("both_good", "both good: highest min(IoU_siamfc, IoU_stark)"),
            ("both_bad", "both bad: lowest max(IoU_siamfc, IoU_stark)"),
            ("stark_better", "STARK better: largest IoU_stark - IoU_siamfc"))},
            "one_clip_per_sequence": True, "min_seconds": a.min_seconds, "partition": "val",
            "n_val_clips": len(stats["siamfc"]), "n_qualifying": len(table),
            "source": RUNS, "picks": picks}, indent=1) + "\n")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
