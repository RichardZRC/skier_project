"""Per-sequence SkiTB AL frame-rate estimate: (last - first MC frame index) / official AthleteTime (as in the Mac
audit's followup.py), snapped to the paper's {25, 30} FPS. Prints the gap between the two groups. Metadata only."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "AL"
out = Path(sys.argv[2])
split = json.loads((root / "AL_train_val_test_athlete_60-40.json").read_text())
part = {s: k for k, v in split.items() for s in v}
meta = {r["ID"]: r for r in csv.DictReader(open(root / "AL_data.csv", encoding="utf-8-sig"))}
rows = []
for s in sorted(meta):
    fr = [int(x) for x in (root / s / "MC" / "frames.txt").read_text().split()]
    a = meta[s]["AthleteTime"].strip()
    t = (60 * float(a.split(":")[0]) + float(a.split(":")[1])) if ":" in a else (float(a) if a else None)
    est = (fr[-1] - fr[0]) / t if t else None
    snap = None if est is None else (25 if abs(est - 25) < abs(est - 30) else 30)
    rows.append({"sequence": s, "partition": part.get(s, ""), "mc_span_frames": fr[-1] - fr[0],
                 "athlete_time_s": t, "fps_estimate": round(est, 2) if est else "", "fps_assigned": snap or ""})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

e = sorted((r["fps_estimate"], r["sequence"], r["partition"]) for r in rows if r["fps_estimate"] != "")
low = [x for x in e if x[0] < 27.5]
high = [x for x in e if x[0] >= 27.5]
print(f"{len(e)} sequences with an estimate; missing: {[r['sequence'] for r in rows if r['fps_estimate'] == '']}")
print(f"near 25: {len(low)}, range {low[0][0]}-{low[-1][0]}; near 30: {len(high)}, range {high[0][0]}-{high[-1][0]}")
print(f"gap between the groups: {low[-1][0]} ({low[-1][1]}) .. {high[0][0]} ({high[0][1]})")
for p in ("val", "test"):
    sel = [r for r in rows if r["partition"] == p]
    print(f"{p}: {len(sel)} sequences, assigned 25: {sum(r['fps_assigned'] == 25 for r in sel)}, "
          f"30: {sum(r['fps_assigned'] == 30 for r in sel)}; closest to 27.5: "
          f"{sorted(sel, key=lambda r: abs(r['fps_estimate'] - 27.5))[0]['sequence']} "
          f"({sorted(sel, key=lambda r: abs(r['fps_estimate'] - 27.5))[0]['fps_estimate']})")
print("val:", [(r["sequence"], r["fps_estimate"], r["fps_assigned"]) for r in rows if r["partition"] == "val"])
print("outliers (>32 or <24):", [(x[1], x[0], x[2]) for x in e if x[0] > 32 or x[0] < 24])
