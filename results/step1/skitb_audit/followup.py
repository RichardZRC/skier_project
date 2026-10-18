import csv, json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1]) / "AL"; out = Path(sys.argv[2])
tvt = json.loads((root / "AL_train_val_test_athlete_60-40.json").read_text())
part = {s: k for k, v in tvt.items() for s in v}
meta = {r["ID"]: r for r in csv.DictReader(open(root / "AL_data.csv", encoding="utf-8-sig"))}
by = {}
for s, r in meta.items():
    by.setdefault((r["AthleteName"].strip(), r["AthleteSurname"].strip()), []).append((part[s], s, r["AthleteGender"], r["AthleteNat"]))
print("athletes in >1 partition:")
for k, v in by.items():
    if len({p for p, *_ in v}) > 1: print("  ", k, v)
clips = list(csv.DictReader(open(out / "skitb_al_clips.csv")))
for p in ("train", "val", "test"):
    n = [int(c["n_frames"]) for c in clips if c["partition"] == p]
    print(p, "clips <=1:", sum(x <= 1 for x in n), " <10:", sum(x < 10 for x in n), " <25:", sum(x < 25 for x in n), " <50:", sum(x < 50 for x in n), "of", len(n))
print("fps estimate = (last MC frame - first MC frame) / AthleteTime:")
est = []
for s in sorted(meta):
    fr = [int(l) for l in (root / s / "MC" / "frames.txt").read_text().split()]
    a = meta[s]["AthleteTime"].strip()
    t = (60 * float(a.split(":")[0]) + float(a.split(":")[1])) if ":" in a else (float(a) if a else None)
    if t: est.append(((fr[-1] - fr[0]) / t, s))
est.sort()
print("  n", len(est), "min", [f"{e:.1f} {s}" for e, s in est[:4]], "max", [f"{e:.1f} {s}" for e, s in est[-4:]])
print("  histogram (rounded to 5):", sorted(Counter(5 * round(e / 5) for e, _ in est).items()))
print("channels:", Counter(r["YouTubeChannel"] for r in meta.values()).most_common(6))
