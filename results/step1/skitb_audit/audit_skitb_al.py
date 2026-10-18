"""One-off, read-only audit of SkiTB AL. Run with python -I; never imports anything from the data dir."""
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1]) / "AL"
out = Path(sys.argv[2])
seqs = sorted(p.name for p in root.iterdir() if p.is_dir())
problems = []

def lines(p):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

# --- split files
splits = {}
for f in sorted(root.glob("AL_*.json")):
    d = json.loads(f.read_text())
    ids = [s for v in d.values() for s in v]
    splits[f.name] = {k: len(v) for k, v in d.items()}
    if len(ids) != len(set(ids)): problems.append(f"{f.name}: duplicate IDs")
    if set(ids) != set(seqs): problems.append(f"{f.name}: IDs != folders ({len(set(ids))} vs {len(seqs)})")
tvt = json.loads((root / "AL_train_val_test_athlete_60-40.json").read_text())
tt = json.loads((root / "AL_train_test_athlete_60-40.json").read_text())
part = {s: k for k, v in tvt.items() for s in v}
same_test = set(tt["test"]) == set(tvt["test"])
train_is_tv = set(tt["train"]) == set(tvt["train"]) | set(tvt["val"])

# --- metadata
meta = {r["ID"]: r for r in csv.DictReader(open(root / "AL_data.csv", encoding="utf-8-sig"))}
athletes = defaultdict(set)
disc = defaultdict(Counter)
for s, r in meta.items():
    athletes[part[s]].add((r["AthleteName"].strip().lower(), r["AthleteSurname"].strip().lower()))
    disc[part[s]][r["Discipline"]] += 1
overlap = {f"{a}&{b}": len(athletes[a] & athletes[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}

# --- sequences
rows, digest_rows = [], []
pat = re.compile(r"^\d{5}\.jpg$")
for s in seqs:
    d = root / s
    for sub in ("frames", "MC", "SC"):
        if not (d / sub).is_dir(): problems.append(f"{s}: missing {sub}/")
    jpgs = sorted(p.name for p in (d / "frames").iterdir() if p.suffix == ".jpg")
    other = sorted(p.name for p in (d / "frames").iterdir() if p.suffix != ".jpg")
    if other: problems.append(f"{s}: non-jpg in frames/: {other[:3]}")
    badname = [j for j in jpgs if not pat.match(j)]
    if badname: problems.append(f"{s}: frame names not 5-digit: {badname[:3]}")
    mc = {n: lines(d / "MC" / f"{n}.txt") for n in ("boxes", "frames", "visibilities", "cameras")}
    if len({len(v) for v in mc.values()}) != 1: problems.append(f"{s}: MC line counts differ")
    mc_frames = [int(x) for x in mc["frames"]]
    jpg_idx = {int(j[:5]) for j in jpgs}
    if set(mc_frames) != jpg_idx: problems.append(f"{s}: MC frames != frames/ files ({len(set(mc_frames) ^ jpg_idx)} differ)")
    contiguous = mc_frames == list(range(mc_frames[0], mc_frames[0] + len(mc_frames)))
    cams = [int(c) for c in mc["cameras"]]
    sc_dirs = sorted((p for p in (d / "SC").iterdir() if p.is_dir()), key=lambda p: int(p.name))
    if [p.name for p in sc_dirs] != [str(i) for i in range(len(sc_dirs))]: problems.append(f"{s}: SC names not 0..n-1")
    if sorted(set(cams)) != list(range(len(sc_dirs))): problems.append(f"{s}: cameras {sorted(set(cams))} vs {len(sc_dirs)} SC clips")
    mc_box = dict(zip(mc_frames, mc["boxes"]))
    mc_vis = dict(zip(mc_frames, mc["visibilities"]))
    w0 = h0 = None
    sizes = set()
    for c in sc_dirs:
        sc = {n: lines(c / f"{n}.txt") for n in ("boxes", "frames", "visibilities")}
        n = len(sc["frames"])
        if len({len(v) for v in sc.values()}) != 1: problems.append(f"{s}/{c.name}: SC line counts differ")
        fr = [int(x) for x in sc["frames"]]
        if any(f not in mc_box for f in fr): problems.append(f"{s}/{c.name}: SC frames not in MC")
        if fr != sorted(fr) or fr != list(range(fr[0], fr[0] + n)): problems.append(f"{s}/{c.name}: SC frames not contiguous")
        if [cams[mc_frames.index(f)] for f in (fr[0], fr[-1])] != [int(c.name)] * 2: problems.append(f"{s}/{c.name}: camera index mismatch")
        if any(mc_box[f] != b for f, b in zip(fr, sc["boxes"])): problems.append(f"{s}/{c.name}: SC boxes != MC boxes")
        if any(mc_vis[f] != v for f, v in zip(fr, sc["visibilities"])): problems.append(f"{s}/{c.name}: SC vis != MC vis")
        b = [[float(t) for t in re.split(r"[,\s]+", x)] for x in sc["boxes"]]
        vis = [v for v in sc["visibilities"]]
        if set(vis) - {"0", "1"}: problems.append(f"{s}/{c.name}: visibilities {set(vis)}")
        for f in (fr[0], fr[-1]):
            with Image.open(d / "frames" / f"{f:05d}.jpg") as im: sizes.add(im.size)
        W, H = max(sizes)
        rows.append(dict(sequence=s, partition=part[s], discipline=meta[s]["Discipline"], clip=c.name, n_frames=n,
                         first_frame=fr[0], visible_frac=round(vis.count("1") / n, 3), first_visible=vis[0] == "1",
                         nonpos_boxes=sum(1 for x in b if x[2] <= 0 or x[3] <= 0),
                         outside_boxes=sum(1 for x in b if x[0] < 0 or x[1] < 0 or x[0] + x[2] > W or x[1] + x[3] > H),
                         width=W, height=H))
    if len(sizes) != 1: problems.append(f"{s}: frame sizes {sizes}")
    # per-sequence digest: sha256 over sorted (relative path, file sha256)
    h = hashlib.sha256(); nbytes = nfiles = 0
    for p in sorted(x for x in d.rglob("*") if x.is_file() and x.name != ".DS_Store"):
        fh = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{p.relative_to(d).as_posix()}\t{fh}\n".encode()); nbytes += p.stat().st_size; nfiles += 1
    digest_rows.append(dict(sequence=s, partition=part[s], n_files=nfiles, n_bytes=nbytes, sha256=h.hexdigest(),
                            mc_frames=len(mc_frames), mc_contiguous=contiguous, n_sc_clips=len(sc_dirs)))

top = hashlib.sha256()
for r in digest_rows: top.update(f"{r['sequence']}\t{r['sha256']}\n".encode())
for f in sorted(root.glob("AL_*")): top.update(f"{f.name}\t{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())

def write(name, rs):
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
write("skitb_al_clips.csv", rows); write("skitb_al_sequences.csv", digest_rows)

summ = {"n_sequences": len(seqs), "split_files": splits, "athlete_tt_test_equals_tvt_test": same_test,
        "athlete_tt_train_equals_tvt_train_plus_val": train_is_tv, "athlete_overlap": overlap,
        "athletes_per_partition": {k: len(v) for k, v in athletes.items()},
        "discipline_per_partition": {k: dict(v) for k, v in disc.items()},
        "al_tree_sha256": top.hexdigest(), "problems": problems}
by = defaultdict(list)
for r in rows: by[r["partition"]].append(r)
summ["clips_per_partition"] = {k: dict(sequences=len({r["sequence"] for r in v}), clips=len(v),
    frames=sum(r["n_frames"] for r in v), min_len=min(r["n_frames"] for r in v), median_len=sorted(r["n_frames"] for r in v)[len(v) // 2],
    max_len=max(r["n_frames"] for r in v), visible_frac=round(sum(r["visible_frac"] * r["n_frames"] for r in v) / sum(r["n_frames"] for r in v), 3),
    clips_first_frame_invisible=sum(not r["first_visible"] for r in v), nonpos_boxes=sum(r["nonpos_boxes"] for r in v),
    outside_boxes=sum(r["outside_boxes"] for r in v)) for k, v in by.items()}
summ["frame_sizes"] = dict(Counter(f"{r['width']}x{r['height']}" for r in digest_rows for r in [next(x for x in rows if x["sequence"] == r["sequence"])]))
summ["mc_contiguous_all"] = all(r["mc_contiguous"] for r in digest_rows)
summ["files"] = sum(r["n_files"] for r in digest_rows); summ["bytes"] = sum(r["n_bytes"] for r in digest_rows)
(out / "skitb_al_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
