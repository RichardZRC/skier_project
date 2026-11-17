"""Near-duplicate check between the D29 pseudo-label source frames and the Ski 2DPose evaluation images (D30).

The rule (D30, written before any distance was computed):
- reference set: every frame of every clip of the 50 SkiTB training races used in D29 (so shared footage would match
  within one frame); each frame is marked as a kept pseudo-label frame or not;
- queries: all Ski 2DPose val, test and follow-cam images, hashed and never displayed;
- 64-bit DCT perceptual hash of the full frame read at 1/4 scale (``skier.data.dedup``); Hamming distance;
- controls first: re-encode (each query vs. itself resized to 1280x720 and JPEG q75), adjacent frames (2,000 random
  same-clip SkiTB pairs 1 frame apart), null (2,000 random reference frames vs. their nearest other-race frame);
- T = max(99th percentile of re-encode, 95th percentile of adjacent); a query is flagged if its nearest reference is
  <= T; a null rate (null frames <= T) above 5% means the hash can't separate footage at T.

Usage (GPU machine, inside Ubuntu): python results/step6b/near_duplicates.py [--out results/step6b/near_duplicates]
Reference hashes are cached in $SKIER_RESULTS/pseudo/near_dup/ (the cache is keyed by the list of frame paths).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.data.dedup import N_BITS, hamming, nearest, phash_bytes, phash_file  # noqa: E402
from skier.io_utils import ensure_dir, read_json, sha256_file, write_json  # noqa: E402
from skier.paths import data_root, repo_root, results_root  # noqa: E402

SEED = 230
N_CONTROL = 2000
REDUCE = 4
REENCODE_SIZE, REENCODE_QUALITY = (1280, 720), 75
P_REENCODE, P_ADJACENT, MAX_NULL_RATE = 99.0, 95.0, 0.05
N_CLOSEST = 10  # listed per partition
PARTITIONS = ("val", "test", "followcam")


def hash_all(paths, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.stack(list(ex.map(lambda p: phash_file(p, REDUCE), paths)))


def reencoded_hash(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, REENCODE_SIZE, interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, REENCODE_QUALITY])
    if not ok:
        raise ValueError(f"cannot re-encode {path}")
    return phash_bytes(buf, REDUCE)


def quantiles(d) -> dict:
    d = np.asarray(d)
    return {"n": int(len(d)), "min": int(d.min()), "max": int(d.max()), "mean": float(d.mean()),
            **{f"p{q:g}": float(np.percentile(d, q)) for q in (1, 5, 50, 95, 99)}}


def histogram(d) -> dict:
    d = np.asarray(d)
    return {str(k): int((d == k).sum()) for k in range(int(d.max()) + 1) if (d == k).any()}


def reference_frames(root: Path):
    """``(paths, sequence per frame, clip key per frame, frame position in its clip)`` for the 50 D29 races."""
    from skier.data.skitb import frame_file, load_skitb_clips
    from skier.pose.pseudo import EXCLUDED_SEQUENCES, PARTITION, drop_sequences

    sk = root / "skitb"
    clips, dropped = drop_sequences(load_skitb_clips(sk, sk / "AL/AL_train_val_test_athlete_60-40.json", PARTITION))
    if dropped != sorted(EXCLUDED_SEQUENCES):
        raise ValueError(f"expected to leave out {sorted(EXCLUDED_SEQUENCES)} (D29), found {dropped}")
    paths, seqs, clip_keys, pos, seen = [], [], [], [], set()
    for c in clips:
        for t, entry in enumerate(c.frame_files):
            p = Path(c.frames_dir) / frame_file(entry)
            if p in seen:
                continue
            seen.add(p)
            paths.append(p)
            seqs.append(c.sequence_id)
            clip_keys.append(f"{c.sequence_id}/{c.clip_id}")
            pos.append(t)
    return paths, np.array(seqs), np.array(clip_keys), np.array(pos), len({c.sequence_id for c in clips})


def cached_ref_hashes(paths, root: Path, cache_dir: Path) -> np.ndarray:
    key = hashlib.sha256("\n".join(str(p.relative_to(root)) for p in paths).encode()).hexdigest()
    f = cache_dir / "ref_hashes.npz"
    if f.is_file():
        with np.load(f) as z:
            if str(z["key"]) == key:
                return z["hashes"]
    t0 = time.perf_counter()
    h = hash_all(paths)
    np.savez(ensure_dir(cache_dir) / "ref_hashes.npz", key=key, hashes=h)
    print(f"hashed {len(paths)} reference frames in {time.perf_counter() - t0:.0f} s")
    return h


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=repo_root() / "results/step6b/near_duplicates")
    a = p.parse_args(argv)
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition
    from skier.schema import load_schema

    root = data_root()
    rng = np.random.default_rng(SEED)
    # --- references ------------------------------------------------------------------------
    paths, seqs, clip_keys, pos, n_races = reference_frames(root)
    rel = [str(q.relative_to(root)) for q in paths]
    labels = results_root() / "pseudo/labels/pseudo_labels.json"
    kept_images = {f["image"] for f in read_json(labels)["frames"]}
    kept = np.array([r in kept_images for r in rel])
    if kept.sum() != len(kept_images):
        raise ValueError(f"{len(kept_images) - int(kept.sum())} kept pseudo-label frames are not among the references")
    refs = cached_ref_hashes(paths, root, results_root() / "pseudo/near_dup")
    print(f"{len(paths)} reference frames from {n_races} races ({int(kept.sum())} kept pseudo-label frames)")
    # --- queries (hashed only) ---------------------------------------------------------------
    schema = load_schema()
    s2d = root / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    manifest = read_json(repo_root() / "data/manifests/pose_split_v1.json")
    queries = {part: select_partition(records, manifest, part)[0] for part in PARTITIONS}
    q_recs = [r for part in PARTITIONS for r in queries[part]]
    q_part = np.array([part for part in PARTITIONS for _ in queries[part]])
    q_hash = hash_all([r.image_path for r in q_recs])
    sizes = sorted({(r.image_width, r.image_height) for r in q_recs})
    # --- controls and the threshold (before any query-to-reference distance) -------------------
    reenc = hamming(q_hash, np.stack([reencoded_hash(r.image_path) for r in q_recs]))
    nxt = np.flatnonzero((clip_keys[:-1] == clip_keys[1:]) & (pos[1:] == pos[:-1] + 1))
    pick = rng.choice(nxt, size=min(N_CONTROL, len(nxt)), replace=False)
    adjacent = hamming(refs[pick], refs[pick + 1])
    null_idx = rng.choice(len(refs), size=N_CONTROL, replace=False)
    null_d, _ = nearest(refs[null_idx], refs, seqs[null_idx], seqs)
    T = max(float(np.percentile(reenc, P_REENCODE)), float(np.percentile(adjacent, P_ADJACENT)))
    null_rate = float((null_d <= T).mean())
    print(f"T = {T:g} (re-encode p{P_REENCODE:g} {np.percentile(reenc, P_REENCODE):g}, adjacent p{P_ADJACENT:g} "
          f"{np.percentile(adjacent, P_ADJACENT):g}); null rate {null_rate:.4f}")
    # --- queries vs. references ---------------------------------------------------------------
    d_all, i_all = nearest(q_hash, refs)
    d_kept, _ = nearest(q_hash, refs[kept])
    res = {"rule": "D30", "seed": SEED, "reduce": REDUCE, "hash_bits": N_BITS,
           "reference": {"races": n_races, "frames": len(paths), "kept_pseudo_label_frames": int(kept.sum()),
                         "pseudo_labels_sha256": sha256_file(labels)},
           "query_image_sizes": [list(s) for s in sizes],
           "controls": {"reencode": quantiles(reenc), "adjacent_frames": quantiles(adjacent),
                        "null_nearest_other_race": quantiles(null_d)},
           "threshold": T, "null_rate": null_rate, "null_rate_max": MAX_NULL_RATE,
           "hash_separates_footage": null_rate <= MAX_NULL_RATE, "partitions": {}}
    separates = res["hash_separates_footage"]

    def entry(k):
        return {"sample_id": q_recs[k].sample_id, "video": q_recs[k].source_video_id, "distance": int(d_all[k]),
                "nearest_frame": rel[i_all[k]], "nearest_is_kept_pseudo_label": bool(kept[i_all[k]])}

    for part in PARTITIONS:
        m = q_part == part
        idx = np.flatnonzero(m)
        flagged = [k for k in idx if d_all[k] <= T]
        res["partitions"][part] = {
            "n_images": int(m.sum()), "nearest_any_frame": quantiles(d_all[m]), "histogram_any_frame": histogram(d_all[m]),
            "nearest_kept_frame": quantiles(d_kept[m]), "n_flagged": len(flagged),
            "n_flagged_kept_frames": int((d_kept[m] <= T).sum()),
            # every flag is listed only when the hash separates footage; otherwise the flags carry no information
            "flagged": [entry(k) for k in flagged] if separates else None,
            "closest": [entry(k) for k in idx[np.argsort(d_all[idx], kind="stable")[:N_CLOSEST]]]}
    if separates:
        verdict = ("no near-duplicates found" if not any(v["n_flagged"] for v in res["partitions"].values())
                   else "flagged images to review (D30)")
    else:
        verdict = "weak evidence: the hash does not separate footage at T (null rate above 5%)"
    res["verdict"] = verdict
    out = ensure_dir(a.out)
    write_json(res, out / "near_duplicates.json")
    c = res["controls"]
    L = ["# Near-duplicate check: D29 pseudo-label source frames vs. Ski 2DPose evaluation images (D30)", "",
         f"Reference: every frame of the {n_races} SkiTB training races used in D29 ({len(paths):,} frames; "
         f"{int(kept.sum()):,} were kept pseudo-label frames). Queries are hashed, never displayed.", "",
         f"64-bit DCT perceptual hash, Hamming distance. Query image sizes: {', '.join(f'{w}x{h}' for w, h in sizes)}.",
         "", "| Control | n | min | median | p95 | p99 | max |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, k in (("re-encode (query vs. itself at 1280x720, JPEG q75)", "reencode"),
                    ("adjacent SkiTB frames (same clip, 1 frame apart)", "adjacent_frames"),
                    ("null: nearest frame from another race", "null_nearest_other_race")):
        q = c[k]
        L.append(f"| {name} | {q['n']} | {q['min']} | {q['p50']:g} | {q['p95']:g} | {q['p99']:g} | {q['max']} |")
    L += ["", f"**Threshold T = {T:g}** (max of re-encode p99 and adjacent p95). Null rate at T: {null_rate:.2%} "
              f"(limit {MAX_NULL_RATE:.0%}).", "",
          "| Partition | images | nearest frame: min | median | images ≤ T | nearest kept frame: min | images ≤ T |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for part, v in res["partitions"].items():
        L.append(f"| {part} | {v['n_images']} | {v['nearest_any_frame']['min']} | {v['nearest_any_frame']['p50']:g} | "
                 f"{v['n_flagged']} | {v['nearest_kept_frame']['min']} | {v['n_flagged_kept_frames']} |")
    L += ["", f"**Verdict: {verdict}.**", ""]
    for part, v in res["partitions"].items():
        listed = v["flagged"] if separates else v["closest"]
        L.append(f"{'Flagged' if separates else f'The {N_CLOSEST} closest'} ({part}):")
        for f in listed:
            L.append(f"- {f['sample_id']} → {f['nearest_frame']} (distance {f['distance']}, "
                     f"kept pseudo-label frame: {f['nearest_is_kept_pseudo_label']})")
        L.append("")
    (out / "near_duplicates.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
