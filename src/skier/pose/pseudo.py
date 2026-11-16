"""Pseudo-labels from a teacher on SkiTB training frames (Step 6b, D29): "noisy student lite".

The teacher is the two E2 models (seeds 230 and 231). For each frame:
- pass 1 crops the start box (the SkiTB GT box, or for calibration a val image's labeled reference
  box) widened by ``K_POSE``, with each model; the two models' points and scores are averaged;
- pass 2 crops the D26 refined box (``refine.refined_boxes`` on the averaged pass-1 prediction),
  with each model; rows without a refined box keep pass 1;
- the pseudo point is the mean of the models' final coordinates.

A body + ski joint is kept only if every model scores it at least ``s_keep`` and the models agree
within ``d_keep`` x the diagonal of the box of the confident points. Poles are never kept. A frame
is kept if at least ``n_keep`` joints survive and they give a valid reference box.

The filter is chosen on Ski 2DPose val labels by the D29 rule (``choose_filter``): among the grid
settings whose kept points are at least 95% correct at PCK@0.05 for body and ski, the one that keeps
the most ski points (ties: higher s_keep, then smaller d_keep); then 90%; else stop.

The core here is numpy only: the models enter as ``Predictor`` callables (``predict_boxes`` around a
heatmap function), so tests use fakes. torch and mmpose are imported only in ``main``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import numpy as np

from skier.data.records import PoseRecord
from skier.geometry import body_ski_mask, box_diagonal, box_from_points, expand_box, reference_box
from skier.integration.refine import N_MIN, refined_boxes
from skier.io_utils import ensure_dir, read_json, sha256_file, write_json
from skier.paths import data_root, repo_root, results_root
from skier.schema import Schema, load_schema

FORMAT = "skier.pseudo_labels.v1"
DATASET = "skitb_pseudo"
PARTITION = "train"  # SkiTB split: pseudo-labels come from training races only (D29)

# --- D29, fixed before any number ------------------------------------------------------------
K_POSE = 1.1  # pass-1 crop: start box x K_POSE, then the model's padding (as for tracker boxes, D24)
S_MIN, K_REFINE = 0.3, 1.3  # pass 2: the D26 refinement setting
S_KEEP_GRID = (0.3, 0.5, 0.7)
D_KEEP_GRID = (0.02, 0.05, 0.1)
N_KEEP = 8  # kept body + ski joints per frame
PRECISION_THR = 0.05  # kept points are scored at PCK@0.05 against the val labels
PRECISION_BARS = (95.0, 90.0)  # percent, body and ski each; tried in order, else stop
MIN_BOX_H = 64.0  # px, SkiTB GT box height
MIN_CLIP_LEN = 10  # frames
TARGET_CANDIDATES = 7500  # frames before filtering; sets the frame step
# SkiTB training races in the Trentino region (AL_data.csv: Madonna di Campiglio x2, Val di Fassa), where the
# Ski 2DPose "Trentino 2013" Universiade videos (2 val, 1 test) were filmed: other years and broadcasts, but
# possibly the same slopes, so they are left out (D29). No other SkiTB training race shares a Ski 2DPose venue.
EXCLUDED_SEQUENCES = ("AL0010", "AL0018", "AL0028")
FILTER_RULE = ("among the grid settings whose kept body and ski points each reach the precision bar (PCK@0.05 vs "
               "the val labels), the one keeping the most ski points; ties: higher s_keep, then smaller d_keep; "
               f"bars {PRECISION_BARS[0]:g}% then {PRECISION_BARS[1]:g}%; if neither is met, stop (no student)")
# ---------------------------------------------------------------------------------------------

Predictor = Callable[[Sequence[np.ndarray], np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Candidate:
    sample_id: str  # "{sequence}/{clip}/{frame:05d}"
    sequence: str
    clip: str
    frame_index: int
    image: str  # relative to $SKIER_DATA, POSIX separators
    gt_box: tuple[float, float, float, float]


def drop_sequences(clips, excluded=EXCLUDED_SEQUENCES) -> tuple[list, list[str]]:
    """``(kept clips, excluded sequence IDs found)``."""
    found = sorted({c.sequence_id for c in clips if c.sequence_id in set(excluded)})
    return [c for c in clips if c.sequence_id not in set(excluded)], found


def candidate_frames(clips, root: Path, min_box_h: float = MIN_BOX_H, min_clip_len: int = MIN_CLIP_LEN,
                     target: int = TARGET_CANDIDATES) -> tuple[list[Candidate], dict]:
    """Every k-th eligible frame of each clip, with one k for all clips (about ``target`` frames in total).

    Eligible: the clip has at least ``min_clip_len`` frames, the skier is visible, and the GT box is at
    least ``min_box_h`` px tall. ``root`` is ``$SKIER_DATA``: image paths are stored relative to it.
    """
    from skier.data.skitb import frame_file

    root = Path(root).resolve()
    eligible, counts = [], {"clips": len(clips), "short_clips": 0, "invisible": 0, "small_box": 0}
    for c in clips:
        if len(c.frame_files) < min_clip_len:
            counts["short_clips"] += 1
            continue
        h = c.boxes_xyxy[:, 3] - c.boxes_xyxy[:, 1]
        ok = c.visible & (h >= min_box_h)
        counts["invisible"] += int((~c.visible).sum())
        counts["small_box"] += int((c.visible & (h < min_box_h)).sum())
        eligible.append((c, np.flatnonzero(ok)))
    n_eligible = sum(len(i) for _, i in eligible)
    k = max(1, round(n_eligible / target)) if target > 0 else 1
    out = []
    for c, idx in eligible:
        rel_dir = PurePosixPath(Path(c.frames_dir).resolve().relative_to(root).as_posix())
        for i in idx[::k]:
            name = frame_file(c.frame_files[i])
            out.append(Candidate(f"{c.sequence_id}/{c.clip_id}/{name[:-4]}", c.sequence_id, c.clip_id,
                                 int(c.frame_files[i]), str(rel_dir / name),
                                 tuple(float(v) for v in c.boxes_xyxy[i])))
    ids = [c.sample_id for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate sample IDs")
    counts.update(eligible=n_eligible, frame_step=k, candidates=len(out))
    return out, counts


def teacher_predict(images: Sequence[np.ndarray], start_boxes, predictors: Sequence[Predictor], schema: Schema,
                    k_pose: float = K_POSE, s_min: float = S_MIN, k_refine: float = K_REFINE,
                    n_min: int = N_MIN) -> dict:
    """Two passes with every teacher model. Returns per-model ``xy [M, N, K, 2]`` and ``scores [M, N, K]``,
    the final ``crop_box [N, 4]`` (the start box x k_pose, or the refined box) and ``refined [N]``."""
    boxes1 = np.stack([expand_box(b, k_pose) for b in np.asarray(start_boxes, dtype=np.float64)])
    p1 = [p(images, boxes1) for p in predictors]
    xy = np.stack([np.asarray(x, dtype=np.float64) for x, _ in p1])
    sc = np.stack([np.asarray(s, dtype=np.float64) for _, s in p1])
    rb = refined_boxes(xy.mean(axis=0), sc.mean(axis=0), schema, s_min, k_refine, n_min)
    refined = np.isfinite(rb).all(axis=1)
    crop = boxes1.copy()
    idx = np.flatnonzero(refined)
    if len(idx):
        sub = [images[i] for i in idx]
        for m, p in enumerate(predictors):
            x2, s2 = p(sub, rb[idx])
            xy[m, idx], sc[m, idx] = x2, s2
        crop[idx] = rb[idx]
    return {"xy": xy, "scores": sc, "crop_box": crop, "refined": refined}


def model_spread(xy: np.ndarray) -> np.ndarray:
    """``[N, K]`` largest distance between any two models' points; inf where a model has no point."""
    xy = np.asarray(xy, dtype=np.float64)
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)  # [M, M, N, K]
    return np.nan_to_num(d.max(axis=(0, 1)), nan=np.inf)


def keep_mask(xy: np.ndarray, scores: np.ndarray, schema: Schema, s_keep: float, d_keep: float,
              n_keep: int = N_KEEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean_xy [N, K, 2], kept [N, K], frame_ok [N])``; ``kept`` is all False on rejected frames."""
    xy = np.asarray(xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    mean = xy.mean(axis=0)
    conf = (body_ski_mask(schema)[None, :] & np.isfinite(xy).all(axis=(0, 3))
            & (np.nan_to_num(scores, nan=-np.inf) >= s_keep).all(axis=0))
    spread = model_spread(xy)
    kept = np.zeros(conf.shape, dtype=bool)
    frame_ok = np.zeros(len(conf), dtype=bool)
    for i in range(len(conf)):
        if conf[i].sum() < 2:
            continue
        diag = box_diagonal(box_from_points(mean[i], conf[i]))
        k_i = conf[i] & (spread[i] <= d_keep * diag)
        if k_i.sum() >= n_keep and reference_box(mean[i], k_i, schema) is not None:
            kept[i], frame_ok[i] = k_i, True
    return mean, kept, frame_ok


def kept_precision(mean_xy: np.ndarray, kept: np.ndarray, records: Sequence[PoseRecord], schema: Schema,
                   thr: float = PRECISION_THR) -> dict:
    """Per group (body, ski): kept points, how many have a reliable label, and how many of those are
    within ``thr`` x the label's reference-box diagonal."""
    from skier.eval.pose_metrics import gt_arrays

    gt, rel, diag, _ = gt_arrays(records)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(np.asarray(mean_xy, dtype=np.float64) - gt, axis=-1)
        close = np.isfinite(dist) & (dist <= thr * diag[:, None])
    out = {}
    for g in ("body", "ski"):
        j = list(schema.groups[g])
        k, r = np.asarray(kept, dtype=bool)[:, j], rel[:, j]
        ver = k & r
        hits = int((ver & close[:, j]).sum())
        out[g] = {"kept": int(k.sum()), "verifiable": int(ver.sum()), "hits": hits, "labels": int(r.sum()),
                  "precision": 100.0 * hits / ver.sum() if ver.sum() else None,
                  "coverage": 100.0 * ver.sum() / r.sum() if r.sum() else None}
    return out


def calibration_table(xy, scores, records, schema: Schema, s_grid=S_KEEP_GRID, d_grid=D_KEEP_GRID,
                      n_keep: int = N_KEEP, thr: float = PRECISION_THR) -> list[dict]:
    """One row per grid setting, plus a ``sanity`` row with no filter (every finite body + ski point kept)."""
    rows = []
    for s in s_grid:
        for d in d_grid:
            mean, kept, ok = keep_mask(xy, scores, schema, s, d, n_keep)
            rows.append({"s_keep": s, "d_keep": d, "n_keep": n_keep, "frames_kept": int(ok.sum()),
                         **kept_precision(mean, kept, records, schema, thr)})
    mean, kept, ok = keep_mask(xy, scores, schema, -np.inf, np.inf, 0)
    rows.append({"sanity": True, "s_keep": None, "d_keep": None, "n_keep": 0, "frames_kept": int(ok.sum()),
                 **kept_precision(mean, kept, records, schema, thr)})
    return rows


def choose_filter(table: Sequence[dict], bars: Sequence[float] = PRECISION_BARS) -> dict:
    """The D29 rule on a calibration table (the sanity row never counts)."""
    rows = [r for r in table if not r.get("sanity")]
    for bar in bars:
        ok = [r for r in rows if all(r[g]["precision"] is not None and r[g]["precision"] >= bar
                                     for g in ("body", "ski"))]
        if ok:
            r = max(ok, key=lambda r: (r["ski"]["kept"], r["s_keep"], -r["d_keep"]))
            return {"stop": False, "bar": bar, "s_keep": r["s_keep"], "d_keep": r["d_keep"], "n_keep": r["n_keep"],
                    "rule": FILTER_RULE}
    return {"stop": True, "bar": None, "rule": FILTER_RULE}


def _num(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def frame_entries(cands: Sequence[Candidate], sizes, mean_xy, kept, frame_ok, crop_box, refined) -> list[dict]:
    """JSON entries for the kept frames (NaN written as null)."""
    out = []
    for i in np.flatnonzero(frame_ok):
        c = cands[i]
        out.append({"sample_id": c.sample_id, "image": c.image, "width": int(sizes[i][0]), "height": int(sizes[i][1]),
                    "sequence": c.sequence, "clip": c.clip, "frame_index": c.frame_index,
                    "gt_box": [float(v) for v in c.gt_box], "crop_box": [_num(v) for v in crop_box[i]],
                    "refined": bool(refined[i]),
                    "xy": [[_num(x), _num(y)] if kept[i, j] else [None, None] for j, (x, y) in enumerate(mean_xy[i])],
                    "kept": [int(v) for v in kept[i]]})
    return out


def _check_rel(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path or (len(path) > 1 and path[1] == ":"):
        raise ValueError(f"pseudo-label image path must be relative to $SKIER_DATA without '..': {path!r}")
    return p


def load_pseudo_records(path: str | Path, root: str | Path, schema: Schema, split_sha256: str) -> list[PoseRecord]:
    """Pseudo-labeled frames as training records (``dataset="skitb_pseudo"``, ``reliable`` = kept)."""
    d = read_json(path)
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    if d.get("skitb_partition") != PARTITION:
        raise ValueError(f"{path}: pseudo-labels must come from the SkiTB {PARTITION!r} split")
    teachers = d.get("teachers") or []
    if not teachers or any(t.get("split_sha256") != split_sha256 for t in teachers):
        raise ValueError(f"{path}: the teachers were trained on a different pose split")
    root = Path(root)
    records, seen = [], set()
    for f in d["frames"]:
        sid = f["sample_id"]
        if sid in seen:
            raise ValueError(f"{path}: duplicate sample {sid}")
        seen.add(sid)
        img = root / Path(*_check_rel(f["image"]).parts)
        if not img.is_file():
            raise FileNotFoundError(f"{path}: image {img} is missing")
        kept = np.array(f["kept"], dtype=bool)
        xy = np.array([[np.nan if v is None else v for v in p] for p in f["xy"]], dtype=np.float64)
        if kept.shape != (schema.num_keypoints,) or xy.shape != (schema.num_keypoints, 2):
            raise ValueError(f"{path}: {sid} has the wrong number of keypoints")
        if not np.isfinite(xy[kept]).all():
            raise ValueError(f"{path}: {sid} has a kept point without coordinates")
        box = reference_box(xy, kept, schema)
        if box is None:
            raise ValueError(f"{path}: {sid} has no valid reference box")
        seq, clip, image_id = sid.split("/")
        records.append(PoseRecord(sample_id=sid, dataset=DATASET, source_video_id=seq, segment_id=clip,
                                  image_id=image_id, frame_index=int(f["frame_index"]), image_path=img,
                                  image_width=int(f["width"]), image_height=int(f["height"]), keypoints_xy=xy,
                                  reliable=kept, raw_visibility=kept.astype(int), reference_box_xyxy=box,
                                  split="skitb_" + PARTITION))
    return records


def pseudo_per_epoch(n_real: int, n_pseudo: int, ratio: float = 1.0) -> int:
    """Pseudo frames drawn per epoch: ``ratio`` x the real images (D29: 1), at most all of them."""
    return min(n_pseudo, int(round(ratio * n_real)))


def epoch_order(n_real: int, n_pseudo: int, n_pe: int, seed: int, epoch: int) -> np.ndarray:
    """Dataset indices for one epoch: every real index ``0..n_real-1`` once plus ``n_pe`` distinct pseudo
    indices (``n_real..n_real+n_pseudo-1``), shuffled. The stream ``[seed, epoch, 0, 1]`` ends in a nonzero
    word, so it never equals a record's augmentation stream ``[seed, epoch, index]``."""
    rng = np.random.default_rng([seed, epoch, 0, 1])
    pick = n_real + rng.choice(n_pseudo, size=min(n_pe, n_pseudo), replace=False)
    return rng.permutation(np.concatenate([np.arange(n_real), pick]))


# --- running the teacher (torch from here on, inside the functions) ---------------------------

def load_teacher(spec: str, hrnet_config: Path, schema: Schema, device: str, split_sha256: str):
    """``NAME=PATH`` -> ``(predictor, info)``."""
    from skier.pose.infer import predict_boxes
    from skier.pose.model import build_hrnet, torch_heatmap_fn
    from skier.pose.train import load_checkpoint

    name, _, path = spec.partition("=")
    if not path:
        raise ValueError(f"--teacher expects NAME=PATH, got {spec!r}")
    model, _ = build_hrnet(schema.num_keypoints, hrnet_config, None, device=device)
    ck = load_checkpoint(Path(path), model, schema=schema, map_location=device)
    if ck.get("split_sha256") != split_sha256:
        raise ValueError(f"teacher {name}: trained on a different pose split")
    expand = float(ck.get("config", {}).get("expand", 1.25))
    fn = torch_heatmap_fn(model, device)
    info = {"name": name, "path": str(path), "sha256": sha256_file(path), "epoch": ck.get("epoch"),
            "split_sha256": ck.get("split_sha256"), "expand": expand}
    return (lambda imgs, boxes: predict_boxes(imgs, boxes, fn, expand)), info


def run_teacher(paths: Sequence[Path], start_boxes, predictors, schema: Schema, batch_size: int, log=print) -> dict:
    """``teacher_predict`` over all images in batches; adds ``sizes [N, 2]`` (w, h) and timing."""
    import cv2

    parts, sizes, t0 = [], [], time.perf_counter()
    for s in range(0, len(paths), batch_size):
        imgs = []
        for p in paths[s:s + batch_size]:
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"cannot read {p}")
            imgs.append(img)
            sizes.append((img.shape[1], img.shape[0]))
        parts.append(teacher_predict(imgs, start_boxes[s:s + batch_size], predictors, schema))
        if (s // batch_size) % 50 == 0:
            log(f"  {s + len(imgs)}/{len(paths)} frames, {time.perf_counter() - t0:.0f} s")
    out = {k: np.concatenate([p[k] for p in parts], axis=1 if k in ("xy", "scores") else 0) for k in parts[0]}
    out["sizes"] = np.array(sizes)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def overlay_sheet(cands, raw, mean_xy, kept, frame_ok, root: Path, schema: Schema, n: int, seed: int = 230):
    """A contact sheet of ``n`` kept frames (white: GT box, yellow dashed: final crop box; kept points)."""
    import cv2

    from skier.viz import contact_sheet, draw_box, draw_edges, draw_points

    idx = np.flatnonzero(frame_ok)
    if not len(idx) or n <= 0:
        return None
    pick = np.sort(np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False))
    tiles = []
    for i in pick:
        img = cv2.imread(str(root / cands[i].image))
        draw_box(img, cands[i].gt_box, (255, 255, 255))
        draw_box(img, raw["crop_box"][i], (0, 255, 255), dashed=True)
        draw_edges(img, mean_xy[i], kept[i], schema)
        draw_points(img, mean_xy[i], kept[i], schema, label_indices=False)
        x1, y1, x2, y2 = cands[i].gt_box
        pad = 0.6 * max(x2 - x1, y2 - y1)
        h, w = img.shape[:2]
        tiles.append(img[max(0, int(y1 - pad)):min(h, int(y2 + pad)), max(0, int(x1 - pad)):min(w, int(x2 + pad))])
    return contact_sheet(tiles, cols=6, thumb_w=320)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pseudo-labels from the E2 teacher on SkiTB training frames (Step 6b, "
                                            "D29); --calibrate chooses the filter on Ski 2DPose val labels.")
    p.add_argument("--teacher", action="append", required=True, metavar="NAME=PATH", help="repeat per teacher model")
    p.add_argument("--hrnet-config", type=Path, required=True)
    p.add_argument("--split", type=Path, help="default: data/manifests/pose_split_v1.json")
    p.add_argument("--calibrate", action="store_true", help="choose the filter on Ski 2DPose val (D29)")
    p.add_argument("--filter", type=Path, help="calibration.json (generate mode)")
    p.add_argument("--skitb-root", type=Path, help="default: $SKIER_DATA/skitb")
    p.add_argument("--skitb-split", type=Path, help="default: <skitb-root>/AL/AL_train_val_test_athlete_60-40.json")
    p.add_argument("--max-clips", type=int, help="trial: only the first N training clips")
    p.add_argument("--overlays", type=int, default=24, help="kept frames on the overlay contact sheet")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pseudo/{calibration,labels}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.calibrate and args.filter is None:
        print("error: generate mode needs --filter calibration.json (run --calibrate first)", file=sys.stderr)
        return 1
    schema = load_schema()
    split_path = args.split or repo_root() / "data/manifests/pose_split_v1.json"
    manifest = read_json(split_path)
    try:
        teachers = [load_teacher(t, args.hrnet_config, schema, args.device, manifest["content_sha256"])
                    for t in args.teacher]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len({t[1]["expand"] for t in teachers}) != 1:
        print("error: the teachers use different crop padding", file=sys.stderr)
        return 1
    predictors, infos = [t[0] for t in teachers], [t[1] for t in teachers]
    from skier.envcheck import _git_commit

    if args.calibrate:
        from skier.data.ski2dpose import load_records
        from skier.pose.eval_generic import select_partition

        s2d = data_root() / "ski2dpose"
        recs = select_partition(load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                                             s2d / "archives/ski2dpose_info.json", schema=schema), manifest, "val")[0]
        raw = run_teacher([r.image_path for r in recs], np.stack([r.reference_box_xyxy for r in recs]), predictors,
                          schema, args.batch_size)
        table = calibration_table(raw["xy"], raw["scores"], recs, schema)
        choice = choose_filter(table)
        out = ensure_dir(args.out or results_root() / "pseudo" / "calibration")
        res = {"rule": FILTER_RULE, "s_keep_grid": S_KEEP_GRID, "d_keep_grid": D_KEEP_GRID, "n_keep": N_KEEP,
               "precision_thr": PRECISION_THR, "settings": {"k_pose": K_POSE, "s_min": S_MIN, "k_refine": K_REFINE,
                                                            "n_min": N_MIN},
               "partition": "val", "n_images": len(recs), "split_sha256": manifest["content_sha256"],
               "teachers": infos, "git_commit": _git_commit(), "refined_share": float(raw["refined"].mean()),
               "table": table, "choice": choice}
        write_json(res, out / "calibration.json")
        np.savez(out / "raw_predictions.npz", sample_ids=np.array([r.sample_id for r in recs]),
                 **{k: raw[k] for k in ("xy", "scores", "crop_box", "refined")})
        for r in table:
            tag = "sanity (no filter)" if r.get("sanity") else f"s_keep {r['s_keep']:g} d_keep {r['d_keep']:g}"
            print(f"{tag:<24} body {r['body']['precision'] or 0:5.1f}% of {r['body']['verifiable']:>5}  "
                  f"ski {r['ski']['precision'] or 0:5.1f}% of {r['ski']['verifiable']:>5} (kept {r['ski']['kept']})  "
                  f"frames {r['frames_kept']}")
        print(f"choice: {choice} -> {out}")
        return 0

    from skier.data.skitb import load_skitb_clips

    cal = read_json(args.filter)
    flt = cal["choice"]
    if flt.get("stop"):
        print("error: the calibration says stop (no setting reached the precision bars; D29)", file=sys.stderr)
        return 1
    if [t["sha256"] for t in cal["teachers"]] != [t["sha256"] for t in infos]:
        print("error: the calibration was made with different teacher checkpoints", file=sys.stderr)
        return 1
    root = data_root()
    sk_root = args.skitb_root or root / "skitb"
    sk_split = args.skitb_split or sk_root / "AL/AL_train_val_test_athlete_60-40.json"
    clips, dropped = drop_sequences(load_skitb_clips(sk_root, sk_split, PARTITION))
    if dropped != sorted(EXCLUDED_SEQUENCES):
        print(f"error: expected to leave out {sorted(EXCLUDED_SEQUENCES)} (D29), found {dropped}", file=sys.stderr)
        return 1
    if args.max_clips:
        clips = clips[:args.max_clips]
    cands, counts = candidate_frames(clips, root)
    counts["excluded_sequences"] = dropped
    print(f"{len(clips)} SkiTB {PARTITION} clips -> {len(cands)} candidate frames (step {counts['frame_step']})")
    raw = run_teacher([root / c.image for c in cands], np.array([c.gt_box for c in cands]), predictors, schema,
                      args.batch_size)
    mean, kept, ok = keep_mask(raw["xy"], raw["scores"], schema, flt["s_keep"], flt["d_keep"], flt["n_keep"])
    frames = frame_entries(cands, raw["sizes"], mean, kept, ok, raw["crop_box"], raw["refined"])
    out = ensure_dir(args.out or results_root() / "pseudo" / "labels")
    counts.update(kept_frames=len(frames), kept_points=int(kept.sum()),
                  kept_ski_points=int(kept[:, list(schema.groups["ski"])].sum()))
    header = {"format": FORMAT, "git_commit": _git_commit(), "skitb_partition": PARTITION,
              "skitb_split": {"path": str(sk_split), "sha256": sha256_file(sk_split)}, "teachers": infos,
              "split_sha256": manifest["content_sha256"],
              "settings": {"k_pose": K_POSE, "s_min": S_MIN, "k_refine": K_REFINE, "n_min": N_MIN,
                           "min_box_h": MIN_BOX_H, "min_clip_len": MIN_CLIP_LEN, "target": TARGET_CANDIDATES,
                           "max_clips": args.max_clips},
              "filter": {**{k: flt[k] for k in ("s_keep", "d_keep", "n_keep", "bar")},
                         "calibration_sha256": sha256_file(args.filter)},
              "counts": counts}
    write_json({**header, "frames": frames}, out / "pseudo_labels.json")
    np.savez(out / "raw_predictions.npz", sample_ids=np.array([c.sample_id for c in cands]),
             **{k: raw[k] for k in ("xy", "scores", "crop_box", "refined", "sizes")})
    per_seq: dict = {}
    for i, c in enumerate(cands):
        s = per_seq.setdefault(c.sequence, {"candidates": 0, "kept": 0})
        s["candidates"] += 1
        s["kept"] += int(ok[i])
    write_json({**header, "seconds": raw["seconds"], "refined_share": float(raw["refined"].mean()),
                "kept_per_joint": {schema.names[j]: int(kept[:, j].sum()) for j in range(schema.num_keypoints)},
                "per_sequence": per_seq}, out / "summary.json")
    sheet = overlay_sheet(cands, raw, mean, kept, ok, root, schema, args.overlays)
    if sheet is not None:
        import cv2

        cv2.imwrite(str(out / "overlays.jpg"), sheet)
    print(f"kept {len(frames)} of {len(cands)} frames ({counts['kept_ski_points']} ski points) in "
          f"{raw['seconds']} s -> {out}")
    return 0
