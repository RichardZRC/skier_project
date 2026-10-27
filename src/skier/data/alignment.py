"""Match labeled images to video frames (guide 1G, code spec §8).

For each labeled image, compare it with decoded frames ``t - window .. t + window``
(``t = start_frame + frame_idx``) using normalized cross-correlation on small
grayscale probes, and report the best offset and how clear the match is.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from skier.data.ski2dpose_raw import annotation_array, iter_raw_labels, safe_name
from skier.io_utils import ensure_dir, read_csv, read_json, write_csv, write_json
from skier.paths import data_root, results_root
from skier.schema import Schema, load_schema
from skier.viz import FONT, render_gt_overlay

# --- FILLED IN AT 1G (2026-10-26, laptop; code spec §13; docs/data_audit_ski2dpose.md §1G) ------
# On 5UHRvqx1iuQ/0 and oKQFABiOTw8/22 all 94 labeled images match their frame exactly (NCC
# 0.9999-1.0, offset 0): the images are frames of these clips. In near-static stretches the
# neighbouring frames score almost as high, so the margin rule leaves those unverified even
# though the best match is the exact frame.
MIN_SCORE = 0.99
MIN_MARGIN = 0.02


def locate_segment(videos_root: Path, video_id: str, segment_id: str, info: Any = None) -> tuple[Path, int]:
    """Return (video file, decode index of the segment's frame_idx 0).

    The videos zip holds one clip per segment at ``Videos/{video_id}/{segment_id}.mp4``
    (158 files). ``frame_idx`` is the 0-based decode index within that clip: confirmed in
    1G (2026-10-26) on two pilot segments, offset 0 for all 94 labeled images.
    """
    path = Path(videos_root) / "Videos" / video_id / f"{segment_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"no clip for {video_id}/{segment_id}: {path}")
    return path, 0
# -----------------------------------------------------------------------------

PROBE_SIZE = (160, 90)  # (w, h)
ALIGN_COLUMNS = [
    "video_id", "segment_id", "image_id", "label_frame_idx", "best_decode_idx",
    "offset", "best_score", "margin", "verified",
]


def to_probe(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale, 160x90 (INTER_AREA), float64, zero mean, unit std (all zeros if constant)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    probe = cv2.resize(gray, PROBE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float64)
    probe -= probe.mean()
    std = probe.std()
    return probe / std if std > 1e-9 else np.zeros_like(probe)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two probes, in [-1, 1]; 0 if either is constant."""
    return float(np.mean(a * b))


def _decode_probes(video_path: Path, needed: set[int]) -> tuple[dict[int, np.ndarray], int]:
    """Decode sequentially up to the largest needed index; keep probes for needed frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    probes: dict[int, np.ndarray] = {}
    last = max(needed) if needed else -1
    idx = 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in needed:
                probes[idx] = to_probe(frame)
            idx += 1
    finally:
        cap.release()
    return probes, idx


def align_segment(
    video_path: str | Path,
    start_frame: int,
    labeled: list[tuple[str, int, Path]],
    window: int,
) -> list[dict]:
    """``labeled`` = [(image_id, frame_idx, image_path)]. One result row per image."""
    targets = [(image_id, int(fi), Path(p), start_frame + int(fi)) for image_id, fi, p in labeled]
    needed = {k for *_, t in targets for k in range(max(0, t - window), t + window + 1)}
    probes, _ = _decode_probes(Path(video_path), needed)

    rows = []
    for image_id, frame_idx, path, t in targets:
        row = {"image_id": image_id, "label_frame_idx": frame_idx, "verified": False}
        img = cv2.imread(str(path))
        if img is None:
            print(f"warning: cannot read {path}", file=sys.stderr)
            rows.append(row)
            continue
        probe = to_probe(img)
        scores = [(k, ncc(probe, probes[k])) for k in range(max(0, t - window), t + window + 1) if k in probes]
        if not scores:  # every candidate frame is past the end of the video
            rows.append(row)
            continue
        scores.sort(key=lambda ks: -ks[1])  # stable: ties keep the earlier frame
        best_k, best_s = scores[0]
        margin = best_s - scores[1][1] if len(scores) > 1 else None
        row.update(
            best_decode_idx=best_k,
            offset=best_k - t,
            best_score=round(best_s, 4),
            margin=round(margin, 4) if margin is not None else None,
            verified=bool(best_s >= MIN_SCORE and margin is not None and margin >= MIN_MARGIN),
        )
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    offsets = Counter(str(r["offset"]) for r in rows if r.get("offset") is not None)
    n_verified = sum(bool(r["verified"]) for r in rows)
    return {
        "n_labeled": len(rows),
        "n_verified": n_verified,
        "offsets": dict(offsets),
        "consistent": bool(rows) and n_verified == len(rows) and len(offsets) == 1,
        "min_score": MIN_SCORE,
        "min_margin": MIN_MARGIN,
    }


def write_gt_clip(
    video_path: Path,
    label_frames: dict[int, tuple[str, np.ndarray]],
    first: int,
    last: int,
    out_path: Path,
    schema: Schema,
) -> int:
    """Write frames ``first..last`` with labels drawn only on ``label_frames``. Returns frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    written = 0
    idx = 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= first:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if idx in label_frames:
                    image_id, ann = label_frames[idx]
                    frame = render_gt_overlay(frame, ann, schema, [f"label {image_id}"])
                h = frame.shape[0]
                cv2.putText(frame, f"frame {idx}", (10, h - 12), FONT, max(0.5, h / 1080), (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                written += 1
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check that labeled images match video frames (guide 1G).")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--segment-id", required=True)
    p.add_argument("--videos-root", type=Path, help="default: $SKIER_DATA/ski2dpose/videos (uses locate_segment)")
    p.add_argument("--info", type=Path, help="info JSON, passed to locate_segment")
    p.add_argument("--video", type=Path, help="override locate_segment: the video file to use")
    p.add_argument("--start-frame", type=int, default=0, help="with --video: decode index of frame_idx 0")
    p.add_argument("--labels", type=Path, metavar="LABELS_JSON", help="also write gt_clip.mp4")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--apply-offset", type=int, default=0, help="draw labels at t + offset in gt_clip.mp4")
    p.add_argument("--schema", type=Path)
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/alignment/<video>__<segment>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vid, seg = args.video_id, args.segment_id
    out = ensure_dir(args.out or results_root() / "step1" / "alignment" / f"{safe_name(vid)}__{safe_name(seg)}")

    if args.video is not None:
        video_path, start_frame = args.video, args.start_frame
    else:
        videos_root = args.videos_root or data_root() / "ski2dpose" / "videos"
        info = read_json(args.info) if args.info else None
        try:
            video_path, start_frame = locate_segment(videos_root, vid, seg, info)
        except (NotImplementedError, FileNotFoundError) as exc:
            print(f"error: {exc}; or pass --video FILE --start-frame N", file=sys.stderr)
            return 1

    rows = [r for r in read_csv(args.inventory) if r["video_id"] == vid and r["segment_id"] == seg]
    labeled = []
    for r in rows:
        if r["frame_idx"] == "" or r["image_path"] == "":
            print(f"warning: {r['sample_id']} has no frame_idx or image_path; skipped", file=sys.stderr)
            continue
        labeled.append((r["image_id"], int(r["frame_idx"]), args.images_root / r["image_path"]))
    if not labeled:
        print(f"error: no usable inventory rows for {vid}/{seg}", file=sys.stderr)
        return 1

    try:
        results = align_segment(video_path, start_frame, labeled, args.window)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for r in results:
        r.update(video_id=vid, segment_id=seg)
    write_csv(results, out / "alignment.csv", ALIGN_COLUMNS)
    summary = summarize(results)
    summary.update(video_id=vid, segment_id=seg, video=str(video_path), start_frame=start_frame, window=args.window)
    write_json(summary, out / "alignment_summary.json")

    if args.labels is not None:
        schema = load_schema(args.schema)
        raw_by_image = {
            raw.image_id: raw
            for raw in iter_raw_labels(read_json(args.labels))
            if raw.video_id == vid and raw.segment_id == seg
        }
        label_frames = {}
        for image_id, frame_idx, _ in labeled:
            ann = annotation_array(raw_by_image[image_id]) if image_id in raw_by_image else None
            if ann is not None:
                label_frames[start_frame + frame_idx + args.apply_offset] = (image_id, ann)
        ts = [start_frame + fi for _, fi, _ in labeled]
        first, last = max(0, min(ts) - args.window), max(ts) + args.window
        n = write_gt_clip(Path(video_path), label_frames, first, last, out / "gt_clip.mp4", schema)
        print(f"gt_clip.mp4: {n} frames")

    print(f"{summary['n_verified']}/{summary['n_labeled']} verified, offsets {summary['offsets']}, "
          f"consistent: {summary['consistent']}  → {out}")
    return 0
