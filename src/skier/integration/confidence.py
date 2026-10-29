"""Tracker-confidence check and per-frame tracking status (plan Step 7; D24).

SiamFC's score is the peak of its sum-normalized, Hann-windowed response map, so it is
tiny (about 1e-5) and only meaningful relative to other frames. Two candidate signals
were fixed before looking at any data (Step 7 plan):
- ``raw``: the score itself;
- ``relative``: the score divided by the median score of the clip's first
  ``REL_WINDOW`` tracked frames.

On SkiTB val (visible frames after initialization), each signal is scored by ROC AUC for
predicting a tracking failure (IoU < ``FAIL_IOU``). The rule: take the signal with the
higher AUC and threshold it at Youden's J (TPR − FPR). If the best AUC is below
``MIN_AUC``, the score is uninformative and nothing is gated (``conf_signal: none``).

Then the replay rule: low_conf frames drive the replay as lost only if that raises the
mean GT-box containment on the same clips.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from skier.integration.config import IntegrationConfig
from skier.io_utils import ensure_dir, write_json
from skier.paths import configs_dir, repo_root, results_root
from skier.pose.crop_error import FINAL_ONLY

REL_WINDOW = 10  # tracked frames
FAIL_IOU = 0.1  # the failure-episode IoU (Step 3)
MIN_AUC = 0.70
GATED = ("raw", "relative")
RULE = (f"failure = IoU < {FAIL_IOU} on visible frames after init; take the signal (raw, relative) with the higher "
        f"ROC AUC, threshold at Youden's J; if the best AUC < {MIN_AUC}, no gating. Replay: low_conf drives the "
        "replay as lost only if that raises mean GT-box containment")


def signal_values(rows: Sequence[dict], signal: str) -> np.ndarray:
    """One value per row (NaN on the init row, which has no score)."""
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    if signal == "raw":
        return scores
    if signal == "relative":
        tracked = scores[[r["status"] != "init" for r in rows]]
        tracked = tracked[np.isfinite(tracked)][:REL_WINDOW]
        ref = float(np.median(tracked)) if len(tracked) else math.nan
        return scores / ref if ref > 0 else np.full_like(scores, np.nan)
    raise ValueError(f"unknown signal {signal!r}")


def tracking_status(rows: Sequence[dict], cfg: IntegrationConfig) -> list[str]:
    """``init`` | ``track`` | ``low_conf`` per row. A tracked frame without a usable signal value is low_conf."""
    cfg.require("conf_signal")
    gated = cfg.conf_signal != "none"
    v = signal_values(rows, cfg.conf_signal) if gated else None
    out = []
    for i, r in enumerate(rows):
        if r["status"] == "init":
            out.append("init")
        elif r["status"] == "lost" or (gated and not v[i] >= cfg.conf_threshold):
            out.append("low_conf")
        else:
            out.append("track")
    return out


def _as_lost(rows: Sequence[dict], status: Sequence[str]) -> list[dict]:
    return [{**r, "status": "lost"} if s == "low_conf" else r for r, s in zip(rows, status)]


def replay_status_rows(rows: Sequence[dict], cfg: IntegrationConfig) -> list[dict]:
    """The rows as the replay controller sees them: low_conf frames become ``lost`` if ``replay_uses_status`` (D24)."""
    cfg.require("conf_signal", "replay_uses_status")
    return _as_lost(rows, tracking_status(rows, cfg)) if cfg.replay_uses_status else list(rows)


def _finite(values, failure) -> tuple[np.ndarray, np.ndarray]:
    v, f = np.asarray(values, dtype=np.float64), np.asarray(failure, dtype=bool)
    keep = np.isfinite(v)
    return v[keep], f[keep]


def _ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv = np.unique(x, return_inverse=True)
    return (np.bincount(inv, ranks) / np.bincount(inv))[inv]


def roc_auc(values, failure) -> float | None:
    """P(a failure frame's value < an ok frame's value), ties counting 1/2 (Mann–Whitney)."""
    v, f = _finite(values, failure)
    n_f, n_ok = int(f.sum()), int((~f).sum())
    if n_f == 0 or n_ok == 0:
        return None
    r = _ranks(-v)  # low values rank high
    return float((r[f].sum() - n_f * (n_f + 1) / 2) / (n_f * n_ok))


def youden_threshold(values, failure) -> dict | None:
    """The ``t`` maximizing TPR − FPR for "flag when value < t"; ties go to the smallest ``t``."""
    v, f = _finite(values, failure)
    if not f.any() or f.all():
        return None
    cand = np.unique(v)
    tpr = np.searchsorted(np.sort(v[f]), cand, side="left") / int(f.sum())
    fpr = np.searchsorted(np.sort(v[~f]), cand, side="left") / int((~f).sum())
    j = tpr - fpr
    i = int(np.flatnonzero(j >= j.max() - 1e-12)[0])  # equal up to rounding is a tie
    return {"threshold": float(cand[i]), "j": float(j[i]), "tpr": float(tpr[i]), "fpr": float(fpr[i])}


def confidence_check(clips: Sequence[tuple[list[dict], np.ndarray, np.ndarray]]) -> dict:
    """``clips``: (track rows, IoU per row, visible per row). Applies the rule above."""
    pooled = {s: [] for s in GATED}
    failure = []
    for rows, ious, visible in clips:
        scored = np.asarray(visible, bool) & np.array([r["status"] != "init" for r in rows])
        for s in GATED:
            pooled[s].append(signal_values(rows, s)[scored])
        failure.append(np.asarray(ious)[scored] < FAIL_IOU)
    fail = np.concatenate(failure) if failure else np.zeros(0, bool)
    values = {s: np.concatenate(pooled[s]) if pooled[s] else np.zeros(0) for s in GATED}
    signals = {s: {"auc": roc_auc(values[s], fail), "youden": youden_threshold(values[s], fail)} for s in GATED}
    best = max(GATED, key=lambda s: -1.0 if signals[s]["auc"] is None else signals[s]["auc"])  # tie: raw
    if signals[best]["auc"] is None or signals[best]["auc"] < MIN_AUC:
        choice = {"conf_signal": "none", "conf_threshold": None}
    else:
        choice = {"conf_signal": best, "conf_threshold": signals[best]["youden"]["threshold"]}
    return {"rule": RULE, "n_frames": int(len(fail)), "n_failure": int(fail.sum()), "signals": signals,
            "choice": choice}


def replay_gating_check(clips, rcfg, icfg: IntegrationConfig, schema) -> dict:
    """Replay on the same clips with and without low_conf frames driving the controller as lost."""
    from skier.replay.run import clip_metrics, compute_track

    def summary(cs):
        ms = [clip_metrics(c, compute_track(c, rcfg)[1], schema) for c in cs]

        def mean(get):
            vals = [v for v in (get(m) for m in ms) if v is not None]
            return float(np.mean(vals)) if vals else None
        return {"box_containment_mean": mean(lambda m: m.get("box_containment", {}).get("mean")),
                "frac_fully_inside": mean(lambda m: m.get("box_containment", {}).get("frac_fully_inside")),
                "acceleration_mean": mean(lambda m: m["acceleration"]["mean"]),
                "area_fraction_mean": mean(lambda m: m["zoom"]["area_fraction_mean"])}

    statuses = [tracking_status(c.rows, icfg) for c in clips]
    gated = [dataclasses.replace(c, rows=_as_lost(c.rows, st)) for c, st in zip(clips, statuses)]
    n = sum(len(st) for st in statuses)
    plain, with_status = summary(clips), summary(gated)
    a, b = plain["box_containment_mean"], with_status["box_containment_mean"]
    return {"low_conf_share": sum(st.count("low_conf") for st in statuses) / n if n else None,
            "status_ignored": plain, "status_drives_replay": with_status,
            "replay_uses_status": bool(icfg.conf_signal != "none" and a is not None and b is not None and b > a)}


def load_skitb_inputs(track_dir: Path, skitb_root: Path, split_json: Path, partition: str, fps_table: Path):
    """Per SkiTB clip with a track CSV: (rows, IoU per row, visible per row, ClipInput for the replay)."""
    import cv2

    from skier.data.skitb import fps_for, frame_file, load_fps_table, load_skitb_clips
    from skier.eval.track_metrics import frame_ious
    from skier.replay.run import ClipInput
    from skier.tracking.run import boxes_of, read_track_csv

    table = load_fps_table(fps_table)
    out = []
    for c in load_skitb_clips(skitb_root, split_json, partition):
        csv = track_dir / f"{c.sequence_id}__{c.clip_id}.csv"
        if not csv.is_file():
            continue
        rows = read_track_csv(csv)
        pos = np.array([r["frame_index"] for r in rows])
        visible = c.visible[pos]
        gt = np.where(visible[:, None], c.boxes_xyxy[pos], np.nan)
        ious = frame_ious(boxes_of(rows), c.boxes_xyxy[pos])
        first = cv2.imread(str(c.frames_dir / frame_file(c.frame_files[0])))
        clip = ClipInput(csv.stem, rows, fps_for(table, c.sequence_id), (float(first.shape[1]), float(first.shape[0])), gt)
        out.append((rows, ious, visible, clip))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tracker-confidence check on SkiTB val (Step 7): which signal, which "
                                            "threshold, and whether low-confidence frames should drive the replay.")
    p.add_argument("--track-dir", type=Path, required=True, help="CSVs from run_tracker.py --skitb-root")
    p.add_argument("--skitb-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--partition", default="val")
    p.add_argument("--fps-table", type=Path, help="default: data/manifests/skitb_fps_v1.csv (D22)")
    p.add_argument("--replay-config", type=Path, help="default: configs/replay/default.yaml")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/integration/confidence-<partition>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition in FINAL_ONLY:
        print("error: never tune on test", file=sys.stderr)
        return 1
    from skier.data.skitb import FPS_TABLE
    from skier.replay.controller import ReplayConfig
    from skier.schema import load_schema

    inputs = load_skitb_inputs(args.track_dir, args.skitb_root, args.split_json, args.partition,
                               args.fps_table or repo_root() / FPS_TABLE)
    if not inputs:
        print(f"error: no track CSVs in {args.track_dir} for {args.partition}", file=sys.stderr)
        return 1
    result = confidence_check([(rows, ious, vis) for rows, ious, vis, _ in inputs])
    icfg = IntegrationConfig(**result["choice"])
    rcfg = ReplayConfig.from_yaml(args.replay_config or configs_dir() / "replay" / "default.yaml")
    result["replay"] = replay_gating_check([c for *_, c in inputs], rcfg, icfg, load_schema())
    result.update(partition=args.partition, n_clips=len(inputs), replay_config=dataclasses.asdict(rcfg))
    out = ensure_dir(args.out or results_root() / "integration" / f"confidence-{args.partition}")
    write_json(result, out / "confidence.json")
    for s, d in result["signals"].items():
        print(f"{s}: AUC {d['auc']}, Youden {d['youden']}")
    print(f"choice: {result['choice']}; replay_uses_status: {result['replay']['replay_uses_status']} → {out}")
    return 0
