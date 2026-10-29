"""Report figures (plan Step 8): charts only, drawn from ``tables.json`` and the training logs.

matplotlib is imported inside ``_plt`` only. The laptop's venv has it; the Mac's light
venv doesn't need it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from skier.io_utils import ensure_dir, read_csv, read_json

RECIPE_COLORS = {"E1": "#4c72b0", "E2": "#dd8452", "E3": "#55a868", "E4": "#c44e52"}
SEED_STYLES = {"230": "-", "231": "--"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_log(path: str | Path) -> dict[str, np.ndarray]:
    """A training ``log.csv`` as arrays: epoch, val body/ski PCK@0.1, train loss, and the fine-tune mask."""
    rows = read_csv(path)
    return {"epoch": np.array([int(r["epoch"]) for r in rows]),
            "finetune": np.array([r["phase"] != "warm" for r in rows]),
            "val_body_pct": np.array([float(r["val_body_pct"]) for r in rows]),
            "val_ski_pct": np.array([float(r["val_ski_pct"]) for r in rows]),
            "train_loss": np.array([float(r["train_loss"]) for r in rows])}


def learning_curves(logs: dict[str, dict], path: Path) -> Path:
    """``logs``: ``{"E2-s230": read_log(...)}``. Val ski and body PCK@0.1 per epoch, one line per run."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, log in sorted(logs.items()):
        recipe, seed = name.split("-s")
        style = dict(color=RECIPE_COLORS.get(recipe, "gray"), linestyle=SEED_STYLES.get(seed, ":"), linewidth=1.2)
        axes[0].plot(log["epoch"], log["val_ski_pct"], label=name, **style)
        axes[1].plot(log["epoch"], log["val_body_pct"], **style)
        axes[2].plot(log["epoch"], log["train_loss"], **style)
    for ax, title in zip(axes, ("val ski PCK@0.1 (%)", "val body PCK@0.1 (%)", "train loss")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def augmentation_bars(aug: dict, path: Path) -> Path:
    """Val seed means per recipe (bars) with per-seed values (dots): ski and body PCK@0.1, clean and moderate crops."""
    plt = _plt()
    rows = aug["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("clean", "ski@0.1", "per_seed_ski@0.1", "ski PCK@0.1, clean crops"),
              ("moderate", "ski@0.1", "per_seed_ski@0.1", "ski PCK@0.1, moderate crop error"),
              ("clean", "body@0.1", "per_seed_body@0.1", "body PCK@0.1, clean crops")]
    for ax, (crops, key, seeds_key, title) in zip(axes, panels):
        sel = [r for r in rows if r["crops"] == crops]
        x = np.arange(len(sel))
        ax.bar(x, [r[key]["pct"] for r in sel], color=[RECIPE_COLORS.get(r["run"], "gray") for r in sel], alpha=0.8)
        for i, r in enumerate(sel):
            ax.plot([i - 0.1, i + 0.1], r[seeds_key], "ko", markersize=4)
        ax.set_xticks(x, [r["run"] for r in sel])
        lo = min(min(r[seeds_key]) for r in sel)
        ax.set_ylim(max(0, lo - 5), 100)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def per_joint_delta(per_joint: list[dict], path: Path, title: str) -> Path:
    """Δ E2 − E1 per joint at PCK@0.1 (seed-pooled)."""
    plt = _plt()
    rows = [r for r in per_joint if r["delta"] is not None and r["total_per_seed"] > 0]
    fig, ax = plt.subplots(figsize=(11, 4))
    d = [r["delta"] for r in rows]
    ax.bar(range(len(rows)), d, color=["#dd8452" if v >= 0 else "#4c72b0" for v in d])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rows)), [r["joint"] for r in rows], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Δ PCK@0.1 (points)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def success_curves(curves: dict[str, tuple[list, list, float]], path: Path) -> Path:
    """``curves``: ``{label: (thresholds, success, auc)}``."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for label, (t, s, auc) in curves.items():
        ax.plot(t, s, label=f"{label} (AUC {auc:.3f})")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("share of frames with IoU above it")
    ax.set_title("SiamFC success curve (SkiTB AL, SC clips)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def propagation_strata(strata: list[dict], path: Path, run: str = "E2") -> Path:
    """Reference vs. tracker crop by tracker-IoU stratum (ski and body PCK@0.1)."""
    plt = _plt()
    order = ["iou>=0.5", "0.1<=iou<0.5", "iou<0.1"]
    rows = {s["value"]: s for s in strata if s["stratum"] == "tracker_iou" and s["run"] == run}
    vals = [v for v in order if v in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, g in zip(axes, ("ski", "body")):
        x = np.arange(len(vals))
        ax.bar(x - 0.2, [rows[v][f"reference_{g}"] for v in vals], 0.4, label="reference crop", color="#999999")
        ax.bar(x + 0.2, [rows[v][f"tracked_{g}"] for v in vals], 0.4, label="tracker crop", color=RECIPE_COLORS[run])
        ax.set_xticks(x, [f"{v}\n(n={rows[v]['n_images']})" for v in vals])
        ax.set_ylim(0, 100)
        ax.set_title(f"{run}: {g} PCK@0.1 by tracker IoU")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report figures (plan Step 8) from tables.json files and training logs.")
    p.add_argument("--tables", type=Path, nargs="+", required=True, help="tables.json files (e.g. val and test)")
    p.add_argument("--d21-dir", type=Path, help="$SKIER_RESULTS/pose/d21 (training logs for the learning curves)")
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = ensure_dir(args.out)
    tables = [read_json(p) for p in args.tables]
    made = []
    if args.d21_dir is not None:
        logs = {d.name: read_log(d / "log.csv") for d in sorted(args.d21_dir.glob("E*-s23*")) if (d / "log.csv").is_file()}
        if logs:
            made.append(learning_curves(logs, out / "learning_curves.png"))
    curves = {}
    for t in tables:
        part = t["partition"]
        if "augmentation" in t:
            made.append(augmentation_bars(t["augmentation"], out / "augmentation_val.png"))
        if "pose" in t:
            made.append(per_joint_delta(t["pose"]["per_joint"], out / f"per_joint_delta_{part}.png",
                                        f"E2 − E1 per joint, PCK@0.1 ({part}, seed-pooled)"))
        if "tracking" in t:
            o = t["tracking"]["ours"]
            curves[part] = (o["success_thresholds"], o["success_curve"], o["auc"])
        if "propagation" in t:
            made.append(propagation_strata(t["propagation"]["strata"], out / f"propagation_strata_{part}.png"))
    if curves:
        made.append(success_curves(curves, out / "tracking_success.png"))
    if not made:
        print("error: nothing to draw", file=sys.stderr)
        return 1
    print(f"{len(made)} figures → {out}")
    return 0
