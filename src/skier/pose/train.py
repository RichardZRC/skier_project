"""E1 training loop (plan Step 5.4).

torch is imported only inside the functions that need it (D13: not installed on the Mac).
Torch-free parts — config, LR schedule, best-checkpoint selection, logging — are unit
tested locally; the loop itself is tested on AWS (``torch``-marked tests) before training.

Phases: ``warm_epochs`` with the backbone frozen (``requires_grad=False`` **and**
``backbone.eval()``, so COCO BatchNorm statistics are kept) and only the head trained,
then fine-tuning with two Adam parameter groups and x0.1 steps at the milestones.
Validation every ``val_every`` epochs uses the same pipeline as E0 (``predict_records``).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from skier.data.records import PoseRecord
from skier.eval.pose_metrics import group_pck, gt_arrays
from skier.io_utils import ensure_dir, read_json, write_json
from skier.paths import configs_dir, data_root, results_root
from skier.pose.augment import AugConfig
from skier.schema import Schema, load_schema

LOG_COLUMNS = ["epoch", "phase", "lr_backbone", "lr_head", "train_loss", "n_steps", "n_dropped", "epoch_s",
               "s_per_step", "data_wait_s", "peak_mem_mb", "val_body_pct", "val_body_hits", "val_body_total",
               "val_ski_pct", "val_ski_hits", "val_ski_total", "is_best"]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    warm_epochs: int = 3
    finetune_epochs: int = 150  # D21 (50 until 2026-10-26)
    lr_head_warm: float = 1e-3
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    milestones: tuple[float, ...] = (0.6, 0.85)
    seed: int = 230
    num_workers: int = 4
    val_every: int = 1
    val_partition: str = "val"
    val_batch_size: int = 32
    expand: float = 1.25
    sigma: float = 2.0
    aug: AugConfig = field(default_factory=AugConfig)

    @property
    def total_epochs(self) -> int:
        return self.warm_epochs + self.finetune_epochs

    @classmethod
    def from_dict(cls, raw: dict) -> "TrainConfig":
        raw = dict(raw or {})
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        if "milestones" in raw:
            raw["milestones"] = tuple(float(m) for m in raw["milestones"])
        raw["aug"] = AugConfig.from_dict(raw.get("aug"))
        cfg = cls(**raw)
        if cfg.val_partition == "test":
            raise ValueError("never validate on test")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["milestones"] = list(self.milestones)
        d["aug"]["scale_range"] = list(self.aug.scale_range)
        return d


def lr_for(epoch: int, cfg: TrainConfig) -> tuple[float, float]:
    """(backbone LR, head LR) for an epoch. Warm-up: (0, lr_head_warm). Fine-tune epoch e:
    base LRs x 0.1 for every milestone with e >= floor(m * finetune_epochs)."""
    if epoch < cfg.warm_epochs:
        return 0.0, cfg.lr_head_warm
    e = epoch - cfg.warm_epochs
    factor = 0.1 ** sum(e >= math.floor(m * cfg.finetune_epochs) for m in cfg.milestones)
    return cfg.lr_backbone * factor, cfg.lr_head * factor


class BestTracker:
    """Best value so far (higher is better); ties keep the earlier epoch; None never wins."""

    def __init__(self, value: float | None = None, epoch: int | None = None):
        self.value, self.epoch = value, epoch

    def update(self, epoch: int, value: float | None) -> bool:
        if value is None or (self.value is not None and value <= self.value):
            return False
        self.value, self.epoch = value, epoch
        return True


def append_log(path: Path, row: dict) -> None:
    extra = set(row) - set(LOG_COLUMNS)
    if extra:
        raise ValueError(f"unknown log columns {sorted(extra)}")
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLUMNS)
        w.writerow(["" if row.get(c) is None else row.get(c) for c in LOG_COLUMNS])


def schema_fingerprint(schema: Schema) -> str:
    content = json.dumps({"names": schema.names, "flip_pairs": schema.flip_pairs}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def select_overfit_records(records: Sequence[PoseRecord], n: int, seed: int) -> list[PoseRecord]:
    rng = random.Random(seed)
    return sorted(rng.sample(list(records), min(n, len(records))), key=lambda r: r.sample_id)


# --- torch from here on ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def param_groups(model) -> list[dict]:
    backbone, head, other = [], [], []
    for name, p in model.named_parameters():
        (backbone if name.startswith("backbone.") else head if name.startswith("head.") else other).append(p)
    if other:
        raise ValueError(f"{len(other)} parameters are neither backbone.* nor head.*")
    return [{"params": backbone, "lr": 0.0, "name": "backbone"}, {"params": head, "lr": 0.0, "name": "head"}]


def set_phase(model, frozen: bool) -> None:
    model.train()
    for p in model.backbone.parameters():
        p.requires_grad_(not frozen)
    if frozen:
        model.backbone.eval()  # keep the COCO BatchNorm running statistics while frozen


def validate(model, records: Sequence[PoseRecord], schema: Schema, device, batch_size: int, expand: float) -> dict:
    from skier.pose.infer import predict_records
    from skier.pose.model import torch_heatmap_fn

    preds = predict_records(records, torch_heatmap_fn(model, device), expand=expand, batch_size=batch_size)
    gt, rel, diag, _ = gt_arrays(records)
    return group_pck(preds.xy, gt, rel, diag, schema, 0.1)


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: TrainConfig, meta: dict) -> None:
    import torch

    torch.save({"version": 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch, "config": cfg.to_dict(), **meta}, str(path))


def load_checkpoint(path: Path, model, optimizer=None, schema: Schema | None = None, map_location="cpu") -> dict:
    import torch

    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)  # config + metadata, not just tensors
    if schema is not None:
        if ckpt.get("num_keypoints") != schema.num_keypoints:
            raise ValueError(f"checkpoint has {ckpt.get('num_keypoints')} keypoints, schema has {schema.num_keypoints}")
        if ckpt.get("schema_fingerprint") != schema_fingerprint(schema):
            raise ValueError("checkpoint was trained with a different landmark schema")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(model, train_records: Sequence[PoseRecord], val_records: Sequence[PoseRecord], cfg: TrainConfig,
          out_dir: str | Path, schema: Schema, device: str = "cpu", *, max_steps: int | None = None,
          profile_steps: int | None = None, split_sha256: str | None = None, cache=None,
          resume: str | Path | None = None, log=print) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from skier.envcheck import _git_commit
    from skier.pose.dataset import PoseDataset, collate
    from skier.pose.loss import masked_mse

    out = ensure_dir(out_dir)
    seed_everything(cfg.seed)
    model.to(device)
    dataset = PoseDataset(train_records, schema, train=True, aug_cfg=cfg.aug, seed=cfg.seed, cache=cache,
                          expand=cfg.expand, sigma=cfg.sigma)
    optimizer = torch.optim.Adam(param_groups(model), lr=0.0)
    meta = {"seed": cfg.seed, "git_commit": _git_commit(), "split_sha256": split_sha256,
            "schema_fingerprint": schema_fingerprint(schema), "num_keypoints": schema.num_keypoints,
            "n_train": len(train_records), "n_val": len(val_records)}
    best, start = BestTracker(), 0
    if resume is not None:
        ck = load_checkpoint(Path(resume), model, optimizer, schema)
        start, best = ck["epoch"] + 1, BestTracker(ck.get("best_value"), ck.get("best_epoch"))
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    history, steps_total = [], 0

    for epoch in range(start, cfg.total_epochs):
        lr_b, lr_h = lr_for(epoch, cfg)
        optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] = lr_b, lr_h
        frozen = epoch < cfg.warm_epochs
        set_phase(model, frozen)
        dataset.set_epoch(epoch)
        gen = torch.Generator()
        gen.manual_seed(cfg.seed * 1000 + epoch)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                            collate_fn=collate, generator=gen)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        losses, n_steps, n_dropped, data_wait = [], 0, 0, 0.0
        t_epoch = t_last = time.perf_counter()
        for batch in loader:
            data_wait += time.perf_counter() - t_last
            if batch is not None:
                n_dropped += batch["n_dropped"]
                pred = model(batch["x"].to(device, non_blocking=True))
                loss = masked_mse(pred, batch["target"].to(device), batch["weight"].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                n_steps += 1
                steps_total += 1
            if use_cuda:
                torch.cuda.synchronize()
            t_last = time.perf_counter()
            if (profile_steps and steps_total >= profile_steps) or (max_steps and steps_total >= max_steps):
                break
        epoch_s = time.perf_counter() - t_epoch
        row = {"epoch": epoch, "phase": "warm" if frozen else "finetune", "lr_backbone": lr_b, "lr_head": lr_h,
               "train_loss": round(float(np.mean(losses)), 8) if losses else None, "n_steps": n_steps,
               "n_dropped": n_dropped, "epoch_s": round(epoch_s, 3),
               "s_per_step": round(epoch_s / n_steps, 5) if n_steps else None, "data_wait_s": round(data_wait, 3),
               "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if use_cuda else None}

        if profile_steps and steps_total >= profile_steps:
            t_val = time.perf_counter()
            if val_records:
                validate(model, val_records, schema, device, cfg.val_batch_size, cfg.expand)
            val_s = time.perf_counter() - t_val
            steps_per_epoch = math.ceil(len(train_records) / cfg.batch_size)
            s_step = epoch_s / max(1, n_steps)
            report = {**row, "profile_steps": steps_total, "data_wait_share": round(data_wait / epoch_s, 3),
                      "val_s": round(val_s, 2), "steps_per_epoch": steps_per_epoch,
                      "estimated_full_run_h": round((cfg.total_epochs * (steps_per_epoch * s_step + val_s)) / 3600, 2)}
            write_json(report, out / "profile.json")
            log(f"profile: {s_step:.3f} s/step, data wait {report['data_wait_share']:.0%}, "
                f"estimated full run {report['estimated_full_run_h']} h")
            return report

        is_best = False
        if val_records and ((epoch + 1) % cfg.val_every == 0 or epoch == cfg.total_epochs - 1):
            v = validate(model, val_records, schema, device, cfg.val_batch_size, cfg.expand)
            row.update(val_body_pct=v["body"]["pct"], val_body_hits=v["body"]["hits"], val_body_total=v["body"]["total"],
                       val_ski_pct=v["ski"]["pct"], val_ski_hits=v["ski"]["hits"], val_ski_total=v["ski"]["total"])
            is_best = best.update(epoch, v["ski"]["pct"])
        row["is_best"] = int(is_best)
        append_log(out / "log.csv", row)
        ck_meta = {**meta, "metrics": row, "best_value": best.value, "best_epoch": best.epoch}
        save_checkpoint(out / "last.pt", model, optimizer, epoch, cfg, ck_meta)
        if is_best:
            save_checkpoint(out / "best.pt", model, None, epoch, cfg, ck_meta)
        history.append(row)
        log(f"epoch {epoch} ({row['phase']}): loss {row['train_loss']}, val ski PCK@0.1 {row.get('val_ski_pct')}"
            f"{' (best)' if is_best else ''}")
        if max_steps and steps_total >= max_steps:
            break

    summary = {"epochs_run": len(history), "best_epoch": best.epoch, "best_val_ski_pct": best.value,
               "config": cfg.to_dict(), **meta}
    write_json(summary, out / "summary.json")
    return summary


def save_val_overlays(model, records: Sequence[PoseRecord], schema: Schema, device, out_dir: Path, n: int = 20,
                      expand: float = 1.25) -> int:
    """Predicted vs. labeled points on ``n`` val images, for the review (plan §10)."""
    import cv2

    from skier.data.ski2dpose_raw import safe_name
    from skier.pose.infer import predict_records
    from skier.pose.model import torch_heatmap_fn
    from skier.viz import prediction_overlay

    chosen = select_overfit_records(records, n, seed=230)
    preds = predict_records(chosen, torch_heatmap_fn(model, device), expand=expand, batch_size=8)
    ensure_dir(out_dir)
    for r, xy in zip(chosen, preds.xy):
        img = prediction_overlay(cv2.imread(str(r.image_path)), xy, r.keypoints_xy, r.reliable, schema)
        cv2.imwrite(str(out_dir / f"{safe_name(r.sample_id)}.png"), img)
    return len(chosen)


def run_name(experiment: str, overfit: int | None = None, seed: int | None = None) -> str:
    """Output folder name: ``E2``, ``E1-seed231``, ``E1-overfit32``."""
    name = experiment
    if seed is not None:
        name += f"-seed{seed}"
    if overfit:
        name += f"-overfit{overfit}"
    return name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the 24-point HRNet (E1 and later experiments).")
    p.add_argument("--config", type=Path, help="default: configs/pose/e1.yaml")
    p.add_argument("--split", type=Path, required=True, help="frozen pose_split_v1.json")
    p.add_argument("--hrnet-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="COCO HRNet-W32 checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--experiment", default="E1")
    p.add_argument("--seed", type=int, help="override the config's seed (extra seeds); output <experiment>-seed<N>")
    p.add_argument("--overfit", type=int, metavar="N", help="N training images, no augmentation, validated on themselves")
    p.add_argument("--epochs", type=int, help="override finetune_epochs (e.g. for --overfit)")
    p.add_argument("--profile-steps", type=int, metavar="N", help="time N steps, estimate the full run, exit")
    p.add_argument("--resume", type=Path)
    p.add_argument("--cache-dir", type=Path, help="use (and build if missing) the crop cache")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/pose/<experiment>")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config or configs_dir() / "pose" / "e1.yaml")
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, finetune_epochs=args.epochs)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    manifest = read_json(args.split)
    schema = load_schema()

    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images",
                           s2d / "archives/ski2dpose_info.json", schema=schema)
    train_recs, train_excl = select_partition(records, manifest, "train")
    val_recs, _ = select_partition(records, manifest, cfg.val_partition)
    name = run_name(args.experiment, args.overfit, args.seed)
    if args.overfit:
        train_recs = val_recs = select_overfit_records(train_recs, args.overfit, cfg.seed)
        cfg = dataclasses.replace(cfg, aug=AugConfig.none())
    out = Path(args.out or results_root() / "pose" / name)

    cache = None
    if args.cache_dir is not None:
        from skier.pose.dataset import CropCache

        cache = (CropCache(args.cache_dir) if (args.cache_dir / "index.json").exists()
                 else CropCache.build(args.cache_dir, train_recs, cfg.aug, cfg.expand))

    from skier.pose.model import build_hrnet

    model, load_info = build_hrnet(schema.num_keypoints, args.hrnet_config, args.checkpoint, device=args.device)
    print(f"train {len(train_recs)} images ({len(train_excl)} excluded), val {len(val_recs)}; "
          f"loaded {load_info['loaded_tensors']} tensors, new head: {load_info['replaced_head']}")
    summary = train(model, train_recs, val_recs, cfg, out, schema, args.device, profile_steps=args.profile_steps,
                    split_sha256=manifest["content_sha256"], cache=cache, resume=args.resume)
    if args.profile_steps:
        return 0
    if (out / "best.pt").exists():
        load_checkpoint(out / "best.pt", model, schema=schema, map_location=args.device)
        n = save_val_overlays(model, val_recs, schema, args.device, out / "val_overlays", expand=cfg.expand)
        print(f"best epoch {summary['best_epoch']}: val ski PCK@0.1 {summary['best_val_ski_pct']}; {n} overlays")
    return 0
