"""Environment check (guide 1C.7 / 1H, code spec §9).

``collect()`` never raises: missing packages are recorded as null, so this runs in
the light dev environment. ``--strict`` (used on AWS) fails unless every check passes.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import ModuleType

from skier.io_utils import ensure_dir, write_json
from skier.paths import repo_root, results_root

PACKAGES = ["torch", "torchvision", "mmcv", "mmengine", "mmpose", "numpy", "opencv-python", "pillow"]


def numpy_ok(version: str) -> bool:
    """torch 2.1 is built against NumPy 1.x."""
    return int(version.split(".")[0]) < 2


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _try_import(name: str) -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # broken CUDA libs raise OSError etc., not just ImportError
        return None, f"{type(exc).__name__}: {exc}"


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _check(name: str, ok: bool | None, detail: str = "") -> dict:
    return {"name": name, "ok": ok, "detail": detail}


def collect() -> dict:
    packages = {p: _version(p) for p in PACKAGES}
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version,
        "git_commit": _git_commit(),
        "packages": packages,
        "torch_cuda_build": None,
        "cuda_available": None,
        "gpu_name": None,
        "vram_gb": None,
        "cudnn": None,
    }
    checks = []

    np_v = packages["numpy"]
    checks.append(_check("numpy_lt_2", numpy_ok(np_v) if np_v else False, np_v or "numpy not installed"))

    torch, torch_err = _try_import("torch") if packages["torch"] else (None, "torch not installed")
    checks.append(_check("torch_present", torch is not None, torch_err or packages["torch"]))
    cuda = False
    if torch is not None:
        info["torch_cuda_build"] = torch.version.cuda
        cuda = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda
        if cuda:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = round(props.total_memory / 2**30, 2)
            info["cudnn"] = torch.backends.cudnn.version()
    checks.append(_check(
        "cuda_available",
        info["cuda_available"],
        "" if torch is not None else "torch not importable",
    ))

    if packages["mmcv"]:
        _, err = _try_import("mmcv.ops")
        checks.append(_check("mmcv_ops_import", err is None, err or "ok"))
    else:
        checks.append(_check("mmcv_ops_import", None, "mmcv not installed"))

    checks.append(_check("mmpose_present", packages["mmpose"] is not None, packages["mmpose"] or "mmpose not installed"))

    if cuda:
        try:
            x = torch.randn(1, 3, 8, 8, device="cuda")
            w = torch.randn(4, 3, 3, 3, device="cuda")
            torch.nn.functional.conv2d(x, w)
            torch.cuda.synchronize()
            checks.append(_check("gpu_conv", True, "3x3 conv on cuda:0"))
        except Exception as exc:
            checks.append(_check("gpu_conv", False, f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(_check("gpu_conv", None, "no CUDA"))

    info["checks"] = checks
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record the software/GPU environment to env.json.")
    p.add_argument("--strict", action="store_true", help="exit 1 unless every check passes (use on AWS)")
    p.add_argument("--out", type=Path, help="default: $SKIER_RESULTS/step1/env")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = ensure_dir(args.out or results_root() / "step1" / "env")
    info = collect()
    write_json(info, out / "env.json")
    for c in info["checks"]:
        mark = {True: "ok  ", False: "FAIL", None: "n/a "}[c["ok"]]
        print(f"[{mark}] {c['name']}: {c['detail']}")
    print(f"wrote {out / 'env.json'}")
    if args.strict and any(c["ok"] is not True for c in info["checks"]):
        print("strict: not every check passed", file=sys.stderr)
        return 1
    return 0
