"""Filesystem locations.

Data, weights and large results live outside the repo and outside any cloud-sync
folder (plan D5). Code never hard-codes absolute paths; it asks this module.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    value = os.environ.get(var)
    return Path(value or default).expanduser()


def data_root() -> Path:
    """Dataset root: ``$SKIER_DATA``, else ``~/skier_data``. Need not exist."""
    return _env_path("SKIER_DATA", "~/skier_data")


def results_root() -> Path:
    """Large-output root: ``$SKIER_RESULTS``, else ``~/skier_results``. Need not exist."""
    return _env_path("SKIER_RESULTS", "~/skier_results")


def repo_root() -> Path:
    """The git checkout. Only valid for the editable install (``pip install -e .``)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").exists():
        raise RuntimeError(
            f"no pyproject.toml at {root}; install the package with `pip install -e .`"
        )
    return root


def configs_dir() -> Path:
    return repo_root() / "configs"
