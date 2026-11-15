"""Which tracker produced a set of tracks (Step 7b, D28): SiamFC (the main result) or STARK-ST50 (a labeled extra).

Every track folder written since D28 holds a ``tracker.json``; a folder without one is SiamFC's (every run before
D28). Settings that depend on the tracker's score scale (the D24 confidence threshold) live in an integration
config with a ``tracker`` field, and the tools that combine the two call ``check_tracker`` first: STARK's scores
are in [0, 1], so SiamFC's raw-scale threshold (3.9e-5) would silently gate nothing.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Callable

from skier.io_utils import read_json, sha256_file, write_json

TRACKERS = ("siamfc", "stark")
TRACKER_FILE = "tracker.json"
DEFAULT = "siamfc"


def add_tracker_args(p: argparse.ArgumentParser, weights_required: bool = True) -> None:
    g = p.add_argument_group("tracker")
    g.add_argument("--tracker", choices=TRACKERS, default=DEFAULT,
                   help="default: siamfc; stark is the D28 post-freeze extra")
    g.add_argument("--siamfc-repo", type=Path, help="clone of the siamfc-pytorch fork (--tracker siamfc)")
    g.add_argument("--stark-repo", type=Path, help="clone of researchmm/Stark at the D28 commit (--tracker stark)")
    g.add_argument("--weights", type=Path, required=weights_required, help="the tracker's weights")


def tracker_repo(args) -> Path:
    """The chosen tracker's clone; refuses a missing one, or the other tracker's flag (a forgotten --tracker)."""
    repos = {"siamfc": args.siamfc_repo, "stark": args.stark_repo}
    other = [t for t in TRACKERS if t != args.tracker and repos[t] is not None]
    if other:
        raise ValueError(f"--{other[0]}-repo given with --tracker {args.tracker}")
    if repos[args.tracker] is None or args.weights is None:
        raise ValueError(f"--tracker {args.tracker} needs --{args.tracker}-repo and --weights")
    return Path(repos[args.tracker])


def tracker_factory(args) -> Callable:
    """A fresh tracker per clip. STARK builds its network once, on the first call, and shares it."""
    repo = tracker_repo(args)
    if args.tracker == "siamfc":
        from skier.tracking.siamfc_adapter import SiamFCTracker

        return lambda: SiamFCTracker(repo, args.weights)
    from skier.tracking.stark_adapter import StarkBackend, StarkTracker

    backend = []

    def make():
        if not backend:
            backend.append(StarkBackend(repo, args.weights))
        return StarkTracker(backend[0])

    return make


def _git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def tracker_info(args) -> dict:
    repo = tracker_repo(args)
    return {"tracker": args.tracker, "repo": str(repo), "repo_commit": _git_head(repo), "weights": str(args.weights),
            "weights_sha256": sha256_file(args.weights)}


def write_tracker_info(info: dict, out_dir: Path) -> None:
    write_json(info, Path(out_dir) / TRACKER_FILE)


def read_tracker_name(track_dir: Path) -> str:
    """The tracker of a track folder; ``siamfc`` when it has no ``tracker.json`` (every run before D28)."""
    path = Path(track_dir) / TRACKER_FILE
    if not path.is_file():
        return DEFAULT
    name = read_json(path).get("tracker")
    if name not in TRACKERS:
        raise ValueError(f"{path}: unknown tracker {name!r}")
    return name


def check_tracker(track_dir: Path, expected: str, what: str = "the integration config") -> None:
    found = read_tracker_name(track_dir)
    if found != expected:
        raise ValueError(f"{track_dir} holds {found} tracks, but {what} is for {expected} (D28); "
                         f"use the config chosen for {found}")
