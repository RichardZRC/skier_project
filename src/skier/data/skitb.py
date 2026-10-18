"""SkiTB single-camera clip reader (plan Step 2.4).

Written against the toolkit README's description, not the real files: SkiTB data
access is still pending (requested 2026-10-17). Every layout assumption sits in the
FILL IN block below, and anything unexpected raises instead of being guessed around.
Boxes are converted from the files' ``xywh`` to ``xyxy`` here, at the edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skier.io_utils import read_json

# --- FILL IN WHEN SKITB ARRIVES (code spec §13) --------------------------------
# Layout per the README: <root>/<discipline>/<sequence>/{frames/, MC/, SC/<clip>/...}.
# The toolkit reportedly reads "ST" in one code path and "SC" in another; check.
SINGLE_CAMERA_DIR = "SC"
FRAMES_DIR = "frames"
BOXES_FILE = "boxes.txt"  # one "x y w h" (or "x,y,w,h") row per frame, pixels
FRAMES_FILE = "frames.txt"  # one entry per row: a frame index or a filename under frames/
VISIBILITIES_FILE = "visibilities.txt"  # one 0/1 per row


def sequences_for_partition(split: dict, partition: str) -> list[str]:
    """Sequence IDs of one partition. Assumed shape: ``{partition: [sequence_id, ...]}``."""
    if partition not in split:
        raise ValueError(f"split JSON has no partition {partition!r}; keys: {sorted(split)}")
    seqs = split[partition]
    if not isinstance(seqs, list) or not all(isinstance(s, (str, int)) for s in seqs):
        raise ValueError(f"split JSON partition {partition!r} is not a list of sequence IDs")
    return [str(s) for s in seqs]
# -------------------------------------------------------------------------------


@dataclass
class Clip:
    sequence_id: str
    clip_id: str
    partition: str
    frames_dir: Path
    frame_files: list[str]  # as listed in frames.txt
    boxes_xyxy: np.ndarray  # [T, 4]
    visible: np.ndarray  # [T] bool


def parse_boxes_txt(path: str | Path) -> np.ndarray:
    """Rows of ``x, y, w, h`` (comma and/or whitespace separated) -> ``[T, 4]`` xyxy."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if len(parts) != 4:
            raise ValueError(f"{path}:{n}: expected 4 numbers (x y w h), got {line!r}")
        x, y, w, h = map(float, parts)
        rows.append([x, y, x + w, y + h])
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_clip(clip_dir: Path, sequence_id: str, partition: str) -> Clip:
    for name in (BOXES_FILE, FRAMES_FILE, VISIBILITIES_FILE):
        if not (clip_dir / name).is_file():
            raise FileNotFoundError(f"{clip_dir}: missing {name}")
    boxes = parse_boxes_txt(clip_dir / BOXES_FILE)
    frames = _lines(clip_dir / FRAMES_FILE)
    vis_raw = _lines(clip_dir / VISIBILITIES_FILE)
    if not len(boxes) == len(frames) == len(vis_raw):
        raise ValueError(f"{clip_dir}: line counts differ (boxes {len(boxes)}, frames {len(frames)}, "
                         f"visibilities {len(vis_raw)})")
    bad = sorted({v for v in vis_raw if v not in ("0", "1")})
    if bad:
        raise ValueError(f"{clip_dir}: visibilities must be 0/1, found {bad}")
    return Clip(
        sequence_id=sequence_id,
        clip_id=clip_dir.name,
        partition=partition,
        frames_dir=clip_dir.parents[1] / FRAMES_DIR,
        frame_files=frames,
        boxes_xyxy=boxes,
        visible=np.array([v == "1" for v in vis_raw]),
    )


def load_skitb_clips(skitb_root: str | Path, split_json: str | Path, partition: str,
                     discipline: str = "AL") -> list[Clip]:
    """Every single-camera clip of the sequences in ``partition``.

    A clip always inherits its parent sequence's partition, so camera segments from
    one run are never assigned independently.
    """
    root = Path(skitb_root) / discipline
    clips = []
    for seq in sequences_for_partition(read_json(split_json), partition):
        sc_dir = root / seq / SINGLE_CAMERA_DIR
        if not sc_dir.is_dir():
            raise FileNotFoundError(f"sequence {seq!r}: no {SINGLE_CAMERA_DIR}/ folder at {sc_dir}")
        for clip_dir in sorted(p for p in sc_dir.iterdir() if p.is_dir()):
            clips.append(load_clip(clip_dir, seq, partition))
    return clips
