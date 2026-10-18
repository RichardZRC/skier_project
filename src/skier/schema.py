"""The 24-landmark Ski 2DPose schema: names, groups, sides, flip pairs, edges.

"Right" and "left" always mean the skier's own side, not the image side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skier.io_utils import read_json
from skier.paths import configs_dir

GROUPS = ("body", "ski", "pole")
SIDES = ("right", "left", "center")


@dataclass(frozen=True)
class Edge:
    a: int
    b: int
    group: str  # "body" | "ski" | "pole"


@dataclass(frozen=True)
class Schema:
    names: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]
    side: tuple[str, ...]
    flip_pairs: tuple[tuple[int, int], ...]  # (right, left)
    flip_index: tuple[int, ...]  # partner of i; i itself for center joints
    edges: tuple[Edge, ...]
    coco_shared: dict[int, int]  # COCO-17 index -> ski24 index
    coco_approx: dict[int, int]  # COCO wrist -> ski hand, reported separately

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    def group_of(self, i: int) -> str:
        for name, members in self.groups.items():
            if i in members:
                return name
        raise KeyError(i)


def load_schema(path: str | Path | None = None) -> Schema:
    """Load and validate the schema JSON (default ``configs/ski24_schema.json``)."""
    path = Path(path) if path is not None else configs_dir() / "ski24_schema.json"
    return schema_from_dict(read_json(path))


def schema_from_dict(raw: dict[str, Any]) -> Schema:
    names = tuple(raw["names"])
    n = len(names)
    if len(set(names)) != n:
        dupes = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"duplicate landmark names: {dupes}")

    groups = {g: tuple(int(i) for i in members) for g, members in raw["groups"].items()}
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    flat = [i for members in groups.values() for i in members]
    if sorted(flat) != list(range(n)):
        raise ValueError(f"groups must partition 0..{n - 1} exactly once each")

    side = tuple(raw["side"])
    if len(side) != n or any(s not in SIDES for s in side):
        raise ValueError(f"side must list one of {SIDES} for each of the {n} landmarks")

    flip_index = list(range(n))
    paired: set[int] = set()
    flip_pairs = []
    for r, l in raw["flip_pairs"]:
        r, l = int(r), int(l)
        if not (0 <= r < n and 0 <= l < n):
            raise ValueError(f"flip pair ({r}, {l}) out of range")
        if side[r] != "right" or side[l] != "left":
            raise ValueError(
                f"flip pair ({r}, {l}) must be (right, left), got ({side[r]}, {side[l]})"
            )
        if r in paired or l in paired:
            raise ValueError(f"landmark in more than one flip pair: ({r}, {l})")
        paired.update((r, l))
        flip_index[r], flip_index[l] = l, r
        flip_pairs.append((r, l))
    unpaired = [i for i in range(n) if side[i] != "center" and i not in paired]
    if unpaired:
        raise ValueError(f"left/right landmarks without a flip partner: {unpaired}")

    edges = []
    for e in raw["edges"]:
        a, b, group = int(e["a"]), int(e["b"]), e["group"]
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"invalid edge {a}-{b}")
        if group not in GROUPS:
            raise ValueError(f"edge {a}-{b} has unknown group {group!r}")
        edges.append(Edge(a, b, group))

    body = set(groups.get("body", ()))

    def coco_map(key: str) -> dict[int, int]:
        mapping = {int(k): int(v) for k, v in raw.get(key, {}).items()}
        bad = {k: v for k, v in mapping.items() if v not in body}
        if bad:
            raise ValueError(f"{key} targets must be body landmarks, got {bad}")
        return mapping

    return Schema(
        names=names,
        groups=groups,
        side=side,
        flip_pairs=tuple(flip_pairs),
        flip_index=tuple(flip_index),
        edges=tuple(edges),
        coco_shared=coco_map("coco_shared"),
        coco_approx=coco_map("coco_approx"),
    )
