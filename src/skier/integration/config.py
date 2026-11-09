"""Integration settings (plan Step 7), chosen on val and recorded in D24.

``None`` means "not chosen yet". Tools refuse an unset value instead of guessing one
(code spec §13). ``conf_signal = "none"`` is a decision: the tracker's score is
uninformative, so nothing is gated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

SIGNALS = ("none", "raw", "relative")


@dataclass(frozen=True)
class IntegrationConfig:
    k_pose: float | None = None  # tracker box expansion before the model's own crop padding
    conf_signal: str | None = None  # "none", "raw" or "relative" (skier.integration.confidence)
    conf_threshold: float | None = None  # a tracked frame is low_conf when its signal is below this
    replay_uses_status: bool | None = None  # low_conf frames drive the replay as lost
    pose_refine: bool | None = None  # pose-guided crop refinement in process_clip (D26)
    refine_min_score: float | None = None  # D26 s_min; the setting Step 8 scores even if pose_refine is false
    refine_k: float | None = None  # D26 k_refine

    def __post_init__(self):
        if self.conf_signal is not None and self.conf_signal not in SIGNALS:
            raise ValueError(f"conf_signal must be one of {SIGNALS}, got {self.conf_signal!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IntegrationConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown integration config keys: {sorted(unknown)}")
        return cls(**raw)

    def with_(self, **changes) -> "IntegrationConfig":
        return dataclasses.replace(self, **changes)

    def require(self, *names: str) -> "IntegrationConfig":
        """Raise unless every named setting is chosen (a gating signal also needs its threshold)."""
        missing = [n for n in names if getattr(self, n) is None]
        if "conf_signal" in names and self.conf_signal in ("raw", "relative") and self.conf_threshold is None:
            missing.append("conf_threshold")
        if "pose_refine" in names and self.pose_refine:
            missing += [n for n in ("refine_min_score", "refine_k") if getattr(self, n) is None and n not in names]
        if missing:
            raise ValueError(f"integration settings not chosen yet: {missing} (Step 7 val runs, D24, D26); set them in "
                             "configs/integration/default.yaml or on the command line")
        return self


def config_sha256(*configs) -> str:
    """One hash over configs (dataclasses or dicts, as canonical JSON), recorded with every output."""
    blob = json.dumps([dataclasses.asdict(c) if dataclasses.is_dataclass(c) else c for c in configs], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
