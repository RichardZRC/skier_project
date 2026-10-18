"""COCO-17 -> Ski-24 mapping for the generic baseline E0 (plan Step 3.4).

Only joints that exist in both schemas are mapped (``schema.coco_shared``): shoulders,
elbows, hips, knees, ankles. COCO wrist -> Ski hand (``schema.coco_approx``) is optional
and always reported as a separate, approximate row. The nose is never used as the head
and no neck is invented; unmapped Ski joints stay NaN.
"""

from __future__ import annotations

import numpy as np

from skier.schema import Schema


def coco17_to_ski24(pred17: np.ndarray, schema: Schema, include_approx: bool = False) -> np.ndarray:
    pred17 = np.asarray(pred17, dtype=np.float64)
    if pred17.ndim != 3 or pred17.shape[1:] != (17, 2):
        raise ValueError(f"expected [N, 17, 2], got {pred17.shape}")
    out = np.full((len(pred17), schema.num_keypoints, 2), np.nan)
    mapping = dict(schema.coco_shared)
    if include_approx:
        mapping.update(schema.coco_approx)
    for coco_idx, ski_idx in mapping.items():
        out[:, ski_idx] = pred17[:, coco_idx]
    return out


def shared_joints(schema: Schema) -> list[int]:
    return sorted(schema.coco_shared.values())


def approx_joints(schema: Schema) -> list[int]:
    return sorted(schema.coco_approx.values())
