"""Box geometry. Step 1 needs only the reference box; Step 2.1 adds the rest."""

from __future__ import annotations

import numpy as np

from skier.schema import Schema


def box_from_points(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tight ``xyxy`` box of ``xy[mask]``. Raises if fewer than 2 points are selected."""
    pts = np.asarray(xy, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(pts) < 2:
        raise ValueError(f"need at least 2 points for a box, got {len(pts)}")
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def reference_box(xy_px: np.ndarray, reliable: np.ndarray, schema: Schema) -> np.ndarray | None:
    """Box of reliable body + ski points (poles excluded); its diagonal is the PCK normalizer.

    Returns None if fewer than 2 such points exist or the diagonal is under 1 px.
    """
    xy_px = np.asarray(xy_px, dtype=np.float64)
    use = np.zeros(schema.num_keypoints, dtype=bool)
    use[list(schema.groups["body"]) + list(schema.groups["ski"])] = True
    mask = use & np.asarray(reliable, dtype=bool) & np.isfinite(xy_px).all(axis=1)
    if mask.sum() < 2:
        return None
    box = box_from_points(xy_px, mask)
    if np.hypot(box[2] - box[0], box[3] - box[1]) < 1.0:
        return None
    return box
