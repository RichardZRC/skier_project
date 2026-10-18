"""Drawing: landmarks, skeleton edges, boxes, headers, contact sheets.

Functions take pixel coordinates and masks (not label records), so later steps can
reuse them for predictions. All ``draw_*`` functions draw in place and return the
image. Markers and lines use ``cv2.LINE_8`` so pixel colors are exact. Colors are BGR.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from skier.data.ski2dpose_raw import to_pixels
from skier.geometry import reference_box
from skier.schema import Schema

COLOR = {"right": (0, 140, 255), "left": (255, 255, 0), "center": (255, 255, 255)}  # orange, cyan, white
REF_BOX_COLOR = (0, 255, 255)  # yellow
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DASH = (8, 6)  # dashed: on, off (px)
DOT = (2, 4)  # dotted
DOT_SPARSE = (2, 8)  # dotted, an endpoint is a guess


def marker_radius(img: np.ndarray) -> int:
    return max(3, round(min(img.shape[:2]) / 200))


def _drawable(xy, img: np.ndarray) -> bool:
    """Finite and not absurdly far outside the image (cv2 needs C ints)."""
    x, y = float(xy[0]), float(xy[1])
    h, w = img.shape[:2]
    lim = 10 * max(h, w)
    return math.isfinite(x) and math.isfinite(y) and abs(x) < lim and abs(y) < lim


def _pt(xy) -> tuple[int, int]:
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def draw_dashed_line(img, p0, p1, color, thickness, dash: int = DASH[0], gap: int = DASH[1]):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1e-9:
        return img
    step = (p1 - p0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        cv2.line(img, _pt(p0 + step * pos), _pt(p0 + step * end), color, thickness, cv2.LINE_8)
        pos = end + gap
    return img


def draw_points(img, xy_px, reliable, schema: Schema, label_indices: bool = True):
    """Filled circle = reliable; hollow circle = occluded guess. Color = skier's side."""
    r = marker_radius(img)
    font_scale = max(0.35, r / 8)
    for i in range(schema.num_keypoints):
        if not _drawable(xy_px[i], img):
            continue
        color = COLOR[schema.side[i]]
        c = _pt(xy_px[i])
        cv2.circle(img, c, r, color, -1 if reliable[i] else 2, cv2.LINE_8)
        if label_indices:
            cv2.putText(img, str(i), (c[0] + r + 2, c[1] - r - 2), FONT, font_scale, color, 1, cv2.LINE_AA)
    return img


def draw_edges(img, xy_px, reliable, schema: Schema):
    """Body thin, skis thick, poles dotted; dashed when an endpoint is a guess."""
    t = max(1, marker_radius(img) // 2)
    for e in schema.edges:
        pa, pb = xy_px[e.a], xy_px[e.b]
        if not (_drawable(pa, img) and _drawable(pb, img)):
            continue
        sa, sb = schema.side[e.a], schema.side[e.b]
        color = COLOR[sa] if sa == sb else COLOR["center"]
        thickness = 3 * t if e.group == "ski" else t
        guess = not (reliable[e.a] and reliable[e.b])
        if e.group == "pole":
            dash, gap = DOT_SPARSE if guess else DOT
            draw_dashed_line(img, pa, pb, color, thickness, dash, gap)
        elif guess:
            draw_dashed_line(img, pa, pb, color, thickness, *DASH)
        else:
            cv2.line(img, _pt(pa), _pt(pb), color, thickness, cv2.LINE_8)
    return img


def draw_box(img, xyxy, color, dashed: bool = False, thickness: int = 2):
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if not dashed:
        cv2.rectangle(img, _pt((x1, y1)), _pt((x2, y2)), color, thickness, cv2.LINE_8)
        return img
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for k in range(4):
        draw_dashed_line(img, corners[k], corners[(k + 1) % 4], color, thickness)
    return img


def draw_header(img, lines: list[str]):
    """Text lines at the top-left, each on a black background strip."""
    scale = max(0.5, 0.9 * img.shape[0] / 1080)
    thickness = 1 if scale < 1 else 2
    y = 0
    for line in lines:
        (w, h), base = cv2.getTextSize(line, FONT, scale, thickness)
        y += h + 8
        cv2.rectangle(img, (4, y - h - 4), (4 + w + 8, y + base + 2), (0, 0, 0), -1)
        cv2.putText(img, line, (8, y), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
        y += base
    return img


def _letterbox(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0, y0 = (out_w - new_w) // 2, (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small
    return canvas


def contact_sheet(images: list[np.ndarray], cols: int = 6, thumb_w: int = 320) -> np.ndarray:
    """Grid of 16:9 thumbnails, letterboxed on black."""
    thumb_h = round(thumb_w * 9 / 16)
    rows = max(1, math.ceil(len(images) / cols))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = _letterbox(img, thumb_w, thumb_h)
    return sheet


def render_gt_overlay(img_bgr: np.ndarray, ann_norm: np.ndarray, schema: Schema, header_lines: list[str]) -> np.ndarray:
    """Ground-truth overlay on a copy of the image.

    ``ann_norm`` is ``(24, 3)`` normalized ``[x, y, visible]``; pixels use this image's
    own size. Draw order: reference box, edges, points (on top), header.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ann = np.asarray(ann_norm, dtype=np.float64)
    xy = to_pixels(ann[:, :2], w, h)
    reliable = ann[:, 2] == 1
    lines = list(header_lines)
    box = reference_box(xy, reliable, schema)
    if box is not None:
        draw_box(out, box, REF_BOX_COLOR, dashed=True, thickness=max(1, marker_radius(out) // 2))
    else:
        lines.append("no ref box")
    draw_edges(out, xy, reliable, schema)
    draw_points(out, xy, reliable, schema)
    draw_header(out, lines)
    return out


GT_MARK_COLOR = (255, 255, 255)  # white crosses: reliable labels


def prediction_overlay(img_bgr: np.ndarray, pred_xy, gt_xy, gt_reliable, schema: Schema,
                       header_lines: list[str] | None = None) -> np.ndarray:
    """Predicted skeleton (side colours, filled points) over reliable labels (white crosses)."""
    out = img_bgr.copy()
    pred = np.asarray(pred_xy, dtype=np.float64)
    shown = np.isfinite(pred).all(axis=1)
    draw_edges(out, pred, shown, schema)
    draw_points(out, pred, shown, schema, label_indices=False)
    r = marker_radius(out)
    for i in np.flatnonzero(np.asarray(gt_reliable, dtype=bool)):
        if _drawable(gt_xy[i], out):
            cv2.drawMarker(out, _pt(gt_xy[i]), GT_MARK_COLOR, cv2.MARKER_CROSS, 3 * r, 1, cv2.LINE_8)
    draw_header(out, list(header_lines or []) + ["points = prediction, crosses = labels"])
    return out
