"""Ski 2DPose -> canonical ``PoseRecord`` list (plan Step 2.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from skier.data.records import PoseRecord
from skier.data.ski2dpose_raw import annotation_array, image_path_for, info_for, iter_raw_labels, to_pixels
from skier.geometry import body_ski_mask, reference_box
from skier.io_utils import read_json
from skier.schema import Schema, load_schema

DATASET = "ski2dpose"
FEWER_THAN_2 = "fewer_than_2_reliable_body_ski"
DEGENERATE_BOX = "degenerate_reference_box"


def load_records(
    labels_json: str | Path,
    images_root: str | Path,
    info_json: str | Path | None = None,
    schema: Schema | None = None,
    path_template: str | None = None,
) -> list[PoseRecord]:
    """Build one record per labeled image, sorted by ``sample_id``.

    Pixels use each image's own size (read from the file header). Records that can't
    give a reference box are kept with ``excluded_reason`` set. A malformed annotation
    or a missing image raises: the Step 1 audit found none, so either means something
    changed and should be looked at.
    """
    schema = schema or load_schema()
    labels = read_json(labels_json)
    info = read_json(info_json) if info_json else None
    images_root = Path(images_root)
    body_ski = body_ski_mask(schema)

    records = []
    for raw in iter_raw_labels(labels):
        ann = annotation_array(raw)
        if ann is None:
            raise ValueError(f"{raw.sample_id}: malformed annotation (run the label audit)")
        if raw.frame_idx is None:
            raise ValueError(f"{raw.sample_id}: missing frame_idx (run the label audit)")
        path = image_path_for(images_root, raw.video_id, raw.segment_id, raw.image_id, path_template)
        if not path.is_file():
            raise FileNotFoundError(f"{raw.sample_id}: no image at {path}")
        with Image.open(path) as im:
            width, height = im.size

        xy = to_pixels(ann[:, :2], width, height)
        reliable = ann[:, 2] == 1
        box = reference_box(xy, reliable, schema)
        if box is not None:
            excluded = None
        elif (reliable & body_ski).sum() < 2:
            excluded = FEWER_THAN_2
        else:
            excluded = DEGENERATE_BOX
        meta = info_for(info, raw.video_id, raw.segment_id) if info is not None else {}

        records.append(PoseRecord(
            sample_id=raw.sample_id,
            dataset=DATASET,
            source_video_id=raw.video_id,
            segment_id=raw.segment_id,
            image_id=raw.image_id,
            frame_index=int(raw.frame_idx),
            image_path=path.resolve(),
            image_width=width,
            image_height=height,
            keypoints_xy=xy,
            reliable=reliable,
            raw_visibility=ann[:, 2].copy(),
            reference_box_xyxy=box,
            discipline=meta.get("discipline"),
            view_angle=meta.get("view_angle"),
            excluded_reason=excluded,
        ))
    records.sort(key=lambda r: r.sample_id)
    return records
