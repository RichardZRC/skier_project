import json

import numpy as np
import pytest
import synth

from skier.data.ski2dpose import DEGENERATE_BOX, FEWER_THAN_2, load_records

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


def make_fixture(tmp_path):
    """2 videos x 2 segments x 2 images; v0 images are 200x100, v1 images are 120x90."""
    labels = synth.make_labels(videos=2, segments=2, images=2)
    images = tmp_path / "images"
    synth.make_blank_images(images, {"v0": labels["v0"]}, size=(200, 100), template=TEMPLATE)
    synth.make_blank_images(images, {"v1": labels["v1"]}, size=(120, 90), template=TEMPLATE)

    ann = labels["v0"]["s0"]["i0"]["annotation"]
    for i in range(24):
        ann[i] = [0.5, 0.5, 0]  # everything unreliable ...
    ann[16] = [0.25, 0.5, 1]  # ... except one ski tip: fewer than 2 reliable body+ski
    ann[5] = [0.9, 0.9, 1]  # a reliable pole basket doesn't count

    ann = labels["v0"]["s1"]["i1"]["annotation"]
    for i in range(24):
        ann[i] = [0.4, 0.4, 1]  # all reliable but identical: zero-size box

    info = {"v0": {"video_info": {"type": "Slalom"}, "splits_info": {"s0": {"front_view": True}}}}
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info))
    return synth.write_labels(labels, tmp_path / "labels.json"), images, info_path, labels


def test_load_records(tmp_path):
    labels_path, images, info_path, labels = make_fixture(tmp_path)
    recs = load_records(labels_path, images, info_path, path_template=TEMPLATE)
    assert [r.sample_id for r in recs] == sorted(r.sample_id for r in recs)
    assert len(recs) == 8
    by_id = {r.sample_id: r for r in recs}

    r = by_id["v1/s0/i1"]
    ann = np.array(labels["v1"]["s0"]["i1"]["annotation"])
    assert (r.image_width, r.image_height) == (120, 90)
    np.testing.assert_allclose(r.keypoints_xy, ann[:, :2] * [120, 90])  # this image's own size
    np.testing.assert_array_equal(r.reliable, ann[:, 2] == 1)
    assert r.frame_index == 10 and r.dataset == "ski2dpose" and r.image_path.is_absolute()
    assert r.excluded_reason is None and r.reference_box_xyxy is not None
    assert by_id["v0/s0/i1"].image_width == 200

    assert by_id["v0/s0/i0"].excluded_reason == FEWER_THAN_2
    assert by_id["v0/s0/i0"].reference_box_xyxy is None
    assert by_id["v0/s1/i1"].excluded_reason == DEGENERATE_BOX
    assert sum(r.excluded for r in recs) == 2

    assert (by_id["v0/s0/i1"].discipline, by_id["v0/s0/i1"].view_angle) == ("Slalom", "front")
    assert (by_id["v1/s0/i0"].discipline, by_id["v1/s0/i0"].view_angle) == (None, None)


def test_reference_box_from_reliable_body_and_ski(tmp_path):
    labels_path, images, info_path, labels = make_fixture(tmp_path)
    r = {r.sample_id: r for r in load_records(labels_path, images, path_template=TEMPLATE)}["v1/s1/i0"]
    ann = np.array(labels["v1"]["s1"]["i0"]["annotation"])
    use = (ann[:, 2] == 1) & ~np.isin(np.arange(24), [5, 9])
    xy = ann[use, :2] * [120, 90]
    np.testing.assert_allclose(r.reference_box_xyxy, [*xy.min(0), *xy.max(0)])


def test_missing_image_raises(tmp_path):
    labels_path, images, info_path, _ = make_fixture(tmp_path)
    (images / "v1/s1/i1.png").unlink()
    with pytest.raises(FileNotFoundError, match="v1/s1/i1"):
        load_records(labels_path, images, path_template=TEMPLATE)


def test_malformed_annotation_raises(tmp_path):
    labels_path, images, info_path, labels = make_fixture(tmp_path)
    labels["v1"]["s1"]["i1"]["annotation"] = labels["v1"]["s1"]["i1"]["annotation"][:20]
    synth.write_labels(labels, labels_path)
    with pytest.raises(ValueError, match="malformed"):
        load_records(labels_path, images, path_template=TEMPLATE)
