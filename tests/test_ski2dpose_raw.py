import numpy as np
import pytest

from skier.data import ski2dpose_raw as raw_mod
from skier.data.ski2dpose_raw import (
    RawLabel,
    annotation_array,
    image_path_for,
    info_for,
    info_preview,
    iter_raw_labels,
    make_sample_id,
    reliable_mask,
    safe_name,
    to_pixels,
)
from skier.geometry import box_from_points, reference_box


def test_iteration_order_is_sorted():
    labels = {"vB": {"s1": {"i2": {"annotation": [], "frame_idx": 0}, "i1": {"annotation": []}}},
              "vA": {"s0": {"i0": "not a dict"}}}
    got = [(r.video_id, r.segment_id, r.image_id) for r in iter_raw_labels(labels)]
    assert got == [("vA", "s0", "i0"), ("vB", "s1", "i1"), ("vB", "s1", "i2")]
    recs = list(iter_raw_labels(labels))
    assert recs[0].annotation is None and recs[0].frame_idx is None  # entry not a dict
    assert recs[1].frame_idx is None  # key missing
    assert recs[2].sample_id == "vB/s1/i2"


def test_bad_nesting_raises():
    with pytest.raises(ValueError):
        list(iter_raw_labels([1, 2]))
    with pytest.raises(ValueError):
        list(iter_raw_labels({"v": ["not", "a", "dict"]}))


def test_ids():
    sid = make_sample_id("v1", "s2", "i3")
    assert sid == "v1/s2/i3"
    assert safe_name(sid) == "v1__s2__i3"


def test_to_pixels():
    xy = np.array([[0, 0], [1, 1], [0.25, 0.5]])
    np.testing.assert_allclose(to_pixels(xy, 200, 100), [[0, 0], [200, 100], [50, 50]])


def test_annotation_array_and_mask():
    good = RawLabel("v", "s", "i", [[0.1, 0.2, 1]] * 23 + [[0.3, 0.4, 0]], 0)
    arr = annotation_array(good)
    assert arr.shape == (24, 3) and arr.dtype == np.float64
    assert reliable_mask(arr).sum() == 23
    assert annotation_array(RawLabel("v", "s", "i", [[0.1, 0.2, 1]] * 23, 0)) is None
    assert annotation_array(RawLabel("v", "s", "i", [[1, 2], [3]], 0)) is None
    assert annotation_array(RawLabel("v", "s", "i", None, 0)) is None


def test_image_path_for(tmp_path, monkeypatch):
    p = image_path_for(tmp_path, "v", "s", "i", "{video_id}/{segment_id}/{image_id}.jpg")
    assert p == tmp_path / "v" / "s" / "i.jpg"
    monkeypatch.setattr(raw_mod, "IMAGE_PATH_TEMPLATE", None)
    with pytest.raises(NotImplementedError, match="1E"):
        image_path_for(tmp_path, "v", "s", "i")
    monkeypatch.setattr(raw_mod, "IMAGE_PATH_TEMPLATE", "{image_id}.png")
    assert image_path_for(tmp_path, "v", "s", "i") == tmp_path / "i.png"


def test_info_preview_has_keys_not_values():
    info = {"v1": {"s1": {"location": "Kitzbuehel", "angle": 45.0}}}
    preview = info_preview(info)
    text = str(preview)
    assert "location" in text and "angle" in text
    assert "Kitzbuehel" not in text and "45.0" not in text


def test_info_for_real_layout():
    info = {"v1": {"video_info": {"type": "Giant Slalom", "location": "x"},
                   "splits_info": {"0": {"front_view": False, "side_view": True, "back_view": True,
                                         "start_time_stamp": "00:00:01.000"}}}}
    assert info_for(info, "v1", "0") == {"discipline": "Giant Slalom", "view_angle": "side+back"}
    assert info_for(info, "v1", "9") == {"discipline": "Giant Slalom", "view_angle": None}
    assert info_for(info, "nope", "0") == {"discipline": None, "view_angle": None}
    none_true = {"v": {"video_info": {"type": "Slalom"}, "splits_info": {"0": {"front_view": False}}}}
    assert info_for(none_true, "v", "0")["view_angle"] is None


def test_default_template_matches_dataset_layout(tmp_path):
    p = image_path_for(tmp_path, "uLW74013Wp0", "0", "uLW74013Wp0_0_00001")
    assert p == tmp_path / "Images_jpg" / "uLW74013Wp0" / "0" / "uLW74013Wp0_0_00001.jpg"


def test_box_from_points():
    xy = np.array([[1, 5], [4, 2], [100, 100]])
    np.testing.assert_allclose(box_from_points(xy, [True, True, False]), [1, 2, 4, 5])
    with pytest.raises(ValueError):
        box_from_points(xy, [True, False, False])


def test_reference_box(schema):
    xy = np.full((24, 2), np.nan)
    reliable = np.zeros(24, dtype=bool)
    xy[0], xy[16] = (10, 10), (30, 50)  # head, ski tip: both count
    reliable[[0, 16]] = True
    xy[5] = (500, 500)  # pole basket: never counts
    reliable[5] = True
    xy[12] = (-50, -50)  # unreliable ankle: excluded
    np.testing.assert_allclose(reference_box(xy, reliable, schema), [10, 10, 30, 50])

    xy[16] = (10, 10)  # two identical points: diagonal 0
    assert reference_box(xy, reliable, schema) is None
    reliable[16] = False  # one point left
    assert reference_box(xy, reliable, schema) is None
