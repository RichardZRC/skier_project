import json

import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.geometry import apply_affine, crop_affine
from skier.pose.coco_map import coco17_to_ski24, shared_joints
from skier.pose.crosscheck import compare_points, pick_pilot_records
from skier.pose.eval_generic import score_e0, select_partition
from skier.pose.heatmap import encode
from skier.pose.infer import predict_records
from skier.pose.model import (FINAL_LAYER_KEYS, check_load_result, check_preprocessor, filter_checkpoint,
                              packaged_coco_config)
from skier.pose.preprocess import INPUT_H, INPUT_W, MEAN, STD, pose_crop_box

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


@pytest.fixture
def records(tmp_path):
    labels = synth.make_labels(videos=2, segments=2, images=3, seed=7)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(640, 360), template=TEMPLATE, value=90)
    return load_records(synth.write_labels(labels, tmp_path / "labels.json"), images, path_template=TEMPLATE)


def oracle_heatmap_fn(records, points_of):
    """A fake model: for the i-th crop, draw Gaussians at that record's ``points_of(r)`` in input space."""
    queue = list(records)

    def fn(batch):
        out = []
        for _ in range(len(batch)):
            r = queue.pop(0)
            M = crop_affine(pose_crop_box(r.reference_box_xyxy), INPUT_W, INPUT_H)
            pts = points_of(r)
            target, _ = encode(apply_affine(M, pts), np.isfinite(pts).all(axis=1))
            out.append(target)
        return np.stack(out)
    return fn


def test_predict_records_round_trip(records):
    recs = [r for r in records if not r.excluded]
    preds = predict_records(recs, oracle_heatmap_fn(recs, lambda r: r.keypoints_xy), batch_size=4)
    assert preds.sample_ids == [r.sample_id for r in recs]
    n_checked = 0
    for r, xy in zip(recs, preds.xy):
        box = pose_crop_box(r.reference_box_xyxy)
        px_per_input = (box[2] - box[0]) / INPUT_W
        hm = apply_affine(crop_affine(box, INPUT_W, INPUT_H), r.keypoints_xy) / 4
        # MMPose's decoder skips the quarter step in the outer heatmap pixels, so the error
        # bound there is sqrt(2^2 + 1^2) input px instead of 1.5.
        edge = (hm[:, 0] < 1.5) | (hm[:, 0] >= 46.5) | (hm[:, 1] < 1.5) | (hm[:, 1] >= 62.5)
        tol = np.where(edge, 2.25, 1.5) * px_per_input + 1e-6
        ok = np.isfinite(xy).all(axis=1)
        assert ok.sum() >= 20
        err = np.linalg.norm(xy[ok] - r.keypoints_xy[ok], axis=1)
        assert (err <= tol[ok]).all()
        n_checked += ok.sum()
    assert n_checked > 200
    assert preds.timing["n_images"] == len(recs)


def test_predict_refuses_excluded(records):
    records[0].excluded_reason = "x"
    with pytest.raises(ValueError, match="excluded"):
        predict_records(records, lambda b: np.zeros((len(b), 24, 64, 48)))


def test_coco_mapping(schema):
    pred17 = np.arange(2 * 17 * 2, dtype=float).reshape(2, 17, 2)
    out = coco17_to_ski24(pred17, schema)
    assert np.isnan(out[:, 0]).all() and np.isnan(out[:, 1]).all()  # no head from the nose, no neck
    np.testing.assert_array_equal(out[:, 6], pred17[:, 5])  # COCO left shoulder -> shoulder_left
    np.testing.assert_array_equal(out[:, 12], pred17[:, 16])  # COCO right ankle -> ankle_right
    assert np.isnan(out[:, 8]).all()  # hand only with include_approx
    np.testing.assert_array_equal(coco17_to_ski24(pred17, schema, include_approx=True)[:, 8], pred17[:, 9])
    assert np.isnan(out[:, list(schema.groups["ski"])]).all()
    assert len(shared_joints(schema)) == 10


def test_score_e0_perfect_and_na(records, schema):
    recs = [r for r in records if not r.excluded]
    pred17 = np.full((len(recs), 17, 2), np.nan)
    for coco, ski in {**schema.coco_shared, **schema.coco_approx}.items():
        pred17[:, coco] = np.stack([r.keypoints_xy[ski] for r in recs])
    m = score_e0(pred17, recs, schema)
    assert m["body_shared"]["pck@0.1"]["pct"] == 100.0
    assert m["approx_wrist_to_hand"]["pck@0.1"]["pct"] == 100.0
    assert m["ski"].startswith("N/A")
    assert set(m["per_video_body_shared_pck@0.1"]) == {"v0", "v1"}
    json.dumps(m)  # serializable


def test_select_partition(records):
    records[0].excluded_reason = "fewer_than_2_reliable_body_ski"
    manifest = {"partitions": {"train": {"videos": ["v0"]}, "val": {"videos": ["v1"]}}}
    eligible, excluded = select_partition(records, manifest, "train")
    assert {r.source_video_id for r in eligible} == {"v0"} and len(excluded) == 1
    with pytest.raises(ValueError, match="no partition"):
        select_partition(records, manifest, "test")


def test_crosscheck_helpers(records):
    a = np.zeros((2, 17, 2))
    b = a.copy()
    b[0, 0] = (3, 4)
    r = compare_points(a, b)
    assert r["max_px"] == 5.0 and r["mean_px"] == pytest.approx(5 / 34)
    chosen = pick_pilot_records(records, ["v1"], n=4)
    assert len(chosen) == 4 and all(c.source_video_id == "v1" for c in chosen)
    assert [c.sample_id for c in chosen] == [c.sample_id for c in pick_pilot_records(records, ["v1"], n=4)]


def test_checkpoint_filter_and_load_check():
    state = {"backbone.conv1.weight": 1, "head.final_layer.weight": 2, "head.final_layer.bias": 3}
    kept, dropped = filter_checkpoint(state)
    assert kept == {"backbone.conv1.weight": 1} and dropped == sorted(FINAL_LAYER_KEYS)
    check_load_result(sorted(FINAL_LAYER_KEYS), [], replaced_head=True)
    check_load_result([], [], replaced_head=False)
    with pytest.raises(RuntimeError, match="missing"):
        check_load_result(["head.final_layer.weight", "backbone.conv1.weight"], [], replaced_head=True)
    with pytest.raises(RuntimeError, match="doesn't"):
        check_load_result([], ["extra.weight"], replaced_head=False)


def test_check_preprocessor():
    # MMPose stores them as 1 x 3 x 1 x 1 float32 (RGB, since bgr_to_rgb=True)
    good = {"data_preprocessor.mean": MEAN.reshape(3, 1, 1), "data_preprocessor.std": STD.reshape(3, 1, 1)}
    check_preprocessor(good)
    check_preprocessor({})  # our own checkpoints have none
    with pytest.raises(RuntimeError, match="mean"):
        check_preprocessor({**good, "data_preprocessor.mean": MEAN[::-1].copy()})  # BGR order
    with pytest.raises(RuntimeError, match="std"):
        check_preprocessor({**good, "data_preprocessor.std": STD + 1})


@pytest.mark.stack
@pytest.mark.parametrize("k", [17, 24])
def test_build_hrnet_output_shape(k):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mmpose")
    config = packaged_coco_config()
    if config is None:
        pytest.skip("HRNet COCO config not found inside the mmpose package")
    from skier.pose.model import build_hrnet

    model, info = build_hrnet(k, config)
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 192))
    assert tuple(out.shape) == (2, k, 64, 48)
    assert info["replaced_head"] == (k != 17)


@pytest.mark.stack
def test_build_hrnet_loads_mmpose_style_checkpoint(tmp_path):
    """A COCO checkpoint also carries ``data_preprocessor.mean/std``, which PoseNet lacks."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mmpose")
    config = packaged_coco_config()
    if config is None:
        pytest.skip("HRNet COCO config not found inside the mmpose package")
    from skier.pose.model import build_hrnet

    coco, _ = build_hrnet(17, config)
    state = dict(coco.state_dict())
    state["data_preprocessor.mean"] = torch.tensor(MEAN).view(1, 3, 1, 1)
    state["data_preprocessor.std"] = torch.tensor(STD).view(1, 3, 1, 1)
    torch.save({"state_dict": state, "meta": {}}, tmp_path / "coco.pth")
    model, info = build_hrnet(24, config, tmp_path / "coco.pth")
    assert info["replaced_head"] and info["skipped_tensors"] == sorted(FINAL_LAYER_KEYS)
    assert info["loaded_tensors"] == len(state) - 4
    torch.testing.assert_close(model.backbone.conv1.weight, coco.backbone.conv1.weight)

    state["data_preprocessor.std"] = state["data_preprocessor.std"] * 2
    torch.save({"state_dict": state}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="std"):
        build_hrnet(24, config, tmp_path / "bad.pth")
