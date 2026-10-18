import cv2
import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.geometry import apply_affine, invert_affine
from skier.pose.augment import AugConfig, AugParams, crop_transform, flip_labels, sample_aug
from skier.pose.dataset import CropCache, PoseDataset, make_sample, read_image, required_region
from skier.pose.preprocess import INPUT_H, INPUT_W, MEAN, STD, make_pose_crop, pose_crop_box

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


@pytest.fixture
def records(tmp_path):
    labels = synth.make_labels(videos=1, segments=2, images=4, seed=11)
    images = tmp_path / "images"
    rng = np.random.default_rng(0)
    for seg, imgs in labels["v0"].items():  # textured images, so warps are comparable pixel by pixel
        for image_id in imgs:
            p = images / TEMPLATE.format(video_id="v0", segment_id=seg, image_id=image_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (5, 5), 0))
    return [r for r in load_records(synth.write_labels(labels, tmp_path / "l.json"), images, path_template=TEMPLATE)
            if not r.excluded]


# --- augmentation -----------------------------------------------------------------------

def test_sample_aug_deterministic_and_in_range():
    cfg = AugConfig()
    a = [sample_aug(np.random.default_rng([230, 0, i]), cfg) for i in range(1000)]
    assert a == [sample_aug(np.random.default_rng([230, 0, i]), cfg) for i in range(1000)]
    assert all(-15 <= p.rot_deg <= 15 and 0.8 <= p.scale <= 1.2 for p in a)
    assert 400 < sum(p.flip for p in a) < 600
    none = sample_aug(np.random.default_rng(0), AugConfig.none())
    assert none == AugParams.identity()
    with pytest.raises(NotImplementedError, match="Step 6"):
        sample_aug(np.random.default_rng(0), AugConfig(blur={"p": 0.5}))
    with pytest.raises(ValueError, match="unknown"):
        AugConfig.from_dict({"flip": 0.5})


def test_flipped_crop_is_mirror_of_unflipped(records, schema):
    r = records[0]
    img = read_image(r.image_path)
    a = make_sample(r, img, AugParams(flip=False, rot_deg=7.0, scale=1.1), schema)
    b = make_sample(r, img, AugParams(flip=True, rot_deg=7.0, scale=1.1), schema)
    raw_a = (a.x.transpose(1, 2, 0) * STD + MEAN)
    raw_b = (b.x.transpose(1, 2, 0) * STD + MEAN)
    d = np.abs(raw_b - raw_a[:, ::-1])  # vs cv2.flip of the unflipped crop
    # OpenCV's fixed-point interpolation rounds mirrored sample positions slightly differently
    # (a few grey levels at isolated pixels); an off-by-one flip would be off by ~10 on average.
    assert d.mean() < 0.1 and d.max() <= 6.0


def test_flip_swaps_left_and_right(records, schema):
    r = records[0]
    img = read_image(r.image_path)
    plain = make_sample(r, img, AugParams(), schema)
    flipped = make_sample(r, img, AugParams(flip=True), schema)
    # new ski_tip_left (20) = mirror of old ski_tip_right (16), and vice versa
    np.testing.assert_allclose(flipped.keypoints_input[20], [INPUT_W - 1 - plain.keypoints_input[16][0],
                                                             plain.keypoints_input[16][1]], atol=1e-9)
    np.testing.assert_allclose(flipped.keypoints_input[16][0], INPUT_W - 1 - plain.keypoints_input[20][0], atol=1e-9)
    np.testing.assert_allclose(flipped.keypoints_input[0][0], INPUT_W - 1 - plain.keypoints_input[0][0], atol=1e-9)  # head
    np.testing.assert_array_equal(flipped.weight[20], plain.weight[16])
    xy, rel = flip_labels(*flip_labels(r.keypoints_xy, r.reliable, schema), schema)
    np.testing.assert_array_equal(xy, r.keypoints_xy)  # flipping twice restores the labels


def test_identity_sample_matches_eval_crop(records, schema):
    """Training with no augmentation sees exactly the crop that evaluation uses."""
    r = records[1]
    img = read_image(r.image_path)
    s = make_sample(r, img, AugParams.identity(), schema)
    x, M = make_pose_crop(img, r.reference_box_xyxy)
    np.testing.assert_array_equal(s.x, x)
    np.testing.assert_allclose(s.M, M)


def test_rotation_and_scale_move_points_like_M(records, schema):
    r = records[2]
    params = AugParams(rot_deg=-12.0, scale=0.85)
    M = crop_transform(r.reference_box_xyxy, params)
    s = make_sample(r, read_image(r.image_path), params, schema)
    np.testing.assert_allclose(s.keypoints_input, apply_affine(M, r.keypoints_xy))
    inside = ((s.keypoints_input >= 0) & (s.keypoints_input < [INPUT_W, INPUT_H])).all(axis=1)
    np.testing.assert_array_equal(s.weight.astype(bool), r.reliable & inside)


def test_points_outside_crop_get_zero_weight(records, schema):
    r = records[3]
    s = make_sample(r, read_image(r.image_path), AugParams(scale=0.5), schema)  # zoom in: some points leave the crop
    outside = ~((s.keypoints_input >= 0) & (s.keypoints_input < [INPUT_W, INPUT_H])).all(axis=1)
    assert outside.any() and (s.weight[outside] == 0).all() and (s.target[outside] == 0).all()


# --- dataset ----------------------------------------------------------------------------

def test_dataset_reproducible_and_epoch_dependent(records, schema):
    a = PoseDataset(records, schema, train=True, seed=1)
    b = PoseDataset(records, schema, train=True, seed=1)
    assert [a.params_for(i) for i in range(len(a))] == [b.params_for(i) for i in range(len(b))]
    e0 = [a.params_for(i) for i in range(len(a))]
    a.set_epoch(1)
    assert [a.params_for(i) for i in range(len(a))] != e0
    item = a[0]
    assert item["x"].shape == (3, INPUT_H, INPUT_W) and item["target"].shape == (24, 64, 48)
    assert item["sample_id"] == records[0].sample_id
    ev = PoseDataset(records, schema, train=False)
    assert all(ev.params_for(i) == AugParams.identity() for i in range(len(ev)))


def test_dataset_refuses_excluded(records, schema):
    records[0].excluded_reason = "x"
    with pytest.raises(ValueError, match="excluded"):
        PoseDataset(records, schema, train=True)


# --- crop cache --------------------------------------------------------------------------

def test_required_region_covers_every_augmented_crop(records):
    aug = AugConfig()
    rng = np.random.default_rng(3)
    corners = np.array([[0, 0], [INPUT_W - 1, 0], [0, INPUT_H - 1], [INPUT_W - 1, INPUT_H - 1]], float)
    for r in records:
        x1, y1, x2, y2 = required_region(r.reference_box_xyxy, aug, 1.25, margin_px=0)
        for _ in range(200):
            back = apply_affine(invert_affine(crop_transform(r.reference_box_xyxy, sample_aug(rng, aug))), corners)
            assert (back[:, 0] >= x1 - 1e-6).all() and (back[:, 0] <= x2 + 1e-6).all()
            assert (back[:, 1] >= y1 - 1e-6).all() and (back[:, 1] <= y2 + 1e-6).all()


def test_cached_equals_uncached(records, schema, tmp_path):
    aug = AugConfig()
    cache = CropCache.build(tmp_path / "cache", records, aug)
    plain = PoseDataset(records, schema, train=True, aug_cfg=aug, seed=5)
    cached = PoseDataset(records, schema, train=True, aug_cfg=aug, seed=5, cache=cache)
    for epoch in range(3):
        plain.set_epoch(epoch)
        cached.set_epoch(epoch)
        for i in range(len(records)):
            a, b = plain[i], cached[i]
            np.testing.assert_array_equal(a["target"], b["target"])
            np.testing.assert_array_equal(a["weight"], b["weight"])
            diff = np.abs((a["x"] - b["x"]).transpose(1, 2, 0) * STD)  # back to 0..255 units
            assert diff.max() <= 2.0 and diff.mean() < 0.05


def test_cache_refuses_wider_augmentation(records, tmp_path, schema):
    cache = CropCache.build(tmp_path / "cache", records[:2], AugConfig(rot_deg=5.0))
    with pytest.raises(ValueError, match="rebuild"):
        PoseDataset(records[:2], schema, train=True, aug_cfg=AugConfig(rot_deg=15.0), cache=cache)
