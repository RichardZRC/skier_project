import dataclasses
import json

import cv2
import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.geometry import apply_affine, box_center_size, invert_affine
from skier.paths import configs_dir
from skier.pose.augment import (N_DRAWS, AugConfig, AugParams, BlurConfig, JitterConfig, OcclusionConfig,
                                blur_kernel, crop_transform, flip_labels, motion_blur, occlude, sample_aug)
from skier.pose.dataset import CropCache, PoseDataset, make_sample, read_image, required_region
from skier.pose.preprocess import INPUT_H, INPUT_W, MEAN, STD, make_pose_crop, pose_crop_box
from skier.pose.train import TrainConfig

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

WIDE_JITTER = AugConfig(jitter={"shift": 0.2, "scale_range": [0.8, 1.3]})


@pytest.mark.parametrize("aug", [AugConfig(), "e4", WIDE_JITTER], ids=["e1", "e4", "wide-jitter"])
def test_required_region_covers_every_augmented_crop(records, aug):
    aug = experiment(4).aug if aug == "e4" else aug
    rng = np.random.default_rng(3)
    corners = np.array([[0, 0], [INPUT_W - 1, 0], [0, INPUT_H - 1], [INPUT_W - 1, INPUT_H - 1]], float)
    for r in records:
        x1, y1, x2, y2 = required_region(r.reference_box_xyxy, aug, 1.25, margin_px=0)
        for _ in range(200):
            back = apply_affine(invert_affine(crop_transform(r.reference_box_xyxy, sample_aug(rng, aug))), corners)
            assert (back[:, 0] >= x1 - 1e-6).all() and (back[:, 0] <= x2 + 1e-6).all()
            assert (back[:, 1] >= y1 - 1e-6).all() and (back[:, 1] <= y2 + 1e-6).all()


@pytest.mark.parametrize("name", ["e1", "e4"])
def test_cached_equals_uncached(records, schema, tmp_path, name):
    aug = experiment(int(name[1])).aug
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
    e1 = CropCache.build(tmp_path / "cache_e1", records[:2], experiment(1).aug)
    with pytest.raises(ValueError, match="rebuild"):  # an E1 cache doesn't cover E2's jitter
        PoseDataset(records[:2], schema, train=True, aug_cfg=experiment(2).aug, cache=e1)
    PoseDataset(records[:2], schema, train=True, aug_cfg=experiment(3).aug, cache=e1)  # blur/occlusion: pixels only
    e4 = CropCache.build(tmp_path / "cache_e4", records[:2], experiment(4).aug)
    for n in (1, 2, 3, 4):  # one E4 cache serves every experiment
        PoseDataset(records[:2], schema, train=True, aug_cfg=experiment(n).aug, cache=e4)


# --- Step 6: crop jitter, motion blur, occlusion (D16) ------------------------------------

def experiment(n: int) -> TrainConfig:
    return TrainConfig.from_yaml(configs_dir() / "pose" / f"e{n}.yaml")


def raw_rgb(x: np.ndarray) -> np.ndarray:
    return x.transpose(1, 2, 0) * STD + MEAN  # back to 0..255, RGB


def test_e2_to_e4_differ_from_e1_only_in_aug():
    base = experiment(1).to_dict()
    base.pop("aug")
    expected = {2: {"jitter"}, 3: {"blur", "occlusion"}, 4: {"jitter", "blur", "occlusion"}}
    for n in (2, 3, 4):
        cfg = experiment(n)
        d = cfg.to_dict()
        aug = d.pop("aug")
        assert d == base
        assert {k for k in ("jitter", "blur", "occlusion") if aug[k] is not None} == expected[n]
        assert (cfg.aug.flip_p, cfg.aug.rot_deg, cfg.aug.scale_range) == (0.5, 15.0, (0.8, 1.2))
        assert TrainConfig.from_dict(json.loads(json.dumps(d | {"aug": aug}))) == cfg  # checkpoint round trip
    e4 = experiment(4).aug
    assert (e4.jitter, e4.blur, e4.occlusion) == (JitterConfig(), BlurConfig(), OcclusionConfig())


@pytest.mark.parametrize("bad", [
    {"jitter": {"shfit": 0.1}}, {"jitter": {"scale_range": [0, 1.1]}}, {"jitter": {"shift": 0.5}}, {"jitter": 0.1},
    {"blur": {"lengths": [3, 4]}}, {"blur": {"p": 1.5}}, {"blur": {"lengths": []}},
    {"occlusion": {"area_range": [0.2, 0.1]}}, {"occlusion": {"area_range": [0.5, 0.9]}},
])
def test_step6_config_validation(bad):
    with pytest.raises(ValueError):
        AugConfig.from_dict(bad)


def test_sample_aug_paired_across_experiments():
    """Same (seed, epoch, index): same flip/rotation/scale everywhere; shared augmentations agree."""
    aug = {n: experiment(n).aug for n in (1, 2, 3, 4)}
    for i in range(300):
        p = {n: sample_aug(np.random.default_rng([230, 2, i]), a) for n, a in aug.items()}
        assert len({(q.flip, q.rot_deg, q.scale) for q in p.values()}) == 1
        assert (p[2].shift, p[2].jitter_scale) == (p[4].shift, p[4].jitter_scale)
        pix = lambda q: (q.blur_length, q.blur_angle, q.occlusion, q.occlusion_color)  # noqa: E731
        assert pix(p[3]) == pix(p[4])
        assert p[1] == AugParams(flip=p[1].flip, rot_deg=p[1].rot_deg, scale=p[1].scale)
        assert p[3].shift == (0.0, 0.0) and p[2].blur_length == 0 and p[2].occlusion is None


class FixedRng:
    """Stands in for a Generator: ``random(n)`` returns a chosen vector."""

    def __init__(self, u):
        self.u = np.asarray(u, dtype=float)

    def random(self, n):
        assert n == N_DRAWS
        return self.u.copy()


def test_each_draw_slot_drives_one_parameter():
    """Slots never share a draw, so augmentations stay independent (and paired across experiments)."""
    cfg = AugConfig(jitter={"p": 0.5}, blur={}, occlusion={})  # every on/off draw matters
    base_u = np.full(N_DRAWS, 0.3)  # everything on
    base = sample_aug(FixedRng(base_u), cfg)
    owners = {}
    for k in range(N_DRAWS):
        u = base_u.copy()
        u[k] = 0.9
        q = sample_aug(FixedRng(u), cfg)
        owners[k] = {f.name for f in dataclasses.fields(AugParams) if getattr(q, f.name) != getattr(base, f.name)}
    jitter, blur, occ = {"shift", "jitter_scale"}, {"blur_length", "blur_angle"}, {"occlusion", "occlusion_color"}
    assert owners == {0: {"flip"}, 1: {"rot_deg"}, 2: {"scale"}, 3: jitter, 4: {"shift"}, 5: {"shift"},
                      6: {"jitter_scale"}, 7: blur, 8: {"blur_length"}, 9: {"blur_angle"}, 10: occ,
                      11: {"occlusion"}, 12: {"occlusion"}, 13: {"occlusion"}, 14: {"occlusion"},
                      15: {"occlusion_color"}, 16: {"occlusion_color"}, 17: {"occlusion_color"}}


def test_step6_params_in_range():
    ps = [sample_aug(np.random.default_rng([230, 0, i]), experiment(4).aug) for i in range(1000)]
    assert all(max(abs(q.shift[0]), abs(q.shift[1])) <= 0.1 and 0.9 <= q.jitter_scale <= 1.1 for q in ps)
    assert len({q.shift for q in ps}) == 1000  # jitter p = 1
    blurred, occluded = [q for q in ps if q.blur_length], [q for q in ps if q.occlusion]
    assert 400 < len(blurred) < 600 and 400 < len(occluded) < 600
    assert {q.blur_length for q in blurred} == {3, 5, 7, 9}
    assert all(0 <= q.blur_angle < 180 for q in blurred)
    for q in occluded:
        x, y, w, h = q.occlusion
        assert x >= 0 and y >= 0 and x + w <= INPUT_W and y + h <= INPUT_H
        assert 0.045 <= w * h / (INPUT_W * INPUT_H) <= 0.155  # whole pixels
        assert 0.45 <= w / h <= 2.2 and all(0 <= c <= 255 for c in q.occlusion_color)


def test_jitter_moves_the_crop_and_the_labels_together(records, schema):
    r = records[2]
    params = AugParams(shift=(0.08, -0.06), jitter_scale=1.07, rot_deg=5.0, scale=0.95)
    M = crop_transform(r.reference_box_xyxy, params)
    c_ref, (w, h) = box_center_size(r.reference_box_xyxy)
    crop_center = apply_affine(invert_affine(M), np.array([[INPUT_W / 2, INPUT_H / 2]]))[0]
    np.testing.assert_allclose(crop_center, c_ref + [0.08 * w, -0.06 * h], atol=1e-6)
    plain = crop_transform(r.reference_box_xyxy, AugParams(rot_deg=5.0, scale=0.95))
    s_m, s_p = np.linalg.norm(M[:, 0]), np.linalg.norm(plain[:, 0])
    assert s_p / s_m == pytest.approx(1.07)  # 7% larger crop region -> 7% smaller pixels
    # A blob drawn at a labeled point lands where the jittered label says.
    k = int(np.flatnonzero(r.reliable)[0])
    x0, y0 = r.keypoints_xy[k]
    ys, xs = np.mgrid[: r.image_height, : r.image_width]
    blob = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * 3.0 ** 2)) * 255
    img = np.repeat(blob[..., None], 3, axis=2).astype(np.uint8)
    s = make_sample(r, img, params, schema)
    g = raw_rgb(s.x)[..., 0]
    gy, gx = np.mgrid[:INPUT_H, :INPUT_W]
    centroid = np.array([(g * gx).sum(), (g * gy).sum()]) / g.sum()
    assert np.linalg.norm(centroid - s.keypoints_input[k]) < 0.3
    np.testing.assert_allclose(s.keypoints_input, apply_affine(M, r.keypoints_xy))


@pytest.mark.parametrize("length", [3, 5, 7, 9])
def test_blur_kernel_is_centered_and_normalized(length):
    yy, xx = np.mgrid[:length, :length]
    c = (length - 1) / 2
    for angle in (0, 30, 45, 90, 137, 179):
        k = blur_kernel(length, angle)
        assert k.shape == (length, length) and abs(k.sum() - 1) < 1e-5 and (k >= 0).all()
        assert abs((k * xx).sum() - c) < 0.05 and abs((k * yy).sum() - c) < 0.05
    with pytest.raises(ValueError):
        blur_kernel(length + 1, 0)


def test_motion_blur_spreads_along_its_direction_without_moving_points():
    img = np.zeros((INPUT_H, INPUT_W, 3), np.float32)
    img[100, 80] = 255.0
    flat = motion_blur(img, 9, 0.0)[..., 0]
    assert (flat[100, 76:85] > 1).all() and max(abs(flat[100, 75]), abs(flat[100, 85]), abs(flat[99, 80])) < 1e-6
    for angle in (0.0, 30.0, 100.0):
        g = motion_blur(img, 7, angle)[..., 0].astype(np.float64)
        gy, gx = np.mgrid[:INPUT_H, :INPUT_W]
        assert abs((g * gx).sum() / g.sum() - 80) < 0.05 and abs((g * gy).sum() / g.sum() - 100) < 0.05
    u8 = (img > 0).astype(np.uint8) * 255
    assert motion_blur(u8, 5, 45.0).dtype == np.uint8
    assert motion_blur(u8, 1, 45.0) is u8 and motion_blur(u8, 0, 45.0) is u8


def test_pixel_augmentations_leave_labels_alone(records, schema):
    r = records[0]
    img = read_image(r.image_path)
    base = AugParams(rot_deg=4.0, shift=(0.03, 0.02))
    a = make_sample(r, img, base, schema)
    k = int(np.flatnonzero(a.weight)[0])  # put the patch over a supervised point
    x0 = int(np.clip(a.keypoints_input[k][0] - 25, 0, INPUT_W - 50))
    y0 = int(np.clip(a.keypoints_input[k][1] - 20, 0, INPUT_H - 40))
    occ = dataclasses.replace(base, occlusion=(x0, y0, 50, 40), occlusion_color=(10, 200, 30))
    blur = dataclasses.replace(base, blur_length=7, blur_angle=60.0)
    b, c = (make_sample(r, img, p, schema) for p in (occ, blur))
    assert b.weight[k] == 1  # still supervised under the patch (D16)
    for s in (b, c):
        np.testing.assert_array_equal(s.target, a.target)
        np.testing.assert_array_equal(s.weight, a.weight)
        np.testing.assert_array_equal(s.keypoints_input, a.keypoints_input)
    ra, rb = raw_rgb(a.x), raw_rgb(b.x)
    patch = (slice(y0, y0 + 40), slice(x0, x0 + 50))
    np.testing.assert_allclose(rb[patch], np.broadcast_to([30, 200, 10], (40, 50, 3)), atol=1e-3)  # BGR -> RGB
    mask = np.ones((INPUT_H, INPUT_W), bool)
    mask[patch] = False
    np.testing.assert_allclose(rb[mask], ra[mask], atol=1e-3)
    assert np.abs(raw_rgb(c.x) - ra).mean() > 0.5  # the blur did something
    with pytest.raises(ValueError, match="inside"):
        occlude(np.zeros((INPUT_H, INPUT_W, 3), np.uint8), (180, 0, 20, 10), (0, 0, 0))


def test_dataset_applies_step6_augmentation_reproducibly(records, schema):
    aug = experiment(4).aug
    a = PoseDataset(records, schema, train=True, aug_cfg=aug, seed=7)
    b = PoseDataset(records, schema, train=True, aug_cfg=aug, seed=7)
    for i in range(len(records)):
        np.testing.assert_array_equal(a[i]["x"], b[i]["x"])
        np.testing.assert_array_equal(a[i]["target"], b[i]["target"])
    assert any(a.params_for(i).occlusion is not None for i in range(len(records)))
    assert all(a.params_for(i).shift != (0.0, 0.0) for i in range(len(records)))



def test_augmentation_preview_is_paired(records, schema):
    from skier.pose.aug_preview import UPSCALE, describe, preview, tile

    idx = [0, 2, 5]
    t2, p2 = preview(records, experiment(2), schema, idx)
    t4, p4 = preview(records, experiment(4), schema, idx)
    assert len(t4) == 3 and t4[0].shape == (INPUT_H * UPSCALE, INPUT_W * UPSCALE, 3) and t4[0].dtype == np.uint8
    assert [p["sample_id"] for p in p4] == [records[i].sample_id for i in idx]
    assert [(p["shift"], p["jitter_scale"], p["rot_deg"]) for p in p2] == \
        [(p["shift"], p["jitter_scale"], p["rot_deg"]) for p in p4]  # same jitter and geometry in both sheets
    sheet = tile(t4, cols=2)
    assert sheet.shape == (2 * INPUT_H * UPSCALE, 2 * INPUT_W * UPSCALE, 3) and not sheet[INPUT_H * UPSCALE:, INPUT_W * UPSCALE:].any()
    lines = describe(AugParams(flip=True, rot_deg=-7.2, shift=(0.05, -0.02), blur_length=5, blur_angle=33.0,
                               occlusion=(0, 0, 96, 64)))
    assert lines == ["flip rot -7 scale 1.00", "jitter +0.05,-0.02 x1.00", "blur 5px @33 occl 12%"]
