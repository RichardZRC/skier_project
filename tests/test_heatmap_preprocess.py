import cv2
import numpy as np
import pytest

from skier.geometry import apply_affine
from skier.pose.heatmap import decode, encode
from skier.pose.preprocess import INPUT_H, INPUT_W, MEAN, STD, make_pose_crop, normalize, pose_crop_box


# --- heatmaps ------------------------------------------------------------------------

def test_encode_decode_round_trip():
    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(12, 180, 200), rng.uniform(12, 244, 200)])
    target, weight = encode(xy, np.ones(200, bool))
    assert weight.all()
    dec, score = decode(target[None])
    err = np.linalg.norm(dec[0] - xy, axis=1)
    assert err.max() <= 1.5
    assert (score > 0.5).all()


def test_peak_location_and_value():
    target, _ = encode([[40.0, 80.0]], [True])  # heatmap (10, 20), exactly on a pixel
    assert np.unravel_index(target[0].argmax(), target[0].shape) == (20, 10)
    assert target[0, 20, 10] == pytest.approx(1.0)
    assert target[0, 20, 11] == pytest.approx(np.exp(-1 / 8))  # one pixel away, sigma 2
    dec, score = decode(target[None])
    np.testing.assert_allclose(dec[0, 0], [40.0, 80.0])  # equal neighbours: no shift
    assert score[0, 0] == pytest.approx(1.0)


def test_quarter_shift_direction():
    target, _ = encode([[41.2, 80.0], [38.8, 80.0]], [True, True])  # heatmap x 10.3 and 9.7
    dec, _ = decode(target[None])
    np.testing.assert_allclose(dec[0, :, 0], [41.0, 39.0])  # 10 + 0.25, 10 - 0.25, times 4


def test_mmpose_edge_conditions():
    hm = np.zeros((1, 3, 64, 48), dtype=np.float32)
    hm[0, 0, 30, 1], hm[0, 0, 30, 2] = 1.0, 0.5  # argmax x = 1: MMPose needs 1 < x, so no x shift
    hm[0, 1, 30, 46], hm[0, 1, 30, 47] = 1.0, 0.5  # x = W-2: shift allowed
    hm[0, 2, 30, 47] = 1.0  # x = W-1: no shift
    dec, _ = decode(hm)
    np.testing.assert_allclose(dec[0, :, 0], [1 * 4, (46 + 0.25) * 4, 47 * 4])


def test_masked_and_outside_points_get_zero_weight():
    xy = [[50.0, 50.0], [50.0, 50.0], [-3.0, 10.0], [192.0, 10.0], [np.nan, 1.0]]
    target, weight = encode(xy, [True, False, True, True, True])
    assert weight.tolist() == [1, 0, 0, 0, 0]
    assert target[1:].max() == 0.0


def test_zero_heatmap_is_missing():
    dec, score = decode(np.zeros((2, 4, 64, 48)))
    assert np.isnan(dec).all() and (score == 0).all()


# --- preprocessing --------------------------------------------------------------------

def test_known_pixel_lands_where_M_says():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[500:503, 900:903] = (0, 0, 255)  # 3x3 red patch centered at (901, 501)
    x, M = make_pose_crop(img, [850, 400, 950, 650])
    u, v = apply_affine(M, [[901, 501]])[0]
    r = x[0] * STD[0] + MEAN[0]  # undo normalization of the R channel
    assert r[int(round(v)), int(round(u))] > 200
    assert x.shape == (3, INPUT_H, INPUT_W) and x.dtype == np.float32


def test_single_normalization():
    img = np.full((300, 300, 3), (10, 20, 30), dtype=np.uint8)  # BGR
    x, _ = make_pose_crop(img, [100, 100, 160, 180])
    np.testing.assert_allclose(x[:, 128, 96], (np.array([30, 20, 10]) - MEAN) / STD, rtol=1e-6)  # RGB order


def test_expand_applied_once():
    box = np.array([100.0, 100.0, 196.0, 228.0])  # 96 x 128, already 192:256
    np.testing.assert_allclose(pose_crop_box(box), [88, 84, 208, 244])  # 1.25x once: 120 x 160
    _, M = make_pose_crop(np.zeros((400, 400, 3), np.uint8), box)
    assert M[0, 0] == pytest.approx(192 / 120)


def test_border_padding_is_black():
    x, _ = make_pose_crop(np.full((200, 200, 3), 255, np.uint8), [-150, -150, -50, -10])  # off-image
    np.testing.assert_allclose(x[:, 5, 5], (0 - MEAN) / STD, rtol=1e-6)
    np.testing.assert_allclose(normalize(np.zeros((2, 2, 3), np.uint8))[:, 0, 0], -MEAN / STD, rtol=1e-6)
