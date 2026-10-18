import cv2
import numpy as np
import pytest

from skier.geometry import (
    apply_affine,
    box_diagonal,
    crop_affine,
    expand_box,
    fit_aspect,
    invert_affine,
    iou_xyxy,
)


def test_expand_box_about_center():
    np.testing.assert_allclose(expand_box([10, 20, 30, 60], 1.5), [5, 10, 35, 70])


@pytest.mark.parametrize("box", [[0, 0, 10, 40], [0, 0, 100, 10], [5, 5, 5.0 + 192 * 0.75, 5.0 + 256 * 0.75]])
def test_fit_aspect_ratio_and_containment(box):
    out = fit_aspect(box, 192 / 256)
    w, h = out[2] - out[0], out[3] - out[1]
    assert w / h == pytest.approx(192 / 256)
    assert out[0] <= box[0] and out[1] <= box[1] and out[2] >= box[2] and out[3] >= box[3]
    np.testing.assert_allclose((out[:2] + out[2:]) / 2, (np.array(box[:2]) + box[2:]) / 2)


@pytest.mark.parametrize("rot", [0.0, 17.0, -90.0])
def test_round_trip(rot):
    box = fit_aspect(expand_box([400, 200, 520, 560], 1.25), 192 / 256)
    M = crop_affine(box, 192, 256, rot_deg=rot)
    pts = np.array([[410.3, 250.7], [500.0, 555.5], [-20.0, 1300.0]])
    back = apply_affine(invert_affine(M), apply_affine(M, pts))
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_crop_maps_box_onto_output():
    box = np.array([100.0, 50.0, 196.0, 178.0])  # 96 x 128 = 192/256 aspect
    M = crop_affine(box, 192, 256)
    np.testing.assert_allclose(apply_affine(M, [[100, 50], [196, 178], [148, 114]]), [[0, 0], [192, 256], [96, 128]])
    assert box_diagonal(box) == pytest.approx(160.0)


def test_rotation_direction_matches_opencv():
    box = np.array([0.0, 0.0, 100.0, 100.0])
    M = crop_affine(box, 100, 100, rot_deg=90)
    # A point right of the center ends up above it (counter-clockwise on screen).
    np.testing.assert_allclose(apply_affine(M, [[75, 50]]), [[50, 25]], atol=1e-9)
    np.testing.assert_allclose(M, cv2.getRotationMatrix2D((50, 50), 90, 1.0), atol=1e-9)


def test_crop_partly_outside_image_pads():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10, 20] = (0, 0, 255)  # one red pixel at (x=20, y=10)
    box = np.array([-40.0, -60.0, 60.0, 40.0])  # mostly outside the image
    M = crop_affine(box, 50, 50)
    crop = cv2.warpAffine(img, M, (50, 50), flags=cv2.INTER_NEAREST, borderValue=(7, 7, 7))
    x, y = apply_affine(M, [[20, 10]])[0]
    assert tuple(crop[int(round(y)), int(round(x))]) == (0, 0, 255)
    assert tuple(crop[0, 0]) == (7, 7, 7)  # padded region


def test_aspect_mismatch_raises():
    with pytest.raises(ValueError, match="fit_aspect"):
        crop_affine([0, 0, 100, 100], 192, 256)


def test_iou():
    assert iou_xyxy([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert iou_xyxy([0, 0, 2, 2], [3, 3, 5, 5]) == 0.0
    assert iou_xyxy([0, 0, 2, 2], [2, 0, 4, 2]) == 0.0  # touching edges
    assert iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)
    assert iou_xyxy([0, 0, 2, 2], [np.nan, 0, 3, 2]) == 0.0
