import math

import numpy as np
import pytest

from skier.paths import configs_dir
from skier.replay.controller import ReplayConfig, run_controller, size_limits

FRAME = (1920.0, 1080.0)


def centre(crop):
    return (crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2


def height(crop):
    return crop[3] - crop[1]


def box_at(cx, cy, w=100.0, h=200.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_default_yaml_matches_dataclass():
    assert ReplayConfig.from_yaml(configs_dir() / "replay" / "default.yaml") == ReplayConfig()


K2 = ReplayConfig(k=2.0)  # the geometry tests below assume k = 2 (the default until D23)


def test_constant_box_gives_constant_crop():
    boxes = [box_at(900, 500)] * 30
    tr = run_controller(boxes, np.arange(30) / 25, FRAME, K2)
    assert np.allclose(tr.crops, tr.crops[0])
    w, h = tr.crops[0][2] - tr.crops[0][0], height(tr.crops[0])
    assert w / h == pytest.approx(16 / 9) and h == pytest.approx(400)  # k=2 on a 200 px tall box
    assert tr.reset[0] and not tr.reset[1:].any() and not tr.urgent.any()


def test_small_step_follows_exponential_with_tau_pos():
    cfg = ReplayConfig()
    fps = 25
    boxes = [box_at(900, 500)] * 5 + [box_at(930, 500)] * 40  # 30 px step, well inside the crop
    tr = run_controller(boxes, np.arange(45) / fps, FRAME, cfg)
    assert not tr.urgent.any()
    for n in range(0, 40):
        expected = 930 + (900 - 930) * math.exp(-(n + 1) / fps / cfg.tau_pos)
        assert centre(tr.crops[5 + n])[0] == pytest.approx(expected, abs=1e-9)


def test_grow_fast_shrink_slow():
    cfg = K2
    fps = 25
    # base crop 600 px tall (k=2 on a 300 px box): 0.7x = 420 stays above the 360 px minimum
    for scale, tau in [(1.3, cfg.tau_grow), (0.7, cfg.tau_shrink)]:
        boxes = [box_at(900, 500, 150, 300)] * 3 + [box_at(900, 500, 150 * scale, 300 * scale)] * 20
        tr = run_controller(boxes, np.arange(23) / fps, FRAME, cfg)
        assert not tr.urgent.any()
        lh0, lh1 = math.log(600), math.log(600 * scale)
        for n in range(20):
            expected = lh1 + (lh0 - lh1) * math.exp(-(n + 1) / fps / tau)
            assert math.log(height(tr.crops[3 + n])) == pytest.approx(expected, abs=1e-9)


def test_big_jump_is_urgent():
    boxes = [box_at(600, 500)] * 3 + [box_at(1300, 500)] * 5
    tr = run_controller(boxes, np.arange(8) / 25, FRAME)
    assert tr.urgent[3] and not tr.urgent[:3].any()
    for t in range(3, 8):  # the fast catch-up keeps the target in view from the jump frame on
        b, c = boxes[t], tr.crops[t]
        assert c[0] <= b[0] and b[2] <= c[2]
    assert not tr.urgent[4:].any()  # then normal smoothing takes over


def trajectory(fps, seconds=4.0):
    t = np.arange(int(seconds * fps)) / fps
    cx = 960 + 150 * np.sin(2 * np.pi * 0.4 * t)
    cy = 540 + 60 * np.sin(2 * np.pi * 0.3 * t)
    return t, [box_at(x, y) for x, y in zip(cx, cy)]


def test_same_motion_at_25_and_50_fps_matches():
    t25, b25 = trajectory(25)
    t50, b50 = trajectory(50)
    c25 = run_controller(b25, t25, FRAME).crops
    c50 = run_controller(b50, t50, FRAME).crops[::2]  # shared timestamps
    assert len(c25) == len(c50)
    diff = np.abs(np.array([centre(c) for c in c25]) - np.array([centre(c) for c in c50]))
    assert diff[5:].max() < 0.005 * FRAME[0]  # after the start-up frames


def test_lost_holds_centre_and_zooms_out():
    cfg = K2
    fps = 25
    boxes = [box_at(900, 500)] * 5 + [[np.nan] * 4] * 100
    tr = run_controller(boxes, np.arange(105) / fps, FRAME, cfg)
    assert tr.lost[5:].all() and not tr.lost[:5].any()
    _, h_max = size_limits(*FRAME, cfg)
    assert math.log(height(tr.crops[6])) == pytest.approx(math.log(400) + 2 * cfg.lost_zoom_out_per_s / fps)
    assert height(tr.crops[-1]) == pytest.approx(h_max)  # capped at the largest crop
    assert centre(tr.crops[10])[0] == pytest.approx(900)  # centre held (no clamp needed yet)
    assert np.isnan(tr.desired[5:]).all()


def test_lost_by_status_and_score():
    boxes = [box_at(900, 500)] * 4
    tr = run_controller(boxes, np.arange(4) / 25, FRAME, status=["init", "track", "lost", "track"])
    assert tr.lost.tolist() == [False, False, True, False]
    cfg = ReplayConfig(lost_score=0.3)
    tr = run_controller(boxes, np.arange(4) / 25, FRAME, cfg, scores=[np.nan, 0.9, 0.1, 0.5])
    assert tr.lost.tolist() == [False, False, True, False]


def test_crop_stays_inside_image_near_corner():
    tr = run_controller([box_at(30, 40)] * 10, np.arange(10) / 25, FRAME)
    assert (tr.crops[:, 0] >= 0).all() and (tr.crops[:, 1] >= 0).all()
    assert (tr.crops[:, 2] <= FRAME[0]).all() and (tr.crops[:, 3] <= FRAME[1]).all()
    assert tr.clamped.all()


def test_tiny_box_gets_minimum_crop():
    cfg = ReplayConfig()
    tr = run_controller([box_at(900, 500, 4, 6)] * 3, np.arange(3) / 25, FRAME, cfg)
    h_min, _ = size_limits(*FRAME, cfg)
    assert h_min == pytest.approx(360)  # max(240, 1080 / 3)
    assert height(tr.crops[0]) == pytest.approx(h_min)


def test_gap_resets():
    boxes = [box_at(900, 500)] * 3 + [box_at(1200, 600)] * 2
    ts = [0, 0.04, 0.08, 2.5, 2.54]
    tr = run_controller(boxes, ts, FRAME)
    assert tr.reset.tolist() == [True, False, False, True, False]
    assert centre(tr.crops[3]) == pytest.approx((1200, 600))


def test_unsmoothed_baseline_equals_desired():
    t, boxes = trajectory(25, seconds=2)
    cfg = ReplayConfig().unsmoothed()
    tr = run_controller(boxes, t, FRAME, cfg)
    np.testing.assert_allclose(tr.crops, tr.desired, atol=1e-9)  # no clamping on this path


def test_bad_inputs():
    with pytest.raises(ValueError, match="increasing"):
        run_controller([box_at(1, 1)] * 2, [0, 0], FRAME)
    with pytest.raises(ValueError, match="timestamps"):
        run_controller([box_at(1, 1)] * 2, [0], FRAME)
