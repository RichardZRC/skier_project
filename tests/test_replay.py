import json

import cv2
import numpy as np
import pytest
import synth

from skier.eval import replay_metrics as rm
from skier.replay.controller import CropTrack, ReplayConfig, run_controller
from skier.replay.render import read_crops_csv, render_debug, render_replay, write_crops_csv
from skier.replay.run import ClipInput, run_clip, run_grid
from skier.tracking.frames import VideoFrames
from skier.tracking.run import write_track_csv

FRAME = (1920.0, 1080.0)


# --- metrics ---------------------------------------------------------------------------

def test_box_containment_and_centering():
    crops = np.array([[0, 0, 100, 100], [0, 0, 100, 100], [0, 0, 100, 100]], float)
    boxes = np.array([[10, 10, 30, 30], [90, 0, 110, 20], [np.nan] * 4])
    c = rm.box_containment(crops, boxes)
    assert c["n_frames"] == 2 and c["mean"] == pytest.approx((1 + 0.5) / 2) and c["frac_fully_inside"] == 0.5
    e = rm.centering_error(crops[:1], np.array([[40, 50, 60, 50]], float))  # centre (50, 50): zero error
    assert e["mean"] == pytest.approx(0.0)
    e = rm.centering_error(crops[:1], np.array([[90, 90, 110, 110]], float))  # centre (100, 100)
    assert e["mean"] == pytest.approx(np.hypot(50, 50) / np.hypot(100, 100))


def test_landmark_containment(schema):
    crops = np.array([[0, 0, 100, 100]], float)
    xy = np.full((1, 24, 2), 50.0)
    xy[0, 16] = (150, 50)  # right ski tip outside
    mask = np.ones((1, 24), bool)
    mask[0, 23] = False  # unreliable left tail doesn't count
    r = rm.landmark_containment(crops, xy, mask, schema)
    assert r["ski_endpoints"] == {"inside": 2, "total": 3, "pct": pytest.approx(200 / 3)}
    assert r["body_ski"]["total"] == 21 and r["body_ski"]["inside"] == 20


@pytest.mark.parametrize("fps", [25, 50])
def test_acceleration_is_exact_and_frame_rate_free(fps):
    t = np.arange(2 * fps) / fps
    a_px = 300.0  # px / s^2 along x
    cx = 500 + 0.5 * a_px * t**2
    crops = np.column_stack([cx - 100, np.full_like(t, 400), cx + 100, np.full_like(t, 512.5)])
    r = rm.crop_acceleration(crops, t, FRAME)
    assert r["mean"] == pytest.approx(a_px / np.hypot(*FRAME), rel=1e-6)


def test_acceleration_exact_with_uneven_timestamps():
    """Dropped frames give uneven steps; the divided-difference formula stays exact for a parabola."""
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.02, 0.12, size=40))
    a_px = 250.0
    cx = 400 + 30 * t + 0.5 * a_px * t**2
    crops = np.column_stack([cx - 80, np.full_like(t, 300), cx + 80, np.full_like(t, 390)])
    r = rm.crop_acceleration(crops, t, FRAME)
    assert r["mean"] == pytest.approx(a_px / np.hypot(*FRAME), rel=1e-6)
    assert r["p95"] == pytest.approx(r["mean"], rel=1e-6)


def test_constant_and_constant_velocity_have_zero_acceleration():
    t = np.arange(30) / 25
    still = np.tile([100, 100, 300, 212.5], (30, 1)).astype(float)
    assert rm.crop_acceleration(still, t, FRAME)["mean"] == pytest.approx(0.0)
    moving = still + np.column_stack([t * 80, t * 0, t * 80, t * 0])
    assert rm.crop_acceleration(moving, t, FRAME)["mean"] == pytest.approx(0.0, abs=1e-9)


def test_gaps_split_segments():
    t = np.array([0, 0.04, 0.08, 5.0, 5.04, 5.08])
    crops = np.tile([0, 0, 160, 90], (6, 1)).astype(float)
    crops[3:] += 1000  # a jump across the gap must not count as acceleration
    assert rm.crop_acceleration(crops, t, FRAME, max_gap_s=1.0) == {"mean": 0.0, "p95": 0.0, "n": 2}
    assert [list(s) for s in rm.segments(t, np.ones(6, bool), 1.0)] == [[0, 1, 2], [3, 4, 5]]


def test_zoom_stats_full_frame():
    t = np.arange(10) / 25
    z = rm.zoom_stats(np.tile([0, 0, *FRAME], (10, 1)), FRAME, t)
    assert z == {"area_fraction_mean": 1.0, "log_zoom_rate_mean": 0.0}


# --- render -----------------------------------------------------------------------------

def make_track(n, crop):
    crops = np.tile(np.asarray(crop, float), (n, 1))
    z = np.zeros(n, bool)
    return CropTrack(crops, crops.copy(), crops.copy(), np.arange(n) / 25, z.copy(), z.copy(), z.copy(), z.copy())


def test_render_replay_maps_crop_centre_to_output_centre(tmp_path):
    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=10, size=(640, 360), step=4, square=20)
    track = make_track(10, [200, 100, 360, 190])  # 160 x 90, 16:9
    out = tmp_path / "replay.mp4"
    n = render_replay(VideoFrames(video), list(range(10)), track, out, fps=25, out_size=(320, 180))
    assert n == 10
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25)
    ok, frame = cap.read()
    assert ok and frame.shape == (180, 320, 3)
    cap.release()


def test_crop_centre_and_keypoints_map_to_output(tmp_path, schema):
    from skier.geometry import apply_affine, crop_affine

    img = np.zeros((360, 640, 3), np.uint8)
    img[143:148, 278:283] = (0, 0, 255)  # red dot at (280, 145) = crop centre
    track = make_track(1, [200, 100, 360, 190])
    M = crop_affine(track.crops[0], 320, 180)  # the same warp render_replay applies
    assert cv2.warpAffine(img, M, (320, 180))[90, 160, 2] > 200  # crop centre -> output centre
    xy = np.full((24, 2), np.nan)
    xy[2] = (280, 145)
    np.testing.assert_allclose(apply_affine(M, xy[2:3])[0], [160, 90])
    n = render_replay([(0, img)], [0], track, tmp_path / "r.mp4", fps=25, out_size=(320, 180),
                      keypoints={0: (xy, np.ones(24, bool))}, schema=schema)
    assert n == 1


def test_crops_csv_round_trip(tmp_path):
    boxes = [[100, 100, 150, 200]] * 5 + [[np.nan] * 4]
    tr = run_controller(boxes, np.arange(6) / 25, FRAME)
    write_crops_csv(tr, list(range(10, 16)), tmp_path / "crops.csv")
    idx, back = read_crops_csv(tmp_path / "crops.csv")
    assert idx == list(range(10, 16))
    np.testing.assert_allclose(back.crops, tr.crops, atol=1e-3)
    assert back.lost.tolist() == tr.lost.tolist() and np.isnan(back.raw_boxes[-1]).all()


def test_render_debug(tmp_path):
    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=6, size=(640, 360), step=4, square=20)
    n = render_debug(VideoFrames(video), list(range(6)), make_track(6, [0, 0, 320, 180]), tmp_path / "d.mp4", fps=25)
    assert n == 6


# --- end to end -------------------------------------------------------------------------

def square_clip(tmp_path, jitter=6.0, n=80, seed=0):
    """Moving-square video + a noisy fake tracker CSV + exact GT boxes."""
    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=n, size=(1280, 720), step=12, square=40)
    rng = np.random.default_rng(seed)
    gt = np.array([[20 + 12 * k, 340, 60 + 12 * k, 380] for k in range(n)], float)
    noisy = gt + rng.normal(0, jitter, size=gt.shape)
    rows = [{"frame_index": k, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "score": float("nan"),
             "status": "init" if k == 0 else "track"} for k, b in enumerate(noisy)]
    write_track_csv(rows, tmp_path / "track.csv")
    return video, ClipInput("square", rows, 25.0, (1280.0, 720.0), gt)


def test_end_to_end_smoothing_beats_unsmoothed(tmp_path, schema):
    video, clip = square_clip(tmp_path)
    out = tmp_path / "out"
    m = run_clip(clip, lambda: iter(VideoFrames(video)), ReplayConfig(), out, schema)
    s, b = m["smoothed"], m["unsmoothed_baseline"]
    assert s["acceleration"]["mean"] < 0.3 * b["acceleration"]["mean"]  # much smoother ...
    assert abs(s["zoom"]["area_fraction_mean"] - b["zoom"]["area_fraction_mean"]) < 0.1  # ... at similar zoom
    assert s["box_containment"]["mean"] >= 0.95 and s["target"] == "ground truth"
    assert m["replay_frames"] == 80 and m["debug_frames"] == 80
    assert json.loads((out / "metrics.json").read_text())["clip"] == "square"
    assert (out / "crops.csv").exists() and (out / "replay.mp4").exists() and (out / "debug.mp4").exists()


def test_grid_table(tmp_path, schema):
    _, clip = square_clip(tmp_path, n=40)
    table = run_grid([clip], ReplayConfig(), schema, taus=(0.0, 0.2), ks=(1.5, 2.5))
    assert [(r["tau"], r["k"]) for r in table] == [(0.0, 1.5), (0.0, 2.5), (0.2, 1.5), (0.2, 2.5)]
    by = {(r["tau"], r["k"]): r for r in table}
    assert by[(0.2, 2.5)]["acceleration_mean"] < by[(0.0, 2.5)]["acceleration_mean"]
    assert by[(0.0, 2.5)]["area_fraction_mean"] >= by[(0.0, 1.5)]["area_fraction_mean"]
