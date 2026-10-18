import numpy as np
import pytest

from skier.eval import pose_metrics as pm
from skier.eval import track_metrics as tm


def setup(n=3, diag=100.0, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0, 500, size=(n, 24, 2))
    rel = np.ones((n, 24), dtype=bool)
    return gt, rel, np.full(n, diag)


def test_perfect_predictor_is_100(schema):
    gt, rel, diag = setup()
    rel[0, 3] = False
    for g, r in pm.group_pck(gt, gt, rel, diag, schema, 0.1).items():
        assert r["pct"] == 100.0
    assert pm.pck(gt, gt, rel, diag, range(24), 0.05) == {"hits": 71, "total": 71, "pct": 100.0}


@pytest.mark.parametrize("direction", [(1, 0), (0, -1), (-0.6, 0.8)])
def test_threshold_boundary_any_direction(direction):
    gt, rel, diag = setup(n=1)
    d = np.array(direction) / np.linalg.norm(direction)
    pred = gt.copy()
    pred[0, 0] += 0.09 * 100 * d  # inside 0.1 d
    pred[0, 1] -= 0.11 * 100 * d  # outside, on the other side
    assert pm.pck(pred, gt, rel, diag, [0], 0.1)["hits"] == 1
    assert pm.pck(pred, gt, rel, diag, [1], 0.1)["hits"] == 0
    assert pm.pck(pred, gt, rel, diag, [0], 0.05)["hits"] == 0


def test_missing_prediction_is_a_miss():
    gt, rel, diag = setup(n=2)
    pred = gt.copy()
    pred[1, :12] = np.nan
    r = pm.pck(pred, gt, rel, diag, range(24), 0.1)
    assert r == {"hits": 36, "total": 48, "pct": 75.0}
    e = pm.normalized_error(pred, gt, rel, diag, range(24))
    assert (e["n_available"], e["n_missing"], e["missing_rate"], e["mean_norm_error"]) == (36, 12, 0.25, 0.0)


def test_unreliable_labels_ignored():
    gt, rel, diag = setup(n=1)
    pred = gt + 1000.0  # every prediction wrong
    rel[0, 5:] = False
    pred[0, 5:] = gt[0, 5:]  # ... the "right" ones are on unreliable labels and must not count
    assert pm.pck(pred, gt, rel, diag, range(24), 0.1) == {"hits": 0, "total": 5, "pct": 0.0}
    rel[:] = False
    assert pm.pck(pred, gt, rel, diag, range(24), 0.1) == {"hits": 0, "total": 0, "pct": None}


def test_groups_and_videos(schema):
    gt, rel, diag = setup(n=4)
    pred = gt.copy()
    pred[:2, list(schema.groups["ski"])] += 50  # ski wrong on the first two samples (= video a)
    g = pm.group_pck(pred, gt, rel, diag, schema, 0.1)
    assert g["body"]["pct"] == 100.0 and g["ski"] == {"hits": 16, "total": 32, "pct": 50.0}
    v = pm.per_video_pck(pred, gt, rel, diag, ["a", "a", "b", "b"], schema.groups["ski"], 0.1)
    assert v["a"]["pct"] == 0.0 and v["b"]["pct"] == 100.0
    s = pm.pose_summary(pred, gt, rel, diag, schema, video_ids=["a", "a", "b", "b"])
    assert s["groups"]["0.1"]["ski"]["pct"] == 50.0 and s["per_joint"]["ski_tip_right"]["total"] == 4
    assert s["per_video"]["ski"]["a"]["hits"] == 0


def test_bad_inputs_raise():
    gt, rel, diag = setup(n=2)
    with pytest.raises(ValueError, match="ref_diag"):
        pm.pck(gt, gt, rel, np.array([100.0, np.nan]), range(24), 0.1)
    with pytest.raises(ValueError):
        pm.pck(gt[:, :10], gt, rel, diag, range(10), 0.1)


def test_gt_arrays_refuses_excluded(tmp_path):
    from skier.data.records import PoseRecord

    def rec(excluded):
        return PoseRecord("v/s/i", "ski2dpose", "v", "s", "i", 0, tmp_path, 10, 10, np.zeros((24, 2)),
                          np.ones(24, bool), np.ones(24), None if excluded else np.array([0, 0, 3, 4.0]),
                          excluded_reason="x" if excluded else None)
    gt, rel, diag, vids = pm.gt_arrays([rec(False)])
    assert diag.tolist() == [5.0] and vids == ["v"]
    with pytest.raises(ValueError, match="excluded"):
        pm.gt_arrays([rec(False), rec(True)])


# --- tracking ------------------------------------------------------------------------

def test_frame_ious_and_clip_mean():
    gt = np.array([[0, 0, 2, 2]] * 4, dtype=float)
    pred = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [np.nan] * 4, [0, 0, 2, 2]])
    np.testing.assert_allclose(tm.frame_ious(pred, gt), [1, 1 / 3, 0, 1])
    c = tm.clip_iou(pred, gt, visible=[True, True, True, False], init_index=0)
    assert c["n_frames"] == 2 and c["mean_iou"] == pytest.approx((1 / 3 + 0) / 2)  # init + invisible excluded


def test_pooled_vs_per_clip():
    long = {"mean_iou": 1.0, "n_frames": 3, "ious": np.ones(3)}
    short = {"mean_iou": 0.0, "n_frames": 1, "ious": np.zeros(1)}
    r = tm.pooled_iou([long, short, {"mean_iou": None, "n_frames": 0, "ious": np.array([])}])
    assert r == {"pooled_mean_iou": 0.75, "per_clip_mean_iou": 0.5, "n_frames": 4, "n_clips": 2}


def test_success_curve():
    r = tm.success_curve([0.0, 0.5, 1.0])
    s = dict(zip(r["thresholds"], r["success"]))
    assert s[0.0] == pytest.approx(2 / 3) and s[0.5] == pytest.approx(1 / 3) and s[1.0] == 0.0
    assert r["auc"] == pytest.approx(np.mean(r["success"]))


def test_failure_episodes():
    fps = 25
    n = 60
    ious = np.full(n, 0.8)
    ious[10:23] = 0.05  # 13 frames = 0.52 s -> an episode
    ious[30:42] = 0.05  # 12 frames = 0.48 s -> too short
    ious[50:57] = 0.05  # 7 failing frames ...
    visible = np.ones(n, bool)
    visible[57:59] = False  # ... an invisible gap doesn't break the run ...
    ious[59] = 0.05  # ... which continues at frame 59: span 50..59 = 0.40 s, still too short
    eps = tm.failure_episodes(ious, visible, np.arange(n), fps)
    assert eps == [{"start_frame": 10, "end_frame": 22, "duration_s": 0.52}]

    ious2 = np.full(n, 0.8)
    ious2[5:11] = 0.0  # 6 failing frames
    visible2 = np.ones(n, bool)
    visible2[11:15] = False  # gap of invisible frames
    ious2[15:20] = 0.0  # 5 more: span 5..19 = 15 frames = 0.6 s
    assert tm.failure_episodes(ious2, visible2, np.arange(n), fps) == [
        {"start_frame": 5, "end_frame": 19, "duration_s": 0.6}]
