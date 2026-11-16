"""Pseudo-labels (Step 6b, D29): frame sampling, the teacher's two passes, the keep rule, the filter
choice, the JSON round trip and the epoch mixing. All torch-free (the models are fakes)."""

import json
from pathlib import Path

import numpy as np
import pytest

from skier.data.records import PoseRecord
from skier.data.skitb import Clip
from skier.geometry import body_ski_mask, box_from_points, expand_box
from skier.integration.refine import refined_boxes
from skier.io_utils import write_json
from skier.pose import pseudo
from skier.pose.pseudo import (FORMAT, candidate_frames, choose_filter, epoch_order, frame_entries, keep_mask,
                               kept_precision, load_pseudo_records, pseudo_per_epoch, teacher_predict)

K = 24


def skeleton(schema, x0=100.0, y0=50.0, w=60.0, h=120.0):
    """A [24, 2] layout spread over a w x h box (every joint at a distinct place)."""
    rng = np.random.default_rng(7)
    return np.column_stack([x0 + w * rng.random(K), y0 + h * rng.random(K)])


# --- frame sampling --------------------------------------------------------------------------

def _clip(root, seq, clip, n, visible=None, h=100.0):
    frames = root / "skitb" / "AL" / seq / "frames"
    hh = np.full(n, h) if np.isscalar(h) else np.asarray(h, float)
    boxes = np.column_stack([np.full(n, 10.0), np.full(n, 20.0), np.full(n, 60.0), 20.0 + hh])
    return Clip(seq, clip, "train", frames, [str(100 + i) for i in range(n)], boxes,
                np.ones(n, bool) if visible is None else np.asarray(visible, bool))


def test_candidate_frames_skips_and_steps(tmp_path):
    vis_b = np.ones(20, bool)
    vis_b[:5] = False
    h_b = np.full(20, 100.0)
    h_b[5:10] = 50.0
    clips = [_clip(tmp_path, "S1", "0", 20), _clip(tmp_path, "S1", "1", 20, vis_b, h_b), _clip(tmp_path, "S2", "0", 5)]
    cands, counts = candidate_frames(clips, tmp_path, min_box_h=64, min_clip_len=10, target=10)
    assert counts == {"clips": 3, "short_clips": 1, "invisible": 5, "small_box": 5, "eligible": 30, "frame_step": 3,
                      "candidates": 11}
    assert [c.frame_index for c in cands] == [100, 103, 106, 109, 112, 115, 118, 110, 113, 116, 119]
    c = cands[7]
    assert c.sample_id == "S1/1/00110" and c.image == "skitb/AL/S1/frames/00110.jpg" and c.gt_box == (10, 20, 60, 120)
    again, _ = candidate_frames(clips, tmp_path, min_box_h=64, min_clip_len=10, target=10)
    assert again == cands
    every, counts = candidate_frames(clips, tmp_path, min_box_h=64, min_clip_len=10, target=10_000)
    assert counts["frame_step"] == 1 and len(every) == 30


def test_drop_sequences(tmp_path):
    clips = [_clip(tmp_path, s, "0", 12) for s in ("AL0010", "AL0001", "AL0028")]
    kept, found = pseudo.drop_sequences(clips)
    assert [c.sequence_id for c in kept] == ["AL0001"] and found == ["AL0010", "AL0028"]
    assert pseudo.EXCLUDED_SEQUENCES == ("AL0010", "AL0018", "AL0028")


# --- the teacher's two passes ----------------------------------------------------------------

def fake_model(schema, offset, low_score_images=()):
    """Points = a fixed skeleton per image + a per-model offset + a box-dependent tag (so pass 2 is visible);
    scores 0.9 on body + ski, 0.05 on poles, and 0.05 everywhere on ``low_score_images``."""
    base = skeleton(schema)
    bs = body_ski_mask(schema)

    def predict(images, boxes):
        xy, sc = [], []
        for img, box in zip(images, np.asarray(boxes)):
            i = int(img[0, 0, 0])
            xy.append(base + i + offset + box[0] * 1e-3)
            s = np.where(bs, 0.9, 0.05)
            sc.append(np.full(K, 0.05) if i in low_score_images else s)
        return np.array(xy), np.array(sc)

    return predict


def test_teacher_predict_two_passes_and_fallback(schema):
    images = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    start = np.array([[90.0, 40.0, 170.0, 180.0]] * 3)
    models = [fake_model(schema, 0.0, low_score_images=(2,)), fake_model(schema, 1.0, low_score_images=(2,))]
    out = teacher_predict(images, start, models, schema, k_pose=1.1, s_min=0.3, k_refine=1.3)
    b1 = expand_box(start[0], 1.1)
    p1 = [m(images, np.stack([b1] * 3)) for m in models]
    rb = refined_boxes(np.mean([p[0] for p in p1], axis=0), np.mean([p[1] for p in p1], axis=0), schema, 0.3, 1.3)
    assert list(out["refined"]) == [True, True, False]
    np.testing.assert_allclose(out["crop_box"][:2], rb[:2])
    np.testing.assert_allclose(out["crop_box"][2], b1)  # fallback keeps the pass-1 crop
    base = skeleton(schema)
    for m, off in enumerate((0.0, 1.0)):
        np.testing.assert_allclose(out["xy"][m, 0], base + 0 + off + rb[0, 0] * 1e-3)  # pass 2, per model
        np.testing.assert_allclose(out["xy"][m, 2], base + 2 + off + b1[0] * 1e-3)  # fallback = pass 1
    assert out["xy"].shape == (2, 3, K, 2) and out["scores"].shape == (2, 3, K)


# --- the keep rule ---------------------------------------------------------------------------

def _two_models(schema, n=1):
    base = skeleton(schema)
    xy = np.stack([np.stack([base] * n), np.stack([base] * n)]).copy()  # [2, n, K, 2]
    sc = np.full((2, n, K), 0.9)
    return xy, sc


def test_keep_rule_scores_models_and_poles(schema):
    xy, sc = _two_models(schema)
    body, ski = list(schema.groups["body"]), list(schema.groups["ski"])
    poles = list(schema.groups["pole"])
    sc[0, 0, body[0]] = 0.5  # exactly the threshold: kept
    sc[1, 0, body[1]] = 0.4999  # one model below: dropped
    sc[0, 0, ski[0]] = np.nan
    xy[1, 0, ski[1]] = np.nan  # a model without a point: dropped
    mean, kept, ok = keep_mask(xy, sc, schema, s_keep=0.5, d_keep=0.05, n_keep=8)
    assert ok[0] and kept[0, body[0]] and not kept[0, body[1]]
    assert not kept[0, ski[0]] and not kept[0, ski[1]]
    assert not kept[0, poles].any()  # poles never, even at 0.9
    assert kept[0].sum() == 22 - 3
    np.testing.assert_allclose(mean[0, body[2]], xy[0, 0, body[2]])


def test_keep_rule_agreement_scales_with_the_box(schema):
    xy, sc = _two_models(schema)
    bs = body_ski_mask(schema)
    diag = np.hypot(*np.ptp(xy[0, 0][bs], axis=0))
    j, j2 = list(schema.groups["ski"])[:2]
    # margins of 10%: moving a point by up to 0.055 x diag moves the mean (and the box) by at most half that
    xy[1, 0, j, 0] += 0.05 * diag * 0.9  # within d_keep x diag
    xy[1, 0, j2, 0] += 0.05 * diag * 1.1  # outside
    _, kept, ok = keep_mask(xy, sc, schema, s_keep=0.3, d_keep=0.05, n_keep=8)
    assert ok[0] and kept[0, j] and not kept[0, j2]
    # the limit is relative to the box of the confident points: on a skeleton twice as large, the same
    # disagreement is within it
    c = xy[0, 0][bs].mean(axis=0)
    scaled = xy.copy()
    scaled[0, 0] = c + 2 * (xy[0, 0] - c)
    scaled[1, 0] = scaled[0, 0]
    scaled[1, 0, j2, 0] += 0.05 * diag * 1.1
    _, kept2, _ = keep_mask(scaled, sc, schema, s_keep=0.3, d_keep=0.05, n_keep=8)
    assert kept2[0, j2]


def test_keep_rule_frame_minimum(schema):
    xy, sc = _two_models(schema, n=2)
    bs = np.flatnonzero(body_ski_mask(schema))
    sc[:, 0, bs[8:]] = 0.0  # frame 0: exactly 8 confident joints
    sc[:, 1, bs[7:]] = 0.0  # frame 1: 7
    _, kept, ok = keep_mask(xy, sc, schema, s_keep=0.3, d_keep=0.05, n_keep=8)
    assert list(ok) == [True, False] and kept[0].sum() == 8 and not kept[1].any()


# --- calibration and the filter choice ------------------------------------------------------

def _record(schema, xy, reliable, sid="v/0/i"):
    box = box_from_points(xy, body_ski_mask(schema) & reliable)
    return PoseRecord(sid, "ski2dpose", "v", "0", "i", 0, Path("x.jpg"), 1280, 720, xy, reliable,
                      reliable.astype(int), box)


def test_kept_precision_counts(schema):
    gt = skeleton(schema)
    rel = np.ones(K, bool)
    ski = list(schema.groups["ski"])
    rel[ski[0]] = False  # kept but unlabeled: not verifiable
    rec = _record(schema, gt, rel)
    diag = np.hypot(*(rec.reference_box_xyxy[2:] - rec.reference_box_xyxy[:2]))
    pred = gt.copy()
    pred[ski[1], 0] += 0.05 * diag * 1.01  # a miss at PCK@0.05
    kept = body_ski_mask(schema).copy()
    out = kept_precision(pred[None], kept[None], [rec], schema, thr=0.05)
    assert out["ski"] == {"kept": 8, "verifiable": 7, "hits": 6, "labels": 7, "precision": pytest.approx(600 / 7),
                          "coverage": 100.0}
    assert out["body"]["precision"] == 100.0 and out["body"]["kept"] == 14


def _row(s, d, body, ski, kept, sanity=False):
    return {"s_keep": s, "d_keep": d, "n_keep": 8, "sanity": sanity, "body": {"precision": body},
            "ski": {"precision": ski, "kept": kept}}


def test_choose_filter_rule():
    t = [_row(0.3, 0.1, 96, 95.5, 900), _row(0.5, 0.05, 97, 96, 900), _row(0.7, 0.02, 99, 99, 500),
         _row(0.3, 0.02, 94, 99, 2000), _row(None, None, 99, 99, 5000, sanity=True)]
    c = choose_filter(t)
    assert (c["stop"], c["bar"], c["s_keep"], c["d_keep"]) == (False, 95.0, 0.5, 0.05)  # tie on kept -> higher s_keep
    t2 = [_row(0.5, 0.05, 92, 91, 700), _row(0.5, 0.02, 92, 91, 700), _row(0.3, 0.1, 89, 99, 9000)]
    c2 = choose_filter(t2)
    assert (c2["bar"], c2["s_keep"], c2["d_keep"]) == (90.0, 0.5, 0.02)  # same s_keep -> smaller d_keep
    assert choose_filter([_row(0.3, 0.1, 89, 99, 10)])["stop"] is True
    assert choose_filter([_row(0.3, 0.1, None, 99, 10)])["stop"] is True


# --- JSON round trip and loading --------------------------------------------------------------

def _labels_file(tmp_path, schema, frames_override=None, header_override=None):
    root = tmp_path / "data"
    img = root / "skitb" / "AL" / "S1" / "frames" / "00197.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"")
    c = pseudo.Candidate("S1/0/00197", "S1", "0", 197, "skitb/AL/S1/frames/00197.jpg", (10.0, 20.0, 60.0, 120.0))
    xy = skeleton(schema)[None]
    kept = body_ski_mask(schema)[None].copy()
    kept[0, list(schema.groups["ski"])[0]] = False
    frames = frame_entries([c], np.array([[1280, 720]]), xy, kept, np.array([True]), np.array([[1.0, 2, 3, 4]]),
                           np.array([True]))
    header = {"format": FORMAT, "skitb_partition": "train", "teachers": [{"split_sha256": "abc"}]}
    header.update(header_override or {})
    p = tmp_path / "pseudo_labels.json"
    write_json({**header, "frames": frames_override(frames) if frames_override else frames}, p)
    return p, root, xy[0], kept[0]


def test_round_trip_to_records(tmp_path, schema):
    p, root, xy, kept = _labels_file(tmp_path, schema)
    assert json.loads(p.read_text())["frames"][0]["xy"][list(schema.groups["ski"])[0]] == [None, None]
    (r,) = load_pseudo_records(p, root, schema, "abc")
    assert r.dataset == "skitb_pseudo" and r.sample_id == "S1/0/00197" and r.source_video_id == "S1"
    assert r.image_path == root / "skitb/AL/S1/frames/00197.jpg" and (r.image_width, r.image_height) == (1280, 720)
    np.testing.assert_array_equal(r.reliable, kept)
    np.testing.assert_allclose(r.keypoints_xy[kept], xy[kept])
    assert np.isnan(r.keypoints_xy[~kept]).all()
    np.testing.assert_allclose(r.reference_box_xyxy, box_from_points(xy, kept))


def test_loaded_records_train_only_on_kept_points(tmp_path, schema):
    from skier.pose.dataset import PoseDataset

    p, root, _, kept = _labels_file(tmp_path, schema)
    recs = load_pseudo_records(p, root, schema, "abc")
    ds = PoseDataset(recs, schema, train=False, image_loader=lambda _: np.zeros((720, 1280, 3), np.uint8))
    np.testing.assert_array_equal(ds[0]["weight"] > 0, kept)


@pytest.mark.parametrize("bad, match", [
    (lambda f: [{**f[0], "image": "/abs/x.jpg"}], "relative"),
    (lambda f: [{**f[0], "image": "skitb/../x.jpg"}], "relative"),
    (lambda f: [f[0], f[0]], "duplicate"),
    (lambda f: [{**f[0], "image": "skitb/AL/S1/frames/missing.jpg"}], "missing"),
    (lambda f: [{**f[0], "xy": [[None, None]] * 24}], "without coordinates"),
])
def test_loader_refuses_bad_frames(tmp_path, schema, bad, match):
    p, root, _, _ = _labels_file(tmp_path, schema, frames_override=bad)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_pseudo_records(p, root, schema, "abc")


@pytest.mark.parametrize("header, match", [
    ({"format": "other"}, "not a"), ({"skitb_partition": "test"}, "train"), ({"teachers": []}, "different pose split")])
def test_loader_refuses_bad_headers(tmp_path, schema, header, match):
    p, root, _, _ = _labels_file(tmp_path, schema, header_override=header)
    with pytest.raises(ValueError, match=match):
        load_pseudo_records(p, root, schema, "abc")
    p2, root2, _, _ = _labels_file(tmp_path / "b", schema)
    with pytest.raises(ValueError, match="different pose split"):
        load_pseudo_records(p2, root2, schema, "another-split")


# --- epoch mixing ------------------------------------------------------------------------------

def test_epoch_order_mixes_all_real_and_distinct_pseudo():
    o = epoch_order(5, 7, 4, seed=230, epoch=3)
    assert len(o) == 9 and sorted(o[o < 5].tolist()) == [0, 1, 2, 3, 4]
    p = o[o >= 5]
    assert len(p) == 4 and len(set(p.tolist())) == 4 and p.max() <= 11
    np.testing.assert_array_equal(o, epoch_order(5, 7, 4, seed=230, epoch=3))
    assert not np.array_equal(o, epoch_order(5, 7, 4, seed=230, epoch=4))
    assert len(epoch_order(5, 3, 10, seed=230, epoch=0)) == 8  # capped at all pseudo frames
    assert pseudo_per_epoch(1194, 6000) == 1194 and pseudo_per_epoch(1194, 500) == 500


def test_train_refuses_pseudo_with_overfit_or_cache(tmp_path):
    from skier.pose.train import main

    base = ["--split", "s.json", "--hrnet-config", "c.py", "--checkpoint", "w.pth", "--pseudo-labels", "p.json"]
    for extra in (["--overfit", "4"], ["--cache-dir", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            main(base + extra)
        assert exc.value.code == 2
