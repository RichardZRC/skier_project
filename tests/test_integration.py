"""Step 7 integration: settings, the tracker-confidence rule, propagation scoring, and process_clip."""

import dataclasses
import itertools
import json
import math

import cv2
import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.geometry import apply_affine, expand_box
from skier.integration import confidence as conf
from skier.integration import process as proc
from skier.integration import propagation as prop
from skier.integration import refine
from skier.integration.config import IntegrationConfig, config_sha256
from skier.paths import configs_dir
from skier.pose.heatmap import encode
from skier.replay.controller import ReplayConfig
from skier.replay.render import read_crops_csv
from skier.replay.run import ClipInput
from skier.tracking.frames import VideoFrames
from skier.tracking.run import read_track_csv, write_track_csv

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


# --- settings -------------------------------------------------------------------------------

def test_integration_config(tmp_path):
    shipped = IntegrationConfig.from_yaml(configs_dir() / "integration" / "default.yaml")
    assert shipped.require("conf_signal").conf_signal in ("none", "raw", "relative")  # chosen on val (D24)
    cfg = IntegrationConfig()
    with pytest.raises(ValueError, match="k_pose"):
        cfg.require("k_pose", "conf_signal")
    with pytest.raises(ValueError, match="conf_threshold"):
        cfg.with_(conf_signal="raw").require("conf_signal")
    assert cfg.with_(conf_signal="none").require("conf_signal").conf_signal == "none"
    with pytest.raises(ValueError, match="conf_signal"):
        IntegrationConfig(conf_signal="peak")
    (tmp_path / "bad.yaml").write_text("k_pose: 1.2\nlost: 3\n")
    with pytest.raises(ValueError, match="unknown"):
        IntegrationConfig.from_yaml(tmp_path / "bad.yaml")
    a = config_sha256(cfg, ReplayConfig(), {"expand": 1.25})
    assert a == config_sha256(IntegrationConfig(), ReplayConfig(), {"expand": 1.25})
    assert a != config_sha256(cfg.with_(k_pose=1.1), ReplayConfig(), {"expand": 1.25})


# --- tracker confidence ---------------------------------------------------------------------

def brute_auc(values, failure):
    pairs = [(f, o) for f, o in itertools.product(values[failure], values[~failure])]
    return sum(1.0 if f < o else 0.5 if f == o else 0.0 for f, o in pairs) / len(pairs)


def test_roc_auc():
    rng = np.random.default_rng(0)
    v = rng.integers(0, 6, size=80).astype(float)  # many ties
    f = rng.random(80) < 0.3
    assert conf.roc_auc(v, f) == pytest.approx(brute_auc(v, f))
    assert conf.roc_auc([1, 2, 3, 4], [True, True, False, False]) == 1.0  # failures have the low values
    assert conf.roc_auc([1, 2, 3, 4], [False, False, True, True]) == 0.0
    assert conf.roc_auc([2, 2, 2], [True, False, False]) == 0.5
    assert conf.roc_auc([1, np.nan, 3], [True, True, False]) == 1.0  # NaN dropped
    assert conf.roc_auc([1, 2], [False, False]) is None


def test_youden_threshold_and_tie():
    v = np.array([1, 2, 3, 2.5, 4, 5], float)
    f = np.array([True, True, True, False, False, False])
    y = conf.youden_threshold(v, f)
    # "flag v < t": t = 2.5 and t = 4 both give J = 2/3; the tie goes to the smaller t
    assert y["threshold"] == 2.5 and y["j"] == pytest.approx(2 / 3)
    assert y["tpr"] == pytest.approx(2 / 3) and y["fpr"] == 0.0
    rng = np.random.default_rng(1)
    v, f = rng.normal(size=60), rng.random(60) < 0.4
    best = max(((v[f] < t).mean() - (v[~f] < t).mean(), -t) for t in np.unique(v))
    assert conf.youden_threshold(v, f)["j"] == pytest.approx(best[0]) and conf.youden_threshold(v, f)["threshold"] == -best[1]
    assert conf.youden_threshold([1, 2], [True, True]) is None


def rows_from(scores, statuses=None):
    statuses = statuses or ["init"] + ["track"] * (len(scores) - 1)
    return [{"frame_index": i, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": s, "status": st}
            for i, (s, st) in enumerate(zip(scores, statuses))]


def test_signal_values_and_status():
    scores = [math.nan] + [2.0] * 5 + [4.0] * 5 + [1.0, 0.5]
    rows = rows_from(scores)
    rel = conf.signal_values(rows, "relative")
    assert math.isnan(rel[0]) and rel[1] == pytest.approx(2 / 3) and rel[-1] == pytest.approx(0.5 / 3)  # median of 10: 3
    np.testing.assert_array_equal(conf.signal_values(rows, "raw")[1:], scores[1:])
    assert np.isnan(conf.signal_values(rows_from([math.nan, 0.0, 0.0]), "relative")).all()
    with pytest.raises(ValueError):
        conf.signal_values(rows, "peak")

    rows = rows_from([math.nan, 0.3, 0.2, 0.1, math.nan, 0.4], ["init", "track", "track", "track", "track", "lost"])
    cfg = IntegrationConfig(conf_signal="raw", conf_threshold=0.2)
    assert conf.tracking_status(rows, cfg) == ["init", "track", "track", "low_conf", "low_conf", "low_conf"]
    assert conf.tracking_status(rows, cfg.with_(conf_signal="none")) == ["init"] + ["track"] * 4 + ["low_conf"]
    with pytest.raises(ValueError, match="not chosen"):
        conf.tracking_status(rows, IntegrationConfig())


def test_confidence_check_rule():
    rng = np.random.default_rng(2)
    clips = []
    for _ in range(4):
        n = 60
        ious = np.where(rng.random(n) < 0.3, 0.0, 0.8)
        scores = np.where(ious < 0.1, 1e-5, 5e-5) * rng.uniform(0.9, 1.1, n)
        scores[0] = math.nan
        visible = np.ones(n, bool)
        visible[5] = False
        clips.append((rows_from(list(scores)), ious, visible))
    r = conf.confidence_check(clips)
    assert r["n_frames"] == 4 * 58  # init row and invisible frame left out
    assert r["signals"]["raw"]["auc"] == 1.0 and r["choice"]["conf_signal"] == "raw"
    assert 5e-5 * 0.9 <= r["choice"]["conf_threshold"] <= 5e-5 * 1.1  # the lowest ok value: J = 1 from there
    noise = [(rows_from([math.nan] + list(rng.random(59))), np.where(rng.random(60) < 0.3, 0.0, 0.8), np.ones(60, bool))
             for _ in range(4)]
    assert conf.confidence_check(noise)["choice"] == {"conf_signal": "none", "conf_threshold": None}


def test_replay_gating_check(schema):
    n = 60
    gt = np.tile([900.0, 500.0, 1000.0, 700.0], (n, 1))
    rows = []
    for i in range(n):  # tracks the skier, then jumps to clutter with a low score
        box = gt[i] if i < 30 else np.array([100.0, 100.0, 200.0, 300.0])
        rows.append({"frame_index": i, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                     "score": math.nan if i == 0 else (5e-5 if i < 30 else 1e-6), "status": "init" if i == 0 else "track"})
    clip = ClipInput("c", rows, 25.0, (1920.0, 1080.0), gt)
    icfg = IntegrationConfig(conf_signal="raw", conf_threshold=2e-5)
    r = conf.replay_gating_check([clip], ReplayConfig(), icfg, schema)
    assert r["low_conf_share"] == pytest.approx(30 / 60)
    assert r["status_drives_replay"]["box_containment_mean"] > r["status_ignored"]["box_containment_mean"]
    assert r["replay_uses_status"] is True
    assert conf.replay_gating_check([clip], ReplayConfig(), icfg.with_(conf_signal="none"), schema)["replay_uses_status"] is False


# --- propagation ----------------------------------------------------------------------------

@pytest.fixture
def seg_records(tmp_path):
    labels = synth.make_labels(videos=1, segments=2, images=4, seed=5)  # frame_idx 0, 10, 20, 30
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(640, 360), template=TEMPLATE, value=90)
    return [r for r in load_records(synth.write_labels(labels, tmp_path / "labels.json"), images,
                                    path_template=TEMPLATE) if not r.excluded]


def write_tracks(records, track_dir, shift_at=20, drop_at=30):
    """Per segment: init at the first label with its box; exact at 10, off the skier (IoU 0) at ``shift_at``,
    and no row at ``drop_at``."""
    init, _ = prop.split_init(records, {(r.source_video_id, r.segment_id) for r in records})
    for (vid, seg), r0 in init.items():
        rows = []
        for r in sorted((r for r in records if (r.source_video_id, r.segment_id) == (vid, seg)), key=lambda r: r.frame_index):
            if r.frame_index == drop_at:
                continue
            w = r.reference_box_xyxy[2] - r.reference_box_xyxy[0]
            box = r.reference_box_xyxy + (np.array([w + 10, 0, w + 10, 0]) if r.frame_index == shift_at else 0)
            rows.append({"frame_index": r.frame_index, **dict(zip(("x1", "y1", "x2", "y2"), box)),
                         "score": math.nan if r is r0 else 1e-5, "status": "init" if r is r0 else "track"})
        write_track_csv(rows, track_dir / f"{vid}__{seg}.csv")
    return init


def test_split_init_excludes_the_first_label(seg_records):
    segs = {(r.source_video_id, r.segment_id) for r in seg_records}
    init, scored = prop.split_init(seg_records, segs)
    assert len(init) == 2 and all(r.frame_index == 0 for r in init.values())
    assert len(scored) == len(seg_records) - 2 and all(r.frame_index > 0 for r in scored)
    only_s0 = [r.sample_id for r in prop.split_init(seg_records, {("v0", "s0")})[1]]
    assert only_s0 == [r.sample_id for r in scored if r.segment_id == "s0"]


def test_propagation_scores_reference_vs_tracked(seg_records, schema, tmp_path):
    init = write_tracks(seg_records, tmp_path)
    _, scored = prop.split_init(seg_records, init)
    fps = {k: 10.0 for k in init}
    trk = prop.tracker_at(scored, init, tmp_path, fps)
    assert np.isnan(trk["boxes"][[r.frame_index == 30 for r in scored]]).all()
    assert trk["seconds"].tolist() == [r.frame_index / 10.0 for r in scored]
    k = 1.1
    boxes = prop.tracker_crop_boxes(trk["boxes"], k)
    finite = np.isfinite(boxes).all(axis=1)
    np.testing.assert_allclose(boxes[0], expand_box(trk["boxes"][0], k))
    calls = [(r, r.reference_box_xyxy) for r in scored] + [(r, b) for r, b, f in zip(scored, boxes, finite) if f]
    fn = synth.box_oracle_heatmap_fn(calls)
    from skier.pose.infer import predict_records

    ref = predict_records(scored, fn, 1.25, batch_size=64).xy
    tracked = prop.predict_on(scored, fn, boxes, 1.25, batch_size=64)
    keys = {"tracker_iou": [prop.iou_bin(v) for v in trk["iou"]], "time_since_init": [prop.time_bin(s) for s in trk["seconds"]]}
    m = prop.compare_crops(scored, schema, ref, tracked, boxes, 1.25, keys)
    for g in ("body", "ski"):
        r, t, cut = m["reference"]["groups"]["0.1"][g], m["tracked"]["groups"]["0.1"][g], m["tracked"]["cut_off"][g]
        assert r["pct"] == 100.0 and r["total"] == t["total"]  # same denominators
        assert t["hits"] == t["total"] - cut["cut"]  # a perfect model misses exactly the cut-off labels
        assert m["delta_pck"]["0.1"][g] == pytest.approx(r["pct"] - t["pct"])
    dropped = [i for i, r in enumerate(scored) if r.frame_index == 30]
    rel = np.array([r.reliable for r in scored])
    assert m["tracked"]["cut_off"]["body"]["cut"] >= rel[dropped][:, list(schema.groups["body"])].sum()  # no box: all cut
    assert m["tracked"]["cut_off"]["ski"]["cut"] > 0
    st = m["strata"]["tracker_iou"]
    assert sum(b["n_images"] for b in st.values()) == len(scored)
    assert sum(b["tracked"]["ski"]["total"] for b in st.values()) == m["tracked"]["groups"]["0.1"]["ski"]["total"]
    assert st["iou<0.1"]["n_images"] == 2 * 2  # frame 20 (shifted) and frame 30 (missing) in both segments
    lost = st["iou<0.1"]
    assert lost["delta"]["ski"] == pytest.approx(lost["reference"]["ski"]["pct"] - lost["tracked"]["ski"]["pct"])
    assert lost["delta"]["ski"] > 0 and st["iou>=0.5"]["delta"]["ski"] == 0.0  # Δ = reference − tracked
    assert set(m["strata"]["time_since_init"]) == {"<2s", "2-5s"}


def test_tracker_at_refuses_a_wrong_init(seg_records, tmp_path):
    init = write_tracks(seg_records, tmp_path)
    vid, seg = next(iter(init))
    rows = read_track_csv(tmp_path / f"{vid}__{seg}.csv")
    rows[0]["x1"] += 5
    write_track_csv(rows, tmp_path / f"{vid}__{seg}.csv")
    _, scored = prop.split_init(seg_records, init)
    with pytest.raises(ValueError, match="not initialized"):
        prop.tracker_at(scored, init, tmp_path, {k: 10.0 for k in init})


def test_time_since_init_counts_from_the_init_label(seg_records, tmp_path):
    later = [r for r in seg_records if r.frame_index >= 10]  # init at frame 10, not 0
    init = write_tracks(later, tmp_path)
    _, scored = prop.split_init(later, init)
    trk = prop.tracker_at(scored, init, tmp_path, {k: 10.0 for k in init})
    assert trk["seconds"].tolist() == [(r.frame_index - 10) / 10.0 for r in scored]


def test_segment_alignment_usable_rule(tmp_path, monkeypatch):
    labels = synth.make_labels(videos=1, segments=3, images=4, seed=6)
    images = tmp_path / "images"
    frames = tmp_path / "frames"
    for seg in ("s0", "s1", "s2"):
        saved = synth.make_moving_square_video(tmp_path / "videos" / "Videos" / "v0" / f"{seg}.avi", n=40,
                                               save_frames=(0, 10, 20, 25, 30), frames_dir=frames / seg)
        for i, fi in enumerate((0, 10, 20, 30)):
            src = saved[25 if (seg == "s1" and fi == 20) else fi]  # s1: one image is another frame
            img = cv2.imread(str(src))
            if seg == "s2" and fi == 30:
                img = cv2.resize(img, (320, 180))  # s2: one image has another size
            path = images / TEMPLATE.format(video_id="v0", segment_id=seg, image_id=f"i{i}")
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)
    recs = load_records(synth.write_labels(labels, tmp_path / "labels.json"), images, path_template=TEMPLATE)
    monkeypatch.setattr("skier.data.alignment.locate_segment",
                        lambda root, v, s, info=None: (root / "Videos" / v / f"{s}.avi", 0))
    image_rows, seg_rows = prop.segment_alignment(recs, tmp_path / "videos", window=6)
    usable = {s["segment_id"]: s["usable"] for s in seg_rows}
    assert usable == {"s0": True, "s1": False, "s2": False}
    s1 = next(s for s in seg_rows if s["segment_id"] == "s1")
    assert s1["n_own_frame"] == 3 and s1["frame_w"] == 640 and s1["fps"] == pytest.approx(25.0)
    assert len(image_rows) == 12


def test_choose_k_pose_and_helpers():
    table = [{"k_pose": 1.0, "body": 80, "ski": 70}, {"k_pose": 1.1, "body": 82, "ski": 71},
             {"k_pose": 1.2, "body": 82, "ski": 71.6}, {"k_pose": 1.35, "body": 70, "ski": 60}]
    r = prop.choose_k_pose(table)
    assert r["best_mean"] == pytest.approx(76.8) and r["pick"] == 1.1  # 76.5 is within 0.5 of 76.8: smaller wins
    assert prop.choose_k_pose(table[:1] + table[3:])["pick"] == 1.0
    assert prop.recipe_of("E2-s230") == "E2" and prop.recipe_of("E2") == "E2"
    assert [prop.iou_bin(v) for v in (0.5, 0.49, 0.1, 0.09)] == ["iou>=0.5", "0.1<=iou<0.5", "0.1<=iou<0.5", "iou<0.1"]
    assert [prop.time_bin(s) for s in (1.99, 2.0, 4.99, 5.0)] == ["<2s", "2-5s", "2-5s", ">=5s"]


def test_eval_propagation_refusals(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"content_sha256": "S", "partitions": {}}))
    assert prop.main(["--split", str(split), "--partition", "test"]) == 1  # held out until Step 8
    assert prop.main(["--split", str(split)]) == 1  # nothing to do
    usable = tmp_path / "usable.json"
    usable.write_text(json.dumps({"partition": "val", "split_sha256": "OTHER", "usable": [], "segments": []}))
    args = ["--split", str(split), "--usable", str(usable), "--track-dir", str(tmp_path), "--checkpoint", "E2-s230=x",
            "--hrnet-config", "x"]
    assert prop.main(args) == 1  # written for another split
    usable.write_text(json.dumps({"partition": "val", "split_sha256": "S", "usable": [], "segments": []}))
    assert prop.main(args + ["--refine-grid", "--k-pose", "1.0", "1.1"]) == 1  # D26 refines one k_pose
    with pytest.raises(SystemExit):
        prop.main(args + ["--refine-grid", "--refine", "0.3", "1.15"])


# --- crop refinement (D26) --------------------------------------------------------------------

def test_refined_boxes_hand_computed(schema):
    body, ski, pole = (list(schema.groups[g]) for g in ("body", "ski", "pole"))
    xy = np.full((1, schema.num_keypoints, 2), 50.0)
    scores = np.full((1, schema.num_keypoints), 0.9)
    xy[0, body[0]], xy[0, ski[0]] = (10.0, 20.0), (110.0, 220.0)  # the extremes that set the box
    xy[0, pole] = (-500.0, 900.0)  # poles never count
    xy[0, body[1]], scores[0, body[1]] = (400.0, 400.0), 0.29  # below s_min
    xy[0, body[2]], scores[0, body[2]] = (np.nan, np.nan), 0.95  # missing prediction
    xy[0, body[3]], scores[0, body[3]] = (-300.0, -300.0), np.nan
    out = refine.refined_boxes(xy, scores, schema, 0.3, 1.0)
    np.testing.assert_allclose(out[0], [10.0, 20.0, 110.0, 220.0])
    np.testing.assert_allclose(refine.refined_boxes(xy, scores, schema, 0.3, 1.2)[0],
                               expand_box([10.0, 20.0, 110.0, 220.0], 1.2))
    assert refine.refined_boxes(xy, scores, schema, 0.29, 1.0)[0, 2] == 400.0  # s_min is inclusive
    few = np.full_like(scores, 0.1)
    few[0, [body[0], ski[0], ski[1]]] = 0.9
    assert np.isnan(refine.refined_boxes(xy, few, schema, 0.3, 1.0)).all()  # 3 < N_MIN points
    few[0, body[4]] = 0.9
    assert np.isfinite(refine.refined_boxes(xy, few, schema, 0.3, 1.0)).all()  # exactly N_MIN
    same = np.full_like(xy, 77.0)
    assert np.isnan(refine.refined_boxes(same, scores, schema, 0.3, 1.0)).all()  # diagonal under 1 px
    assert refine.refined_boxes(np.zeros((0, schema.num_keypoints, 2)), np.zeros((0, schema.num_keypoints)), schema,
                                0.3, 1.0).shape == (0, 4)


def skier_record(r, schema):
    """A fixed skier: body in a column, the skis flat below it (left ski x 220-280, right ski 360-420)."""
    xy = np.zeros((schema.num_keypoints, 2))
    body, ski, pole = (list(schema.groups[g]) for g in ("body", "ski", "pole"))
    xy[body] = np.stack([np.linspace(300, 340, len(body)), np.linspace(100, 250, len(body))], axis=1)
    xy[ski] = np.stack([[220, 240, 260, 280, 360, 380, 400, 420], [270] * 4 + [272] * 4], axis=1)
    xy[pole] = [(250, 200), (390, 200)]
    rel = np.ones(schema.num_keypoints, dtype=bool)
    return dataclasses.replace(r, keypoints_xy=xy, reliable=rel, reference_box_xyxy=np.array([220.0, 100.0, 420.0, 272.0]))


def test_refine_on_recovers_a_cut_off_skier(seg_records, schema):
    _, scored = prop.split_init(seg_records, {(r.source_video_id, r.segment_id) for r in seg_records})
    scored = [skier_record(r, schema) for r in scored]
    trk = np.tile([290.0, 100.0, 490.0, 272.0], (len(scored), 1))  # 0.35 box widths right: x < 252.5 is cut off
    trk[1] += [800, 0, 800, 0]  # far off the skier: no label in the crop, so it falls back
    trk[2] = np.nan  # no tracker box
    trk[3] = scored[3].reference_box_xyxy
    boxes = prop.tracker_crop_boxes(trk, 1.1)
    finite = np.isfinite(boxes).all(axis=1)
    fn1 = synth.box_oracle_heatmap_fn([(r, b) for r, b, f in zip(scored, boxes, finite) if f])
    xy1, s1 = prop.predict_on_scored(scored, fn1, boxes, 1.25, batch_size=64)
    assert np.isnan(xy1[[1, 2]]).all() and np.isnan(s1[2]).all()
    s1[3] = 0.1  # found, but not confident: falls back with finite pass-1 points
    expect = refine.refined_boxes(xy1, s1, schema, 0.3, 1.3)
    used_expect = np.isfinite(expect).all(axis=1)
    assert used_expect.tolist() == [True, False, False, False, True, True] and np.isfinite(xy1[3]).all()
    fn2 = synth.box_oracle_heatmap_fn([(r, b) for r, b, u in zip(scored, expect, used_expect) if u])
    xy2, crop, used = refine.refine_on(scored, fn2, xy1, s1, boxes, schema, 0.3, 1.3, 1.25, batch_size=64)
    np.testing.assert_array_equal(used, used_expect)
    np.testing.assert_array_equal(xy2[~used], xy1[~used])  # fallback keeps pass 1 and its crop
    np.testing.assert_array_equal(crop[~used], boxes[~used])
    np.testing.assert_allclose(crop[used], expect[used])

    ref = np.array([r.keypoints_xy for r in scored])
    one = prop.compare_crops(scored, schema, ref, xy1, boxes, 1.25, {})
    two = prop.compare_crops(scored, schema, ref, xy2, crop, 1.25, {})
    for m in (one, two):  # a perfect model misses exactly the labels its crop cuts off
        for g in ("body", "ski"):
            t, cut = m["tracked"]["groups"]["0.1"][g], m["tracked"]["cut_off"][g]
            assert t["hits"] == t["total"] - cut["cut"]
    cut1, cut2 = one["tracked"]["cut_off"]["ski"]["cut"], two["tracked"]["cut_off"]["ski"]["cut"]
    lost = 2 * len(schema.groups["ski"])  # rows 1 and 2 have no skier in their crop
    assert cut1 == lost + 3 * 2 and cut2 == lost  # the 2 cut-off ski points in each shifted row come back
    assert two["tracked"]["groups"]["0.1"]["body"]["hits"] == one["tracked"]["groups"]["0.1"]["body"]["hits"]


def test_choose_refine_rule():
    rows = [{"s_min": s, "k_refine": k, "body": 80.0, "ski": 70.0} for s, k in refine.REFINE_GRID]
    best = {"s_min": 0.2, "k_refine": 1.3, "body": 82.0, "ski": 72.0}  # mean 77
    rows[[(r["s_min"], r["k_refine"]) for r in rows].index((0.2, 1.3))] = best
    r = refine.choose_refine(rows, {"body": 80.0, "ski": 70.0})
    assert r["pick"] == {"s_min": 0.2, "k_refine": 1.3} and r["gain"] == pytest.approx(2.0) and r["adopt"]
    tie = dict(best, s_min=0.1, k_refine=1.15, ski=71.2)  # mean 76.6: a tie, and k_refine is nearer 1.0
    rows2 = rows + [tie]
    assert refine.choose_refine(rows2, {"body": 80.0, "ski": 70.0})["pick"] == {"s_min": 0.1, "k_refine": 1.15}
    rows3 = rows2 + [dict(tie, s_min=0.5, ski=71.0)]  # mean 76.5: still a tie, same k_refine, higher s_min
    pick = refine.choose_refine(rows3, {"body": 80.0, "ski": 70.0})
    assert pick["pick"] == {"s_min": 0.5, "k_refine": 1.15} and pick["best_mean"] == pytest.approx(77.0)
    assert pick["pick_mean"] == pytest.approx(76.5) and pick["gain"] == pytest.approx(1.5)
    gate = refine.choose_refine(rows, {"body": 81.0, "ski": 72.0})  # gain 0.5: not more than the gate
    assert gate["gain"] == pytest.approx(0.5) and gate["adopt"] is False
    assert refine.setting_key(0.3, 1.15) == "s0.3_k1.15" and len(refine.REFINE_GRID) == 12


def test_integration_config_refine_settings(tmp_path):
    cfg = IntegrationConfig(pose_refine=True)
    with pytest.raises(ValueError, match="refine_min_score"):
        cfg.require("pose_refine")
    with pytest.raises(ValueError, match="refine_k"):
        cfg.with_(refine_min_score=0.3).require("pose_refine")
    assert cfg.with_(refine_min_score=0.3, refine_k=1.15).require("pose_refine").refine_k == 1.15
    assert IntegrationConfig(pose_refine=False).require("pose_refine").pose_refine is False
    with pytest.raises(ValueError, match="pose_refine"):
        IntegrationConfig().require("pose_refine")
    (tmp_path / "r.yaml").write_text("pose_refine: true\nrefine_min_score: 0.3\nrefine_k: 1.15\n")
    assert IntegrationConfig.from_yaml(tmp_path / "r.yaml").require("pose_refine").refine_min_score == 0.3


# --- process_clip ---------------------------------------------------------------------------

class SquareTracker:
    """Follows synth.square_frame exactly; the score drops after frame 20."""

    def init(self, frame_bgr, box_xyxy):
        self.k = 0

    def update(self, frame_bgr):
        self.k += 1
        x = 20 + 8 * self.k
        return np.array([x, 160.0, x + 40, 200.0]), (5e-5 if self.k < 20 else 1e-6)


INPUT_POINTS = np.array([[40 + 4 * j, 60 + 6 * j] for j in range(24)], float)


def fixed_heatmap_fn(batch):
    """A fake model: every crop gets each joint at the same input-space point."""
    target, _ = encode(INPUT_POINTS, np.ones(24, bool))
    return np.stack([target] * len(batch))


def run_square_clip(tmp_path, schema, icfg, videos=True):
    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=30)
    rows = proc.track_stage(SquareTracker(), VideoFrames(video), 0, (20, 160, 60, 200), tmp_path / "track.csv")
    m = proc.process_clip(lambda: iter(VideoFrames(video)), rows, 25.0, (640.0, 360.0), fixed_heatmap_fn, 24, 1.25,
                          icfg, ReplayConfig(), schema, tmp_path / "out", "video/v", "a" * 64, videos=videos)
    return rows, m


ICFG = IntegrationConfig(k_pose=1.2, conf_signal="raw", conf_threshold=1e-5, replay_uses_status=False,
                         pose_refine=False)


def same_rows(a, b):
    return ([{k: v for k, v in r.items() if k != "score"} for r in a] == [{k: v for k, v in r.items() if k != "score"}
                                                                           for r in b]
            and np.array_equal([r["score"] for r in a], [r["score"] for r in b], equal_nan=True))


def test_process_clip_end_to_end(tmp_path, schema):
    rows, m = run_square_clip(tmp_path, schema, ICFG)
    assert same_rows(rows, read_track_csv(tmp_path / "track.csv"))  # what --track-csv would give
    assert rows[1]["score"] == 5e-5  # full precision through the CSV
    out = tmp_path / "out"
    assert proc.validate_jsonl(out / "frames.jsonl", 24) == 30
    lines = [json.loads(t) for t in (out / "frames.jsonl").read_text().splitlines()]
    assert [d["tracking_status"] for d in lines] == ["init"] + ["track"] * 19 + ["low_conf"] * 10
    assert m["status_counts"] == {"init": 1, "track": 19, "low_conf": 10} and m["overlay_frames"] == 20
    assert lines[0]["tracker_score"] is None  # NaN is written as null
    # a 40 px box x k_pose 1.2 x padding 1.25 = 60 px, fit to 3:4 -> 80 px tall -> 256/80 = 3.2 input px per px
    assert lines[5]["pose_crop_M"][0][0] == pytest.approx(3.2) and lines[5]["pose_crop_M"][1][1] == pytest.approx(3.2)
    for d in lines:
        xy = np.array(d["raw_keypoints_xy"], float)
        assert np.isfinite(xy).all()  # raw predictions kept on low_conf frames too
        assert d["overlay_visible"] == (d["tracking_status"] != "low_conf")
        back = apply_affine(np.array(d["pose_crop_M"]), xy)  # original px → model input
        assert np.abs(back - INPUT_POINTS).max() < 1.5  # decoder error bound inside the heatmap
    assert m["replay_frames"] == 30 and m["debug_frames"] == 30
    _, track = read_crops_csv(out / "crops.csv")
    assert not track.lost.any()  # replay_uses_status false: low_conf doesn't drive the replay
    assert (out / "replay.mp4").stat().st_size > 0
    assert not any(d["pose_refined"] for d in lines) and m["refined_frames"] == 0


def test_process_clip_with_crop_refinement(tmp_path, schema):
    _, m = run_square_clip(tmp_path, schema, ICFG.with_(pose_refine=True, refine_min_score=0.3, refine_k=1.3),
                           videos=False)
    out = tmp_path / "out"
    assert proc.validate_jsonl(out / "frames.jsonl", 24) == 30
    lines = [json.loads(t) for t in (out / "frames.jsonl").read_text().splitlines()]
    assert all(d["pose_refined"] for d in lines) and m["refined_frames"] == 30
    # pass 1 is the 3.2 px/px tracker crop (test above); pass 2 crops around its points, so M changes
    assert lines[5]["pose_crop_M"][0][0] != pytest.approx(3.2)
    body_ski = list(schema.groups["body"]) + list(schema.groups["ski"])
    pts = INPUT_POINTS[body_ski]
    span = pts.max(axis=0) - pts.min(axis=0)  # input px of pass 1 = 3.2 x original px
    w, h = span / 3.2 * 1.3 * 1.25  # the refined box, padded
    # fit to 3:4 grows the short side; pass-1 decoding moves each point by up to ~1.4 input px
    assert lines[5]["pose_crop_M"][0][0] == pytest.approx(min(192 / w, 256 / h), rel=0.05)
    for d in lines:  # the fake model put each joint at INPUT_POINTS in the crop it was given: the refined one
        back = apply_affine(np.array(d["pose_crop_M"]), np.array(d["raw_keypoints_xy"], float))
        assert np.abs(back - INPUT_POINTS).max() < 1.5
    assert m["integration_config"]["pose_refine"] is True


def test_process_clip_status_can_drive_the_replay(tmp_path, schema):
    run_square_clip(tmp_path, schema, ICFG.with_(replay_uses_status=True), videos=False)
    _, track = read_crops_csv(tmp_path / "out" / "crops.csv")
    assert track.lost.tolist() == [False] * 20 + [True] * 10


def test_process_clip_needs_chosen_settings(tmp_path, schema):
    with pytest.raises(ValueError, match="not chosen"):
        run_square_clip(tmp_path, schema, ICFG.with_(k_pose=None), videos=False)
    with pytest.raises(ValueError, match="pose_refine"):
        run_square_clip(tmp_path, schema, ICFG.with_(pose_refine=None), videos=False)


def test_validate_jsonl_catches_problems(tmp_path, schema):
    run_square_clip(tmp_path, schema, ICFG, videos=False)
    path = tmp_path / "out" / "frames.jsonl"
    good = [json.loads(t) for t in path.read_text().splitlines()]

    def broken(i, change):
        lines = [dict(d) for d in good]
        change(lines[i])
        p = tmp_path / "bad.jsonl"
        p.write_text("".join(json.dumps(d) + "\n" for d in lines))
        return p

    cases = [
        (3, lambda d: d.pop("timestamp")),
        (25, lambda d: d.update(overlay_visible=True)),  # a low_conf frame with the overlay on
        (3, lambda d: d.update(raw_keypoints_xy=d["raw_keypoints_xy"][:23])),
        (3, lambda d: d.update(checkpoint_sha256="abc")),
        (3, lambda d: d.update(frame_index=1)),
        (3, lambda d: d.update(tracking_status="init")),
        (3, lambda d: d.update(tracker_box=None)),  # a crop matrix without a box
        (3, lambda d: d.update(replay_crop=[0, 0, 1])),
        (3, lambda d: d.update(pose_refined=1)),
        (3, lambda d: d.update(tracker_box=None, pose_crop_M=None, pose_refined=True)),  # refined without a crop
    ]
    for i, change in cases:
        with pytest.raises(ValueError):
            proc.validate_jsonl(broken(i, change), 24)
    assert proc.validate_jsonl(broken(3, lambda d: None), 24) == 30


def test_process_cli_refusals(tmp_path):
    base = ["--video", str(tmp_path / "v.avi"), "--checkpoint", "x", "--hrnet-config", "x"]
    assert proc.main(base) == 1  # the integration settings aren't chosen yet
    chosen = ["--k-pose", "1.2", "--conf-signal", "none", "--replay-uses-status", "false"]
    assert proc.main(base + chosen + ["--partition", "test"]) == 1  # held out until Step 8
    assert proc.main(base + chosen) == 1  # --video needs --init-frame and --init-box


@pytest.mark.stack
def test_process_clip_with_a_random_init_hrnet(tmp_path, schema):
    pytest.importorskip("torch")
    pytest.importorskip("mmpose")
    from skier.pose.model import build_hrnet, packaged_coco_config, torch_heatmap_fn

    config = packaged_coco_config()
    if config is None:
        pytest.skip("HRNet COCO config not found inside the mmpose package")
    model, _ = build_hrnet(24, config)
    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=8)
    rows = proc.track_stage(SquareTracker(), VideoFrames(video), 0, (20, 160, 60, 200), tmp_path / "track.csv")
    proc.process_clip(lambda: iter(VideoFrames(video)), rows, 25.0, (640.0, 360.0), torch_heatmap_fn(model, "cpu"),
                      24, 1.25, ICFG, ReplayConfig(), schema, tmp_path / "out", "video/v", "b" * 64, videos=False)
    assert proc.validate_jsonl(tmp_path / "out" / "frames.jsonl", 24) == 8
