"""Step 8: bootstrap, per-sample hits, the SkiTB toolkit port, partition-wide replay, report tables and figures."""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.eval import bootstrap as bs
from skier.eval import skitb_official as off
from skier.eval.pose_metrics import gt_arrays, pck, sample_hits
from skier.integration import propagation as prop
from skier.io_utils import write_csv, write_json
from skier.pose.evaluate import score
from skier.replay import run as replay_run
from skier.replay.controller import ReplayConfig
from skier.replay.run import ClipInput, evaluate_clips, flat_metrics
from skier.report import figures as figs
from skier.report import tables as tab
from skier.tracking.run import run_skitb, write_track_csv

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"
# read before the autouse fixture points SKIER_DATA at tmp_path: the real vendored toolkit, if this machine has it
TOOLKIT = Path(os.environ.get("SKIER_DATA", "~/skier_data")).expanduser() / "vendor" / "SkiTB-toolkit"


# --- bootstrap ------------------------------------------------------------------------------

def test_bootstrap_basics():
    clusters = np.repeat(np.arange(10), 5)
    h = np.random.default_rng(0).integers(0, 10, size=50)
    n = np.full(50, 10)
    same = bs.ratio_diff_ci(h, n, h, n, clusters, n_boot=200)
    assert same["estimate"] == 0.0 and same["ci"] == [0.0, 0.0]
    shifted = bs.ratio_diff_ci(h + 2, n, h, n, clusters, n_boot=200)
    assert shifted["estimate"] == pytest.approx(20.0) and shifted["ci"] == [pytest.approx(20.0), pytest.approx(20.0)]
    a = bs.ratio_ci(h, n, clusters, n_boot=300, seed=5)
    assert a == bs.ratio_ci(h, n, clusters, n_boot=300, seed=5) and a["ci"] != bs.ratio_ci(h, n, clusters, n_boot=300, seed=6)["ci"]
    assert a["estimate"] == pytest.approx(100 * h.sum() / n.sum()) and a["n_clusters"] == 10
    with pytest.raises(ValueError, match="at least 2"):
        bs.ratio_ci(h, n, np.zeros(50), n_boot=10)
    with pytest.raises(ValueError, match="spans several strata"):
        bs.ratio_ci(h, n, clusters, np.arange(50) % 2, n_boot=10)


def test_stratified_resampling_keeps_clusters_per_stratum():
    clusters = np.array(["a1", "a1", "a2", "b1", "b2", "b3"])
    strata = np.array(["A", "A", "A", "B", "B", "B"])
    counts = []
    bs.cluster_bootstrap(lambda i: counts.append((int((strata[i] == "A").sum()), int((strata[i] == "B").sum()))) or 0.0,
                         clusters, strata, n_boot=100)
    resamples = counts[1:]  # the first call is the point estimate
    assert all(b == 3 for _, b in resamples)  # B: three one-item clusters, drawn three times
    a_items = {a for a, _ in resamples}  # A: two draws of a1 (2 items) or a2 (1 item)
    assert a_items <= {2, 3, 4} and len(a_items) > 1


def test_bootstrap_coverage_on_synthetic_segments():
    """About 95% of intervals cover the true difference when segments vary (200 simulations)."""
    rng = np.random.default_rng(230)
    covered = 0
    for sim in range(200):
        n_seg, per = 30, 8
        effect = rng.normal(0, 0.08, n_seg)  # segment-level variation shared by both runs
        p_b = np.clip(0.75 + effect, 0.05, 0.95)
        p_a = np.clip(p_b + 0.04, 0.05, 0.99)
        seg = np.repeat(np.arange(n_seg), per)
        vid = seg % 3
        tot = np.full(n_seg * per, 10)
        ha, hb = rng.binomial(10, p_a[seg]), rng.binomial(10, p_b[seg])
        ci = bs.ratio_diff_ci(ha, tot, hb, tot, seg, vid, n_boot=400, seed=sim)["ci"]
        covered += ci[0] <= 100 * np.mean(p_a - p_b) <= ci[1]
    assert 0.88 <= covered / 200 <= 0.99


# --- per-sample hits ------------------------------------------------------------------------

def test_sample_hits_sum_to_pck(schema):
    rng = np.random.default_rng(1)
    gt = rng.uniform(0, 100, (20, 24, 2))
    pred = gt + rng.normal(0, 4, gt.shape)
    pred[3, 5] = np.nan
    rel = rng.random((20, 24)) < 0.9
    diag = rng.uniform(50, 150, 20)
    for joints in schema.groups.values():
        for t in (0.1, 0.05):
            h, n = sample_hits(pred, gt, rel, diag, joints, t)
            ref = pck(pred, gt, rel, diag, joints, t)
            assert (h.sum(), n.sum()) == (ref["hits"], ref["total"]) and len(h) == 20


# --- SkiTB toolkit port ---------------------------------------------------------------------

def test_toolkit_metrics_hand_cases():
    anno = np.array([[10, 10, 20, 20]] * 4, float)
    vis = np.ones(4)
    perfect = off.clip_metrics(anno, np.ones(4), anno, vis)
    assert perfect["f_score"] == pytest.approx(1.0) and perfect["gsr_score"] == 1.0

    boxes = anno.copy()
    boxes[2:] = [100, 100, 20, 20]  # off the target on the second half
    m = off.clip_metrics(boxes, np.array([1.0, 1.0, 0.5, 0.5]), anno, vis)
    # t <= 0.5: P = R = 0.5; t > 0.5 keeps the first half only: P = 1, R = 0.5, F = 2/3 from t = 0.51
    assert m["f_score"] == pytest.approx(2 / 3, abs=1e-6) and m["f_score_thresh"] == pytest.approx(0.51)
    assert m["precision"] == pytest.approx(1.0, abs=1e-6) and m["recall"] == pytest.approx(0.5, abs=1e-6)
    assert m["gsr_score"] == pytest.approx(0.5)  # the first frame with IoU <= any threshold is frame 2 of 4

    wrong_first = boxes.copy()
    wrong_first[0] = [500, 500, 5, 5]  # frame 0 is forced to the GT box
    assert off.clip_metrics(wrong_first, np.array([1.0, 1.0, 0.5, 0.5]), anno, vis)["f_score"] == pytest.approx(m["f_score"])

    invisible = off.clip_metrics(anno, np.ones(4), anno, np.array([1, 1, 0, 0]))
    assert invisible["precision"] == pytest.approx(1.0, abs=1e-6) and invisible["recall"] == pytest.approx(2.0, abs=1e-6)

    rows = [{"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": math.nan, "status": "init"},
            {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 22.0, "score": 3.94461912e-05, "status": "track"}]
    b, c = off.to_toolkit(rows)
    assert b.tolist() == [[1, 2, 10, 20]] * 2 and c.tolist() == [1.0, 3.945e-05]  # %.8f grid
    with pytest.raises(ValueError, match="no confidence"):
        off.to_toolkit([rows[0], {**rows[1], "score": math.nan}])


def make_toolkit_case(root: Path, results: Path, seed: int = 3):
    """Synthetic SC clips in the toolkit's layout, plus tracker results with spread-out confidences."""
    rng = np.random.default_rng(seed)
    names, ours = [], {}
    for seq, n_clips in (("AL0001", 2), ("AL0002", 1)):
        for k in range(n_clips):
            n = 30 + 10 * k
            clip = root / "AL" / seq / "SC" / str(k)
            clip.mkdir(parents=True)
            anno = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 300, n), rng.uniform(20, 80, (n, 2))]).round(1)
            anno[7] = -1  # a frame without GT: skipped by _calc_metrics
            vis = (rng.random(n) > 0.15).astype(int)
            vis[0] = 1
            np.savetxt(clip / "boxes.txt", anno, fmt="%.1f", delimiter=",")
            (clip / "frames.txt").write_text("".join(f"{i}\n" for i in range(n)))
            (clip / "visibilities.txt").write_text("".join(f"{v}\n" for v in vis))
            boxes = anno + rng.normal(0, 8, anno.shape)
            boxes[rng.random(n) < 0.3] += 200
            conf = rng.random(n)
            conf[0] = 1.0
            name = f"{seq}_{k}"
            rec = results / "SkiTB" / "fake" / "SC" / "AL" / "ope" / name
            rec.mkdir(parents=True)
            np.savetxt(rec / "boxes.txt", boxes, fmt="%.3f", delimiter=",")
            np.savetxt(rec / "confidences.txt", conf, fmt="%.8f")
            np.savetxt(rec / "times.txt", np.zeros(n), fmt="%.8f")
            names.append(name)
            # our port gets exactly what the toolkit reads back from the files
            ours[name] = off.clip_metrics(np.loadtxt(rec / "boxes.txt", delimiter=","), np.loadtxt(rec / "confidences.txt"),
                                          np.loadtxt(clip / "boxes.txt", delimiter=","), vis)
    return names, ours


@pytest.mark.skipif(not (TOOLKIT / "toolkit").is_dir(), reason="needs the vendored SkiTB toolkit (laptop)")
def test_toolkit_port_matches_the_real_report(tmp_path, monkeypatch):
    names, ours = make_toolkit_case(tmp_path / "skitb", tmp_path / "results")
    real_loadtxt = np.loadtxt
    monkeypatch.setattr(np, "loadtxt", lambda *a, **kw: real_loadtxt(*a, **{k: v for k, v in kw.items()
                                                                              if not (k == "delimiter" and v == "\n")}))
    monkeypatch.syspath_prepend(str(TOOLKIT))
    from toolkit.experiments import ExperimentSkiTB

    exp = ExperimentSkiTB(str(tmp_path / "skitb"), discipline="AL", mode="SC", test_videos=names,
                          result_dir=str(tmp_path / "results"), report_dir=str(tmp_path / "reports"))
    perf = exp.report(["fake"], fn_note="check")["fake"]
    for name in names:
        t, o = perf["seq_wise"][name], ours[name]
        for key in ("f_score", "precision", "recall", "f_score_thresh"):
            assert o[key] == pytest.approx(t[key], abs=1e-12), (name, key)
        assert o["gsr_score"] == pytest.approx(t["generalized_success_robustness_score"], abs=1e-12)
    mine = off.overall(ours, names)
    for key in ("f_score", "precision", "recall", "f_score_thresh"):
        assert mine[key] == pytest.approx(perf["overall"][key], abs=1e-12)
    assert mine["gsr_score"] == pytest.approx(perf["overall"]["generalized_success_robustness_score"], abs=1e-12)


# --- partition-wide replay ------------------------------------------------------------------

def test_evaluate_clips_matches_run_clip_and_counts_landmarks(tmp_path, schema):
    n = 40
    rows = [{"frame_index": i, "x1": 600.0 + 5 * i, "y1": 400.0, "x2": 700.0 + 5 * i, "y2": 600.0,
             "score": math.nan if i == 0 else 1e-4, "status": "init" if i == 0 else "track"} for i in range(n)]
    gt = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows])
    xy = np.full((n, 24, 2), np.nan)
    mask = np.zeros((n, 24), bool)
    for i in (0, 10, 20):
        xy[i] = np.column_stack([np.linspace(610, 690, 24) + 5 * i, np.linspace(410, 590, 24)])
        mask[i] = True
    xy[20, 16] = [5000.0, 5000.0]  # one ski tip far outside any crop
    clip = ClipInput("c", rows, 25.0, (1920.0, 1080.0), gt, (xy, mask))
    out = evaluate_clips([("skitb", "SEQ", clip)], ReplayConfig(), schema)
    ref = replay_run.run_clip(clip, lambda: iter(()), ReplayConfig(), tmp_path / "r", schema, videos=False)
    assert out["smoothed"][0] == {"clip": "c", "source": "skitb", "sequence": "SEQ", **flat_metrics(ref["smoothed"])}
    assert out["unsmoothed_baseline"][0]["acceleration_mean"] == pytest.approx(ref["unsmoothed_baseline"]["acceleration"]["mean"])
    s = out["smoothed"][0]
    assert (s["ski_end_total"], s["ski_end_inside"]) == (12, 11)  # 4 ends x 3 labelled frames, one far off
    assert s["body_ski_total"] == 3 * 22


def make_skitb_multi(root: Path):
    """Three train sequences with one or two SC clips each; GT moves like ShiftTracker (+2 px per frame)."""
    seqs = {"AL0001": 2, "AL0002": 1, "AL0003": 1}
    for seq, n_clips in seqs.items():
        (root / "AL" / seq / "frames").mkdir(parents=True)
        for i in range(8):
            cv2.imwrite(str(root / "AL" / seq / "frames" / f"{i:05d}.jpg"), np.zeros((360, 640, 3), np.uint8))
        for k in range(n_clips):
            clip = root / "AL" / seq / "SC" / str(k)
            clip.mkdir(parents=True)
            (clip / "boxes.txt").write_text("".join(f"{2 * i + k},0,10,10\n" for i in range(8)))
            (clip / "frames.txt").write_text("".join(f"{i}\n" for i in range(8)))
            # clips score different numbers of frames, so a mean of clip means differs from the pooled mean
            (clip / "visibilities.txt").write_text("1\n" * 6 + "0\n1\n" if k == 0 else "1\n" * 3 + "0\n" * 5)
    split = root / "split.json"
    split.write_text(json.dumps({"train": list(seqs)}))
    fps = root / "fps.csv"
    write_csv([{"sequence": s, "fps": 25.0, "fps_estimate": 25.0} for s in seqs], fps, ["sequence", "fps", "fps_estimate"])
    return split, fps


class DriftTracker:
    """Moves +3 px per frame (GT moves +2), score 0.5."""

    def init(self, frame, box):
        self.box = np.asarray(box, float)

    def update(self, frame):
        self.box = self.box + np.array([3, 0, 3, 0])
        return self.box, 0.5


def test_replay_evaluate_cli_and_test_gate(tmp_path):
    split, fps = make_skitb_multi(tmp_path / "skitb")
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    run_skitb(DriftTracker, tmp_path / "skitb", split, "train", tracks, fps=25.0)
    icfg = tmp_path / "integration.yaml"
    icfg.write_text("k_pose: 1.1\nconf_signal: raw\nconf_threshold: 0.4\nreplay_uses_status: true\n")
    args = ["--evaluate", "--track-dir", str(tracks), "--skitb-root", str(tmp_path / "skitb"), "--split-json", str(split),
            "--fps", "25", "--integration-config", str(icfg), "--partition", "train", "--out", str(tmp_path / "out")]
    assert replay_run.main(args) == 0
    data = json.loads((tmp_path / "out" / "replay_eval.json").read_text())
    assert data["n_clips"] == {"skitb": 4, "ski2dpose": 0} and len(data["smoothed"]) == 4
    assert {r["sequence"] for r in data["smoothed"]} == {"AL0001", "AL0002", "AL0003"}
    assert replay_run.main(args[:-4] + ["--partition", "test", "--out", str(tmp_path / "o2")]) == 1
    # every frame below the threshold: with replay_uses_status the crop holds and zooms out instead of following
    icfg.write_text("k_pose: 1.1\nconf_signal: raw\nconf_threshold: 0.6\nreplay_uses_status: true\n")
    assert replay_run.main(args[:-1] + [str(tmp_path / "lost")]) == 0
    icfg.write_text("k_pose: 1.1\nconf_signal: raw\nconf_threshold: 0.6\nreplay_uses_status: false\n")
    assert replay_run.main(args[:-1] + [str(tmp_path / "follow")]) == 0
    lost, follow = (json.loads((tmp_path / d / "replay_eval.json").read_text())["smoothed"] for d in ("lost", "follow"))
    assert [r["log_zoom_rate_mean"] for r in lost] != [r["log_zoom_rate_mean"] for r in follow]


# --- tables ---------------------------------------------------------------------------------

@pytest.fixture
def pose_records(tmp_path):
    labels = synth.make_labels(videos=2, segments=3, images=4, seed=11)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(640, 360), template=TEMPLATE)
    return [r for r in load_records(synth.write_labels(labels, tmp_path / "labels.json"), images,
                                    path_template=TEMPLATE) if not r.excluded]


def fake_pose_runs(pose_dir: Path, records, schema, noise=None):
    gt, rel, diag, _ = gt_arrays(records)
    rng = np.random.default_rng(4)
    ids = np.array([r.sample_id for r in records])
    preds = {}
    for name, sd in (noise or {"E1-s230": 9.0, "E1-s231": 10.0, "E2-s230": 6.0, "E2-s231": 7.0}).items():
        pred = gt + rng.normal(0, sd, gt.shape)
        d = pose_dir / name
        d.mkdir(parents=True)
        np.savez(d / "predictions.npz", sample_ids=ids, clean=pred)
        write_json({"clean": score(pred, records, schema)}, d / "metrics.json")
        preds[name] = pred
    return preds


def test_pose_tables(tmp_path, pose_records, schema):
    preds = fake_pose_runs(tmp_path / "pose", pose_records, schema)
    by_id = {r.sample_id: r for r in pose_records}
    t = tab.pose_tables(tmp_path / "pose", by_id, schema, n_boot=200)
    gt, rel, diag, _ = gt_arrays(pose_records)
    ski = list(schema.groups["ski"])
    row = next(r for r in t["rows"] if (r["run"], r["group"], r["threshold"]) == ("E2", "ski", 0.1))
    s230, s231 = (pck(preds[f"E2-s{s}"], gt, rel, diag, ski, 0.1) for s in (230, 231))
    assert (row["hits"], row["total"]) == (s230["hits"] + s231["hits"], s230["total"] + s231["total"])
    assert row["per_seed"] == {230: pytest.approx(s230["pct"]), 231: pytest.approx(s231["pct"])}
    assert sum(v["total"] for v in row["per_video"].values()) == row["total"]
    d = next(x for x in t["deltas"] if (x["a"], x["b"], x["group"], x["threshold"]) == ("E2", "E1", "ski", 0.1))
    e1 = next(r for r in t["rows"] if (r["run"], r["group"], r["threshold"]) == ("E1", "ski", 0.1))
    assert d["delta"] == pytest.approx(row["pct"] - e1["pct"]) and d["ci95"][0] <= d["delta"] <= d["ci95"][1]
    assert d["n_clusters"] == 6 and t["n_segments"] == 6 and t["n_videos"] == 2
    assert len(t["per_joint"]) == 24 and "| E2 |" in tab.to_markdown({"partition": "val", "pose": t})
    tip = next(r for r in t["per_joint"] if r["joint"] == "ski_tip_left")
    j = schema.names.index("ski_tip_left")
    one = [pck(preds[f"E2-s{s}"], gt, rel, diag, [j], 0.1) for s in (230, 231)]
    assert tip["E2"] == pytest.approx(100 * sum(o["hits"] for o in one) / sum(o["total"] for o in one))
    m = json.loads((tmp_path / "pose" / "E2-s231" / "metrics.json").read_text())
    m["clean"]["groups"]["0.1"]["ski"]["hits"] += 1
    (tmp_path / "pose" / "E2-s231" / "metrics.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="E2-s231"):
        tab.pose_tables(tmp_path / "pose", by_id, schema, n_boot=50)


def test_pose_tables_with_e0(tmp_path, pose_records, schema):
    fake_pose_runs(tmp_path / "pose", pose_records, schema)
    gt, rel, diag, _ = gt_arrays(pose_records)
    shared = tab.joint_sets(schema)["body_shared"]
    pred24 = np.full_like(gt, np.nan)
    pred24[:, shared] = gt[:, shared]
    (tmp_path / "pose" / "E0").mkdir()
    np.savez(tmp_path / "pose" / "E0" / "predictions.npz", sample_ids=np.array([r.sample_id for r in pose_records]),
             pred24=pred24)
    write_json({"body_shared": {f"pck@{t:g}": pck(pred24, gt, rel, diag, shared, t) for t in (0.1, 0.05)}},
               tmp_path / "pose" / "E0" / "metrics.json")
    t = tab.pose_tables(tmp_path / "pose", {r.sample_id: r for r in pose_records}, schema, n_boot=100)
    e0 = [r for r in t["rows"] if r["run"] == "E0"]
    assert {r["group"] for r in e0} == {"body_shared"} and all(r["pct"] == 100.0 for r in e0)
    assert "N/A" in tab.to_markdown({"partition": "val", "pose": t})  # E0's ski PCK is N/A, not 0
    assert any(d["b"] == "E0" and d["a"] == "E2" for d in t["deltas"])


def test_propagation_tables(tmp_path, pose_records, schema):
    gt, rel, diag, _ = gt_arrays(pose_records)
    rng = np.random.default_rng(8)
    boxes = np.array([r.reference_box_xyxy for r in pose_records])
    keys = {"tracker_iou": ["iou>=0.5"] * len(pose_records)}
    ckpts, npz = {}, {"sample_ids": np.array([r.sample_id for r in pose_records])}
    for name, sd in (("E1-s230", 5), ("E1-s231", 5), ("E2-s230", 4), ("E2-s231", 4)):
        ref = gt + rng.normal(0, sd, gt.shape)
        trk = gt + rng.normal(0, 3 * sd, gt.shape)
        npz[f"{name}__reference"], npz[f"{name}__k1.1"] = ref, trk
        ckpts[name] = {"by_k_pose": {"1.1": prop.compare_crops(pose_records, schema, ref, trk, boxes, 1.25, keys)}}
    (tmp_path / "prop").mkdir()
    write_json({"checkpoints": ckpts}, tmp_path / "prop" / "propagation.json")
    np.savez(tmp_path / "prop" / "predictions.npz", **npz)
    t = tab.propagation_tables(tmp_path / "prop", {r.sample_id: r for r in pose_records}, schema, 1.1, n_boot=200)
    d = next(x for x in t["deltas"] if x["a"] == "E2 reference" and x["group"] == "ski" and x["threshold"] == 0.1)
    ref_row = next(r for r in t["rows"] if (r["run"], r["crop"], r["group"], r["threshold"]) == ("E2", "reference", "ski", 0.1))
    trk_row = next(r for r in t["rows"] if (r["run"], r["crop"], r["group"], r["threshold"]) == ("E2", "tracked", "ski", 0.1))
    assert d["delta"] == pytest.approx(ref_row["pct"] - trk_row["pct"]) and d["delta"] > 0
    st = [s for s in t["strata"] if s["run"] == "E2"]
    assert len(st) == 1 and st[0]["tracked_ski"] == pytest.approx(trk_row["pct"])
    ckpts["E1-s230"]["by_k_pose"]["1.1"]["tracked"]["groups"]["0.1"]["body"]["hits"] -= 1
    write_json({"checkpoints": ckpts}, tmp_path / "prop" / "propagation.json")
    with pytest.raises(ValueError, match="E1-s230 tracked"):
        tab.propagation_tables(tmp_path / "prop", {r.sample_id: r for r in pose_records}, schema, 1.1, n_boot=50)


def test_tracking_tables_match_run_skitb_and_the_port(tmp_path):
    split, fps = make_skitb_multi(tmp_path / "skitb")
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    s = run_skitb(DriftTracker, tmp_path / "skitb", split, "train", tracks, fps=25.0)
    t = tab.tracking_tables(tracks, tmp_path / "skitb", split, "train", fps, n_boot=200)
    assert t["ours"]["pooled_mean_iou"] == pytest.approx(s["pooled_mean_iou"]) and t["n_clips"] == 4
    assert t["ours"]["auc"] == pytest.approx(s["success"]["auc"]) and t["n_sequences"] == 3
    lo, hi = t["ci95"]["pooled_mean_iou"]
    assert lo <= t["ours"]["pooled_mean_iou"] <= hi
    assert 0 < t["official_toolkit"]["f_score"] <= 1 and t["official_toolkit"]["n_clips"] == 4
    assert "**Toolkit** F-score" in tab.to_markdown({"partition": "train", "tracking": t})
    with open(tracks / "summary.json") as f:
        bad = json.load(f)
    bad["pooled_mean_iou"] += 0.01
    write_json(bad, tracks / "summary.json")
    with pytest.raises(ValueError, match="differ"):
        tab.tracking_tables(tracks, tmp_path / "skitb", split, "train", fps, n_boot=50)


def test_replay_tables(tmp_path):
    cols = replay_run.EVAL_COLUMNS
    rows = []
    for i, (src, seq) in enumerate([("skitb", "A"), ("skitb", "A"), ("skitb", "B"), ("ski2dpose", "v/1"),
                                    ("ski2dpose", "v/2"), ("ski2dpose", "w/1"), ("ski2dpose", "w/2")]):
        rows.append({c: "" for c in cols} | {"clip": f"c{i}", "source": src, "sequence": seq, "n_frames": 10,
                                             "box_containment_mean": 0.8 + 0.05 * i, "box_containment_min": 0.5,
                                             "frac_fully_inside": 0.7, "acceleration_mean": 1.0 + i,
                                             "acceleration_p95": 3.0, "area_fraction_mean": 0.3,
                                             "log_zoom_rate_mean": 0.1, "centering_error_mean": 0.2,
                                             "centering_error_p95": 0.4,
                                             **({"ski_end_inside": 7, "ski_end_total": 8, "body_ski_inside": 20,
                                                 "body_ski_total": 22} if src == "ski2dpose" else {})})
    for v in ("smoothed", "unsmoothed_baseline"):
        write_csv(rows, tmp_path / f"{v}.csv", cols)
    t = tab.replay_tables(tmp_path, n_boot=200)
    b = t["smoothed"]
    assert b["skitb_clips"] == 3 and b["skitb"]["box_containment_mean"] == pytest.approx(np.mean([0.8, 0.85, 0.9]))
    lm = b["ski2dpose_landmarks"]["ski_endpoints"]
    assert (lm["inside"], lm["total"]) == (28, 32) and lm["pct"] == pytest.approx(87.5) and lm["meets_target"] is False
    assert "Ski ends inside" in tab.to_markdown({"partition": "val", "replay": t})


# --- figures and gates ----------------------------------------------------------------------

def test_read_log_and_figures(tmp_path, pose_records, schema):
    cols = ["epoch", "phase", "lr_backbone", "lr_head", "train_loss", "n_steps", "n_dropped", "epoch_s", "s_per_step",
            "data_wait_s", "peak_mem_mb", "val_body_pct", "val_body_hits", "val_body_total", "val_ski_pct",
            "val_ski_hits", "val_ski_total", "is_best"]
    rows = [{c: 0 for c in cols} | {"epoch": e, "phase": "warm" if e < 2 else "finetune", "train_loss": 0.01 / (e + 1),
                                    "val_body_pct": 80 + e, "val_ski_pct": 50 + 3 * e} for e in range(6)]
    for name in ("E1-s230", "E2-s231"):
        (tmp_path / "d21" / name).mkdir(parents=True)
        write_csv(rows, tmp_path / "d21" / name / "log.csv", cols)
    log = figs.read_log(tmp_path / "d21" / "E1-s230" / "log.csv")
    assert log["val_ski_pct"].tolist() == [50.0 + 3 * e for e in range(6)] and log["finetune"].sum() == 4
    pytest.importorskip("matplotlib")
    fake_pose_runs(tmp_path / "pose", pose_records, schema)
    t = {"partition": "val", "pose": tab.pose_tables(tmp_path / "pose", {r.sample_id: r for r in pose_records}, schema,
                                                     n_boot=50)}
    write_json(t, tmp_path / "tables.json")
    assert figs.main(["--tables", str(tmp_path / "tables.json"), "--d21-dir", str(tmp_path / "d21"),
                      "--out", str(tmp_path / "figs")]) == 0
    assert (tmp_path / "figs" / "learning_curves.png").stat().st_size > 0
    assert (tmp_path / "figs" / "per_joint_delta_val.png").stat().st_size > 0


def test_final_only_gates(tmp_path):
    from skier.pose import eval_generic
    from skier.tracking import run as tracking_run

    assert eval_generic.main(["--config", "x", "--checkpoint", "y", "--split", "z", "--partition", "test"]) == 1
    assert eval_generic.main(["--config", "x", "--checkpoint", "y", "--split", "z", "--partition", "followcam"]) == 1
    assert tracking_run.main(["--siamfc-repo", "x", "--weights", "y", "--skitb-root", "r", "--split-json", "s",
                              "--partition", "test"]) == 1
