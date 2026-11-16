"""The PL vs. E2 comparison (Step 6b, D29) and the shared event groups (skier.report.events)."""

from pathlib import Path

import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.eval.pose_metrics import gt_arrays, pck
from skier.integration import propagation as prop
from skier.io_utils import write_json
from skier.pose.evaluate import score
from skier.report import compare_pseudo as cp
from skier.report import events as ev

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


@pytest.fixture
def records(tmp_path):
    labels = synth.make_labels(videos=2, segments=3, images=4, seed=11)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(640, 360), template=TEMPLATE)
    return [r for r in load_records(synth.write_labels(labels, tmp_path / "labels.json"), images,
                                    path_template=TEMPLATE) if not r.excluded]


def eval_runs(root: Path, records, schema, noise: dict, seed: int, levels=("clean",)) -> dict:
    """``eval_pose``-style folders: predictions.npz (per level) + metrics.json with the stored counts."""
    gt, _, _, _ = gt_arrays(records)
    rng = np.random.default_rng(seed)
    ids = np.array([r.sample_id for r in records])
    preds = {}
    for name, sd in noise.items():
        d = root / name
        d.mkdir(parents=True)
        p = {lv: gt + rng.normal(0, sd, gt.shape) for lv in levels}
        m = {"clean": score(p["clean"], records, schema)}
        if len(levels) > 1:
            m["crop_error"] = {"levels": {lv: score(p[lv], records, schema) for lv in levels if lv != "clean"}}
        np.savez(d / "predictions.npz", sample_ids=ids, **p)
        write_json(m, d / "metrics.json")
        preds[name] = p
    return preds


def groups_of(records):
    v1, v2 = sorted({r.source_video_id for r in records})
    return {ev.SAME: {v1}, ev.OTHER: {v2}, ev.ALL: {v1, v2}}, v1, v2


def pooled_pct(preds, names, records, joints, mask, level="clean"):
    gt, rel, diag, _ = gt_arrays(records)
    res = [pck(preds[n][level][mask], gt[mask], rel[mask], diag[mask], joints, 0.1) for n in names]
    return 100 * sum(r["hits"] for r in res) / sum(r["total"] for r in res)


def test_pose_compare_is_paired_per_group(tmp_path, records, schema):
    e2 = eval_runs(tmp_path / "e2", records, schema, {"E2-s230": 25.0, "E2-s231": 28.0}, 1, ("clean", "moderate"))
    pl = eval_runs(tmp_path / "pl", records, schema, {"PL-s230": 6.0, "PL-s231": 7.0}, 2, ("clean", "moderate"))
    by_id = {r.sample_id: r for r in records}
    groups, v1, v2 = groups_of(records)
    ski = list(schema.groups["ski"])
    vids = np.array([r.source_video_id for r in records])
    for level in ("clean", "moderate"):
        res = cp.pose_compare([tmp_path / "e2" / n for n in e2], [tmp_path / "pl" / n for n in pl], by_id, schema,
                              groups, False, level, n_boot=200)
        for gname, mask in ((ev.SAME, vids == v1), (ev.OTHER, vids == v2), (ev.ALL, np.ones(len(vids), bool))):
            g = res[gname]
            want = pooled_pct(pl, list(pl), records, ski, mask, level) - pooled_pct(e2, list(e2), records, ski, mask,
                                                                                     level)
            d = next(x for x in g["deltas"] if x["group"] == "ski@0.1")
            assert (d["a"], d["b"]) == ("PL", "E2") and d["delta"] == pytest.approx(want)
            assert d["delta"] > 0 and d["ci95"][0] <= d["delta"] <= d["ci95"][1]
            assert g["runs"]["PL"]["ski@0.1"]["pct"] == pytest.approx(pooled_pct(pl, list(pl), records, ski, mask, level))


def test_pose_compare_refuses_other_samples_and_bad_counts(tmp_path, records, schema):
    import json

    e2 = eval_runs(tmp_path / "e2", records, schema, {"E2-s230": 8.0, "E2-s231": 9.0}, 1)
    pl = eval_runs(tmp_path / "pl", records[:-1], schema, {"PL-s230": 3.0, "PL-s231": 4.0}, 2)
    by_id = {r.sample_id: r for r in records}
    groups, _, _ = groups_of(records)
    with pytest.raises(ValueError, match="different samples"):
        cp.pose_compare([tmp_path / "e2" / n for n in e2], [tmp_path / "pl" / n for n in pl], by_id, schema, groups,
                        False, "clean", 50)
    m = json.loads((tmp_path / "e2" / "E2-s231" / "metrics.json").read_text())
    m["clean"]["groups"]["0.1"]["ski"]["hits"] += 1
    (tmp_path / "e2" / "E2-s231" / "metrics.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="E2-s231"):
        cp.pose_compare([tmp_path / "e2" / n for n in e2], [tmp_path / "e2" / n for n in e2], by_id, schema, groups,
                        False, "clean", 50)


def prop_run(d: Path, records, schema, names, ref_sd, trk_sd, seed):
    gt, _, _, _ = gt_arrays(records)
    rng = np.random.default_rng(seed)
    boxes = np.array([r.reference_box_xyxy for r in records])
    keys = {"tracker_iou": ["iou>=0.5"] * len(records)}
    key = "s0.3_k1.3"
    ckpts, npz = {}, {"sample_ids": np.array([r.sample_id for r in records])}
    for name in names:
        ref, trk, rfd = (gt + rng.normal(0, s, gt.shape) for s in (ref_sd, trk_sd, trk_sd / 2))
        npz[f"{name}__reference"], npz[f"{name}__k1.1"], npz[f"{name}__k1.1__{key}"] = ref, trk, rfd
        block = prop.compare_crops(records, schema, ref, trk, boxes, 1.25, keys)
        block["refined"] = {key: {"s_min": 0.3, "k_refine": 1.3, "n_refined": len(records), "fallback_share": 0.0,
                                  **prop.compare_crops(records, schema, ref, rfd, boxes, 1.25, keys)}}
        ckpts[name] = {"by_k_pose": {"1.1": block}}
    d.mkdir(parents=True)
    write_json({"checkpoints": ckpts, "refine_settings": [key]}, d / "propagation.json")
    np.savez(d / "predictions.npz", **npz)
    return npz


def test_propagation_compare(tmp_path, records, schema):
    a = prop_run(tmp_path / "e2", records, schema, ("E1-s230", "E1-s231", "E2-s230", "E2-s231"), 6, 30, 1)
    b = prop_run(tmp_path / "pl", records, schema, ("PL-s230", "PL-s231"), 3, 12, 2)
    by_id = {r.sample_id: r for r in records}
    groups, _, _ = groups_of(records)
    res = cp.propagation_compare(tmp_path / "e2", tmp_path / "pl", by_id, schema, 1.1, groups, n_boot=200)
    g = res[ev.ALL]
    assert set(g["runs"]) == {f"{m} {c}" for m in ("E2", "PL") for c in ("reference", "tracked", "refined")}
    gt, rel, diag, _ = gt_arrays(records)
    ski = list(schema.groups["ski"])
    want = (np.mean([pck(b[f"PL-s{s}__k1.1"], gt, rel, diag, ski, 0.1)["pct"] for s in (230, 231)]),
            np.mean([pck(a[f"E2-s{s}__k1.1"], gt, rel, diag, ski, 0.1)["pct"] for s in (230, 231)]))
    d = next(x for x in g["deltas"] if (x["a"], x["b"], x["group"]) == ("PL tracked", "E2 tracked", "ski@0.1"))
    assert d["delta"] == pytest.approx(want[0] - want[1]) and d["delta"] > 0
    prop_run(tmp_path / "short", records[:-1], schema, ("PL-s230", "PL-s231"), 3, 12, 2)
    with pytest.raises(ValueError, match="different labels"):
        cp.propagation_compare(tmp_path / "e2", tmp_path / "short", by_id, schema, 1.1, groups, n_boot=50)


def test_groups_must_add_up_and_partition_groups(records, schema):
    e2 = {("ski", 0.1): (np.ones(len(records)), np.ones(len(records)) * 2)}
    v1, v2 = sorted({r.source_video_id for r in records})
    with pytest.raises(ValueError, match="don't add up"):
        ev.grouped(records, {"E2": e2}, (("ski", 0.1),), {"only one": {v1}}, n_boot=20)
    manifest = {"partitions": {"train": {"videos": ["t"]}, "val": {"videos": ["a"]}, "test": {"videos": ["b", "c"]},
                               "followcam": {"videos": ["f1", "f2"]}}}
    loc = {"t": "Almaty", "a": "Trentino", "b": "Almaty", "c": "Granada", "f1": "Training", "f2": "Training"}
    groups, by_video = ev.partition_groups("test", manifest, loc, {"Almaty"})
    assert groups == {ev.SAME: {"b"}, ev.OTHER: {"c"}, ev.ALL: {"b", "c"}} and not by_video
    assert ev.partition_groups("followcam", manifest, loc, {"Almaty"}) == ({ev.FOLLOWCAM: {"f1", "f2"}}, True)
    with pytest.raises(ValueError, match="shares the training event"):
        ev.partition_groups("val", manifest, {**loc, "a": "Almaty"}, {"Almaty"})
    assert ev.video_events({v: {"video_info": {"location": l}} for v, l in loc.items()}, manifest)[1] == {"Almaty"}
