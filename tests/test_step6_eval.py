"""Step 6 evaluation tools: the crop-error set, checkpoint scoring and the comparison (D16)."""

import json
import math

import numpy as np
import pytest
import synth

from skier.data.ski2dpose import load_records
from skier.pose import compare as cmp
from skier.pose import crop_error as ce_mod
from skier.pose.augment import jitter_box
from skier.pose.crop_error import LEVELS, check_crop_error_set, make_crop_error_set, perturbed_boxes
from skier.pose.evaluate import evaluate, sample_ids_sha256

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


@pytest.fixture
def records(tmp_path):
    labels = synth.make_labels(videos=2, segments=2, images=3, seed=7)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(640, 360), template=TEMPLATE, value=90)
    recs = [r for r in load_records(synth.write_labels(labels, tmp_path / "labels.json"), images,
                                    path_template=TEMPLATE) if not r.excluded]
    for i, r in enumerate(recs):  # metadata for the strata
        r.discipline = ("Slalom", "Giant Slalom")[i % 2]
        r.view_angle = ("front", "front+side", "back")[i % 3]
    return recs


# --- crop-error set ------------------------------------------------------------------------

def test_crop_error_set_is_deterministic_and_exact():
    ids = [f"v/s/{i}" for i in range(20)]
    a = make_crop_error_set(ids, "SPLIT", "val")
    assert a == make_crop_error_set(list(reversed(ids)), "SPLIT", "val")
    assert a["items"] != make_crop_error_set(ids, "SPLIT", "val", seed=231)["items"]
    assert len(a["items"]) == 20 * len(LEVELS) and a["levels"] == LEVELS
    for it in a["items"]:
        r = LEVELS[it["level"]]
        assert set(it) == {"sample_id", "level", "dx", "dy", "scale"}  # no coordinates, so no label data
        assert math.hypot(it["dx"], it["dy"]) == pytest.approx(r, abs=1e-12)
        assert 1 / (1 + r) - 1e-12 <= it["scale"] <= 1 + r + 1e-12
    # a sample's perturbation doesn't depend on which other samples are in the set
    b = make_crop_error_set(ids[5:], "SPLIT", "val")
    pick = lambda s: {(i["sample_id"], i["level"]): i for i in s["items"]}  # noqa: E731
    assert all(pick(a)[k] == v for k, v in pick(b).items())
    angles = [math.atan2(i["dy"], i["dx"]) for i in a["items"]]
    assert np.std(angles) > 1.0  # directions vary


def test_crop_error_set_checks():
    ids = ["v/s/1", "v/s/2"]
    ce = make_crop_error_set(ids, "SPLIT", "val")
    check_crop_error_set(json.loads(json.dumps(ce)), "SPLIT", "val", ids)  # survives a JSON round trip
    with pytest.raises(ValueError, match="different split"):
        check_crop_error_set(ce, "OTHER", "val", ids)
    with pytest.raises(ValueError, match="partition"):
        check_crop_error_set(ce, "SPLIT", "test", ids)
    with pytest.raises(ValueError, match="missing"):
        check_crop_error_set(ce, "SPLIT", "val", ids + ["v/s/3"])
    edited = json.loads(json.dumps(ce))
    edited["items"][0]["dx"] = 0.0
    with pytest.raises(ValueError, match="edited"):
        check_crop_error_set(edited, "SPLIT", "val", ids)


def test_perturbed_boxes(records):
    ce = make_crop_error_set([r.sample_id for r in records], "SPLIT", "val")
    boxes = perturbed_boxes(records, ce, "severe")
    it = next(i for i in ce["items"] if i["sample_id"] == records[0].sample_id and i["level"] == "severe")
    np.testing.assert_allclose(boxes[0], jitter_box(records[0].reference_box_xyxy, it["dx"], it["dy"], it["scale"]))
    with pytest.raises(KeyError):
        perturbed_boxes(records, ce, "extreme")


def test_make_crop_error_set_cli(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"content_sha256": "S", "partitions": {
        "val": {"sample_ids": ["a/1/x", "a/1/y"]}, "test": {"sample_ids": ["b/1/z"]}}}))
    out = tmp_path / "ce.json"
    assert ce_mod.main(["--split", str(split), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["n_samples"] == 2 and written["split_sha256"] == "S"
    assert ce_mod.main(["--split", str(split), "--out", str(out)]) == 1  # frozen
    assert ce_mod.main(["--split", str(split), "--out", str(out), "--force"]) == 0
    t = tmp_path / "ce_test.json"
    assert ce_mod.main(["--split", str(split), "--partition", "test", "--out", str(t)]) == 1  # Step 8 only
    assert not t.exists()
    assert ce_mod.main(["--split", str(split), "--partition", "test", "--final-test", "--out", str(t)]) == 0


# --- evaluation ----------------------------------------------------------------------------

def run_oracle_eval(records, schema, ce=None):
    calls = [(r, r.reference_box_xyxy) for r in records]
    if ce is not None:
        for level in ce["levels"]:
            calls += list(zip(records, perturbed_boxes(records, ce, level)))
    return evaluate(records, synth.box_oracle_heatmap_fn(calls), schema, ce, batch_size=4)


def test_oracle_scores_100_clean_and_cut_off_points_are_misses(records, schema):
    ce = make_crop_error_set([r.sample_id for r in records], "SPLIT", "val")
    m, preds = run_oracle_eval(records, schema, ce)
    clean = m["clean"]["groups"]["0.1"]
    assert clean["body"]["pct"] == 100.0 and clean["ski"]["pct"] == 100.0
    assert m["clean"]["cut_off"]["ski"]["cut"] == 0 and m["clean"]["missing_rate"] == 0.0
    assert m["clean"]["body_shared"]["0.1"]["pct"] == 100.0
    assert m["n_samples"] == len(records)
    assert m["sample_ids_sha256"] == sample_ids_sha256([r.sample_id for r in records])
    cuts = {}
    for level, d in m["crop_error"]["levels"].items():
        for g in ("body", "ski"):
            block, cut = d["groups"]["0.1"][g], d["cut_off"][g]
            assert block["total"] == clean[g]["total"]  # same denominators
            assert block["hits"] == block["total"] - cut["cut"]  # a perfect model misses exactly the cut-off points
        cuts[level] = d["cut_off"]["body"]["cut"] + d["cut_off"]["ski"]["cut"]
    assert cuts["severe"] > 0 and cuts["mild"] <= cuts["severe"]
    assert set(preds) == {"clean", *LEVELS} and preds["severe"].shape == preds["clean"].shape


def test_strata_add_up(records, schema):
    m, _ = run_oracle_eval(records, schema)
    st = m["strata"]
    assert sum(b["n_images"] for b in st["discipline"].values()) == len(records)
    for g in ("body", "ski"):
        assert sum(b[g]["total"] for b in st["discipline"].values()) == m["clean"]["groups"]["0.1"][g]["total"]
    expected_front = sum("front" in r.view_angle.split("+") for r in records)
    assert st["view"]["front"]["n_images"] == expected_front and st["view"]["side"]["n_images"] >= 1
    assert "crop_error" not in m


# --- comparison ----------------------------------------------------------------------------

def blk(pct, total=1000):
    return {"hits": round(pct * total / 100), "total": total, "pct": pct}


def fake_metrics(body, ski, ski005=None, moderate=None, split="S", ids="I", ce="C"):
    m = {"split_sha256": split, "partition": "val", "n_samples": 300, "sample_ids_sha256": ids,
         "clean": {"groups": {"0.1": {"body": blk(body), "ski": blk(ski)}, "0.05": {"body": blk(body - 10),
                                                                              "ski": blk(ski005 or ski - 20)}},
                   "body_shared": {"0.1": blk(body + 1)},
                   "per_joint": {"head": blk(body), "ski_tip_right": blk(ski)}}}
    if moderate is not None:
        m["crop_error"] = {"content_sha256": ce, "levels": {
            "mild": {"radius": 0.05, "groups": {"0.1": {"ski": blk(ski - 1)}}, "cut_off": {"ski": {"share": 0.01}}},
            "moderate": {"radius": 0.1, "groups": {"0.1": {"ski": blk(moderate)}}, "cut_off": {"ski": {"share": 0.03}}},
            "severe": {"radius": 0.2, "groups": {"0.1": {"ski": blk(moderate - 10)}}, "cut_off": {"ski": {"share": 0.1}}}}}
    return m


def test_compare_deltas_flag_and_clear_winner():
    r = cmp.compare({"E1": fake_metrics(80, 55, moderate=50), "E2": fake_metrics(77, 61, moderate=58),
                     "E3": fake_metrics(80, 57, moderate=52), "E4": fake_metrics(79.5, 59, moderate=56)})
    assert cmp.delta(r["rows"]["E3"]["ski"], r["rows"]["E1"]["ski"]) == pytest.approx(2.0)  # sign: E3 - E1
    sel = r["selection"]
    assert sel["disqualified_body_drop"] == ["E2"]  # body -3 points
    assert sel["pick"] == "E4" and sel["beat_baseline"] == ["E4", "E3"]
    assert r["levels"] == ["mild", "moderate", "severe"]
    md = r["markdown"]
    assert "| E2 |" in md and "-3.0 ⚠" in md and "+4.0" in md and "**Rule's suggestion: E4**" in md
    row = next(x for x in r["long"] if x["run"] == "E2" and x["metric"] == "ski@0.1_moderate")
    assert row["delta_vs_E1"] == pytest.approx(8.0)
    pj = {x["joint"]: x for x in r["per_joint"]}
    assert pj["ski_tip_right"]["E2_delta"] == pytest.approx(6.0)


def test_selection_tie_break_and_negative_result():
    tie = cmp.compare({"E1": fake_metrics(80, 55, moderate=50), "E2": fake_metrics(80, 60.5, moderate=54),
                       "E4": fake_metrics(80, 60.0, moderate=57)})
    assert tie["selection"]["pick"] == "E4" and "within" in tie["selection"]["reason"]
    no_ce = cmp.compare({"E1": fake_metrics(80, 55), "E2": fake_metrics(80, 60.5), "E4": fake_metrics(80, 60.0)})
    assert no_ce["selection"]["pick"] == "E2" and "no moderate crop-error" in no_ce["selection"]["reason"]
    neg = cmp.compare({"E1": fake_metrics(80, 55), "E2": fake_metrics(80, 54), "E3": fake_metrics(70, 70)})
    assert neg["selection"]["pick"] == "E1" and neg["selection"]["disqualified_body_drop"] == ["E3"]
    assert "negative" in neg["selection"]["reason"]


def test_compare_refuses_mismatched_runs_and_accepts_e0():
    with pytest.raises(ValueError, match="split_sha256"):
        cmp.compare({"E1": fake_metrics(80, 55), "E2": fake_metrics(80, 56, split="T")})
    with pytest.raises(ValueError, match="sample_ids_sha256"):
        cmp.compare({"E1": fake_metrics(80, 55), "E2": fake_metrics(80, 56, ids="J")})
    with pytest.raises(ValueError, match="ce"):
        cmp.compare({"E1": fake_metrics(80, 55, moderate=50), "E2": fake_metrics(80, 56, moderate=50, ce="D")})
    e0 = {"experiment": "E0", "split_sha256": "S", "partition": "val", "n_samples": 300,
          "body_shared": {"pck@0.1": blk(70)}}
    r = cmp.compare({"E0": e0, "E1": fake_metrics(80, 55)})
    assert r["rows"]["E0"]["ski"] is None and "| E0 | — | 70.0% (700/1000) | — |" in r["markdown"]


def test_compare_cli(tmp_path):
    paths = {}
    for name, m in {"E1": fake_metrics(80, 55, moderate=50), "E4": fake_metrics(80, 58, moderate=55)}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(m))
    out = tmp_path / "cmp"
    assert cmp.main([f"{n}={p}" for n, p in paths.items()] + ["--out", str(out)]) == 0
    assert json.loads((out / "selection.json").read_text())["pick"] == "E4"
    assert (out / "comparison.md").exists() and (out / "comparison.csv").exists() and (out / "per_joint.csv").exists()
    assert cmp.main(["E1", "--out", str(out)]) == 1
    with pytest.raises(ValueError, match="twice"):
        cmp.parse_runs(["E1=a", "E1=b"])
