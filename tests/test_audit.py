import json
from pathlib import Path

import pytest
import synth

from skier.data import audit
from skier.io_utils import read_csv, read_json, sha256_file

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"

# (sample_id, problem) seeded by make_fixture, one per record (plus one stray file)
SEEDED = {
    ("v0/s0/i0", "bad_shape"),
    ("v0/s0/i1", "coord_out_of_range"),
    ("v0/s0/i2", "non_finite"),
    ("v0/s1/i0", "bad_visibility"),
    ("v0/s1/i1", "missing_frame_idx"),
    ("v1/s0/i1", "duplicate_frame_idx"),
    ("v1/s1/i0", "label_without_image"),
    ("v1/s1/extra.png", "image_without_label"),
}


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    labels = synth.make_labels(videos=2, segments=2, images=3)
    images_root = tmp_path / "images"
    synth.make_blank_images(images_root, labels, size=(64, 48), template=TEMPLATE)

    labels["v0"]["s0"]["i0"]["annotation"] = labels["v0"]["s0"]["i0"]["annotation"][:23]
    labels["v0"]["s0"]["i1"]["annotation"][3][0] = 1.2
    labels["v0"]["s0"]["i2"]["annotation"][5][1] = float("nan")
    labels["v0"]["s1"]["i0"]["annotation"][7][2] = 2
    del labels["v0"]["s1"]["i1"]["frame_idx"]
    labels["v1"]["s0"]["i1"]["frame_idx"] = labels["v1"]["s0"]["i0"]["frame_idx"]
    (images_root / "v1/s1/i0.png").unlink()
    (images_root / "v1/s1/extra.png").write_bytes((images_root / "v1/s1/i1.png").read_bytes())

    labels_path = synth.write_labels(labels, tmp_path / "labels.json")
    return labels_path, images_root


def run_audit(tmp_path, *extra):
    labels_path, images_root = make_fixture(tmp_path)
    out, inv = tmp_path / "out", tmp_path / "inventory.csv"
    args = [str(labels_path), "--out", str(out), "--inventory", str(inv), *extra]
    return labels_path, images_root, out, inv, args


def test_each_seeded_problem_reported_once(tmp_path):
    labels_path, images_root, out, inv, args = run_audit(tmp_path)
    before = sha256_file(labels_path)
    assert audit.main(args + ["--images-root", str(images_root), "--path-template", TEMPLATE]) == 0
    assert sha256_file(labels_path) == before  # inputs untouched

    problems = read_csv(out / "problems.csv")
    got = [(p["sample_id"], p["problem"]) for p in problems]
    assert len(got) == len(set(got))
    assert set(got) == SEEDED
    detail = {(p["sample_id"], p["problem"]): p["detail"] for p in problems}
    assert detail[("v0/s0/i1", "coord_out_of_range")] == "joints 3"
    assert detail[("v1/s0/i1", "duplicate_frame_idx")] == "same frame_idx as v1/s0/i0"

    summary = read_json(out / "label_summary.json")
    assert (summary["n_videos"], summary["n_segments"], summary["n_images"]) == (2, 4, 12)
    assert summary["matches_expected"] is False
    assert summary["images_per_video"] == {"v0": 6, "v1": 6}
    assert set(summary["visibility_values"]) == {"0", "1", "2"}
    assert sum(summary["problem_counts"].values()) == len(SEEDED)


def test_inventory(tmp_path):
    labels_path, images_root, out, inv, args = run_audit(tmp_path)
    audit.main(args + ["--images-root", str(images_root), "--path-template", TEMPLATE])
    rows = read_csv(inv)
    assert list(rows[0]) == audit.INVENTORY_COLUMNS
    assert [r["sample_id"] for r in rows] == sorted(r["sample_id"] for r in rows)
    assert len(rows) == 12
    by_id = {r["sample_id"]: r for r in rows}
    assert by_id["v0/s1/i2"]["image_path"] == "v0/s1/i2.png"  # relative, '/' separators
    assert (by_id["v0/s1/i2"]["width"], by_id["v0/s1/i2"]["height"]) == ("64", "48")
    assert by_id["v1/s1/i0"]["width"] == ""  # image deleted
    assert by_id["v0/s0/i0"]["n_reliable_body"] == ""  # bad shape: unknown
    assert int(by_id["v0/s1/i2"]["n_reliable_body"]) + int(by_id["v0/s1/i2"]["n_reliable_ski"]) \
        + int(by_id["v0/s1/i2"]["n_reliable_pole"]) == 22  # make_annotation marks 2 of 24 unreliable


def test_pass1_only(tmp_path):
    labels_path, images_root, out, inv, args = run_audit(tmp_path)
    assert audit.main(args) == 0
    problems = {p["problem"] for p in read_csv(out / "problems.csv")}
    assert not problems & {"label_without_image", "image_without_label", "unreadable_image"}
    assert all(r["width"] == "" and r["image_path"] == "" for r in read_csv(inv))
    assert read_json(out / "label_summary.json")["images_checked"] is False


def test_images_without_template_fails(tmp_path, capsys):
    labels_path, images_root, out, inv, args = run_audit(tmp_path)
    assert audit.main(args + ["--images-root", str(images_root)]) == 1
    assert "1E" in capsys.readouterr().err


def test_broken_nesting_fails(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text("[1, 2, 3]")
    assert audit.main([str(p), "--out", str(tmp_path / "o"), "--inventory", str(tmp_path / "i.csv")]) == 1


def test_duplicate_key_reported(tmp_path):
    entry = json.dumps({"annotation": [[0.5, 0.5, 1]] * 24, "frame_idx": 0})
    p = tmp_path / "labels.json"
    p.write_text(f'{{"v0": {{"s0": {{"i0": {entry}, "i0": {entry}}}}}}}')
    out = tmp_path / "o"
    assert audit.main([str(p), "--out", str(out), "--inventory", str(tmp_path / "i.csv")]) == 0
    problems = read_csv(out / "problems.csv")
    assert [(p["sample_id"], p["problem"]) for p in problems] == [("", "duplicate_key")]


def test_info_preview_in_summary(tmp_path):
    labels_path, images_root, out, inv, args = run_audit(tmp_path)
    info = tmp_path / "info.json"
    info.write_text(json.dumps({"v0": {"s0": {"view": "front"}}}))
    audit.main(args + ["--info", str(info)])
    preview = read_json(out / "label_summary.json")["info_preview"]
    assert "view" in json.dumps(preview) and "front" not in json.dumps(preview)
