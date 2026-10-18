import cv2
import numpy as np
import pytest
import synth

from skier.data import alignment
from skier.data.alignment import ALIGN_COLUMNS, ncc, to_probe
from skier.io_utils import read_csv, read_json, write_csv
from skier.data.audit import INVENTORY_COLUMNS


@pytest.fixture
def square_video(tmp_path):
    images_root = tmp_path / "images"
    frames_dir = images_root / "vA" / "sA"
    video = tmp_path / "video.avi"
    saved = synth.make_moving_square_video(video, n=60, save_frames=(2, 10, 30, 50), frames_dir=frames_dir)
    return video, images_root, saved


def write_inventory(path, entries):
    """entries: [(image_id, frame_idx, saved_frame)]"""
    rows = [{"sample_id": f"vA/sA/{image_id}", "video_id": "vA", "segment_id": "sA", "image_id": image_id,
             "frame_idx": frame_idx, "image_path": f"vA/sA/frame_{saved:03d}.png"}
            for image_id, frame_idx, saved in entries]
    write_csv(rows, path, INVENTORY_COLUMNS)
    return path


def run(tmp_path, video, images_root, entries, *extra):
    inv = write_inventory(tmp_path / "inventory.csv", entries)
    out = tmp_path / "out"
    code = alignment.main(["--inventory", str(inv), "--images-root", str(images_root), "--video-id", "vA",
                           "--segment-id", "sA", "--video", str(video), "--start-frame", "0",
                           "--out", str(out), *extra])
    return code, out


def test_exact_labels_give_offset_zero(tmp_path, square_video):
    video, images_root, _ = square_video
    code, out = run(tmp_path, video, images_root, [("i10", 10, 10), ("i30", 30, 30), ("i50", 50, 50)])
    assert code == 0
    rows = read_csv(out / "alignment.csv")
    assert list(rows[0]) == ALIGN_COLUMNS
    assert [int(r["offset"]) for r in rows] == [0, 0, 0]
    assert all(r["verified"] == "True" for r in rows)
    assert all(float(r["margin"]) > 0 for r in rows)
    summary = read_json(out / "alignment_summary.json")
    assert summary["consistent"] is True and summary["offsets"] == {"0": 3}


def test_shifted_labels_give_minus_two(tmp_path, square_video):
    video, images_root, _ = square_video
    code, out = run(tmp_path, video, images_root, [("i10", 12, 10), ("i30", 32, 30), ("i50", 52, 50)])
    assert code == 0
    assert [int(r["offset"]) for r in read_csv(out / "alignment.csv")] == [-2, -2, -2]


def test_window_clipped_at_start_and_eof(tmp_path, square_video):
    video, images_root, _ = square_video
    code, out = run(tmp_path, video, images_root, [("i2", 2, 2), ("late", 70, 50)])
    assert code == 0
    rows = {r["image_id"]: r for r in read_csv(out / "alignment.csv")}
    assert rows["i2"]["offset"] == "0"
    assert rows["late"]["best_decode_idx"] == "" and rows["late"]["verified"] == "False"
    assert read_json(out / "alignment_summary.json")["consistent"] is False


def test_gt_clip(tmp_path, square_video):
    video, images_root, _ = square_video
    labels = {"vA": {"sA": {f"i{k}": {"annotation": synth.make_annotation(np.random.default_rng(k)), "frame_idx": k}
                            for k in (10, 30, 50)}}}
    labels_path = synth.write_labels(labels, tmp_path / "labels.json")
    code, out = run(tmp_path, video, images_root, [("i10", 10, 10), ("i30", 30, 30), ("i50", 50, 50)],
                    "--labels", str(labels_path))
    assert code == 0
    cap = cv2.VideoCapture(str(out / "gt_clip.mp4"))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == (50 + 5) - (10 - 5) + 1


def test_locate_segment_unfilled(tmp_path, square_video, capsys):
    _, images_root, _ = square_video
    inv = write_inventory(tmp_path / "inventory.csv", [("i10", 10, 10)])
    code = alignment.main(["--inventory", str(inv), "--images-root", str(images_root),
                           "--video-id", "vA", "--segment-id", "sA", "--out", str(tmp_path / "o")])
    assert code == 1
    assert "--video" in capsys.readouterr().err


def test_ncc():
    rng = np.random.default_rng(0)
    p = to_probe(rng.integers(0, 255, size=(90, 160, 3), dtype=np.uint8))
    assert ncc(p, p) == pytest.approx(1.0, abs=1e-6)
    assert ncc(p, -p) == pytest.approx(-1.0, abs=1e-6)
    flat = to_probe(np.full((90, 160, 3), 7, dtype=np.uint8))
    assert ncc(p, flat) == 0.0 and not np.isnan(ncc(flat, flat))
