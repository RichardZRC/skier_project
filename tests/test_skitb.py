import json

import numpy as np
import pytest

from skier.data.skitb import frame_file, load_skitb_clips, parse_boxes_txt


def make_tree(root, clips_per_seq=None):
    """AL/<seq>/{frames/, MC/, SC/<clip>/{boxes,frames,visibilities}.txt}, as in the real
    files (frames.txt holds bare indices into the source video); 2 sequences, 3 clips."""
    clips_per_seq = clips_per_seq or {"001": ["0", "1"], "002": ["0"]}
    for seq, clips in clips_per_seq.items():
        (root / "AL" / seq / "frames").mkdir(parents=True)
        (root / "AL" / seq / "MC").mkdir()
        (root / "AL" / seq / "SC").mkdir()
        (root / "AL" / seq / "SC" / ".DS_Store").write_text("")  # Finder litter: a file, ignored
        for c in clips:
            d = root / "AL" / seq / "SC" / c
            d.mkdir()
            (d / "boxes.txt").write_text("10,20,30,40\n11 21 30 40\n12, 22, 30, 40\n")
            (d / "frames.txt").write_text("197\n198\n199\n")
            (d / "visibilities.txt").write_text("1\n0\n1\n")
    split = root / "split.json"
    split.write_text(json.dumps({"train": ["001"], "test": ["002"]}))
    return split


def test_parse_boxes_mixed_separators(tmp_path):
    p = tmp_path / "boxes.txt"
    p.write_text("10,20,30,40\n\n1.5 2.5 3 4\n")
    np.testing.assert_allclose(parse_boxes_txt(p), [[10, 20, 40, 60], [1.5, 2.5, 4.5, 6.5]])
    p.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="4 numbers"):
        parse_boxes_txt(p)


def test_clips_inherit_sequence_partition(tmp_path):
    split = make_tree(tmp_path)
    train = load_skitb_clips(tmp_path, split, "train")
    test = load_skitb_clips(tmp_path, split, "test")
    assert [(c.sequence_id, c.clip_id, c.partition) for c in train] == [("001", "0", "train"), ("001", "1", "train")]
    assert [(c.sequence_id, c.clip_id) for c in test] == [("002", "0")]
    c = train[0]
    np.testing.assert_allclose(c.boxes_xyxy[0], [10, 20, 40, 60])
    assert c.visible.tolist() == [True, False, True]
    assert c.frames_dir == tmp_path / "AL" / "001" / "frames"
    assert [frame_file(e) for e in c.frame_files] == ["00197.jpg", "00198.jpg", "00199.jpg"]


def test_frame_file():
    assert frame_file("197") == "00197.jpg"
    assert frame_file("0") == "00000.jpg"
    assert frame_file("99999") == "99999.jpg"
    for bad in ("00197.jpg", "100000", "-1", "1.0", "", "１２"):  # the last one is full-width digits
        with pytest.raises(ValueError, match="frame index"):
            frame_file(bad)


def test_clips_in_camera_order(tmp_path):
    split = make_tree(tmp_path, {"001": ["0", "1", "2", "10", "9"], "002": ["0"]})
    assert [c.clip_id for c in load_skitb_clips(tmp_path, split, "train")] == ["0", "1", "2", "9", "10"]
    (tmp_path / "AL" / "001" / "SC" / "extra").mkdir()
    with pytest.raises(ValueError, match="camera indices"):
        load_skitb_clips(tmp_path, split, "train")


def test_surprises_fail_loudly(tmp_path):
    split = make_tree(tmp_path)
    with pytest.raises(ValueError, match="no partition"):
        load_skitb_clips(tmp_path, split, "val")
    (tmp_path / "AL" / "001" / "SC" / "1" / "visibilities.txt").write_text("1\n0\n")
    with pytest.raises(ValueError, match="line counts differ"):
        load_skitb_clips(tmp_path, split, "train")


def test_missing_clip_file(tmp_path):
    split = make_tree(tmp_path)
    (tmp_path / "AL" / "002" / "SC" / "0" / "boxes.txt").unlink()
    with pytest.raises(FileNotFoundError, match="boxes.txt"):
        load_skitb_clips(tmp_path, split, "test")
