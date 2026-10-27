import json
import math

import cv2
import numpy as np
import pytest
import synth

from skier.eval.track_metrics import frame_ious
from skier.tracking.frames import ImageFrames, VideoFrames
from skier.tracking.run import (
    TRACK_COLUMNS,
    evaluate_clip,
    read_track_csv,
    run_skitb,
    track_clip,
    write_track_csv,
)
from skier.io_utils import read_csv


class ShiftTracker:
    """Fake tracker: moves the box by (+2, 0) per update; score falls with each frame."""

    def __init__(self):
        self.inits, self.box, self.t = 0, None, 0

    def init(self, frame_bgr, box_xyxy):
        self.inits += 1
        self.box = np.asarray(box_xyxy, dtype=np.float64)

    def update(self, frame_bgr):
        assert frame_bgr.ndim == 3
        self.t += 1
        self.box = self.box + np.array([2, 0, 2, 0])
        return self.box, 1.0 / self.t


def fake_frames(n):
    return ((i, np.zeros((20, 30, 3), np.uint8)) for i in range(n))


def test_track_clip_inits_once_and_marks_status():
    tr = ShiftTracker()
    rows = track_clip(tr, fake_frames(10), init_index=3, init_box=(0, 0, 10, 10), max_frames=5)
    assert [r["frame_index"] for r in rows] == [3, 4, 5, 6, 7]
    assert [r["status"] for r in rows] == ["init"] + ["track"] * 4
    assert math.isnan(rows[0]["score"]) and rows[1]["score"] == 1.0
    assert rows[-1]["x1"] == 8.0 and tr.inits == 1
    with pytest.raises(ValueError, match="never reached"):
        track_clip(ShiftTracker(), fake_frames(3), init_index=5, init_box=(0, 0, 1, 1))


def test_track_csv_round_trip(tmp_path):
    rows = track_clip(ShiftTracker(), fake_frames(4), 0, (1.2345, 2, 3, 4))
    write_track_csv(rows, tmp_path / "c.csv")
    assert list(read_csv(tmp_path / "c.csv")[0]) == TRACK_COLUMNS
    back = read_track_csv(tmp_path / "c.csv")
    assert [r["status"] for r in back] == [r["status"] for r in rows]
    assert math.isnan(back[0]["score"]) and back[2]["score"] == pytest.approx(0.5)
    assert back[0]["x1"] == pytest.approx(1.2345, abs=5e-4)


def test_evaluate_clip_matches_hand_iou():
    rows = track_clip(ShiftTracker(), fake_frames(4), 0, (0, 0, 10, 10))
    gt = np.array([[0, 0, 10, 10]] * 4, dtype=float)
    stats = evaluate_clip(rows, gt, np.ones(4, bool), init_pos=0, fps=25)
    expected = frame_ious(np.array([[2, 0, 12, 10], [4, 0, 14, 10], [6, 0, 16, 10]], float), gt[:3])
    assert stats["n_frames"] == 3 and stats["mean_iou"] == pytest.approx(expected.mean())


def test_frame_sources(tmp_path):
    video = tmp_path / "v.avi"
    saved = synth.make_moving_square_video(video, n=12, size=(320, 180), step=4, square=20,
                                           save_frames=(0, 5), frames_dir=tmp_path / "frames")
    vf = VideoFrames(video)
    assert [i for i, _ in vf] == list(range(12)) and vf.fps == pytest.approx(25)
    imf = ImageFrames(tmp_path / "frames", [saved[0].name, saved[5].name], fps=30)
    got = list(imf)
    assert [i for i, _ in got] == [0, 1] and got[1][1].shape == (180, 320, 3)
    with pytest.raises(FileNotFoundError):
        list(ImageFrames(tmp_path / "frames", ["missing.png"]))


def make_skitb(root, n_frames=6):
    """One sequence 001 in train with one SC clip whose GT box moves like ShiftTracker."""
    seq = root / "AL" / "001"
    (seq / "frames").mkdir(parents=True)
    for i in range(n_frames):
        cv2.imwrite(str(seq / "frames" / f"{100 + i:05d}.jpg"), np.zeros((40, 60, 3), np.uint8))
    clip = seq / "SC" / "0"
    clip.mkdir(parents=True)
    (clip / "boxes.txt").write_text("".join(f"{2 * i},0,10,10\n" for i in range(n_frames)))
    (clip / "frames.txt").write_text("".join(f"{100 + i}\n" for i in range(n_frames)))  # bare indices, as in SkiTB
    (clip / "visibilities.txt").write_text("0\n" + "1\n" * (n_frames - 1))  # first frame invisible
    split = root / "split.json"
    split.write_text(json.dumps({"train": ["001"]}))
    return split


def test_run_skitb_end_to_end(tmp_path):
    split = make_skitb(tmp_path / "skitb")
    out = tmp_path / "out"
    out.mkdir()
    s = run_skitb(ShiftTracker, tmp_path / "skitb", split, "train", out, fps=25)
    rows = read_track_csv(out / "001__0.csv")
    assert rows[0]["frame_index"] == 1 and rows[0]["status"] == "init"  # init at the first visible frame
    assert s["n_clips"] == 1 and s["pooled_mean_iou"] == pytest.approx(1.0)  # fake moves exactly like GT
    assert s["n_failure_episodes"] == 0
    assert json.loads((out / "summary.json").read_text())["partition"] == "train"


class StillTracker(ShiftTracker):
    def update(self, frame_bgr):
        return self.box, 0.5


def test_run_skitb_uses_each_sequences_frame_rate(tmp_path):
    """Two identical sequences at 25 and 30 fps (D22): the same failing frames give different durations."""
    root, n = tmp_path / "skitb", 40
    for seq in ("001", "002"):
        d = root / "AL" / seq
        (d / "frames").mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / "frames" / f"{i:05d}.jpg"), np.zeros((40, 60, 3), np.uint8))
        clip = d / "SC" / "0"
        clip.mkdir(parents=True)
        (clip / "boxes.txt").write_text("".join(f"{2 * i},0,10,10\n" for i in range(n)))  # GT moves away
        (clip / "frames.txt").write_text("".join(f"{i}\n" for i in range(n)))
        (clip / "visibilities.txt").write_text("1\n" * n)
    split = root / "split.json"
    split.write_text(json.dumps({"val": ["001", "002"]}))
    out = tmp_path / "out"
    out.mkdir()
    s = run_skitb(StillTracker, root, split, "val", out, fps={"001": 25.0, "002": 30.0})
    # IoU of the still box with GT is 0 from frame 5 on: frames 5..39 fail (35 frames)
    ep = {k: v["failure_episodes"] for k, v in s["per_clip"].items()}
    assert ep["001__0"] == [{"start_frame": 5, "end_frame": 39, "duration_s": round(35 / 25, 4)}]
    assert ep["002__0"] == [{"start_frame": 5, "end_frame": 39, "duration_s": round(35 / 30, 4)}]
    assert s["per_clip"]["002__0"]["fps"] == 30.0 and s["fps"] == "per sequence (D22)"
    with pytest.raises(KeyError, match="002"):
        run_skitb(StillTracker, root, split, "val", out, fps={"001": 25.0})
    assert run_skitb(StillTracker, root, split, "val", out, fps=25)["per_clip"]["002__0"]["fps"] == 25.0
