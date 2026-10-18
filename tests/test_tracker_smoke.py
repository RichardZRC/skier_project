import numpy as np
import pytest
import synth

from skier.tracking.smoke import read_boxes_csv, render_track, run_on_video, write_boxes_csv, xywh1_to_xyxy, xyxy_to_xywh1


class FakeTracker:
    """Shifts the box by (+3, +1) on every update."""

    def __init__(self):
        self.box = None
        self.inits = 0

    def init(self, frame_bgr, box_xyxy):
        self.inits += 1
        self.box = np.asarray(box_xyxy, dtype=np.float64)

    def update(self, frame_bgr):
        assert frame_bgr.ndim == 3
        self.box = self.box + np.array([3, 1, 3, 1])
        return self.box, 0.5


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.avi"
    synth.make_moving_square_video(path, n=20, size=(320, 180), step=4, square=20)
    return path


def test_run_on_video(video):
    tracker = FakeTracker()
    init_box = (10.0, 20.0, 50.0, 80.0)
    rows = run_on_video(tracker, video, init_frame=5, init_box_xyxy=init_box, max_frames=10)
    assert [r[0] for r in rows] == list(range(5, 15))
    assert rows[0][1:] == init_box
    np.testing.assert_allclose(rows[-1][1:], np.array(init_box) + 9 * np.array([3, 1, 3, 1]))
    assert tracker.inits == 1  # never re-initialized


def test_stops_at_eof(video):
    rows = run_on_video(FakeTracker(), video, init_frame=15, init_box_xyxy=(0, 0, 10, 10), max_frames=300)
    assert [r[0] for r in rows] == list(range(15, 20))


def test_init_past_eof_raises(video):
    with pytest.raises(ValueError, match="past the end"):
        run_on_video(FakeTracker(), video, init_frame=25, init_box_xyxy=(0, 0, 10, 10), max_frames=5)


def test_boxes_csv_round_trip(tmp_path):
    rows = [(3, 1.23456, 2.0, 30.5, 40.25), (4, -1.0, 0.0, 10.0, 10.0)]
    write_boxes_csv(rows, tmp_path / "boxes.csv")
    back = read_boxes_csv(tmp_path / "boxes.csv")
    assert [r[0] for r in back] == [3, 4]
    np.testing.assert_allclose([r[1:] for r in back], [r[1:] for r in rows], atol=5e-4)


def test_box_conversion():
    np.testing.assert_allclose(xyxy_to_xywh1((10, 20, 30, 60)), (11, 21, 20, 40))
    box = np.array([12.5, 7.25, 99.0, 150.75])
    np.testing.assert_allclose(xywh1_to_xyxy(xyxy_to_xywh1(box)), box)


def test_render_track(tmp_path, video):
    import cv2

    rows = run_on_video(FakeTracker(), video, init_frame=2, init_box_xyxy=(10, 10, 40, 40), max_frames=8)
    out = tmp_path / "track.mp4"
    render_track(video, rows, out)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == len(rows)
