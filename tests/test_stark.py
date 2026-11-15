"""Step 7b (D28): the STARK adapter, the tracker choice, tracker.json and the tracker guard."""

import argparse
import hashlib
import importlib
import io
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import synth

from skier.integration.config import IntegrationConfig
from skier.paths import configs_dir
from skier.tracking import stark_adapter as sa
from skier.tracking import trackers as trk

# read before the autouse fixture points SKIER_DATA at tmp_path: the real STARK clone and weights, if this machine has them
STARK = Path(os.environ.get("SKIER_DATA", "~/skier_data")).expanduser() / "vendor" / "Stark"
HAS_STARK = (STARK / sa.CHECKPOINT).is_file()


def parse(argv, weights_required=True):
    p = argparse.ArgumentParser()
    trk.add_tracker_args(p, weights_required)
    return p.parse_args(argv)


def test_box_conversion_is_zero_based():
    np.testing.assert_allclose(sa.xyxy_to_xywh0((10, 20, 30, 60)), (10, 20, 20, 40))
    box = np.array([12.5, 7.25, 99.0, 150.75])
    np.testing.assert_allclose(sa.xywh0_to_xyxy(sa.xyxy_to_xywh0(box)), box)


# --- import shims (torch-free) -------------------------------------------------------------------

FAKE_STARK = {
    "lib/__init__.py": "",
    "lib/models/__init__.py": "",
    "lib/models/stark/__init__.py": "raise ImportError('timm (STARK-Lightning)')",
    "lib/models/stark/stark_st.py": "from .helper import VALUE\n\ndef build_starkst(cfg):\n    return ('net', VALUE)\n",
    "lib/models/stark/helper.py": "VALUE = 7\n",
    "lib/train/__init__.py": "raise ImportError('lib.train.admin')",
    "lib/train/data/__init__.py": "raise ImportError('torch._six')",
    "lib/train/data/processing_utils.py": "def sample_target():\n    return 'ok'\n",
}


def write_tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.fixture
def clean_lib(monkeypatch):
    """The fake ``lib`` packages never leak into other tests (and a real one loaded earlier comes back after)."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {k: v for k, v in sys.modules.items() if k == "lib" or k.startswith("lib.")}
    for k in saved:
        del sys.modules[k]
    yield
    for k in [k for k in sys.modules if k == "lib" or k.startswith("lib.")]:
        del sys.modules[k]
    sys.modules.update(saved)


def test_install_shims_skips_the_package_inits(tmp_path, clean_lib):
    repo = write_tree(tmp_path / "stark", FAKE_STARK)
    with pytest.raises(ImportError, match="timm"):  # what STARK's own package init does under our stack
        sys.path.insert(0, str(repo))
        importlib.import_module("lib.models.stark.stark_st")
    for k in [k for k in sys.modules if k.startswith("lib.")]:
        del sys.modules[k]
    sa.install_shims(repo)
    assert importlib.import_module("lib.models.stark.stark_st").build_starkst(None) == ("net", 7)  # relative imports
    assert importlib.import_module("lib.train.data.processing_utils").sample_target() == "ok"
    sa.install_shims(repo)  # a second backend: nothing changes
    assert sys.modules["lib.models.stark"].__path__ == [str((repo / "lib/models/stark").resolve())]


def test_install_shims_refuses_another_lib(tmp_path, clean_lib):
    other = write_tree(tmp_path / "other", {"lib/__init__.py": ""})
    sys.path.insert(0, str(other))
    importlib.import_module("lib")
    with pytest.raises(ImportError, match="another package named lib"):
        sa.install_shims(write_tree(tmp_path / "stark", FAKE_STARK))


def test_alias_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "skier_test_gone", raising=False)
    sa.alias_module("skier_test_gone", f=len)
    assert importlib.import_module("skier_test_gone").f is len
    monkeypatch.delitem(sys.modules, "skier_test_gone")
    sa.alias_module("json", dumps=None)  # a module that imports is left alone
    assert json.dumps({}) == "{}"


def test_attrdict_does_what_starks_config_needs():
    """STARK's config.py: ``cfg = edict(); cfg.TEST = edict(); ...``, then ``edict(yaml.safe_load(f))`` merged by item
    assignment, and ``hasattr(cfg.TEST.UPDATE_INTERVALS, name)`` to pick an update interval."""
    E = sa.AttrDict
    cfg = E()
    cfg.TEST = E()
    cfg.TEST.SEARCH_SIZE = 256
    cfg.TEST.UPDATE_INTERVALS = E()
    exp = E({"TEST": {"SEARCH_SIZE": 320, "UPDATE_INTERVALS": {"LASOT": [200]}}, "DATA": {"T": [{"a": 1}]}})
    assert isinstance(exp.TEST, E) and isinstance(exp.DATA.T[0], E) and exp.DATA.T[0].a == 1  # nested, also in lists
    assert isinstance(exp, dict) and exp["TEST"] is exp.TEST

    def update(base, new):  # STARK's _update_config
        for k, v in new.items():
            if not isinstance(v, dict):
                base[k] = v
            else:
                update(base[k], v)

    update(cfg, E({"TEST": exp.TEST}))
    assert cfg.TEST.SEARCH_SIZE == 320 and cfg["TEST"]["SEARCH_SIZE"] == 320  # item writes read as attributes
    assert hasattr(cfg.TEST.UPDATE_INTERVALS, "LASOT") and not hasattr(cfg.TEST.UPDATE_INTERVALS, "SKITB")
    assert cfg.TEST.UPDATE_INTERVALS.LASOT == [200]
    with pytest.raises(AttributeError):
        cfg.MISSING


def fake_training_class(monkeypatch):
    """A class living in ``lib.train.admin.settings``, as in STARK's checkpoints; remove the modules to forget it."""
    names = ["lib", "lib.train", "lib.train.admin", "lib.train.admin.settings"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    Settings = type("Settings", (), {"__module__": "lib.train.admin.settings"})
    sys.modules["lib.train.admin.settings"].Settings = Settings
    return Settings, names


def test_stub_unpickler_turns_training_objects_into_placeholders(monkeypatch, clean_lib):
    Settings, names = fake_training_class(monkeypatch)
    s = Settings()
    s.project_path = "train/stark_st2/baseline"
    blob = pickle.dumps({"net": {"w": [1, 2]}, "settings": s, "when": np.float64(3.5)}, protocol=2)
    for n in names:
        monkeypatch.delitem(sys.modules, n)
    out = sa.STUB_PICKLE.load(io.BytesIO(blob))
    assert out["net"] == {"w": [1, 2]} and out["when"] == 3.5  # other globals still resolve
    assert isinstance(out["settings"], sa._Placeholder) and out["settings"].state["project_path"].startswith("train")


@pytest.mark.torch
def test_load_net_state_reads_a_legacy_checkpoint(tmp_path, monkeypatch, clean_lib):
    import torch

    Settings, names = fake_training_class(monkeypatch)
    torch.save({"epoch": 50, "net": {"w": torch.arange(3.0)}, "settings": Settings()}, tmp_path / "ck.pth.tar",
               _use_new_zipfile_serialization=False)  # STARK's checkpoints are in the legacy format
    for n in names:
        monkeypatch.delitem(sys.modules, n)
    state = sa.load_net_state(tmp_path / "ck.pth.tar")
    assert list(state) == ["w"] and state["w"].tolist() == [0.0, 1.0, 2.0]
    torch.save({"model": {}}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="not a STARK checkpoint"):
        sa.load_net_state(tmp_path / "other.pth")


# --- the tracker wrapper -------------------------------------------------------------------------

class FakeStarkST:
    """STARK_ST's interface: ``initialize(rgb, {"init_bbox": list})``; ``track(rgb)``. A still target, as STARK
    reports it (0-based xywh)."""

    def __init__(self):
        self.seen, self.state = [], None

    def initialize(self, image, info):
        assert isinstance(info["init_bbox"], list)
        self.seen.append(image.copy())
        self.state = info["init_bbox"]

    def track(self, image):
        self.seen.append(image.copy())
        return {"target_bbox": list(self.state), "conf_score": np.float32(0.75)}


def test_stark_tracker_conventions():
    backend = types.SimpleNamespace(tracker=FakeStarkST(), owner=None)
    t = sa.StarkTracker(backend)
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    box = np.array([12.5, 7.25, 30.0, 33.75])
    t.init(frame, box)
    assert backend.tracker.state == pytest.approx([12.5, 7.25, 17.5, 26.5])  # 0-based x, y, w, h
    out, score = t.update(frame)
    np.testing.assert_allclose(out, box)  # a still target comes back as the same xyxy box
    assert score == 0.75 and type(score) is float
    assert all((img[..., 2] == 255).all() and (img[..., 0] == 0).all() for img in backend.tracker.seen)  # RGB in
    sa.StarkTracker(backend).init(frame, box)  # the next clip claims the shared backend
    with pytest.raises(RuntimeError, match="one at a time"):
        t.update(frame)


# --- tracker choice, tracker.json and the guard --------------------------------------------------

def test_tracker_args():
    a = parse(["--siamfc-repo", "r", "--weights", "w"])
    assert a.tracker == "siamfc" and trk.tracker_repo(a) == Path("r")  # the default: every pre-D28 command
    assert trk.tracker_repo(parse(["--tracker", "stark", "--stark-repo", "s", "--weights", "w"])) == Path("s")
    with pytest.raises(ValueError, match="needs --stark-repo"):
        trk.tracker_repo(parse(["--tracker", "stark", "--weights", "w"]))
    with pytest.raises(ValueError, match="--stark-repo given with --tracker siamfc"):
        trk.tracker_repo(parse(["--siamfc-repo", "r", "--stark-repo", "s", "--weights", "w"]))
    with pytest.raises(ValueError, match="needs --siamfc-repo and --weights"):
        trk.tracker_repo(parse(["--siamfc-repo", "r"], weights_required=False))


def test_stark_factory_builds_one_backend_lazily(monkeypatch):
    built = []

    class FakeBackend:
        def __init__(self, repo, weights):
            built.append((repo, weights))
            self.tracker, self.owner = FakeStarkST(), None

    monkeypatch.setattr(sa, "StarkBackend", FakeBackend)
    make = trk.tracker_factory(parse(["--tracker", "stark", "--stark-repo", "s", "--weights", "w"]))
    assert built == []  # nothing loads before the first clip
    a, b = make(), make()
    assert built == [(Path("s"), Path("w"))] and a._backend is b._backend and isinstance(b, sa.StarkTracker)


def test_tracker_json_and_check(tmp_path):
    w = tmp_path / "w.pth"
    w.write_bytes(b"abc")
    info = trk.tracker_info(parse(["--tracker", "stark", "--stark-repo", str(tmp_path), "--weights", str(w)]))
    assert info["tracker"] == "stark" and info["weights_sha256"] == hashlib.sha256(b"abc").hexdigest()
    d = tmp_path / "tracks"
    d.mkdir()
    assert trk.read_tracker_name(d) == "siamfc"  # a folder from before D28
    trk.check_tracker(d, "siamfc")
    trk.write_tracker_info(info, d)
    assert trk.read_tracker_name(d) == "stark"
    with pytest.raises(ValueError, match="holds stark tracks"):
        trk.check_tracker(d, "siamfc")
    (d / "tracker.json").write_text('{"tracker": "kcf"}')
    with pytest.raises(ValueError, match="unknown tracker"):
        trk.read_tracker_name(d)


def test_run_tracker_writes_tracker_json(tmp_path, monkeypatch):
    from test_tracking_run import ShiftTracker, make_skitb

    from skier.tracking import run

    split = make_skitb(tmp_path / "skitb")
    w = tmp_path / "w.pth"
    w.write_bytes(b"w")
    args = ["--tracker", "stark", "--stark-repo", str(tmp_path), "--weights", str(w), "--skitb-root",
            str(tmp_path / "skitb"), "--split-json", str(split), "--partition", "train", "--fps", "25"]
    monkeypatch.setattr(run, "tracker_factory", lambda a: ShiftTracker)
    assert run.main(args + ["--out", str(tmp_path / "out")]) == 0
    assert trk.read_tracker_name(tmp_path / "out") == "stark" and (tmp_path / "out" / "001__0.csv").is_file()
    monkeypatch.undo()
    assert run.main(args[:2] + args[4:] + ["--out", str(tmp_path / "o2")]) == 1  # --tracker stark, no --stark-repo


def test_integration_config_tracker():
    default = IntegrationConfig.from_yaml(configs_dir() / "integration" / "default.yaml")
    assert IntegrationConfig().tracker == default.tracker == "siamfc"  # default.yaml stays as frozen in D27
    with pytest.raises(ValueError, match="tracker must be one of"):
        IntegrationConfig(tracker="kcf")
    s = IntegrationConfig.from_yaml(configs_dir() / "integration" / "stark.yaml")
    assert s.tracker == "stark"
    fixed = ("k_pose", "pose_refine", "refine_min_score", "refine_k")  # only the tracker changes (D28)
    assert [getattr(s, k) for k in fixed] == [getattr(default, k) for k in fixed]
    s.require("k_pose", "conf_signal", "replay_uses_status", "pose_refine")  # filled in by the val check (D28)
    assert s.conf_signal == "raw" and 0 < s.conf_threshold < 1  # on STARK's [0, 1] scale, not SiamFC's 3.9e-5
    with pytest.raises(ValueError, match="for stark"):
        s.with_(conf_threshold=None).require("conf_signal")


def test_replay_evaluate_refuses_another_trackers_config(tmp_path):
    from test_report import DriftTracker, make_skitb_multi

    from skier.replay import run as replay_run
    from skier.tracking.run import run_skitb

    split, fps = make_skitb_multi(tmp_path / "skitb")
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    run_skitb(DriftTracker, tmp_path / "skitb", split, "train", tracks, fps=25.0)
    (tracks / "tracker.json").write_text('{"tracker": "stark"}')
    icfg = tmp_path / "integration.yaml"
    icfg.write_text("k_pose: 1.1\nconf_signal: raw\nconf_threshold: 0.4\nreplay_uses_status: true\n")
    args = ["--evaluate", "--track-dir", str(tracks), "--skitb-root", str(tmp_path / "skitb"), "--split-json", str(split),
            "--fps", "25", "--integration-config", str(icfg), "--partition", "train", "--out", str(tmp_path / "out")]
    assert replay_run.main(args) == 1  # SiamFC's settings on STARK tracks
    icfg.write_text("tracker: stark\nk_pose: 1.1\nconf_signal: raw\nconf_threshold: 0.4\nreplay_uses_status: true\n")
    assert replay_run.main(args) == 0
    assert json.loads((tmp_path / "out" / "replay_eval.json").read_text())["tracker"] == "stark"


def test_propagation_refuses_another_trackers_config(tmp_path, capsys):
    from skier.integration import propagation

    split = tmp_path / "split.json"
    split.write_text('{"content_sha256": "x"}')
    usable = tmp_path / "usable.json"
    usable.write_text('{"partition": "val", "split_sha256": "x", "usable": [], "segments": []}')
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "tracker.json").write_text('{"tracker": "stark"}')
    assert propagation.main(["--split", str(split), "--usable", str(usable), "--track-dir", str(tracks),
                             "--checkpoint", "E2-s230=x.pt", "--hrnet-config", "h.py", "--out", str(tmp_path / "o")]) == 1
    assert "holds stark tracks" in capsys.readouterr().err


def test_process_clip_refuses_another_trackers_config(tmp_path, capsys):
    from skier.integration import process

    video = tmp_path / "v.avi"
    synth.make_moving_square_video(video, n=10, size=(320, 180), step=4, square=20)
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "tracker.json").write_text('{"tracker": "stark"}')
    base = ["--video", str(video), "--init-frame", "0", "--init-box", "20,80,40,100", "--checkpoint", "c.pt",
            "--hrnet-config", "h.py", "--out", str(tmp_path / "out"), "--no-video"]
    assert process.main(base + ["--track-csv", str(tracks / "track.csv")]) == 1
    assert "holds stark tracks" in capsys.readouterr().err
    assert process.main(base + ["--tracker", "stark", "--stark-repo", "s", "--weights", "w"]) == 1
    assert "the integration config is for siamfc" in capsys.readouterr().err


def test_tables_refuse_mixed_trackers(tmp_path):
    from skier.report.tables import check_root_tracker

    (tmp_path / "tracking" / "skitb").mkdir(parents=True)
    (tmp_path / "propagation").mkdir()
    (tmp_path / "propagation" / "propagation.json").write_text('{"tracker": "stark"}')
    with pytest.raises(ValueError, match="propagation.json was made from stark"):
        check_root_tracker(tmp_path, "siamfc")
    with pytest.raises(ValueError, match="holds siamfc tracks"):
        check_root_tracker(tmp_path, "stark")  # the SkiTB folder has no tracker.json: SiamFC's
    (tmp_path / "tracking" / "skitb" / "tracker.json").write_text('{"tracker": "stark"}')
    check_root_tracker(tmp_path, "stark")


# --- the real STARK (laptop) ---------------------------------------------------------------------

def textured_clip(n=30, size=(640, 360), x0=40, step=5, side=60, seed=3):
    """Frames with a fixed noisy background and a textured square moving sideways; returns frames and GT xyxy."""
    rng = np.random.default_rng(seed)
    w, h = size
    assert 0 <= min(x0, x0 + step * (n - 1)) and max(x0, x0 + step * (n - 1)) + side <= w
    background = rng.integers(60, 140, (h, w, 3), dtype=np.uint8)
    target = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    frames, gt = [], []
    for k in range(n):
        x, y = x0 + step * k, (h - side) // 2
        f = background.copy()
        f[y:y + side, x:x + side] = target
        frames.append((k, f))
        gt.append([x, y, x + side, y + side])
    return frames, np.array(gt, float)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_STARK, reason="needs the D28 STARK clone and its weights (laptop)")
def test_real_stark_tracks_deterministically_and_shares_its_backend():
    from skier.eval.track_metrics import frame_ious
    from skier.tracking.run import boxes_of, track_clip

    backend = sa.StarkBackend(STARK)
    assert backend.tracker.update_intervals == sa.UPDATE_INTERVALS
    a_frames, a_gt = textured_clip(seed=3)
    b_frames, b_gt = textured_clip(seed=5, x0=500, step=-4)  # another target, moving left

    def run(frames, gt, be):
        rows = track_clip(sa.StarkTracker(be), iter(frames), 0, gt[0])
        return boxes_of(rows), np.array([r["score"] for r in rows[1:]])

    box_a, score_a = run(a_frames, a_gt, backend)
    assert frame_ious(box_a, a_gt)[1:].mean() > 0.7 and ((score_a >= 0) & (score_a <= 1)).all()
    again, _ = run(a_frames, a_gt, backend)
    np.testing.assert_array_equal(again, box_a)  # deterministic
    box_b, score_b = run(b_frames, b_gt, backend)  # clip B after clip A on the shared backend
    fresh, fresh_score = run(b_frames, b_gt, sa.StarkBackend(STARK))
    np.testing.assert_array_equal(box_b, fresh)
    np.testing.assert_array_equal(score_b, fresh_score)
