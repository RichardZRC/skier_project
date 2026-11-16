import subprocess
import sys

import pytest

from skier import envcheck, splits
from skier.data import alignment, audit, overlays
from skier.integration import confidence, process, propagation
from skier.paths import repo_root
from skier.report import compare_pseudo, compare_trackers, figures, tables
from skier.pose import aug_preview, compare, crop_error, crosscheck, eval_generic, evaluate, train as pose_train
from skier.pose import pseudo
from skier.pose import smoke as pose_smoke
from skier.replay import run as replay_run
from skier.tracking import run as tracking_run
from skier.tracking import smoke as tracker_smoke

TOOLS = {
    "audit_pose_labels": audit,
    "draw_gt_overlays": overlays,
    "check_alignment": alignment,
    "check_env": envcheck,
    "smoke_pose": pose_smoke,
    "smoke_tracker": tracker_smoke,
    "make_pose_split": splits,
    "eval_generic_pose": eval_generic,
    "crosscheck_mmpose": crosscheck,
    "run_tracker": tracking_run,
    "make_replay": replay_run,
    "train_pose": pose_train,
    "make_crop_error_set": crop_error,
    "eval_pose": evaluate,
    "compare_experiments": compare,
    "show_augmentations": aug_preview,
    "check_tracker_confidence": confidence,
    "eval_propagation": propagation,
    "process_clip": process,
    "make_tables": tables,
    "make_figures": figures,
    "compare_trackers": compare_trackers,
    "make_pseudo_labels": pseudo,
    "compare_pseudo": compare_pseudo,
}


@pytest.mark.parametrize("name", TOOLS)
def test_help_in_process(name, capsys):
    with pytest.raises(SystemExit) as exc:
        TOOLS[name].main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out


@pytest.mark.parametrize("name", TOOLS)
def test_script_help_and_lazy_imports(name):
    """The script runs `--help` without importing torch, mmpose, siamfc or STARK (``lib``, easydict)."""
    script = repo_root() / "scripts" / f"{name}.py"
    code = (
        "import runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0, e.code\n"
        "heavy = [m for m in ('torch', 'mmpose', 'mmcv', 'siamfc', 'got10k', 'lib', 'easydict') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_smoke_pose_parser():
    args = pose_smoke.build_parser().parse_args(
        ["--config", "c.py", "--checkpoint", "w.pth", "--image", "i.jpg", "--box", "1", "2", "30.5", "40"]
    )
    assert args.box == [1.0, 2.0, 30.5, 40.0]
    assert (args.warmup, args.runs, args.device) == (5, 20, "cuda:0")


def test_smoke_tracker_parser():
    args = tracker_smoke.build_parser().parse_args(
        ["--siamfc-repo", "r", "--weights", "w.pth", "--video", "v.mp4", "--init-frame", "3",
         "--init-box", "1", "2", "3", "4"]
    )
    assert args.init_box == [1.0, 2.0, 3.0, 4.0] and args.max_frames == 300 and args.tracker == "siamfc"
