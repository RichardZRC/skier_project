from importlib import metadata

from skier import envcheck
from skier.io_utils import read_json

KEYS = {"timestamp_utc", "os", "python", "git_commit", "packages", "torch_cuda_build",
        "cuda_available", "gpu_name", "vram_gb", "cudnn", "checks"}
CHECKS = ["numpy_lt_2", "torch_present", "cuda_available", "mmcv_ops_import", "mmpose_present", "gpu_conv"]


def test_runs_without_crashing(tmp_path):
    assert envcheck.main(["--out", str(tmp_path)]) == 0
    info = read_json(tmp_path / "env.json")
    assert KEYS <= set(info)
    assert [c["name"] for c in info["checks"]] == CHECKS
    assert info["packages"]["numpy"]
    assert info["cuda_available"] in (True, False, None)


def test_missing_package_is_null(monkeypatch):
    real = metadata.version

    def fake(dist):
        if dist == "mmpose":
            raise metadata.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(metadata, "version", fake)
    info = envcheck.collect()
    assert info["packages"]["mmpose"] is None
    assert next(c for c in info["checks"] if c["name"] == "mmpose_present")["ok"] is False


def test_strict_fails_on_any_check(tmp_path, monkeypatch):
    real = envcheck.collect

    def failing():
        info = real()
        info["checks"] = [{"name": "x", "ok": True, "detail": ""}, {"name": "y", "ok": None, "detail": ""}]
        return info

    monkeypatch.setattr(envcheck, "collect", failing)
    assert envcheck.main(["--strict", "--out", str(tmp_path)]) == 1
    assert envcheck.main(["--out", str(tmp_path)]) == 0


def test_numpy_ok():
    assert envcheck.numpy_ok("1.26.4")
    assert not envcheck.numpy_ok("2.0.1")


def test_no_gpu_reports_false_or_null():
    info = envcheck.collect()
    if info["packages"]["torch"] is None:
        assert info["cuda_available"] is None
        assert next(c for c in info["checks"] if c["name"] == "gpu_conv")["ok"] is None
    elif not info["cuda_available"]:
        assert info["cuda_available"] is False
