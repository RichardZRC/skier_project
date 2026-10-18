import importlib.util

import pytest

from skier.schema import load_schema


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def pytest_collection_modifyitems(config, items):
    has_mmpose = _has("mmpose")
    cuda = False
    if any("gpu" in item.keywords for item in items) and _has("torch"):
        import torch

        cuda = torch.cuda.is_available()
    for item in items:
        if "stack" in item.keywords and not has_mmpose:
            item.add_marker(pytest.mark.skip(reason="needs the pinned torch/mmcv/mmpose stack"))
        if "gpu" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch):
    """Tests must never write to the real $SKIER_DATA / $SKIER_RESULTS."""
    monkeypatch.setenv("SKIER_DATA", str(tmp_path / "skier_data"))
    monkeypatch.setenv("SKIER_RESULTS", str(tmp_path / "skier_results"))


@pytest.fixture(scope="session")
def schema():
    return load_schema()
