"""STARK-ST50 tracker adapter (Step 7b, D28): an off-the-shelf comparison for SiamFC.

``StarkTracker.update`` returns ``(box_xyxy, score)`` like ``SiamFCTracker``. STARK's own code runs unedited:
``STARK_ST.__init__``, ``initialize`` and ``track`` from ``lib/test/tracker/stark_st.py``, with its published test
parameters. Its conventions are converted here, at the edge: RGB input, 0-based ``[x, y, w, h]`` boxes, and
``conf_score`` (a sigmoid in [0, 1]) as the score. torch and STARK are imported only when a ``StarkBackend`` is built.

STARK targets PyTorch 1.7. Under our pinned torch 2.1.2, loading it hits six snags. Each is handled here, and no
file of the clone is edited, and nothing is installed:

1. ``lib/models/stark/__init__.py`` also imports STARK-Lightning, then Swin, then ``timm`` (not installed). Fix:
   register ``lib.models.stark`` as a bare package and import ``stark_st`` itself (``install_shims``).
2. ``resnet.py`` imports ``torchvision.models.utils``, removed in torchvision 0.13. Fix: alias it to ``torch.hub``.
3. ``backbone.py`` builds ResNet-50 with ``pretrained=is_main_process()``, which downloads ImageNet weights. Fix:
   patch the name to return False. Every weight comes from the STARK checkpoint, loaded ``strict=True``.
4. The tracker imports ``lib.train.data.processing_utils``. That package's ``__init__`` needs ``torch._six``
   (removed in torch 2.0) and ``jpeg4py``. Fix: register ``lib.train`` and ``lib.train.data`` as bare packages.
5. The checkpoint (legacy torch format) pickles training objects of ``lib.train.admin.{local, settings, stats}``.
   Fix: unpickle those as placeholders and use only ``["net"]`` (``load_net_state``).
6. ``lib/config/stark_st2/config.py`` imports ``easydict`` (not in the pinned venv). It only needs an attribute-style
   dict, so ``AttrDict`` stands in when the package is missing; the real one is used if installed.

One loaded network serves every clip, because ``initialize`` resets all per-clip state. Clips run one at a time.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np


# --- Filled in 2026-11-12 from researchmm/Stark @ 162e1d3 (lib/test/tracker/stark_st.py, lib/test/parameter/
# stark_st.py, experiments/stark_st2/baseline.yaml). ``STARK_ST(params, dataset_name)``;
# ``initialize(rgb, {"init_bbox": [x, y, w, h]})``; ``track(rgb)`` returns ``{"target_bbox": [x, y, w, h],
# "conf_score": float}``. Boxes are 0-based (``sample_target`` indexes the array from x, y) and clipped to the image
# with a 10 px minimum side. A dataset not in ``cfg.TEST.UPDATE_INTERVALS`` gets ``cfg.DATA.MAX_SAMPLE_INTERVAL``.
STARK_COMMIT = "162e1d37a38f92251cb380e90ccdb59608f2df3c"
YAML = "experiments/stark_st2/baseline.yaml"
CHECKPOINT = "checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar"
DATASET_NAME = "skitb"
UPDATE_INTERVALS = [200]  # what STARK picks for DATASET_NAME (D28); checked when the backend is built
TEST_PARAMS = {"template_factor": 2.0, "template_size": 128, "search_factor": 5.0, "search_size": 320}  # baseline.yaml
BARE_PACKAGES = ("lib.models.stark", "lib.train", "lib.train.data")  # snags 1 and 4
STUB_MODULES = ("lib.train",)  # snag 5: pickled classes under these modules become placeholders
# ------------------------------------------------------------------------------------------------------------------


def xyxy_to_xywh0(box) -> np.ndarray:
    """``xyxy`` to 0-based ``[x, y, w, h]`` (STARK / pytracking convention)."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return np.array([x1, y1, x2 - x1, y2 - y1])


def xywh0_to_xyxy(box) -> np.ndarray:
    x, y, w, h = np.asarray(box, dtype=np.float64)
    return np.array([x, y, x + w, y + h])


def install_shims(repo: Path) -> None:
    """Put the clone on ``sys.path`` and register ``BARE_PACKAGES`` without running their ``__init__`` (torch-free).

    Refuses a ``lib`` package that isn't the clone's, since the name is generic.
    """
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    lib = importlib.import_module("lib")
    where = [Path(p).resolve() for p in getattr(lib, "__path__", [])]
    if repo / "lib" not in where:
        raise ImportError(f"'lib' resolves to {where}, not {repo / 'lib'}: another package named lib is on sys.path")
    for name in BARE_PACKAGES:
        if name in sys.modules:
            continue
        parent, _, child = name.rpartition(".")
        mod = types.ModuleType(name)
        mod.__path__ = [str(repo / Path(*name.split(".")))]
        mod.__package__ = name
        sys.modules[name] = mod
        setattr(importlib.import_module(parent), child, mod)


def alias_module(name: str, **attrs) -> None:
    """Make ``import name`` work by giving it a stand-in with ``attrs``, unless the real module imports."""
    try:
        importlib.import_module(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


class AttrDict(dict):
    """What STARK's config needs from ``easydict.EasyDict`` (snag 6): keys read and write as attributes, and nested
    dicts (also inside lists) become ``AttrDict``. Reads of a missing attribute raise AttributeError, so ``hasattr``
    works as STARK expects for its update-interval table."""

    def __init__(self, d=None, **kwargs):
        super().__init__()
        for k, v in {**(d or {}), **kwargs}.items():
            self[k] = v

    def __setitem__(self, key, value):
        super().__setitem__(key, _attr(value))

    __setattr__ = __setitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _attr(value):
    if isinstance(value, dict) and not isinstance(value, AttrDict):
        return AttrDict(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_attr(v) for v in value)
    return value


class _Placeholder:
    """Stands in for a pickled STARK training object (settings, statistics). Never used."""

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module == m or module.startswith(m + ".") for m in STUB_MODULES):
            return _Placeholder
        return super().find_class(module, name)


STUB_PICKLE = types.ModuleType("skier_stark_pickle")  # torch.load(pickle_module=...) needs Unpickler and load
STUB_PICKLE.Unpickler = _StubUnpickler
STUB_PICKLE.load = lambda f, **kw: _StubUnpickler(f, **kw).load()


def load_net_state(weights: Path) -> dict:
    """The ``net`` state dict of a STARK checkpoint; its pickled training objects are skipped (snag 5)."""
    import torch

    ck = torch.load(str(weights), map_location="cpu", pickle_module=STUB_PICKLE)
    if not isinstance(ck, dict) or "net" not in ck:
        raise ValueError(f"{weights} is not a STARK checkpoint (no 'net' entry)")
    return ck["net"]


class StarkBackend:
    """The loaded STARK-ST50 network and STARK's tracker object, shared across clips. CUDA only (STARK's code)."""

    def __init__(self, stark_repo: Path, weights: Path | None = None):
        import torch
        from unittest import mock

        repo = Path(stark_repo).resolve()
        weights = Path(weights) if weights is not None else repo / CHECKPOINT
        alias_module("torchvision.models.utils", load_state_dict_from_url=torch.hub.load_state_dict_from_url)  # 2
        alias_module("easydict", EasyDict=AttrDict)  # 6
        self.easydict = "stand-in" if sys.modules["easydict"].EasyDict is AttrDict else "package"
        install_shims(repo)  # 1, 4
        from lib.config.stark_st2 import config as stark_config

        stark_config.update_config_from_file(str(repo / YAML))
        cfg = stark_config.cfg
        backbone = importlib.import_module("lib.models.stark.backbone")
        backbone.is_main_process = lambda: False  # 3: no ImageNet download; the checkpoint has every weight
        model = importlib.import_module("lib.models.stark.stark_st")
        sys.modules["lib.models.stark"].build_starkst = model.build_starkst  # what the skipped __init__ exports
        tracker_module = importlib.import_module("lib.test.tracker.stark_st")

        # lib/test/parameter/stark_st.py's parameters(), without its environment file (local.py)
        self.params = types.SimpleNamespace(
            cfg=cfg, template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
            search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE, checkpoint=str(weights),
            save_all_boxes=False)
        got = {k: getattr(self.params, k) for k in TEST_PARAMS}
        if got != TEST_PARAMS:
            raise ValueError(f"STARK's config gave test parameters {got}, D28 expects {TEST_PARAMS}")
        state = load_net_state(weights)  # 5

        def stark_load(path, *args, **kwargs):  # STARK_ST.__init__ calls torch.load(params.checkpoint)["net"]
            if str(path) != self.params.checkpoint:
                raise RuntimeError(f"unexpected torch.load({path!r}) while building STARK")
            return {"net": state}

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with mock.patch.object(torch, "load", stark_load):
            self.tracker = tracker_module.STARK_ST(self.params, DATASET_NAME)  # loads strict=True, moves to CUDA
        if list(self.tracker.update_intervals) != UPDATE_INTERVALS:
            raise ValueError(f"STARK chose update intervals {self.tracker.update_intervals}, D28 expects "
                             f"{UPDATE_INTERVALS}")
        self.owner = None  # the StarkTracker whose clip is running


class StarkTracker:
    """One clip's view of a shared ``StarkBackend``; ``init`` claims the backend for this clip."""

    def __init__(self, backend: StarkBackend):
        self._backend = backend

    def init(self, frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> None:
        self._backend.owner = self
        box = [float(v) for v in xyxy_to_xywh0(box_xyxy)]  # a list: STARK indexes and copies it
        self._backend.tracker.initialize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), {"init_bbox": box})

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        if self._backend.owner is not self:
            raise RuntimeError("another clip re-initialized the shared STARK backend; run clips one at a time")
        out = self._backend.tracker.track(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return xywh0_to_xyxy(out["target_bbox"]), float(out["conf_score"])
