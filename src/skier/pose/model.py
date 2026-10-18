"""HRNet-W32 wrapper (plan Step 3.1; decision D3: MMPose only builds the model).

``PoseNet.forward(x) = head(backbone(x))`` returns ``N x K x 64 x 48`` heatmaps and never
touches MMPose's data preprocessor (``skier.pose.preprocess`` normalizes, once).
torch and mmpose are imported inside the functions, so this module imports in the
light environment; ``filter_checkpoint`` and ``check_load_result`` are plain Python.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FINAL_LAYER_PREFIX = "head.final_layer."
FINAL_LAYER_KEYS = {"head.final_layer.weight", "head.final_layer.bias"}
COCO_CONFIG = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"


def filter_checkpoint(state_dict: Mapping[str, Any], drop_prefix: str = FINAL_LAYER_PREFIX) -> tuple[dict, list[str]]:
    """Split a state dict into (kept, sorted dropped keys) by key prefix."""
    kept = {k: v for k, v in state_dict.items() if not k.startswith(drop_prefix)}
    return kept, sorted(k for k in state_dict if k.startswith(drop_prefix))


def check_load_result(missing: list[str], unexpected: list[str], replaced_head: bool) -> None:
    """Every tensor except (when the head was replaced) the final layer must have loaded."""
    if unexpected:
        raise RuntimeError(f"checkpoint has {len(unexpected)} tensors the model doesn't: {sorted(unexpected)[:5]}")
    expected = FINAL_LAYER_KEYS if replaced_head else set()
    if set(missing) != expected:
        raise RuntimeError(f"missing tensors {sorted(missing)[:5]} (expected exactly {sorted(expected)})")


@lru_cache(maxsize=None)
def _posenet_class():
    from torch import nn

    class PoseNet(nn.Module):
        """Backbone + heatmap head, called directly on normalized ``N x 3 x 256 x 192`` input."""

        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    return PoseNet


def build_hrnet(num_keypoints: int, config_path: str | Path, checkpoint: str | Path | None = None,
                device: str = "cpu"):
    """Build HRNet-W32 with MMPose, optionally with a new ``num_keypoints`` final layer.

    Returns ``(model, info)``; ``info`` records what was loaded and skipped.
    """
    import torch
    from mmpose.apis import init_model
    from torch import nn

    estimator = init_model(str(config_path), None, device="cpu")
    model = _posenet_class()(estimator.backbone, estimator.head)
    old = model.head.final_layer
    replaced = num_keypoints != old.out_channels
    if replaced:
        layer = nn.Conv2d(old.in_channels, num_keypoints, kernel_size=1)
        nn.init.normal_(layer.weight, std=0.001)  # MMPose's head init
        nn.init.zeros_(layer.bias)
        model.head.final_layer = layer
    info = {"num_keypoints": num_keypoints, "replaced_head": replaced, "checkpoint": None,
            "loaded_tensors": 0, "skipped_tensors": []}
    if checkpoint is not None:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)  # MMPose checkpoints carry metadata
        state = ckpt.get("state_dict", ckpt)
        kept, dropped = filter_checkpoint(state) if replaced else (dict(state), [])
        result = model.load_state_dict(kept, strict=False)
        check_load_result(list(result.missing_keys), list(result.unexpected_keys), replaced)
        info.update(checkpoint=Path(checkpoint).name, loaded_tensors=len(kept), skipped_tensors=dropped)
    return model.to(device), info


def torch_heatmap_fn(model, device: str):
    """Wrap the model as ``heatmap_fn(np [N,3,256,192]) -> np [N,K,64,48]`` (eval mode, no grad)."""
    import torch

    model.eval()

    def heatmap_fn(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(device)
            return model(x).float().cpu().numpy()

    return heatmap_fn


def packaged_coco_config() -> Path | None:
    """The HRNet COCO config shipped inside the mmpose package, if present."""
    try:
        import mmpose
    except ImportError:
        return None
    path = (Path(mmpose.__file__).parent / ".mim" / "configs" / "body_2d_keypoint" / "topdown_heatmap"
            / "coco" / f"{COCO_CONFIG}.py")
    return path if path.is_file() else None
