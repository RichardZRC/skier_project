"""Step 5 tests: loss, training-loop helpers (light) and the training loop itself (``torch``).

The ``torch`` tests are skipped on the Mac (D13) and run on AWS as the first action,
before any training. They use a tiny stand-in network with the same backbone/head split.
"""

import csv
import math

import numpy as np
import pytest
import synth

from skier.paths import configs_dir
from skier.pose.augment import AugConfig
from skier.pose.loss import masked_mse_np
from skier.pose.train import (
    LOG_COLUMNS,
    BestTracker,
    TrainConfig,
    append_log,
    lr_for,
    run_name,
    schema_fingerprint,
    select_overfit_records,
)

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


# --- loss (numpy reference) -------------------------------------------------------------

def test_loss_hand_computed():
    pred = np.zeros((2, 3, 4, 4))
    c = np.array([[1, 2, 3], [4, 5, 6]], float)
    pred += c[:, :, None, None]  # constant heatmaps: per-joint mean squared error = c^2
    w = np.array([[1, 0, 1], [1, 1, 0]], float)
    assert masked_mse_np(pred, np.zeros_like(pred), w) == pytest.approx((1 + 9 + 16 + 25) / 4)


def test_loss_masked_joints_contribute_nothing():
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=(2, 3, 4, 4)), rng.normal(size=(2, 3, 4, 4))
    w = np.array([[1, 0, 1], [0, 1, 0]], float)
    base = masked_mse_np(pred, target, w)
    pred2 = pred.copy()
    pred2[0, 1] += 100  # a masked joint changes wildly
    assert masked_mse_np(pred2, target, w) == pytest.approx(base)


def test_loss_all_masked_is_zero():
    pred = np.ones((2, 3, 4, 4))
    out = masked_mse_np(pred, np.zeros_like(pred), np.zeros((2, 3)))
    assert out == 0.0 and not math.isnan(out)


# --- training helpers (no torch) -----------------------------------------------------------

def test_e1_yaml_is_the_default_config():
    cfg = TrainConfig.from_yaml(configs_dir() / "pose" / "e1.yaml")
    assert cfg == TrainConfig()
    assert cfg.total_epochs == 153 and cfg.aug == AugConfig()  # D21
    assert [3 + e for e in (90, 127)] == [e for e in range(cfg.total_epochs)
                                          if e > 3 and lr_for(e, cfg) != lr_for(e - 1, cfg)]
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_unknown_keys_and_test_partition():
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({"batchsize": 8})
    with pytest.raises(ValueError, match="test"):
        TrainConfig.from_dict({"val_partition": "test"})


def test_run_name():
    assert run_name("E2") == "E2"
    assert run_name("E1", seed=231) == "E1-seed231"
    assert run_name("E1", overfit=32) == "E1-overfit32"
    assert run_name("E4", overfit=8, seed=232) == "E4-seed232-overfit8"


def test_lr_schedule_boundaries():
    cfg = TrainConfig(finetune_epochs=50)  # 3 warm + 50 fine-tune; milestones at fine-tune epochs 30 and 42
    assert lr_for(0, cfg) == (0.0, 1e-3) and lr_for(2, cfg) == (0.0, 1e-3)
    assert lr_for(3, cfg) == (1e-4, 5e-4)
    assert lr_for(3 + 29, cfg) == (1e-4, 5e-4)
    assert lr_for(3 + 30, cfg) == pytest.approx((1e-5, 5e-5))
    assert lr_for(3 + 41, cfg) == pytest.approx((1e-5, 5e-5))
    assert lr_for(3 + 42, cfg) == pytest.approx((1e-6, 5e-6))
    assert lr_for(52, cfg) == pytest.approx((1e-6, 5e-6))


def test_best_tracker():
    b = BestTracker()
    assert [b.update(e, v) for e, v in enumerate([None, 40.0, 38.0, 40.0, 41.5, None])] == \
        [False, True, False, False, True, False]
    assert (b.value, b.epoch) == (41.5, 4)


def test_log_csv(tmp_path):
    p = tmp_path / "log.csv"
    append_log(p, {"epoch": 0, "phase": "warm", "train_loss": 0.01})
    append_log(p, {"epoch": 1, "phase": "finetune", "val_ski_pct": 50.0, "is_best": 1})
    rows = list(csv.DictReader(p.open()))
    assert list(rows[0]) == LOG_COLUMNS and rows[1]["val_ski_pct"] == "50.0" and rows[0]["val_ski_pct"] == ""
    with pytest.raises(ValueError, match="unknown"):
        append_log(p, {"epochz": 2})


def test_schema_fingerprint_and_overfit_selection(schema):
    assert schema_fingerprint(schema) == schema_fingerprint(schema) and len(schema_fingerprint(schema)) == 16

    class R:
        def __init__(self, i):
            self.sample_id = f"s{i:02d}"

    recs = [R(i) for i in range(20)]
    a = select_overfit_records(recs, 5, seed=230)
    assert [r.sample_id for r in a] == [r.sample_id for r in select_overfit_records(recs, 5, seed=230)]
    assert len(a) == 5 and len(select_overfit_records(recs, 50, 1)) == 20


def test_prediction_overlay(schema):
    from skier.viz import COLOR, prediction_overlay

    img = np.zeros((200, 300, 3), np.uint8)
    pred = np.full((24, 2), np.nan)
    pred[2] = (100, 100)
    gt = np.full((24, 2), 50.0)
    out = prediction_overlay(img, pred, gt, np.ones(24, bool), schema)
    assert tuple(out[100, 100]) == COLOR["right"] and tuple(out[50, 50]) == (255, 255, 255)


# --- torch (AWS) ----------------------------------------------------------------------------

def _tiny_net(k=24, bias_map=False):
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, k, 1)
            # A learnable per-pixel output bias lets the overfit test fit fixed heatmaps without
            # relying on a tiny conv net to localize points in a blank synthetic image.
            self.bias_map = nn.Parameter(torch.zeros(1, k, 64, 48)) if bias_map else None

        def forward(self, x):
            y = self.conv(x)
            return y + self.bias_map if self.bias_map is not None else y

    class TinyNet(nn.Module):
        """Same backbone/head split as PoseNet; 256x192 -> 64x48 via two stride-2 convs."""

        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                                          nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                          nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
            self.head = Head()

        def forward(self, x):
            return self.head(self.backbone(x))

    return TinyNet()


@pytest.fixture
def train_records(tmp_path):
    from skier.data.ski2dpose import load_records

    labels = synth.make_labels(videos=1, segments=1, images=4, seed=3)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(160, 120), template=TEMPLATE, value=100)
    return [r for r in load_records(synth.write_labels(labels, tmp_path / "l.json"), images, path_template=TEMPLATE)
            if not r.excluded]


@pytest.mark.torch
def test_torch_loss_matches_numpy():
    torch = pytest.importorskip("torch")
    from skier.pose.loss import masked_mse

    rng = np.random.default_rng(1)
    p, t = rng.normal(size=(3, 5, 8, 6)), rng.normal(size=(3, 5, 8, 6))
    w = (rng.random((3, 5)) > 0.4).astype(float)
    pt = torch.tensor(p, requires_grad=True)
    loss = masked_mse(pt, torch.tensor(t), torch.tensor(w))
    assert float(loss) == pytest.approx(masked_mse_np(p, t, w), rel=1e-9)
    loss.backward()
    masked = w == 0
    assert torch.all(pt.grad[torch.tensor(masked)] == 0)


@pytest.mark.torch
def test_collate_drops_empty_samples():
    pytest.importorskip("torch")
    from skier.pose.dataset import collate

    s = {"x": np.zeros((3, 256, 192), np.float32), "target": np.zeros((24, 64, 48), np.float32),
         "M": np.eye(2, 3), "sample_id": "a", "index": 0}
    full, empty = {**s, "weight": np.ones(24, np.float32)}, {**s, "sample_id": "b", "weight": np.zeros(24, np.float32)}
    batch = collate([full, empty])
    assert batch["x"].shape == (1, 3, 256, 192) and batch["sample_ids"] == ["a"] and batch["n_dropped"] == 1
    assert collate([empty]) is None


@pytest.mark.torch
def test_two_step_smoke_and_checkpoint_reload(tmp_path, train_records, schema):
    torch = pytest.importorskip("torch")
    from skier.pose.train import load_checkpoint, train

    cfg = TrainConfig(batch_size=2, warm_epochs=1, finetune_epochs=1, num_workers=0, val_batch_size=2)
    model = _tiny_net()
    summary = train(model, train_records, train_records, cfg, tmp_path / "run", schema, "cpu", max_steps=2)
    rows = list(csv.DictReader((tmp_path / "run" / "log.csv").open()))
    assert math.isfinite(float(rows[0]["train_loss"])) and (tmp_path / "run" / "last.pt").exists()
    x = torch.zeros(1, 3, 256, 192)
    model.eval()
    reloaded = _tiny_net()
    ck = load_checkpoint(tmp_path / "run" / "last.pt", reloaded, schema=schema)
    reloaded.eval()
    assert torch.allclose(model(x), reloaded(x)) and ck["seed"] == 230 and ck["num_keypoints"] == 24
    assert summary["epochs_run"] == 1


@pytest.mark.torch
def test_warm_up_freezes_backbone_weights_and_batchnorm(tmp_path, train_records, schema):
    torch = pytest.importorskip("torch")
    from skier.pose.train import train

    model = _tiny_net()
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}  # includes BN running stats
    head_before = model.head.conv.weight.detach().clone()
    cfg = TrainConfig(batch_size=2, warm_epochs=1, finetune_epochs=0, num_workers=0)
    train(model, train_records, [], cfg, tmp_path / "run", schema, "cpu")
    after = model.backbone.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not torch.equal(head_before, model.head.conv.weight)


@pytest.mark.torch
def test_same_seed_same_first_loss(tmp_path, train_records, schema):
    pytest.importorskip("torch")
    from skier.pose.train import train

    cfg = TrainConfig(batch_size=2, warm_epochs=0, finetune_epochs=1, num_workers=0)
    losses = []
    for run in range(2):
        import torch

        torch.manual_seed(0)
        train(_tiny_net(), train_records, [], cfg, tmp_path / f"r{run}", schema, "cpu", max_steps=1)
        losses.append(next(csv.DictReader((tmp_path / f"r{run}" / "log.csv").open()))["train_loss"])
    assert losses[0] == losses[1]


@pytest.mark.torch
def test_pseudo_records_add_steps_and_provenance(tmp_path, train_records, schema):
    """Step 6b (D29): each epoch = every real record + an equal-size subset of pseudo records."""
    import dataclasses

    torch = pytest.importorskip("torch")
    from skier.pose.train import load_checkpoint, train

    pseudo = [dataclasses.replace(r, sample_id=f"P/{i}/x", dataset="skitb_pseudo") for i, r in
              enumerate(train_records * 2)]  # 8 pseudo, 4 real -> 4 drawn per epoch
    cfg = TrainConfig(batch_size=2, warm_epochs=0, finetune_epochs=1, num_workers=0)
    train(_tiny_net(), train_records, [], cfg, tmp_path / "run", schema, "cpu", pseudo_records=pseudo,
          pseudo_meta={"pseudo_labels": "p.json", "pseudo_labels_sha256": "feed"})
    row = next(csv.DictReader((tmp_path / "run" / "log.csv").open()))
    assert int(row["n_steps"]) == math.ceil((4 + 4) / 2)
    ck = load_checkpoint(tmp_path / "run" / "last.pt", _tiny_net(), schema=schema)
    assert (ck["n_train"], ck["n_pseudo"], ck["n_pseudo_per_epoch"], ck["pseudo_labels_sha256"]) == (4, 8, 4, "feed")
    with pytest.raises(ValueError, match="different pseudo-labels"):
        train(_tiny_net(), train_records, [], cfg, tmp_path / "run2", schema, "cpu", resume=tmp_path / "run" / "last.pt")


@pytest.mark.torch
def test_no_pseudo_records_trains_exactly_as_before(tmp_path, train_records, schema):
    torch = pytest.importorskip("torch")
    from skier.pose.train import train

    cfg = TrainConfig(batch_size=2, warm_epochs=0, finetune_epochs=1, num_workers=0)
    states = []
    for run, kw in enumerate(({}, {"pseudo_records": ()})):
        torch.manual_seed(0)
        model = _tiny_net()
        train(model, train_records, [], cfg, tmp_path / f"r{run}", schema, "cpu", max_steps=2, **kw)
        states.append(model.state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


@pytest.mark.torch
def test_tiny_overfit_learns_the_labels(tmp_path, train_records, schema):
    """The loop really learns: one image, no augmentation, decoded points end up on the labels.

    Checks that loss, gradients, targets and the decoder fit together (not model capacity)."""
    torch = pytest.importorskip("torch")
    from skier.eval.pose_metrics import gt_arrays, group_pck
    from skier.pose.infer import predict_records
    from skier.pose.model import torch_heatmap_fn
    from skier.pose.train import train

    torch.manual_seed(0)
    one = train_records[:1]
    model = _tiny_net(bias_map=True)
    cfg = TrainConfig(batch_size=1, warm_epochs=0, finetune_epochs=300, lr_backbone=1e-3, lr_head=2e-2,
                      milestones=(), num_workers=0, aug=AugConfig.none(), val_every=1000)
    train(model, one, [], cfg, tmp_path / "run", schema, "cpu")
    rows = list(csv.DictReader((tmp_path / "run" / "log.csv").open()))
    assert float(rows[-1]["train_loss"]) < 0.1 * float(rows[0]["train_loss"])
    preds = predict_records(one, torch_heatmap_fn(model, "cpu"), batch_size=1)
    gt, rel, diag, _ = gt_arrays(one)
    for group in ("body", "ski"):
        assert group_pck(preds.xy, gt, rel, diag, schema, 0.05)[group]["pct"] == 100.0
