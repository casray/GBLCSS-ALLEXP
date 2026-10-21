from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gb_stage_b.config import StageBConfig
from gb_stage_b.data import dataset_summary, load_graph, make_synthetic_graph, synthetic_graph


def _config(**overrides):
    params = {
        "dataset": "Synthetic",
        "epochs": 1,
        "warmup_epochs": 0,
        "seed": 7,
        "num_nodes": 300,
        "num_classes": 3,
        "feature_dim": 6,
        "knn_k": 5,
        "label_noise": 0.3,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
    }
    params.update(overrides)
    return StageBConfig(**params)


def test_synthetic_graph_shapes_dtypes_and_masks():
    cfg = _config()

    data = make_synthetic_graph(cfg)

    assert isinstance(data, Data)
    assert data.x.shape == (cfg.num_nodes, cfg.feature_dim)
    assert data.x.dtype is torch.float32
    assert data.edge_index.shape == (2, cfg.num_nodes * cfg.knn_k)
    assert data.edge_index.dtype is torch.long
    assert data.y.shape == (cfg.num_nodes,)
    assert data.y_true.shape == (cfg.num_nodes,)
    assert data.y_noisy.shape == (cfg.num_nodes,)
    assert data.y.dtype is torch.long
    assert torch.equal(data.y, data.y_noisy)
    for mask in (data.train_mask, data.val_mask, data.test_mask, data.noise_mask):
        assert mask.shape == (cfg.num_nodes,)
        assert mask.dtype is torch.bool

    covered = data.train_mask | data.val_mask | data.test_mask
    assert covered.all()
    assert not (data.train_mask & data.val_mask).any()
    assert not (data.train_mask & data.test_mask).any()
    assert not (data.val_mask & data.test_mask).any()


def test_knn_edges_are_directed_without_self_loops_and_fixed_out_degree():
    cfg = _config(num_nodes=120, knn_k=4, seed=3)

    data = make_synthetic_graph(cfg)

    assert not (data.edge_index[0] == data.edge_index[1]).any()
    out_degree = torch.bincount(data.edge_index[0], minlength=cfg.num_nodes)
    assert torch.equal(out_degree, torch.full((cfg.num_nodes,), cfg.knn_k))


def test_synthetic_graph_is_reproducible_with_same_seed():
    cfg = _config(seed=11)

    first = synthetic_graph(seed=cfg.seed, config=cfg)
    second = synthetic_graph(seed=cfg.seed, config=cfg)

    for key in ("x", "edge_index", "y", "y_true", "y_noisy", "train_mask", "val_mask", "test_mask", "noise_mask"):
        assert torch.equal(getattr(first, key), getattr(second, key))


def test_label_noise_only_affects_training_labels():
    cfg = _config(label_noise=0.35, seed=13)

    data = make_synthetic_graph(cfg)

    assert (data.noise_mask & ~data.train_mask).sum().item() == 0
    assert torch.equal(data.y_noisy[data.val_mask], data.y_true[data.val_mask])
    assert torch.equal(data.y_noisy[data.test_mask], data.y_true[data.test_mask])
    assert torch.equal(data.noise_mask, data.y_noisy != data.y_true)


def test_dataset_summary_reports_train_noise_rate():
    cfg = _config(num_nodes=600, label_noise=0.25, seed=17)
    data = make_synthetic_graph(cfg)

    summary = dataset_summary(data)

    assert summary["num_nodes"] == cfg.num_nodes
    assert summary["num_features"] == cfg.feature_dim
    assert summary["num_classes"] == cfg.num_classes
    assert summary["train_size"] == int(cfg.num_nodes * cfg.train_ratio)
    assert summary["val_size"] == int(cfg.num_nodes * cfg.val_ratio)
    assert summary["test_size"] == cfg.num_nodes - summary["train_size"] - summary["val_size"]
    assert summary["num_edges"] == data.edge_index.shape[1]
    assert abs(summary["train_noise_rate"] - cfg.label_noise) <= 0.05


def test_load_graph_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Synthetic, Cora, CiteSeer, or PubMed"):
        load_graph("Unknown")


def test_load_graph_planetoid_branch_uses_expected_root(monkeypatch, tmp_path):
    fake = Data(x=torch.ones(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0, 1]))
    calls = {}

    class FakePlanetoid:
        def __init__(self, root, name):
            calls["root"] = root
            calls["name"] = name

        def __getitem__(self, index):
            assert index == 0
            return fake

    monkeypatch.setattr("gb_stage_b.data.Planetoid", FakePlanetoid)

    out = load_graph("Cora", seed=99, root=tmp_path)

    assert out is fake
    assert calls == {"root": str(Path(tmp_path) / "Cora"), "name": "Cora"}


def test_load_graph_planetoid_ratio_split_overrides_standard_masks(monkeypatch, tmp_path):
    labels = torch.arange(100) % 5
    fake = Data(
        x=torch.ones(100, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=labels,
        train_mask=torch.zeros(100, dtype=torch.bool),
        val_mask=torch.zeros(100, dtype=torch.bool),
        test_mask=torch.zeros(100, dtype=torch.bool),
    )
    fake.train_mask[:5] = True
    fake.val_mask[5:10] = True
    fake.test_mask[10:] = True

    class FakePlanetoid:
        def __init__(self, root, name):
            pass

        def __getitem__(self, index):
            assert index == 0
            return fake

    monkeypatch.setattr("gb_stage_b.data.Planetoid", FakePlanetoid)
    cfg = StageBConfig(
        dataset="Cora",
        split_mode="ratio",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        noise_rate=0.4,
        seed=3,
    )

    out = load_graph("Cora", seed=3, root=tmp_path, noise_rate=0.4, config=cfg)

    assert int(out.train_mask.sum().item()) == 60
    assert int(out.val_mask.sum().item()) == 20
    assert int(out.test_mask.sum().item()) == 20
    assert (out.noise_mask & ~out.train_mask).sum().item() == 0
    assert int(out.noise_mask.sum().item()) == 24
    assert torch.equal(out.y_noisy[out.val_mask], out.y_true[out.val_mask])
    assert torch.equal(out.y_noisy[out.test_mask], out.y_true[out.test_mask])


def test_unstratifiable_split_raises_clear_value_error():
    cfg = _config(num_nodes=9, num_classes=3, train_ratio=0.34, val_ratio=0.33, test_ratio=0.33)

    with pytest.raises(ValueError, match="stratified split"):
        make_synthetic_graph(cfg)
