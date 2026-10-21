from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.ablations import build_self_vote_ablation_targets
from gblcss.data import GraphData
from gblcss.granular_ball import build_granular_balls
from gblcss.targets import build_leave_one_out_targets


METHODS = (
    "gblcss_self_vote", "uniform_smoothing", "prior_smoothing", "knn",
    "kmeans", "raw_features", "propagated_features",
)


def test_public_api_exists():
    assert importlib.util.find_spec("gblcss.ablation_targets") is not None, "missing ablation API module"


@pytest.fixture
def build():
    spec = importlib.util.find_spec("gblcss.ablation_targets")
    assert spec is not None, "missing current-paper ablation target construction"
    module = importlib.import_module("gblcss.ablation_targets")
    assert hasattr(module, "build_ablation_targets"), "missing public ablation API"
    return module.build_ablation_targets


@pytest.fixture
def config():
    with (ROOT / "configs/main/gblcss.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _graph(seed=7):
    rng = np.random.RandomState(seed)
    x = torch.from_numpy(rng.normal(size=(30, 6)).astype(np.float32))
    h = torch.from_numpy(rng.normal(size=(30, 5)).astype(np.float32))
    y = torch.arange(30) % 4
    order = torch.tensor([19, 1, 7, 3, 12, 5, 22, 9, 2, 16, 21, 26, 27, 0, 4])
    mask = torch.zeros(30, dtype=torch.bool)
    mask[order] = True
    # Directed and irregular so A^2 differs from normalized/looped propagation.
    edges = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 7, 8, 12, 16, 19, 22, 26],
                          [1, 2, 3, 4, 0, 5, 7, 8, 12, 16, 19, 22, 26, 27]])
    graph = GraphData("toy", x, edges, y, mask, ~mask, ~mask,
                      noisy_y=(y + (torch.arange(30) % 3 == 0)) % 4,
                      train_index=order)
    return graph, h


def _small_config(config):
    config = copy.deepcopy(config)
    config["method"].update(max_balls=5, min_ball_size=2, split_improvement=0.0)
    return config


def _assignment(h, config, seed):
    method = config["method"]
    return build_granular_balls(
        F.normalize(h.float(), dim=1), max_balls=method["max_balls"],
        min_ball_size=method["min_ball_size"],
        split_improvement=method["split_improvement"], random_state=seed,
    ).assignment


def _hash(assignment):
    return hashlib.sha256(np.ascontiguousarray(assignment, dtype="<i8").tobytes()).hexdigest()


@pytest.mark.parametrize("method", METHODS)
def test_all_node_contract_no_mutation_or_global_rng_consumption(build, config, method):
    data, h = _graph()
    before_data, before_config, before_h = copy.deepcopy(data), copy.deepcopy(config), h.clone()
    np.random.seed(91)
    torch.manual_seed(91)
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    targets, diagnostics = build(data, h.requires_grad_(), method, config, seed=4)
    assert targets.shape == (data.num_nodes, data.num_classes)
    assert targets.dtype == torch.float32
    assert not targets.requires_grad
    assert torch.isfinite(targets).all() and (targets >= 0).all()
    torch.testing.assert_close(targets.sum(1), torch.ones(data.num_nodes))
    assert torch.equal(targets[~data.train_mask], F.one_hot(data.noisy_y, 4).float()[~data.train_mask])
    assert diagnostics["method"] == method
    json.dumps(diagnostics, allow_nan=False)
    assert config == before_config and torch.equal(h, before_h)
    for key in ("x", "y", "noisy_y", "edge_index", "train_mask", "train_index"):
        assert torch.equal(getattr(data, key), getattr(before_data, key))
    after_numpy = np.random.get_state()
    assert np.array_equal(after_numpy[1], numpy_state[1]) and after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("method,reference", [("uniform_smoothing", "uniform"), ("prior_smoothing", "prior")])
@pytest.mark.parametrize("epsilon", [None, 0.0, 0.35, 1.0])
def test_smoothing_fixed_epsilon_and_train_only_prior(build, config, method, reference, epsilon):
    data, _ = _graph()
    if epsilon is not None:
        config["method"]["smoothing_epsilon"] = epsilon
    actual_epsilon = 0.1 if epsilon is None else epsilon
    targets, diag = build(data, None, method, config, seed=0)
    expected = F.one_hot(data.noisy_y, 4).float()
    prior = (torch.full((4,), 0.25) if reference == "uniform" else
             torch.bincount(data.noisy_y[data.train_index], minlength=4).float() / data.train_index.numel())
    expected[data.train_index] = (1 - actual_epsilon) * expected[data.train_index] + actual_epsilon * prior
    torch.testing.assert_close(targets, expected, rtol=0, atol=0)
    assert diag["smoothing_epsilon"] == actual_epsilon
    assert diag["smoothing_reference"] == prior.tolist()


def test_self_vote_uses_inclusive_support_and_tie_distribution(build, config):
    data, h = _graph()
    config["method"]["max_balls"] = 1
    data.train_mask[:] = False
    data.train_index = torch.tensor([0, 1, 2, 3])
    data.train_mask[data.train_index] = True
    data.noisy_y[:4] = torch.tensor([0, 0, 1, 1])
    result, diag = build(data, h, "gblcss_self_vote", config, seed=3)
    expected = F.one_hot(data.noisy_y, 4).float()
    alpha = torch.tensor((0.5 - 0.25) / 0.75, dtype=torch.float32)
    expected[:4] = (1.0 - alpha) * expected[:4] + alpha * torch.tensor([0.5, 0.5, 0.0, 0.0])
    assert torch.equal(result, expected)
    assert diag["leave_one_out"] == 0 and diag["tie_mode"] == "distribution_only"
    assert diag["tied_train_nodes"] == 4
    data.train_index = torch.tensor([0, 1])
    data.train_mask[2:4] = False
    data.noisy_y[:2] = 1
    _, diag = build(data, h, "gblcss_self_vote", config, seed=3)
    assert diag["eligible_train_nodes"] == 2


def test_self_vote_preserves_original_float32_alpha_subtraction(build, config):
    data, h = _graph()
    config["method"]["max_balls"] = 1
    data.train_index = torch.arange(6)
    data.train_mask[:] = False
    data.train_mask[:6] = True
    data.noisy_y[:6] = torch.tensor([0, 0, 0, 1, 2, 3])
    counts = torch.tensor([3, 1, 1, 1]).float()
    local = 0.5 * (counts / 6) + 0.5 * torch.tensor([1, 0, 0, 0]).float()
    alpha = torch.full((6,), (0.5 - 0.25) / 0.75)
    expected = F.one_hot(data.noisy_y, 4).float()
    expected[:6] = (1 - alpha[:, None]) * expected[:6] + alpha[:, None] * local
    actual, _ = build(data, h, "gblcss_self_vote", config, seed=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_knn_excludes_self_matches_rounded_nonempty_group_support(build, config):
    data, h = _graph()
    config = _small_config(config)
    assignment = _assignment(h, config, 7)
    counts = np.bincount(assignment[data.train_index], minlength=int(assignment.max()) + 1)
    k = min(round(float(counts[counts > 0].mean())), len(data.train_index) - 1)
    targets, diag = build(data, h, "knn", config, seed=7)
    assert diag["matched_k"] == k
    assert diag["knn_exclude_self"] == 1
    assert diag["reference_assignment_sha256"] == _hash(assignment)
    normalized = F.normalize(h, dim=1)[data.train_index]
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.topk(k, dim=1).indices
    expected = F.one_hot(data.noisy_y, 4).float()
    for row, node in enumerate(data.train_index):
        histogram = np.bincount(data.noisy_y[data.train_index[neighbors[row]]].numpy(), minlength=4)
        p = histogram / histogram.sum()
        alpha = max(0.0, (p.max() - 0.25) / 0.75)
        local = p if (histogram == histogram.max()).sum() > 1 else 0.5 * p + 0.5 * np.eye(4)[p.argmax()]
        # Original kNN computes local probabilities in float64, then casts once.
        a = torch.tensor(alpha, dtype=torch.float32)
        expected[node] = (1 - a) * expected[node] + a * torch.tensor(local, dtype=torch.float32)
    torch.testing.assert_close(targets, expected, rtol=0, atol=0)


def test_knn_preserves_ordered_train_indices_on_distance_ties(build, config):
    data, h = _graph()
    config = _small_config(config)
    h = F.one_hot(torch.arange(30) % 3, 3).float()
    targets, _ = build(data, h, "knn", config, seed=2)
    ordered = data.train_index.clone()
    data.train_mask = ordered
    data.train_index = None
    from_index_mask, _ = build(data, h, "knn", config, seed=2)
    assert torch.equal(targets, from_index_mask)


def test_kmeans_same_reference_group_count_original_seed_and_loo(build, config):
    data, h = _graph()
    config = _small_config(config)
    reference = _assignment(h, config, 17)
    groups = len(np.unique(reference))
    assignment = KMeans(n_clusters=groups, n_init=10, random_state=np.random.RandomState(17)).fit_predict(
        F.normalize(h, dim=1).numpy())
    expected = build_leave_one_out_targets(data.noisy_y, assignment, data.train_index, 4)
    targets, diag = build(data, h, "kmeans", config, seed=17)
    assert torch.equal(targets, expected.soft_target)
    assert diag["kmeans_n_clusters"] == diag["matched_num_balls"] == groups
    assert diag["assignment_sha256"] == _hash(assignment)
    assert diag["leave_one_out"] == 1


@pytest.mark.parametrize("method", ["raw_features", "propagated_features"])
@pytest.mark.parametrize("sparse_x", [False, True])
def test_representation_uses_raw_x_or_raw_adjacency_twice_then_normalizes(build, config, method, sparse_x):
    data, _ = _graph()
    config = _small_config(config)
    source = data.x.clone()
    if method == "propagated_features":
        a = torch.zeros((30, 30))
        a.index_put_(tuple(data.edge_index), torch.ones(data.edge_index.shape[1]), accumulate=True)
        source = a @ (a @ source)
    assignment = _assignment(source, config, 1)
    expected = build_leave_one_out_targets(data.noisy_y, assignment, data.train_index, 4)
    if sparse_x:
        data.x = data.x.to_sparse()
    targets, diag = build(data, None, method, config, seed=1)
    assert torch.equal(targets, expected.soft_target)
    assert diag["assignment_sha256"] == _hash(assignment)
    assert diag["propagation_steps"] == (2 if method == "propagated_features" else 0)
    assert diag["leave_one_out"] == 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_train", [0, 1])
def test_empty_and_single_train_node(build, config, method, n_train):
    data, h = _graph()
    data.train_index = torch.arange(n_train)
    data.train_mask[:] = False
    data.train_mask[:n_train] = True
    targets, diag = build(data, h, method, config, seed=0)
    assert torch.isfinite(targets).all()
    if n_train == 0 or method not in {"uniform_smoothing", "prior_smoothing"}:
        assert torch.equal(targets, F.one_hot(data.noisy_y, 4).float())
    if method == "knn":
        assert diag["matched_k"] == 0


@pytest.mark.parametrize("method", METHODS)
def test_heldout_labels_do_not_affect_training_targets(build, config, method):
    data, h = _graph()
    first, _ = build(data, h, method, config, seed=1)
    data.noisy_y[~data.train_mask] = (data.noisy_y[~data.train_mask] + 2) % 4
    data.y[data.train_mask] = (data.y[data.train_mask] + 1) % 4
    second, _ = build(data, h, method, config, seed=1)
    assert torch.equal(first[data.train_mask], second[data.train_mask])


def test_reject_unknown_method_missing_or_invalid_representation(build, config):
    data, h = _graph()
    with pytest.raises(ValueError, match="method"):
        build(data, h, "unknown", config, seed=0)
    for invalid in (None, h[:2], h * float("nan")):
        with pytest.raises(ValueError, match="representation"):
            build(data, invalid, "knn", config, seed=0)


@pytest.mark.parametrize("key,value", [("tie_mode", "min_index"), ("leave_one_out", False), ("beta", 1.2)])
def test_reject_nonformal_target_settings(build, config, key, value):
    data, h = _graph()
    config["method"][key] = value
    with pytest.raises(ValueError, match=key):
        build(data, h, "kmeans", config, seed=0)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan")])
def test_reject_invalid_smoothing_epsilon(build, config, epsilon):
    data, _ = _graph()
    config["method"]["smoothing_epsilon"] = epsilon
    with pytest.raises(ValueError, match="epsilon"):
        build(data, None, "uniform_smoothing", config, seed=0)
