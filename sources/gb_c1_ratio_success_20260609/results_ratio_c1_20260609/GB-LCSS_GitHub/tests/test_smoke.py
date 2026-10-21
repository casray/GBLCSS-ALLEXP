from __future__ import annotations

import subprocess
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.experiment import load_config, make_smoke_graph, run_gblcss
from gblcss.baselines import train_noisy_label_baseline
from gblcss.data import (
    GraphData,
    inject_train_label_noise,
    percent_split_indices,
    set_seed,
    unique_undirected_edge_statistics,
)
from gblcss.granular_ball import build_granular_balls
from gblcss.ablations import build_self_vote_ablation_targets
from gblcss.targets import build_leave_one_out_targets
from gblcss.representation import load_frozen_representation


def test_tie_safe_leave_one_out_target() -> None:
    labels = torch.tensor([0, 1, 1, 2, 2])
    assignment = np.zeros(5, dtype=np.int64)
    train_mask = torch.ones(5, dtype=torch.bool)
    result = build_leave_one_out_targets(labels, assignment, train_mask, 4, beta=0.5)
    # Removing node 0 leaves a 2:2 tie. The tied classes receive equal mass.
    assert torch.isclose(result.soft_target[0, 1], result.soft_target[0, 2])
    assert result.soft_target[0, 3] == 0
    # Removing node 1 leaves class 2 as the unique majority.
    assert result.soft_target[1, 2] > result.soft_target[1, 0]
    assert result.diagnostics["tie_mode"] == "distribution_only"


def test_smoke_forward_is_x_only() -> None:
    graph, representation = make_smoke_graph(0)
    config = load_config(ROOT / "configs" / "main" / "gblcss.yaml")
    config["training"]["epochs"] = 3
    config["training"]["device"] = "cpu"
    output = run_gblcss(graph, representation, config, seed=0)
    assert output.result.logits.shape == (graph.num_nodes, graph.num_classes)
    assert output.diagnostics["leave_one_out"] == 1
    assert output.diagnostics["tie_mode"] == "distribution_only"


def test_method_result_is_independent_of_execution_order() -> None:
    graph, representation = make_smoke_graph(0)
    config = load_config(ROOT / "configs" / "main" / "gblcss.yaml")
    config["training"]["epochs"] = 3

    set_seed(0)
    standalone = train_noisy_label_baseline(graph, "mlp_ce", noise_rate=0.4, epochs=3)

    run_gblcss(graph, representation, config, seed=0)
    set_seed(0)
    after_gblcss = train_noisy_label_baseline(graph, "mlp_ce", noise_rate=0.4, epochs=3)

    assert torch.equal(standalone.logits, after_gblcss.logits)
    assert standalone.test_accuracy == after_gblcss.test_accuracy


def _index_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def test_cora_seed0_split_signature() -> None:
    train, validation, test = percent_split_indices(2708, seed=0)
    assert train[:10].tolist() == [1808, 1918, 891, 2052, 565, 1633, 615, 2055, 124, 1414]
    assert _index_fingerprint(train, validation, test) == (
        "fe241a5bd04c1aba7c5312aa410fc4c653a2c14dd68e58e9015b5271eb4c67f9"
    )


def test_independent_noise_uses_ordered_indices_and_multinomial() -> None:
    labels = torch.arange(140, dtype=torch.long) % 7
    train, validation, test = percent_split_indices(labels.numel(), seed=3)
    mask = lambda index: torch.zeros(labels.numel(), dtype=torch.bool).scatter_(
        0, torch.as_tensor(index, dtype=torch.long), True
    )
    graph = GraphData(
        "fixture",
        torch.zeros(labels.numel(), 2),
        torch.empty((2, 0), dtype=torch.long),
        labels,
        mask(train),
        mask(validation),
        mask(test),
        train_index=torch.as_tensor(train, dtype=torch.long),
        val_index=torch.as_tensor(validation, dtype=torch.long),
        test_index=torch.as_tensor(test, dtype=torch.long),
    )
    inject_train_label_noise(graph, "uniform", 0.4, seed=3)
    assert graph.noisy_y is not None
    assert _index_fingerprint(graph.noisy_y.numpy()) == (
        "350ab2b99cb1174f4acaa102ef25d00fe741649c4067e035625b2695875dc669"
    )


def test_unique_undirected_edge_statistics_remove_loops_and_directions() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 2], [1, 0, 2, 1, 2, 0]],
        dtype=torch.long,
    )
    labels = torch.tensor([0, 0, 1])
    edge_count, homophily = unique_undirected_edge_statistics(edge_index, labels)
    assert edge_count == 3
    assert abs(homophily - (1.0 / 3.0)) < 1e-7


def test_self_vote_ablation_is_distinct_from_formal_target() -> None:
    labels = torch.tensor([0, 0, 1])
    assignment = np.zeros(3, dtype=np.int64)
    train_mask = torch.ones(3, dtype=torch.bool)
    formal = build_leave_one_out_targets(labels, assignment, train_mask, 2)
    ablation = build_self_vote_ablation_targets(labels, assignment, train_mask, 2)
    assert not torch.allclose(formal.soft_target, ablation.soft_target)
    assert formal.diagnostics["leave_one_out"] == 1
    assert ablation.diagnostics["leave_one_out"] == 0




def test_frozen_representations_match_reported_node_counts() -> None:
    expected = {
        "amazoncomputers": 13752,
        "amazonphoto": 7650,
        "attributedwiki": 2405,
        "citeseer": 3327,
        "citeseerfull": 4230,
        "coauthorcs": 18333,
        "coauthorphysics": 34493,
        "cora": 2708,
        "corafull": 19793,
        "coraml": 2995,
        "pubmed": 19717,
    }
    for key, node_count in expected.items():
        config = yaml.safe_load((ROOT / "configs" / "datasets" / f"{key}.yaml").read_text())
        representation = load_frozen_representation(ROOT / config["representation"], node_count)
        assert representation.shape[0] == node_count
        assert torch.isfinite(representation).all()








