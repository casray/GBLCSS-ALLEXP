from pathlib import Path

import pytest
import torch

from scripts.run_mlp43_representation_source import (
    DATASETS,
    _run_args,
    build_source_tensors,
    formal_label_free_cells,
    sha256_file,
)


def test_two_step_adjacency_propagation_is_exact():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    adjacency = torch.tensor([[0.5, 0.5], [0.25, 0.75]]).to_sparse()
    sources = build_source_tensors(features, adjacency, propagation_steps=2)
    expected = torch.tensor([[0.375, 0.625], [0.3125, 0.6875]])
    assert torch.equal(sources["raw_x"], features)
    assert torch.allclose(sources["adj2_x"], expected)


def test_formal_label_free_grid_has_two_sources_five_datasets_five_seeds():
    cells = formal_label_free_cells()
    assert len(cells) == 50
    assert set(cells) == {
        (dataset, source, seed)
        for dataset in DATASETS
        for source in ("raw_x", "adj2_x")
        for seed in range(5)
    }


def test_sha256_file_changes_with_artifact(tmp_path: Path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(torch.ones(2, 2), first)
    torch.save(torch.zeros(2, 2), second)
    assert sha256_file(first) != sha256_file(second)


def test_source_builder_rejects_invalid_steps():
    features = torch.ones(2, 2)
    adjacency = torch.eye(2).to_sparse()
    with pytest.raises(ValueError, match="positive"):
        build_source_tensors(features, adjacency, propagation_steps=0)


def test_representation_runner_can_request_formal_loo_targets():
    args = _run_args("cpu", "H.pt", leave_one_out=True)
    assert args.config_overrides["gb.tie_mode"] == "distribution_only"
    assert args.config_overrides["gb.leave_one_out"] is True
