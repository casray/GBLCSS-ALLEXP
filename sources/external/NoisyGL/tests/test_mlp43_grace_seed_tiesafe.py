from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.run_mlp43_grace_seed_tiesafe import (
    DATASETS,
    DOWNSTREAM_SEEDS,
    H_SEEDS,
    _run_args,
    formal_run_cells,
    validate_embedding,
)
from scripts.summarize_mlp43_grace_seed_tiesafe import summarize_variance


def test_formal_grid_contains_exactly_27_cells():
    cells = formal_run_cells()
    assert len(cells) == 27
    assert set(cells) == {
        (dataset, h_seed, downstream_seed)
        for dataset in DATASETS
        for h_seed in H_SEEDS
        for downstream_seed in DOWNSTREAM_SEEDS
    }


def test_grace_seed_runner_can_request_formal_loo_targets():
    args = _run_args("cpu", "H.pt", leave_one_out=True)
    assert args.config_overrides["gb.tie_mode"] == "distribution_only"
    assert args.config_overrides["gb.leave_one_out"] is True


def test_validate_embedding_rejects_nonfinite_and_wrong_node_count(tmp_path: Path):
    good = tmp_path / "good.pt"
    torch.save(torch.ones(4, 3), good)
    diagnostics = validate_embedding(good, expected_nodes=4)
    assert diagnostics["h_num_nodes"] == 4
    assert diagnostics["h_dim"] == 3

    bad_shape = tmp_path / "bad_shape.pt"
    torch.save(torch.ones(3, 3), bad_shape)
    with pytest.raises(ValueError, match="node count"):
        validate_embedding(bad_shape, expected_nodes=4)

    bad_finite = tmp_path / "bad_finite.pt"
    tensor = torch.ones(4, 3)
    tensor[0, 0] = float("nan")
    torch.save(tensor, bad_finite)
    with pytest.raises(ValueError, match="non-finite"):
        validate_embedding(bad_finite, expected_nodes=4)


def test_summarize_variance_separates_h_and_downstream_variation():
    rows = []
    for h_seed, h_offset in enumerate((0.00, 0.01, 0.02)):
        for downstream_seed, downstream_offset in enumerate((0.00, 0.001, -0.001)):
            rows.append(
                {
                    "dataset": "cora",
                    "h_seed": h_seed,
                    "seed": downstream_seed,
                    "test": 0.70 + h_offset + downstream_offset,
                }
            )
    summary = summarize_variance(pd.DataFrame(rows), datasets=("cora",))
    row = summary.iloc[0]
    assert row["h_seed_std"] == pytest.approx(0.01)
    assert row["downstream_rms_std"] == pytest.approx(0.001)
    assert row["h_to_downstream_std_ratio"] == pytest.approx(10.0)
