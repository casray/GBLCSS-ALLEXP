from __future__ import annotations

import hashlib

import pytest
import torch

from scripts.audit_grace_embedding_provenance_20260711 import (
    audit_embedding_file,
    summarize_ball_rows,
)


def test_audit_embedding_file_handles_tensor_and_payload(tmp_path):
    tensor = torch.nn.functional.normalize(torch.randn(7, 3), p=2, dim=1)
    tensor_path = tmp_path / "tensor.pt"
    payload_path = tmp_path / "payload.pt"
    torch.save(tensor, tensor_path)
    torch.save({"H": tensor, "checkpoint_path": str(tmp_path / "missing.pt")}, payload_path)

    tensor_row = audit_embedding_file("tensor", tensor_path)
    payload_row = audit_embedding_file("payload", payload_path)

    assert tensor_row["nodes"] == 7
    assert tensor_row["dim"] == 3
    assert tensor_row["finite"] is True
    assert tensor_row["max_row_norm_error"] == pytest.approx(0.0, abs=1e-6)
    assert payload_row["checkpoint_recorded"] is True
    assert payload_row["checkpoint_exists"] is False
    assert tensor_row["sha256"] == hashlib.sha256(tensor_path.read_bytes()).hexdigest()


def test_summarize_ball_rows_reports_cap_frequency_and_range():
    rows = [
        {"dataset": "a", "seed": 0, "num_balls": 52, "cap_hit": 0},
        {"dataset": "a", "seed": 1, "num_balls": 64, "cap_hit": 1},
        {"dataset": "b", "seed": 0, "num_balls": 64, "cap_hit": 1},
    ]

    summary = summarize_ball_rows(rows, max_balls=64)

    assert summary["global_min"] == 52
    assert summary["global_max"] == 64
    assert summary["cap_hits"] == 2
    assert summary["total"] == 3
    assert summary["by_dataset"]["a"]["range"] == "52-64"


def test_summarize_ball_rows_rejects_impossible_count_above_cap():
    with pytest.raises(ValueError, match="exceeds max_balls"):
        summarize_ball_rows(
            [{"dataset": "a", "seed": 0, "num_balls": 65, "cap_hit": 0}],
            max_balls=64,
        )
