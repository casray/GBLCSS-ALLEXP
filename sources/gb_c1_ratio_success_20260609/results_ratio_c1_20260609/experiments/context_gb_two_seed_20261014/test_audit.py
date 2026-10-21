"""Focused tests for the independent Context-GB artifact auditor."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit


def _registered_config(data_path: Path) -> dict:
    return {
        "datasets": ["tiny"],
        "seeds": [0],
        "arms": audit.DEFAULT_ARMS,
        "spaces": audit.DEFAULT_SPACES,
        "budgets": [64, 128, 256],
        "noise": audit.DEFAULT_NOISE,
        "scope": "two complete seeds exploratory; no KNN/KMeans comparator",
        "data": {"tiny": {"path": str(data_path), "sha256": audit.file_sha256(data_path)}},
    }


def test_expected_two_seed_grid_is_684_students():
    cells = audit.expected_student_cells(
        ["cora", "citeseer", "corafull"], [0, 1], audit.DEFAULT_SPACES,
        [64, 128, 256], list(audit.DEFAULT_NOISE),
    )
    assert len(cells) == 684


def test_existing_split_keeps_remainder_unassigned():
    train, validation, test = audit.split_indices(2708, 0)
    assert (len(train), len(validation), len(test)) == (541, 270, 1895)
    joined = np.concatenate([train, validation, test])
    assert len(np.unique(joined)) == 2706
    assert 2708 - len(joined) == 2


def test_training_noise_is_deterministic_and_train_only():
    clean = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2])
    train = np.asarray([0, 2, 4, 6, 8, 10])
    first = audit.noisy_labels(clean, train, "uniform", 0.4, 3)
    second = audit.noisy_labels(clean, train, "uniform", 0.4, 3)
    torch.testing.assert_close(first, second)
    held_out = torch.ones(len(clean), dtype=torch.bool)
    held_out[torch.as_tensor(train)] = False
    torch.testing.assert_close(first[held_out], clean[held_out])
    torch.testing.assert_close(audit.noisy_labels(clean, train, "uniform", 0, 3), clean)


def test_accuracy_rejects_nonfinite_logits():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert audit.accuracy(logits, torch.tensor([0, 0])) == 0.5
    with pytest.raises(ValueError, match="non-finite"):
        audit.accuracy(torch.tensor([[float("nan"), 0.0]]), torch.tensor([0]))


def test_summary_uses_sample_standard_deviation_across_seeds(tmp_path: Path):
    auditor = audit.Auditor(tmp_path, partial=True)
    for accuracy_pct in (50.0, 54.0):
        auditor.rows.append({
            "dataset": "cora", "noise": "clean", "head": "mlp",
            "space": "CE", "budget": 0,
            "test_accuracy_pct": accuracy_pct,
        })
    row = auditor._summaries()[0]
    assert row["seed_count"] == 2
    assert row["mean_accuracy_pct"] == 52.0
    assert row["sample_std_pp"] == pytest.approx(np.sqrt(8.0))


def test_pretrain_audit_rejects_missing_epochs_and_nonfinite_values(tmp_path: Path, monkeypatch):
    folder = tmp_path / "pretrain" / "tiny" / "seed0" / "grace"
    folder.mkdir(parents=True)
    torch.save({"h": torch.tensor([[float('nan')], [0.]]), "c": torch.zeros(2,1)}, folder / "representation.pt")
    torch.save({}, folder / "checkpoint.pt")
    (folder / "complete.json").write_text(json.dumps({"sha256": audit.file_sha256(folder / "representation.pt")}), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps({"dataset":"tiny", "seed":0, "arm":"grace", "epochs":2}), encoding="utf-8")
    (folder / "loss.csv").write_text("epoch,node_loss,context_loss,valid_roots\n1,nan,0,0\n", encoding="utf-8")
    checker = audit.Auditor(tmp_path, partial=True)
    checker.datasets, checker.seeds, checker.arms = ["tiny"], [0], ["grace"]
    checker.config = {"pretrain_epochs":2}
    monkeypatch.setattr(checker, "_graph_info", lambda *args: {"clean":torch.tensor([0,1])})
    checker._verify_pretrains()
    assert any("epoch coverage" in error for error in checker.errors)
    assert any("non-finite pretrain" in error for error in checker.errors)
    assert any("non-finite saved" in error for error in checker.errors)


def test_partial_audit_recomputes_student_logits_and_hashes(tmp_path: Path):
    output = tmp_path / "smoke"
    dataset_path = tmp_path / "tiny.pt"
    labels = torch.tensor([0, 1] * 10, dtype=torch.long)
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    torch.save([{"y": labels, "edge_index": edges}], dataset_path)

    config = _registered_config(dataset_path)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({str(Path(audit.__file__).resolve()): audit.file_sha256(audit.__file__)}),
        encoding="utf-8",
    )

    train, validation, test = audit.split_indices(len(labels), 0)
    split_hash = audit.fingerprint(train, validation, test)
    protocol = {
        "split_hash": split_hash,
        "nodes": len(labels),
        "classes": 2,
        "edges": edges.shape[1],
    }
    protocol_path = output / "protocol" / "tiny" / "seed0.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    noisy = audit.noisy_labels(labels, train, "uniform", 0.0, 0)
    logits = torch.zeros((len(labels), 2), dtype=torch.float32)
    logits[torch.arange(len(labels)), labels] = 1.0
    noise_hash = audit.fingerprint(noisy[torch.as_tensor(train)])
    target_hash = audit.fingerprint(torch.nn.functional.one_hot(noisy, 2).float()[train])
    result = {
        "dataset": "tiny", "seed": 0, "noise": "clean", "head": "mlp",
        "space": "CE", "budget": 0,
        "train_accuracy": audit.accuracy(logits[torch.as_tensor(train)], noisy[torch.as_tensor(train)]),
        "validation_accuracy": audit.accuracy(logits[torch.as_tensor(validation)], labels[torch.as_tensor(validation)]),
        "test_accuracy": audit.accuracy(logits[torch.as_tensor(test)], labels[torch.as_tensor(test)]),
        "best_epoch": 1, "split_hash": split_hash, "noise_hash": noise_hash,
        "target_hash": target_hash,
    }
    student_dir = output / "students" / "tiny" / "seed0" / "clean" / "mlp" / "CE_B0"
    student_dir.mkdir(parents=True)
    (student_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (student_dir / "config.json").write_text(json.dumps({
        "dataset": "tiny", "seed": 0, "noise": "clean", "head": "mlp",
        "space": "CE", "budget": 0, "split_hash": split_hash,
        "noise_hash": noise_hash, "target_hash": target_hash,
        "training": {"epochs": 3},
    }), encoding="utf-8")
    torch.save(logits, student_dir / "logits.pt")

    # A writer may publish the result before updating its progress snapshot.
    (output / "status.json").write_text(json.dumps({
        "status": "running", "completed_pretrains": 0, "completed_students": 0,
    }), encoding="utf-8")

    report = audit.audit_output(output, partial=True)
    assert report["coverage_complete"] is False
    assert report["audited_students"] == 1
    assert report["checked"]["logits"] == 1
    assert report["checked"]["target_hashes"] == 1
    assert report["errors"] == []
    assert any("completed_students=0" in warning for warning in report["warnings"])
    assert (output / "audit" / "report.json").is_file()

