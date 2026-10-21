from __future__ import annotations

import importlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _pretrain_module():
    return importlib.import_module("scripts.pretrain_mlp37_grace_seeds")


def _summary_module():
    return importlib.import_module("scripts.summarize_mlp37_grace_seed_variance")


def _runner_module():
    return importlib.import_module("scripts.run_mlp37_grace_seed_variance")


def test_grace_seed_output_directories_are_isolated(tmp_path: Path):
    module = _pretrain_module()

    first = module.seed_output_dir(tmp_path, "Cora", 0)
    second = module.seed_output_dir(tmp_path, "Cora", 1)
    other_dataset = module.seed_output_dir(tmp_path, "CoraFull", 0)

    assert first == tmp_path / "pretrain" / "cora" / "hseed0"
    assert second != first
    assert other_dataset != first


def test_corafull_uses_the_frozen_dblp_config_fallback(monkeypatch):
    module = _pretrain_module()
    requested = []

    def fake_loader(name):
        requested.append(name)
        return {"num_epochs": 7, "num_hidden": 16}

    monkeypatch.setattr(module, "load_grace_config", fake_loader)
    config, source = module.resolve_grace_config("CoraFull")

    assert source == "DBLP"
    assert requested == ["DBLP"]
    assert config["num_epochs"] == 7


def test_checkpoint_payload_records_reproducibility_and_cost_fields():
    module = _pretrain_module()
    payload = module.build_checkpoint_payload(
        dataset="Cora",
        h_seed=2,
        config={"num_epochs": 2},
        config_source="Cora",
        model_state_dict={"weight": np.array([1.0])},
        optimizer_state_dict={"state": {}},
        losses=[{"epoch": 1, "loss": 1.0}],
        wall_time_seconds=3.5,
        peak_memory_allocated_bytes=1024,
        peak_memory_reserved_bytes=2048,
    )

    required = {
        "dataset",
        "h_seed",
        "config",
        "config_source_dataset",
        "model_state_dict",
        "optimizer_state_dict",
        "losses",
        "wall_time_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "label_free",
        "used_labels_or_masks_in_training",
    }
    assert required.issubset(payload)
    assert payload["label_free"] is True
    assert payload["used_labels_or_masks_in_training"] is False


def test_rocm_peak_memory_measurement_initializes_the_device_first(monkeypatch):
    module = _pretrain_module()
    calls = []

    monkeypatch.setattr(module.torch.cuda, "set_device", lambda device: calls.append(("set", device)))
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: calls.append(("empty", None)))
    monkeypatch.setattr(
        module.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )

    module.prepare_device_measurement(module.torch.device("cuda:0"))

    assert [name for name, _ in calls] == ["set", "empty", "reset"]


def _synthetic_raw() -> pd.DataFrame:
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for downstream_seed in range(3):
            split_hash = f"split-{dataset}-{downstream_seed}"
            noise_hash = f"noise-{dataset}-{downstream_seed}"
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "method": "mlp_ce",
                        "seed": downstream_seed,
                        "h_seed": "",
                        "test": 0.60,
                        "status": "ok",
                        "noise_scope": "train",
                        "modified_val_count": 0,
                        "modified_test_count": 0,
                        "split_hash": split_hash,
                        "noise_hash": noise_hash,
                    },
                    {
                        "dataset": dataset,
                        "method": "mlp_jocor",
                        "seed": downstream_seed,
                        "h_seed": "",
                        "test": 0.64,
                        "status": "ok",
                        "noise_scope": "train",
                        "modified_val_count": 0,
                        "modified_test_count": 0,
                        "split_hash": split_hash,
                        "noise_hash": noise_hash,
                    },
                ]
            )
            for h_seed in range(3):
                rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "method": "frozengrace_gb_mlp_softdist_beta05",
                            "seed": downstream_seed,
                            "h_seed": h_seed,
                            "test": 0.70 + 0.001 * h_seed,
                            "status": "ok",
                            "noise_scope": "train",
                            "modified_val_count": 0,
                            "modified_test_count": 0,
                            "split_hash": split_hash,
                            "noise_hash": noise_hash,
                        },
                        {
                            "dataset": dataset,
                            "method": "frozengrace_gb_mlp_softdist_beta05_shuffled",
                            "seed": downstream_seed,
                            "h_seed": h_seed,
                            "test": 0.66,
                            "status": "ok",
                            "noise_scope": "train",
                            "modified_val_count": 0,
                            "modified_test_count": 0,
                            "split_hash": split_hash,
                            "noise_hash": noise_hash,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def test_summary_rejects_an_incomplete_three_by_three_design():
    module = _summary_module()
    frame = _synthetic_raw()
    incomplete = frame.drop(frame.index[-1])

    with pytest.raises(ValueError, match="incomplete"):
        module.validate_raw(incomplete)


def test_summary_preregistered_gate_passes_stable_synthetic_data():
    module = _summary_module()
    frame = _synthetic_raw()

    module.validate_raw(frame)
    variance, gates = module.summarize_variance_and_gates(frame)

    assert set(variance["dataset"]) == {"cora", "citeseer", "corafull"}
    assert gates["primary_pass"].all()
    assert gates["strong_baseline_pass"].all()
    assert gates["all_datasets_primary_pass"].all()


def test_assignment_agreement_is_permutation_invariant():
    module = _summary_module()
    first = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])

    assert module.assignment_agreement(first, permuted) == pytest.approx(1.0)


def test_formal_run_grid_contains_72_unique_cells():
    module = _runner_module()
    cells = module.formal_run_cells()

    assert len(cells) == 72
    assert len(set(cells)) == 72
    assert sum(cell[1] == "mlp_ce" for cell in cells) == 9
    assert sum(cell[1] == "mlp_jocor" for cell in cells) == 9


def test_pretrain_manifest_rejects_a_missing_or_tampered_embedding(tmp_path: Path):
    module = _runner_module()
    rows = []
    for dataset in ("Cora", "CiteSeer", "CoraFull"):
        for h_seed in range(3):
            path = tmp_path / f"{dataset}-{h_seed}.pt"
            path.write_bytes(f"{dataset}-{h_seed}".encode("utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "h_seed": h_seed,
                    "embedding_path": str(path),
                    "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    loaded = module.load_pretrain_manifest(manifest)
    assert len(loaded) == 9

    Path(rows[-1]["embedding_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        module.load_pretrain_manifest(manifest)


def test_decorated_h_row_records_pretraining_provenance():
    module = _runner_module()
    row = module.decorate_h_row(
        {"dataset": "cora", "method": "candidate", "seed": 1},
        h_seed=2,
        manifest_row={
            "embedding_path": "H.pt",
            "embedding_sha256": "abc",
            "config_source_dataset": "Cora",
        },
    )

    assert row["h_seed"] == 2
    assert row["h_sha256"] == "abc"
    assert row["h_config_source_dataset"] == "Cora"


def test_geometry_summary_reports_all_pairwise_h_seed_agreements(tmp_path: Path):
    module = _summary_module()
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for seed in range(3):
            for h_seed in range(3):
                path = tmp_path / f"{dataset}-{seed}-{h_seed}.npy"
                assignment = np.array([h_seed, h_seed, (h_seed + 1) % 3, (h_seed + 1) % 3])
                np.save(path, assignment)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "h_seed": h_seed,
                        "assignment_path": str(path),
                    }
                )

    agreements = module.summarize_assignment_agreement(pd.DataFrame(rows))

    assert len(agreements) == 27
    assert set(agreements["h_seed_pair"]) == {"0-1", "0-2", "1-2"}
    assert agreements["adjusted_rand_index"].between(-1.0, 1.0).all()


def test_report_labels_the_result_as_post_freeze_conditional_not_held_out():
    module = _summary_module()
    frame = _synthetic_raw()
    variance, gates = module.summarize_variance_and_gates(frame)
    report = module.build_report(frame, variance, gates, pd.DataFrame(), pd.DataFrame())

    assert "参数冻结后的条件评测" in report
    assert "held-out" not in report
    assert "留出数据集" not in report
