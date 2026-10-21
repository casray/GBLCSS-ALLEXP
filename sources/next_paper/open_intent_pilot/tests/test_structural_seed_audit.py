import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


def load_audit_module():
    return importlib.import_module("next_paper.open_intent_pilot.structural_seed_audit")


def test_summarize_metric_reports_sample_std_and_positive_count():
    audit = load_audit_module()
    rows = [
        {"dataset": "toy", "full_delta": -0.02},
        {"dataset": "toy", "full_delta": -0.01},
        {"dataset": "toy", "full_delta": 0.01},
    ]

    result = audit.summarize_metric(rows, "full_delta")

    assert result["n"] == 3
    assert result["mean"] == pytest.approx(-0.0066666667)
    assert result["sample_std"] == pytest.approx(0.0152752523)
    assert result["positive_count"] == 1


def test_exact_sign_flip_p_uses_all_sign_assignments():
    audit = load_audit_module()

    result = audit.exact_sign_flip_p(np.array([-1.0, -1.0, -1.0]))

    assert result == pytest.approx(0.25)


def test_validate_config_requires_five_distinct_seeds():
    audit = load_audit_module()

    with pytest.raises(ValueError, match="five distinct structural seeds"):
        audit.validate_config({"seeds": [0, 1, 2]})


def test_paired_bootstrap_mean_ci_is_deterministic():
    audit = load_audit_module()
    deltas = np.array([-0.03, -0.02, -0.01, 0.0, 0.01])

    result = audit.paired_bootstrap_mean_ci(deltas, replicates=1000, seed=7)

    assert result == pytest.approx([-0.022, 0.002])


def test_prepare_output_dir_refuses_to_overwrite_nonempty_directory(tmp_path):
    audit = load_audit_module()
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    (output_dir / "result.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        audit.prepare_output_dir(output_dir)


def test_summarize_dataset_applies_the_preregistered_gate():
    audit = load_audit_module()
    rows = []
    for seed, delta in enumerate([0.02, 0.01, 0.03, 0.02, -0.01]):
        rows.append(
            {
                "dataset": "toy",
                "structural_seed": seed,
                "full_real": 0.80 + delta,
                "full_kmeans": 0.80,
                "full_shuffle": 0.70,
                "full_delta": delta,
                "all_safe_real": 0.75 + delta,
                "all_safe_kmeans": 0.75,
                "all_safe_shuffle": 0.65,
                "all_safe_delta": delta,
            }
        )

    result = audit.summarize_dataset(rows, bootstrap_replicates=1000, bootstrap_seed=7)

    assert result["dataset"] == "toy"
    assert result["n_seeds"] == 5
    assert result["bootstrap_seed"] == 7
    assert result["full_delta_positive_count"] == 4
    assert result["all_safe_delta_positive_count"] == 4
    assert result["full_delta_mean"] == pytest.approx(0.014)
    assert result["gate_pass"] is True


def test_write_artifacts_creates_complete_machine_readable_outputs(tmp_path):
    audit = load_audit_module()
    output_dir = tmp_path / "run"
    audit.prepare_output_dir(output_dir)
    raw_rows = [{"dataset": "toy", "structural_seed": 0, "full_delta": -0.1}]
    summaries = [{"dataset": "toy", "n_seeds": 1, "gate_pass": False}]
    statistics = {"gate_pass": False, "datasets": {"toy": summaries[0]}}
    resolved_config = {"seeds": [0, 1, 2, 3, 4]}

    audit.write_artifacts(
        output_dir=output_dir,
        raw_rows=raw_rows,
        summaries=summaries,
        statistics=statistics,
        resolved_config=resolved_config,
        log_lines=["audit-start", "gate=FAIL"],
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "config.resolved.json",
        "raw_seed_metrics.csv",
        "run.log",
        "statistics.json",
        "summary.csv",
    ]
    loaded = json.loads((output_dir / "statistics.json").read_text(encoding="utf-8"))
    assert loaded["gate_pass"] is False
    assert (output_dir / "run.log").read_text(encoding="utf-8").endswith("gate=FAIL\n")


def test_evaluate_dataset_runs_five_real_structural_seeds(tmp_path):
    audit = load_audit_module()
    source_workspace = Path(
        os.environ.get(
            "OPEN_INTENT_SOURCE_WORKSPACE",
            r"C:\Users\Admin\Desktop\实验8去噪5.19-",
        )
    )
    rng = np.random.default_rng(11)
    train_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(20, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(20, 4)),
        ]
    ).astype(np.float32)
    train_labels = np.repeat([0, 1], 20)
    test_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=0.0, scale=2.0, size=(20, 4)),
        ]
    ).astype(np.float32)
    test_open_labels = np.concatenate([np.zeros(10), np.ones(10), -np.ones(20)]).astype(int)
    test_logits = np.vstack(
        [
            np.tile([4.0, -1.0], (10, 1)),
            np.tile([-1.0, 4.0], (10, 1)),
            np.tile([0.1, 0.1], (20, 1)),
        ]
    )
    npz_path = tmp_path / "synthetic.npz"
    np.savez(
        npz_path,
        train_features=train_features,
        train_labels=train_labels,
        train_pred=train_labels.copy(),
        test_features=test_features,
        test_open_labels=test_open_labels,
        test_logits=test_logits,
    )
    config = {
        "seeds": [0, 1, 2, 3, 4],
        "kmeans_seed_offset": 13,
        "shuffle_seed_offset": 1013,
        "gb": {
            "max_balls_per_class": 4,
            "min_ball_size": 2,
            "split_improvement": 0.0,
        },
    }

    rows = audit.evaluate_dataset(
        dataset="synthetic",
        npz_path=npz_path,
        source_workspace=source_workspace,
        config=config,
    )

    assert len(rows) == 5
    assert [row["structural_seed"] for row in rows] == [0, 1, 2, 3, 4]
    assert all(row["n_balls"] >= 2 for row in rows)
    assert all(
        key in row
        for row in rows
        for key in ("full_real", "full_kmeans", "full_shuffle", "all_safe_delta")
    )


def test_run_audit_fingerprints_input_and_writes_five_raw_rows(tmp_path):
    audit = load_audit_module()
    source_workspace = Path(
        os.environ.get(
            "OPEN_INTENT_SOURCE_WORKSPACE",
            r"C:\Users\Admin\Desktop\实验8去噪5.19-",
        )
    )
    rng = np.random.default_rng(17)
    train_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(20, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(20, 4)),
        ]
    ).astype(np.float32)
    train_labels = np.repeat([0, 1], 20)
    test_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 4)),
            rng.normal(loc=0.0, scale=2.0, size=(20, 4)),
        ]
    ).astype(np.float32)
    test_open_labels = np.concatenate([np.zeros(10), np.ones(10), -np.ones(20)]).astype(int)
    test_logits = np.vstack(
        [
            np.tile([4.0, -1.0], (10, 1)),
            np.tile([-1.0, 4.0], (10, 1)),
            np.tile([0.1, 0.1], (20, 1)),
        ]
    )
    npz_path = tmp_path / "synthetic.npz"
    np.savez(
        npz_path,
        train_features=train_features,
        train_labels=train_labels,
        train_pred=train_labels.copy(),
        test_features=test_features,
        test_open_labels=test_open_labels,
        test_logits=test_logits,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "audit_date": "2026-07-14",
                "seeds": [0, 1, 2, 3, 4],
                "kmeans_seed_offset": 13,
                "shuffle_seed_offset": 1013,
                "bootstrap": {"replicates": 100, "seed": 7},
                "gb": {
                    "max_balls_per_class": 4,
                    "min_ball_size": 2,
                    "split_improvement": 0.0,
                },
                "datasets": {"synthetic": {"npz": str(npz_path)}},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit-output"

    statistics = audit.run_audit(
        source_workspace=source_workspace,
        config_path=config_path,
        output_dir=output_dir,
    )

    assert statistics["randomness_scope"] == "frozen-representation structural seeds"
    assert statistics["inputs"]["synthetic"]["sha256"]
    assert len((output_dir / "raw_seed_metrics.csv").read_text(encoding="utf-8").splitlines()) == 6
    assert (output_dir / "statistics.json").is_file()
