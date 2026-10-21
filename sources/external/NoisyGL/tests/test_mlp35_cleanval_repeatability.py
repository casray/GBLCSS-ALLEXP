import copy
import subprocess
import sys
from pathlib import Path

import pytest


DATASETS = ("cora", "citeseer")
NOISE_RATES = (0.2, 0.4)
SEEDS = (0,)
FORGET_RATES = (0.0, 0.2, 0.4, 0.5, 0.6)


def _row(dataset, noise_rate, forget_rate, *, valid=None, test=None, seed=0):
    if valid is None:
        valid = 0.60 + 0.02 * forget_rate
    if test is None:
        test = 0.55 + 0.01 * forget_rate
    return {
        "dataset": dataset,
        "method": "frozengrace_gb_mlp_softdist_beta05_small_loss",
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "split_type": "percent",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "requested_forget_rate": forget_rate,
        "forget_rate": forget_rate,
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{dataset}-{noise_rate}-{seed}",
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "status": "ok",
        "valid": valid,
        "test": test,
    }


def _complete_rows():
    return [
        _row(dataset, noise_rate, forget_rate)
        for dataset in DATASETS
        for noise_rate in NOISE_RATES
        for forget_rate in FORGET_RATES
    ]


def test_clean_val_selection_uses_validation_and_smaller_rate_tie_break():
    from scripts.summarize_mlp35_cleanval_grid import select_by_clean_validation

    rows = [
        _row("cora", 0.4, 0.0, valid=0.70, test=0.99),
        _row("cora", 0.4, 0.2, valid=0.75, test=0.10),
        _row("cora", 0.4, 0.4, valid=0.75, test=0.95),
        _row("cora", 0.4, 0.6, valid=0.72, test=0.96),
    ]

    selected = select_by_clean_validation(rows, candidate_rates=(0.0, 0.2, 0.4, 0.6))

    assert selected["requested_forget_rate"] == 0.2
    assert selected["test"] == 0.10


def test_clean_val_selection_is_invariant_to_test_values():
    from scripts.summarize_mlp35_cleanval_grid import select_by_clean_validation

    rows = [
        _row("cora", 0.4, 0.0, valid=0.70, test=0.90),
        _row("cora", 0.4, 0.2, valid=0.74, test=0.10),
        _row("cora", 0.4, 0.4, valid=0.73, test=0.99),
        _row("cora", 0.4, 0.6, valid=0.72, test=0.98),
    ]
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["test"] = float(index)

    selected = select_by_clean_validation(rows)
    changed_selected = select_by_clean_validation(changed)

    assert selected["requested_forget_rate"] == 0.2
    assert changed_selected["requested_forget_rate"] == 0.2


def test_clean_val_summary_validates_complete_paired_protocol():
    from scripts.summarize_mlp35_cleanval_grid import build_cleanval_summary

    candidates, selected, aggregate, gates = build_cleanval_summary(
        _complete_rows(),
        expected_datasets=DATASETS,
        expected_noise_rates=NOISE_RATES,
        expected_seeds=SEEDS,
    )

    assert len(candidates) == len(DATASETS) * len(NOISE_RATES) * len(FORGET_RATES)
    assert len(selected) == len(DATASETS) * len(NOISE_RATES) * len(SEEDS)
    assert len(aggregate) == len(DATASETS) * len(NOISE_RATES)
    assert gates["seed0"]["complete"] is True


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "val_pollution"])
def test_clean_val_summary_rejects_invalid_protocol(mutation):
    from scripts.summarize_mlp35_cleanval_grid import build_cleanval_summary

    rows = _complete_rows()
    if mutation == "missing":
        rows.pop()
        match = "missing"
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
        match = "duplicate"
    elif mutation == "hash":
        rows[1]["noise_hash"] = "wrong"
        match = "hash"
    else:
        rows[0]["modified_val_count"] = 1
        match = "validation/test"

    with pytest.raises(ValueError, match=match):
        build_cleanval_summary(
            rows,
            expected_datasets=DATASETS,
            expected_noise_rates=NOISE_RATES,
            expected_seeds=SEEDS,
        )


def test_grid_cells_have_deterministic_order_and_cardinality():
    from scripts.run_mlp35_cleanval_grid import grid_cells

    cells = grid_cells(
        datasets=("cora", "pubmed"),
        noise_rates=(0.2, 0.6),
        seeds=(0, 1),
        forget_rates=(0.0, 0.5),
    )

    assert len(cells) == 16
    assert cells[0] == ("cora", 0.2, 0, 0.0)
    assert cells[-1] == ("pubmed", 0.6, 1, 0.5)


def test_fiveseed_summary_keeps_phase_a_gate_seed0_only():
    from scripts.summarize_mlp35_cleanval_grid import build_cleanval_summary

    rows = []
    for seed in range(5):
        for forget_rate in FORGET_RATES:
            valid = 0.90 if forget_rate == 0.2 else 0.80
            test = 0.70
            if seed == 1 and forget_rate == 0.2:
                test = 0.40
            rows.append(
                _row(
                    "cora",
                    0.2,
                    forget_rate,
                    valid=valid,
                    test=test,
                    seed=seed,
                )
            )

    _, _, _, gates = build_cleanval_summary(
        rows,
        expected_datasets=("cora",),
        expected_noise_rates=(0.2,),
        expected_seeds=tuple(range(5)),
    )

    assert gates["seed0"]["n_settings"] == 1
    assert gates["seed0"]["pass"] is True


def _repeatability_rows():
    rows = []
    specs = {
        "cora_pair0p4": {
            "dataset": "cora",
            "baseline_method": "mlp_small_loss",
            "combo_method": "frozengrace_gb_mlp_softdist_beta05_small_loss",
        }
    }
    for replicate in range(3):
        for method, test in (
            ("mlp_small_loss", 0.50 + 0.001 * replicate),
            ("frozengrace_gb_mlp_softdist_beta05_small_loss", 0.52 + 0.001 * replicate),
        ):
            rows.append(
                {
                    "setting_id": "cora_pair0p4",
                    "dataset": "cora",
                    "method": method,
                    "replicate": replicate,
                    "seed": 0,
                    "noise_type": "pair",
                    "noise_rate": 0.4,
                    "noise_scope": "train",
                    "split_hash": "split-cora",
                    "noise_hash": "noise-cora",
                    "config_id": f"config-{method}",
                    "modified_count": 10,
                    "modified_train_count": 10,
                    "modified_val_count": 0,
                    "modified_test_count": 0,
                    "status": "ok",
                    "test": test,
                }
            )
    return specs, rows


def test_repeatability_summary_reports_ranges_and_paired_deltas():
    from scripts.summarize_mlp35_repeatability_audit import build_repeatability_summary

    specs, rows = _repeatability_rows()
    method_rows, paired_rows, gate = build_repeatability_summary(
        rows,
        setting_specs=specs,
        expected_replicates=(0, 1, 2),
    )

    assert len(method_rows) == 2
    assert len(paired_rows) == 1
    assert paired_rows[0]["positive_replicates"] == 3
    assert paired_rows[0]["delta_mean"] == pytest.approx(0.02)
    assert gate["pass"] is True


@pytest.mark.parametrize("mutation", ["missing", "hash"])
def test_repeatability_summary_rejects_incomplete_or_unpaired_rows(mutation):
    from scripts.summarize_mlp35_repeatability_audit import build_repeatability_summary

    specs, rows = _repeatability_rows()
    if mutation == "missing":
        rows.pop()
        match = "missing"
    else:
        rows[1]["noise_hash"] = "wrong"
        match = "hash"
    with pytest.raises(ValueError, match=match):
        build_repeatability_summary(
            rows,
            setting_specs=specs,
            expected_replicates=(0, 1, 2),
        )


def test_repeatability_summary_script_runs_directly():
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_mlp35_repeatability_audit.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_forget0_lcss_audit_requires_matching_protocol_hashes():
    from scripts.summarize_mlp35_cleanval_grid import audit_forget0_against_lcss

    forget0 = _row("cora", 0.2, 0.0, test=0.61)
    anchor = dict(forget0)
    anchor.pop("requested_forget_rate")
    anchor.pop("forget_rate")
    anchor["method"] = "frozengrace_gb_mlp_softdist_beta05"
    anchor["test"] = 0.60

    rows = audit_forget0_against_lcss([forget0], [anchor])

    assert rows[0]["forget0_minus_lcss"] == pytest.approx(0.01)
    anchor["noise_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        audit_forget0_against_lcss([forget0], [anchor])


def test_forget0_lcss_audit_only_requires_keys_covered_by_anchor_file():
    from scripts.summarize_mlp35_cleanval_grid import audit_forget0_against_lcss

    seed0 = _row("cora", 0.2, 0.0, test=0.61, seed=0)
    seed1 = _row("cora", 0.2, 0.0, test=0.62, seed=1)
    anchor = dict(seed0)
    anchor.pop("requested_forget_rate")
    anchor.pop("forget_rate")
    anchor["method"] = "frozengrace_gb_mlp_softdist_beta05"

    rows = audit_forget0_against_lcss([seed0, seed1], [anchor])

    assert len(rows) == 1
    assert rows[0]["seed"] == 0
