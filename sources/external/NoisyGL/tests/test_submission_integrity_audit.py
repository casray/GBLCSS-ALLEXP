from __future__ import annotations

import math

import pytest
import scripts.audit_submission_integrity_20260710 as audit

from scripts.audit_submission_integrity_20260710 import (
    REPO_ROOT,
    aggregate_seed_rows,
    assert_paired_hashes,
    assert_train_only_protocol,
    assert_unique_rows,
    build_core_audit_report,
    enrich_protocol_from_source,
    load_core_rows,
    summarize_core_claims,
)


def _row(method: str, seed: int, test: float, split_hash: str = "split", noise_hash: str = "noise"):
    return {
        "dataset": "cora",
        "method": method,
        "noise_type": "uniform",
        "noise_rate": "0.4",
        "seed": str(seed),
        "test": str(test),
        "split_hash": split_hash,
        "noise_hash": noise_hash,
    }


def test_aggregate_seed_rows_uses_sample_standard_deviation():
    rows = [_row("ours", 0, 0.5), _row("ours", 1, 0.7)]

    aggregate = aggregate_seed_rows(rows)
    item = aggregate[("cora", "uniform", 0.4, "ours")]

    assert item["n"] == 2
    assert item["mean"] == pytest.approx(0.6)
    assert item["std"] == pytest.approx(math.sqrt(0.02))


def test_assert_unique_rows_rejects_duplicate_seed_records():
    rows = [_row("ours", 0, 0.5), _row("ours", 0, 0.6)]

    with pytest.raises(ValueError, match="duplicate experiment row"):
        assert_unique_rows(rows)


def test_assert_paired_hashes_rejects_mismatched_noise_realizations():
    rows = [
        _row("ours", 0, 0.5, noise_hash="noise-a"),
        _row("baseline", 0, 0.4, noise_hash="noise-b"),
    ]

    with pytest.raises(ValueError, match="noise_hash mismatch"):
        assert_paired_hashes(rows)


def test_assert_train_only_protocol_rejects_validation_label_corruption():
    row = _row("ours", 0, 0.5)
    row.update(
        {
            "status": "ok",
            "noise_scope": "train",
            "modified_count": "11",
            "modified_train_count": "10",
            "modified_val_count": "1",
            "modified_test_count": "0",
        }
    )

    with pytest.raises(ValueError, match="validation/test labels were modified"):
        assert_train_only_protocol([row])


def test_summarize_core_claims_counts_best_settings_and_shuffled_wins():
    rows = []
    for dataset, ours_values, baseline_values in (
        ("cora", (0.8, 0.9), (0.7, 0.8)),
        ("citeseer", (0.6, 0.7), (0.65, 0.75)),
    ):
        for seed, (ours, baseline) in enumerate(zip(ours_values, baseline_values)):
            rows.extend(
                [
                    _row("ours", seed, ours),
                    _row("baseline", seed, baseline),
                    _row("shuffled", seed, ours - 0.2),
                ]
            )
            for row in rows[-3:]:
                row["dataset"] = dataset

    summary = summarize_core_claims(
        rows,
        ours_method="ours",
        shuffled_method="shuffled",
    )

    assert summary["settings"] == 2
    assert summary["ours_best_settings"] == 1
    assert summary["ours_above_shuffled_settings"] == 2


def test_enrich_protocol_from_source_recovers_raw_provenance(tmp_path):
    source = tmp_path / "formal.csv"
    source.write_text(
        "dataset,method,seed,noise_type,noise_rate,noise_scope,split_hash,noise_hash,"
        "modified_count,modified_train_count,modified_val_count,modified_test_count,status,test\n"
        "cora,ours,0,uniform,0.4,train,split-0,noise-0,10,10,0,0,ok,0.5\n",
        encoding="utf-8",
    )
    row = _row("ours", 0, 0.5)
    row["source"] = str(source)

    enriched = enrich_protocol_from_source(row, tmp_path)

    assert enriched["split_hash"] == "split-0"
    assert enriched["noise_hash"] == "noise-0"
    assert enriched["noise_scope"] == "train"
    assert enriched["modified_count"] == "10"


def test_current_core_evidence_matrix_is_complete_and_train_only():
    rows = load_core_rows(REPO_ROOT)

    assert len(rows) == 5 * 5 * 13 * 5
    assert_unique_rows(rows)
    assert_paired_hashes(rows)
    assert_train_only_protocol(rows)


def test_core_report_records_the_pre_registered_25_setting_claim():
    rows = load_core_rows(REPO_ROOT)

    report = build_core_audit_report(rows)

    assert "25/25" in report
    assert "24/25" in report
    assert "Coauthor-CS | 4/5" in report


def test_table3_evidence_is_complete_paired_and_train_only():
    rows = audit.load_table3_rows(REPO_ROOT)

    assert len(rows) == 5 * 9 * 5
    methods = {row["method"] for row in rows}
    assert "graphmlp_reimplemented" in methods
    assert "ffg2m_reimplemented" in methods
    assert "graphmlp" not in methods
    assert "ffg2m_adapted" not in methods
    assert_unique_rows(rows)
    assert_paired_hashes(rows)
    assert_train_only_protocol(rows)


def test_table4_teacher_ablation_is_complete_paired_and_train_only():
    rows = audit.load_table4_rows(REPO_ROOT)

    assert len(rows) == 3 * 2 * 5 * 5
    assert_unique_rows(rows)
    assert_paired_hashes(rows)
    assert_train_only_protocol(rows)


def test_table5_graph_reference_is_complete_and_train_only():
    rows = audit.load_table5_rows(REPO_ROOT)

    assert len(rows) == 5 * 8 * 5
    assert_unique_rows(rows)
    assert_paired_hashes(rows)
    assert_train_only_protocol(rows)


def test_manuscript_table3_to_table5_cells_match_recomputed_seed_rows():
    manuscript = REPO_ROOT / "docs" / "paper_final_zh_v11_faithful_claims_20260712.md"

    mismatches = audit.compare_manuscript_tables(
        manuscript,
        table3_rows=audit.load_table3_rows(REPO_ROOT),
        table4_rows=audit.load_table4_rows(REPO_ROOT),
        table5_rows=audit.load_table5_rows(REPO_ROOT),
    )

    assert mismatches == []


def test_extended_report_records_complete_tables_and_zero_cell_mismatches():
    manuscript = REPO_ROOT / "docs" / "paper_final_zh_v11_faithful_claims_20260712.md"

    report = audit.build_extended_audit_report(
        manuscript,
        audit.load_table3_rows(REPO_ROOT),
        audit.load_table4_rows(REPO_ROOT),
        audit.load_table5_rows(REPO_ROOT),
    )

    assert "Table 3 rows: 225" in report
    assert "Table 4 rows: 150" in report
    assert "Table 5 rows: 200" in report
    assert "Manuscript cell mismatches: 0" in report
