import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_gagb_v2_class_failure import build_method_targets
from noisy_label_gb_grace.diagnostics.class_failure import (
    class_confusion_matrix,
    node_profile_stats,
    per_class_target_diagnostics,
)


def test_per_class_ncr_and_cdr_are_computed_from_clean_classes():
    clean = torch.tensor([0, 0, 1, 1, 2, 2])
    noisy = torch.tensor([1, 0, 0, 1, 1, 2])
    target_pred = torch.tensor([0, 0, 1, 2, 0, 2])
    train_mask = torch.tensor([True, True, True, True, True, True])

    rows = per_class_target_diagnostics(
        clean,
        noisy,
        target_pred,
        train_mask,
        num_classes=3,
        gated_mask=torch.tensor([True, False, True, True, True, False]),
        teacher_pred=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    by_class = {row["class_id"]: row for row in rows}

    assert by_class[0]["noisy_train_count"] == 1
    assert by_class[0]["correction_by_class_ncr"] == 1.0
    assert by_class[1]["noisy_train_count"] == 1
    assert by_class[1]["correction_by_class_ncr"] == 1.0
    assert by_class[2]["noisy_train_count"] == 1
    assert by_class[2]["correction_by_class_ncr"] == 0.0
    assert by_class[1]["clean_train_count"] == 1
    assert by_class[1]["clean_damage_by_class_cdr"] == 1.0
    assert by_class[2]["target_changed_by_clean_class"] == 1
    assert by_class[1]["gated_noisy_by_class"] == 1
    assert by_class[1]["gated_clean_by_class"] == 1


def test_confusion_matrix_uses_clean_label_rows_and_prediction_columns():
    clean = torch.tensor([0, 0, 1, 1, 2])
    pred = torch.tensor([0, 2, 1, 0, 2])
    mask = torch.tensor([True, True, True, False, True])

    matrix = class_confusion_matrix(clean, pred, mask, num_classes=3)

    assert matrix == [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_node_profile_stats_handles_empty_masks_with_nan_values():
    values = {
        "degree": torch.tensor([1.0, 2.0]),
        "teacher_conf": torch.tensor([0.7, 0.8]),
    }
    mask = torch.tensor([False, False])

    stats = node_profile_stats(values, mask, prefix="corrected")

    assert stats["corrected_count"] == 0
    assert torch.isnan(torch.tensor(stats["corrected_degree_mean"]))
    assert torch.isnan(torch.tensor(stats["corrected_teacher_conf_mean"]))


def test_class_failure_script_reuses_build_method_targets():
    import diagnose_gagb_v2_class_failure

    source = inspect.getsource(diagnose_gagb_v2_class_failure)

    assert "build_method_targets(" in source
    assert build_method_targets is not None


def test_v23_class_safe_script_reuses_build_method_targets():
    import diagnose_gagb_v2_3_class_safe_calibration

    source = inspect.getsource(diagnose_gagb_v2_3_class_safe_calibration)

    assert "build_method_targets(" in source
    assert build_method_targets is not None


def test_grace_first_grouping_script_reuses_build_method_targets():
    import diagnose_grace_first_gagb_grouping

    source = inspect.getsource(diagnose_grace_first_gagb_grouping)

    assert "build_method_targets(" in source
    assert build_method_targets is not None


def test_quality_filter_script_reuses_build_method_targets():
    import diagnose_gagb_v2_6_quality_filter

    source = inspect.getsource(diagnose_gagb_v2_6_quality_filter)

    assert "build_method_targets(" in source
    assert build_method_targets is not None
