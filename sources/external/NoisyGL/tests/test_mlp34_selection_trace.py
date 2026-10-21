import numpy as np
import torch

from utils.selection_trace import SelectionTraceRecorder
from predictor.MLP34SelectionTrace_Predictor import (
    lcss_small_loss_shuffled_trace_Predictor,
    lcss_small_loss_trace_Predictor,
    mlp_small_loss_trace_Predictor,
)
from scripts.run_mlp34_selection_trace import TRACE_METHOD_SPECS


def test_trace_maps_local_selection_to_global_node_mask(tmp_path):
    recorder = SelectionTraceRecorder(n_nodes=6, target_epochs=(1, 10))
    recorder.record(
        epoch_zero_based=0,
        global_train_indices=torch.tensor([1, 3, 4, 5]),
        selected_local_indices=torch.tensor([0, 2]),
    )

    mask = recorder.masks[1]
    assert mask.tolist() == [False, True, False, False, True, False]

    output = tmp_path / "trace.npz"
    recorder.save(
        output,
        train_indices=np.array([1, 3, 4, 5]),
        noisy_train_labels=np.array([0, 1, 1, 0]),
        train_targets=np.eye(2, dtype=np.float32)[[0, 1, 1, 0]],
        metadata={"dataset": "toy", "seed": 0},
    )
    saved = np.load(output, allow_pickle=False)
    assert saved["epochs"].tolist() == [1]
    assert saved["selected_masks"].shape == (1, 6)
    assert saved["metadata_json"].item() == '{"dataset": "toy", "seed": 0}'


def test_trace_ignores_epochs_outside_registered_schedule():
    recorder = SelectionTraceRecorder(n_nodes=4, target_epochs=(1, 10))
    recorder.record(
        epoch_zero_based=4,
        global_train_indices=np.array([0, 2]),
        selected_local_indices=np.array([1]),
    )

    assert recorder.masks == {}


def test_trace_rejects_out_of_range_local_index():
    recorder = SelectionTraceRecorder(n_nodes=4, target_epochs=(1,))

    try:
        recorder.record(
            epoch_zero_based=0,
            global_train_indices=np.array([0, 2]),
            selected_local_indices=np.array([2]),
        )
    except ValueError as exc:
        assert "selected_local_indices" in str(exc)
    else:
        raise AssertionError("out-of-range selection should fail")


def test_trace_predictors_preserve_registered_training_strategies():
    assert mlp_small_loss_trace_Predictor.strategy == "small_loss"
    assert lcss_small_loss_trace_Predictor.robust_strategy == "soft_small_loss"
    assert lcss_small_loss_shuffled_trace_Predictor.random_control is True
    assert set(TRACE_METHOD_SPECS) == {
        "small_loss",
        "lcss_soft_small_loss",
        "lcss_soft_small_loss_shuffled",
    }
