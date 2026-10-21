import json

import numpy as np
import torch
import torch.nn.functional as F

from predictor.LCSSRobust_Predictor import (
    frozengrace_gb_mlp_softdist_beta05_small_loss_Predictor,
    frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled_Predictor,
)
from predictor.MLPNoiseBaselines_Predictor import mlp_small_loss_Predictor
from utils.selection_trace import SelectionTraceRecorder


def _mask_to_indices(mask) -> np.ndarray:
    values = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if values.dtype == np.bool_:
        return np.flatnonzero(values).astype(np.int64)
    return values.astype(np.int64, copy=False).reshape(-1)


class _SelectionTraceMixin:
    trace_epochs = (1, 10, 50, 100, 200)

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.selection_trace = SelectionTraceRecorder(self.n_nodes, self.trace_epochs)

    def _record_selection(self, epoch, global_train_indices, selected_local_indices):
        self.selection_trace.record(
            epoch_zero_based=epoch,
            global_train_indices=global_train_indices,
            selected_local_indices=selected_local_indices,
        )

    def _save_selection_trace(self):
        output_path = getattr(self, "trace_output_path", None)
        if not output_path:
            raise ValueError("trace_output_path must be set before training a trace predictor")
        train_idx = _mask_to_indices(self.train_mask)
        noisy_train = self.noisy_label[torch.as_tensor(train_idx, device=self.device)].detach().cpu()
        if getattr(self, "soft_target", None) is not None:
            train_targets = self.soft_target[
                torch.as_tensor(train_idx, device=self.device)
            ].detach().cpu().numpy()
        else:
            train_targets = F.one_hot(
                noisy_train.long(), num_classes=int(self.n_classes)
            ).float().numpy()
        soft_alpha = None
        if getattr(self, "soft_alpha", None) is not None:
            soft_alpha = self.soft_alpha[
                torch.as_tensor(train_idx, device=self.device)
            ].detach().cpu().numpy()
        metadata = {
            "dataset": str(getattr(self, "dataset_name", "")),
            "method": str(getattr(self, "trace_method", self.__class__.__name__)),
            "seed": int(getattr(self, "seed", -1)),
            "noise_type": str(getattr(self, "noise_type", "")),
            "noise_rate": float(getattr(self, "noise_rate", float("nan"))),
            "split_hash": str(getattr(self, "split_hash", "")),
            "noise_hash": str(getattr(self, "noise_hash", "")),
        }
        self.selection_trace.save(
            output_path,
            train_indices=train_idx,
            noisy_train_labels=noisy_train.numpy(),
            train_targets=train_targets,
            metadata=metadata,
            soft_alpha=soft_alpha,
            assignment=getattr(self, "gb_assignment", None),
        )
        self.noise_baseline_diagnostics.update(
            {
                "trace_output_path": str(output_path),
                "trace_recorded_epochs": ",".join(
                    str(epoch) for epoch in sorted(self.selection_trace.masks)
                ),
            }
        )

    def train(self):
        result = super().train()
        self._save_selection_trace()
        return result


class mlp_small_loss_trace_Predictor(_SelectionTraceMixin, mlp_small_loss_Predictor):
    pass


class lcss_small_loss_trace_Predictor(
    _SelectionTraceMixin,
    frozengrace_gb_mlp_softdist_beta05_small_loss_Predictor,
):
    pass


class lcss_small_loss_shuffled_trace_Predictor(
    _SelectionTraceMixin,
    frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled_Predictor,
):
    pass
