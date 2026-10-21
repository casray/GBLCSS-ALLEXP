import json
from pathlib import Path

import numpy as np
import torch


def _to_int_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


class SelectionTraceRecorder:
    def __init__(self, n_nodes, target_epochs=(1, 10, 50, 100, 200)):
        self.n_nodes = int(n_nodes)
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        self.target_epochs = tuple(sorted({int(epoch) for epoch in target_epochs}))
        if not self.target_epochs or self.target_epochs[0] <= 0:
            raise ValueError("target_epochs must contain positive one-based epochs")
        self.masks = {}

    def record(self, epoch_zero_based, global_train_indices, selected_local_indices) -> None:
        epoch = int(epoch_zero_based) + 1
        if epoch not in self.target_epochs:
            return
        global_indices = _to_int_array(global_train_indices)
        selected_local = _to_int_array(selected_local_indices)
        if global_indices.size and (
            global_indices.min() < 0 or global_indices.max() >= self.n_nodes
        ):
            raise ValueError("global_train_indices contains an out-of-range node")
        if selected_local.size and (
            selected_local.min() < 0 or selected_local.max() >= global_indices.size
        ):
            raise ValueError("selected_local_indices contains an out-of-range index")
        mask = np.zeros(self.n_nodes, dtype=np.bool_)
        mask[global_indices[selected_local]] = True
        self.masks[epoch] = mask

    def save(
        self,
        path,
        train_indices,
        noisy_train_labels,
        train_targets,
        metadata,
        soft_alpha=None,
        assignment=None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        epochs = [epoch for epoch in self.target_epochs if epoch in self.masks]
        selected_masks = (
            np.stack([self.masks[epoch] for epoch in epochs], axis=0)
            if epochs
            else np.zeros((0, self.n_nodes), dtype=np.bool_)
        )
        payload = {
            "epochs": np.asarray(epochs, dtype=np.int64),
            "selected_masks": selected_masks,
            "train_indices": _to_int_array(train_indices),
            "noisy_train_labels": _to_int_array(noisy_train_labels),
            "train_targets": np.asarray(train_targets, dtype=np.float32),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        if soft_alpha is not None:
            payload["soft_alpha"] = np.asarray(soft_alpha, dtype=np.float32)
        if assignment is not None:
            payload["assignment"] = _to_int_array(assignment)
        np.savez_compressed(path, **payload)
