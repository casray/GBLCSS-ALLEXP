from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import torch
import torch.nn.functional as F

from predictor.NRGNN_Predictor import nrgnn_Predictor
from predictor.module.GNNs import MLP
from utils.logger import SingleExpRecorder


def soft_target_distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.shape != teacher_probs.shape:
        raise ValueError("student logits and teacher probabilities must have the same shape")
    if not torch.isfinite(teacher_probs).all():
        raise ValueError("teacher probabilities must be finite")
    row_sums = teacher_probs.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("teacher probabilities must sum to one")
    temperature = max(1e-6, float(temperature))
    log_prob = F.log_softmax(student_logits / temperature, dim=1)
    return -(teacher_probs * log_prob).sum(dim=1).mean() * (temperature ** 2)


class _NRGNNDistillBase(nrgnn_Predictor):
    use_xh_features = False

    def _gb_value(self, key, default):
        gb_conf = getattr(self.conf, "gb", {})
        if isinstance(gb_conf, dict):
            value = gb_conf.get(key, default)
        else:
            value = getattr(gb_conf, key, default)
        return default if value is None else value

    def method_init(self, conf, data):
        self.student_feats = self.feats
        super().method_init(conf, data)
        self.student_model = MLP(
            in_channels=int(self.student_feats.shape[1]),
            hidden_channels=int(self._gb_value("student_hidden", 64)),
            out_channels=self.n_classes,
            n_layers=int(self._gb_value("student_n_layer", 2)),
            dropout=float(self._gb_value("student_dropout", 0.5)),
        ).to(self.device)
        self.student_optim = torch.optim.Adam(
            self.student_model.parameters(),
            lr=float(self._gb_value("student_lr", 1e-2)),
            weight_decay=float(self._gb_value("student_weight_decay", 5e-4)),
        )
        self.method_diagnostics = {}

    def _best_teacher_probs(self):
        self.model.eval()
        reformed_adj = torch.sparse_coo_tensor(
            self.best_model_index,
            self.best_graph,
            [self.n_nodes, self.n_nodes],
        )
        temperature = float(self._gb_value("distill_temperature", 1.0))
        with torch.no_grad():
            logits = self.model(self.feats, reformed_adj)
            probs = F.softmax(logits / max(1e-6, temperature), dim=1).detach()
        if not torch.isfinite(probs).all():
            raise ValueError("NRGNN teacher produced non-finite probabilities")
        row_sum_error = float((probs.sum(dim=1) - 1.0).abs().max().item())
        if row_sum_error > 1e-5:
            raise ValueError(
                f"NRGNN teacher probabilities do not sum to one: {row_sum_error}"
            )
        return probs, temperature, row_sum_error

    def _train_student(self, teacher_probs, temperature):
        student_epochs = int(self._gb_value("student_epochs", 200))
        student_recorder = SingleExpRecorder(
            self._gb_value("student_patience", self.conf.training["patience"]),
            self._gb_value("student_criterion", self.conf.training["criterion"]),
        )
        best_state = None
        best_epoch = -1
        best_val = -1.0
        best_train = -1.0

        for epoch in range(student_epochs):
            self.student_model.train()
            self.student_optim.zero_grad()
            logits = self.student_model(self.student_feats)
            loss = soft_target_distillation_loss(
                logits,
                teacher_probs,
                temperature,
            )
            train_acc = self.metric(
                self.noisy_label[self.train_mask].cpu().numpy(),
                logits[self.train_mask].detach().cpu().numpy(),
            )
            loss.backward()
            self.student_optim.step()

            self.student_model.eval()
            with torch.no_grad():
                logits = self.student_model(self.student_feats)
                val_loss = self.loss_fn(
                    logits[self.val_mask],
                    self.noisy_label[self.val_mask],
                )
                val_acc = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
            improved, should_stop = student_recorder.add(val_loss, val_acc)
            if improved:
                best_state = deepcopy(self.student_model.state_dict())
                best_epoch = epoch + 1
                best_val = float(val_acc)
                best_train = float(train_acc)
            elif should_stop:
                break

        if best_state is None:
            raise RuntimeError("NRGNN-distilled student did not produce a checkpoint")
        self.student_model.load_state_dict(best_state)
        self.student_model.eval()
        with torch.no_grad():
            logits = self.student_model(self.student_feats)
            test_loss = self.loss_fn(
                logits[self.test_mask],
                self.clean_label[self.test_mask],
            )
            test_acc = self.metric(
                self.clean_label[self.test_mask].cpu().numpy(),
                logits[self.test_mask].detach().cpu().numpy(),
            )
        return {
            "train": best_train,
            "valid": best_val,
            "test": float(test_acc),
            "test_loss": float(test_loss.item()),
            "best_epoch": int(best_epoch),
        }

    def train(self):
        teacher_started = perf_counter()
        teacher_result = dict(super().train())
        teacher_training_seconds = perf_counter() - teacher_started
        teacher_parameter_count = int(
            sum(parameter.numel() for parameter in self.model.parameters())
        )
        teacher_probs, temperature, row_sum_error = self._best_teacher_probs()
        student_started = perf_counter()
        student_result = self._train_student(teacher_probs, temperature)
        student_training_seconds = perf_counter() - student_started

        self.nrgnn_teacher = self.model
        self.model = self.student_model
        self.result = {
            "train": student_result["train"],
            "valid": student_result["valid"],
            "test": student_result["test"],
        }
        self.method_diagnostics.update(
            {
                "teacher_family": "nrgnn",
                "teacher_best_val": float(teacher_result["valid"]),
                "teacher_best_test": float(teacher_result["test"]),
                "teacher_use_xh": int(self.use_xh_features),
                "teacher_input_dim": int(self.feats.shape[1]),
                "teacher_parameter_count": teacher_parameter_count,
                "teacher_training_seconds": float(teacher_training_seconds),
                "student_input_dim": int(self.student_feats.shape[1]),
                "student_inference_uses_graph": 0,
                "student_training_seconds": float(student_training_seconds),
                "distill_temperature": float(temperature),
                "teacher_probability_row_sum_max_error": row_sum_error,
                "teacher_probability_min": float(teacher_probs.min().item()),
                "teacher_probability_max": float(teacher_probs.max().item()),
                "distill_student_best_epoch": student_result["best_epoch"],
                "student_parameter_count": int(
                    sum(parameter.numel() for parameter in self.student_model.parameters())
                ),
            }
        )
        return self.result


class mlp_distill_nrgnn_Predictor(_NRGNNDistillBase):
    use_xh_features = False


class mlp_distill_nrgnn_xh_Predictor(_NRGNNDistillBase):
    use_xh_features = True
