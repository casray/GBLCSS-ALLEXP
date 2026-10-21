import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.FrozenGraceGB_GCN_Predictor import (
    concat_l2_normalized_embeddings,
    load_frozen_embeddings,
    mlp_distill_gcn_xh_Predictor,
)
from predictor.LCSSRobust_Predictor import (
    select_soft_small_loss_indices,
    soft_cross_entropy_per_sample,
)


def robust_student_distill_loss(
    student_logits,
    teacher_probs,
    soft_targets,
    train_mask,
    epoch,
    noise_rate,
    temperature=1.0,
    kd_weight=1.0,
    robust_weight=1.0,
    num_gradual=10,
    exponent=1.0,
):
    temperature = max(1e-6, float(temperature))
    kd_loss = -(
        teacher_probs * F.log_softmax(student_logits / temperature, dim=1)
    ).sum(dim=1).mean() * (temperature ** 2)
    per_sample = soft_cross_entropy_per_sample(
        student_logits[train_mask],
        soft_targets[train_mask],
    )
    selected = select_soft_small_loss_indices(
        per_sample,
        epoch=int(epoch),
        noise_rate=float(noise_rate),
        num_gradual=int(num_gradual),
        exponent=float(exponent),
    )
    robust_loss = per_sample[selected].mean()
    total = float(kd_weight) * kd_loss + float(robust_weight) * robust_loss
    return total, kd_loss, robust_loss, selected, per_sample


class _LCSSRobustStudentDistillBase(mlp_distill_gcn_xh_Predictor):
    random_control = False
    use_soft_targets = True
    skip_ball_targets = False
    teacher_use_xh = True
    teacher_target = "ce"
    strategy = "gcn_xh_kd_lcss_soft_small_loss"

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.noise_baseline_diagnostics = {
            "student_robust_strategy": "lcss_soft_small_loss",
            "student_kd_weight": float(self._conf_value("student_kd_weight", 1.0)),
            "student_robust_weight": float(self._conf_value("student_robust_weight", 1.0)),
        }

    def _active_student_noise_rate(self):
        configured = self._conf_value("student_forget_rate", "auto")
        if str(configured).lower() != "auto":
            return float(configured)
        return float(getattr(self, "noise_rate", 0.4))

    def _teacher_input(self):
        if hasattr(self, "_teacher_feats"):
            return self._teacher_feats
        embedding_path = self._conf_value(
            "teacher_embedding_path",
            self._feature_embedding_path(),
        )
        embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
        self._teacher_feats = concat_l2_normalized_embeddings(self.feats, embeddings).to(self.device)
        return self._teacher_feats

    def train(self):
        self._make_corrected_labels()
        teacher_probs, temperature = self._train_teacher_probs()
        self.noise_baseline_diagnostics.update(
            {
                "teacher_best_test": self.gb_diagnostics.get("teacher_best_test", ""),
                "teacher_best_val": self.gb_diagnostics.get("teacher_best_val", ""),
                "teacher_best_epoch": self.gb_diagnostics.get("teacher_best_epoch", ""),
                "teacher_use_xh": self.gb_diagnostics.get("teacher_use_xh", 1),
                "teacher_input_dim": self.gb_diagnostics.get("teacher_input_dim", ""),
                "student_input_dim": int(self.feats.shape[1]),
                "distill_temperature": float(temperature),
            }
        )

        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output = self.model(self.feats, self.adj)
            loss_train, kd_loss, robust_loss, selected, per_sample = robust_student_distill_loss(
                output,
                teacher_probs,
                self.soft_target,
                self.train_mask,
                epoch=epoch,
                noise_rate=self._active_student_noise_rate(),
                temperature=temperature,
                kd_weight=self._conf_value("student_kd_weight", 1.0),
                robust_weight=self._conf_value("student_robust_weight", 1.0),
                num_gradual=self._conf_value("student_num_gradual", 10),
                exponent=self._conf_value("student_exponent", 1.0),
            )
            acc_train = self.metric(
                self.noisy_label[self.train_mask].cpu().numpy(),
                output[self.train_mask].detach().cpu().numpy(),
            )
            loss_train.backward()
            self.optim.step()
            self.noise_baseline_diagnostics.update(
                {
                    "student_forget_rate": self._active_student_noise_rate(),
                    "student_selected_count": int(selected.numel()),
                    "student_selected_ratio": float(selected.numel() / max(1, per_sample.numel())),
                    "student_last_kd_loss": float(kd_loss.detach().item()),
                    "student_last_robust_loss": float(robust_loss.detach().item()),
                }
            )

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = "*"
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.best_epoch = epoch + 1
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | "
                    "KD {:.4f} | Robust {:.4f} | Acc(train) {:.4f} | "
                    "Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
                        kd_loss.item(),
                        robust_loss.item(),
                        acc_train,
                        loss_val,
                        acc_val,
                        improve,
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self.noise_baseline_diagnostics["distill_student_best_epoch"] = int(self.best_epoch)
        self._append_diagnostics()
        return self.result


class mlp_distill_gcn_xh_lcss_small_loss_Predictor(_LCSSRobustStudentDistillBase):
    pass


class mlp_distill_gcn_xh_lcss_small_loss_shuffled_Predictor(_LCSSRobustStudentDistillBase):
    random_control = True
    strategy = "gcn_xh_kd_lcss_soft_small_loss_shuffled"
