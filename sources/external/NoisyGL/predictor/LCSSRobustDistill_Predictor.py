import torch

from predictor.FrozenGraceGB_GCN_Predictor import mlp_distill_gcn_Predictor
from predictor.LCSSRobust_Predictor import (
    select_soft_small_loss_indices,
    soft_cross_entropy_per_sample,
)


def soft_small_teacher_loss(
    logits,
    targets,
    epoch,
    noise_rate,
    num_gradual=10,
    exponent=1.0,
):
    per_sample = soft_cross_entropy_per_sample(logits, targets)
    selected = select_soft_small_loss_indices(
        per_sample,
        epoch=epoch,
        noise_rate=noise_rate,
        num_gradual=num_gradual,
        exponent=exponent,
    )
    return per_sample[selected].mean(), selected, per_sample


class _Beta05SoftDistillBase(mlp_distill_gcn_Predictor):
    random_control = False
    use_soft_targets = True
    skip_ball_targets = False
    teacher_use_xh = True
    teacher_target = "soft_target"
    soft_target_builder = "ball"
    robust_teacher = False


class _Beta05SoftSmallTeacherDistillBase(_Beta05SoftDistillBase):
    robust_teacher = True

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.noise_baseline_diagnostics = {
            "teacher_robust_strategy": "soft_small_loss",
        }

    def _active_teacher_noise_rate(self):
        configured = self._conf_value("teacher_forget_rate", "auto")
        if str(configured).lower() != "auto":
            return float(configured)
        return float(getattr(self, "noise_rate", 0.4))

    def _teacher_loss(self, logits):
        if self.soft_target is None:
            raise ValueError("robust SoftDist teacher requires soft targets")
        loss, selected, per_sample = soft_small_teacher_loss(
            logits[self.train_mask],
            self.soft_target[self.train_mask],
            epoch=int(getattr(self, "teacher_epoch", 0)),
            noise_rate=self._active_teacher_noise_rate(),
            num_gradual=int(self._conf_value("teacher_num_gradual", 10)),
            exponent=float(self._conf_value("teacher_exponent", 1.0)),
        )
        self.noise_baseline_diagnostics.update(
            {
                "teacher_forget_rate": self._active_teacher_noise_rate(),
                "teacher_selected_count": int(selected.numel()),
                "teacher_selected_ratio": float(selected.numel() / max(1, per_sample.numel())),
            }
        )
        return loss


class mlp_distill_softdist_beta05_xh_Predictor(_Beta05SoftDistillBase):
    pass


class mlp_distill_knn_beta05_xh_Predictor(_Beta05SoftDistillBase):
    soft_target_builder = "knn"


class mlp_distill_softdist_beta05_small_loss_xh_Predictor(_Beta05SoftSmallTeacherDistillBase):
    pass


class mlp_distill_softdist_beta05_small_loss_xh_shuffled_Predictor(_Beta05SoftSmallTeacherDistillBase):
    random_control = True

