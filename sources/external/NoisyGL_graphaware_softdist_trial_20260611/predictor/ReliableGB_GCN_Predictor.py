import csv
import os
import time
from copy import deepcopy

import nni
import numpy as np
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from utils.reliable_gb import build_reliable_granular_balls, build_random_size_matched_assignment
from utils.soft_targets import build_gb_soft_targets


class _ReliableGBGCNBase(Predictor):
    """Reciprocal-kNN reliable granular balls + conservative reweight on a GCN host.

    Two changes vs OrginGB that target its `best_phase=warmup` failure:
      1. balls are grown from reciprocal-kNN (reliable construction), not embedding-space splitting;
      2. balls + weights are REBUILT periodically on an EMA of the hidden state, so the denoising
         signal tracks the evolving representation instead of being fixed once after warmup.

    Clean labels are diagnostics/eval only; never used in balls, suspicion, gate, or weights.
    """

    random_control = False
    target_mode = "reweight"
    suspicion_mode = "gb"

    def method_init(self, conf, data):
        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'], n_layers=conf.model['n_layer'],
                         dropout=conf.model['dropout'], norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.training['lr'],
                                      weight_decay=self.conf.training['weight_decay'])
        self.dataset_name = data.name
        self.soft_targets = None
        self.sample_weights = None
        self.gb_diagnostics = {}
        self.best_epoch = -1
        self.best_phase = ""
        self._ema_hidden = None

    def _gb_conf(self):
        return getattr(self.conf, "gb", {})

    def _conf_value(self, key, default):
        gb_conf = self._gb_conf()
        if isinstance(gb_conf, dict):
            value = gb_conf.get(key, default)
        else:
            value = getattr(gb_conf, key, default)
        return default if value is None else value

    def _make_gb_targets(self):
        self.model.eval()
        with torch.no_grad():
            logits, hidden = self.model.forward_with_hidden(self.feats, self.adj)
            probs = F.softmax(logits, dim=1)

        # EMA over the hidden state damps cross-epoch ball jitter (warm-up + EMA before GB).
        ema_decay = float(self._conf_value("gb_ema_decay", 0.8))
        if self._ema_hidden is None or ema_decay <= 0.0:
            self._ema_hidden = hidden.detach()
        else:
            self._ema_hidden = ema_decay * self._ema_hidden + (1.0 - ema_decay) * hidden.detach()
        ball_embeddings = self._ema_hidden

        seed = int(torch.initial_seed() % (2 ** 32 - 1))
        gb_result = build_reliable_granular_balls(
            ball_embeddings,
            k=int(self._conf_value("k", 15)),
            min_ball_size=int(self._conf_value("min_ball_size", 3)),
            max_ball_size=int(self._conf_value("max_ball_size", 128)),
            random_state=seed,
        )
        assignment = gb_result.assignment
        if self.random_control:
            assignment = build_random_size_matched_assignment(assignment, random_state=seed + 17)

        soft_result = build_gb_soft_targets(
            noisy_label=self.noisy_label,
            clean_label=self.clean_label,
            probs=probs,
            assignment=assignment,
            train_mask=self.train_mask,
            n_classes=self.n_classes,
            alpha_max=self._conf_value("alpha_max", 0.5),
            gamma=self._conf_value("gamma", 0.5),
            eta_max=self._conf_value("eta_max", 0.4),
            min_train_per_ball=self._conf_value("min_train_per_ball", 2),
            full_train_per_ball=self._conf_value("full_train_per_ball", 4),
            label_prior_smoothing=self._conf_value("label_prior_smoothing", 1.0),
            target_mode=self._conf_value("target_mode", self.target_mode),
            suspicion_mode=self._conf_value("suspicion_mode", self.suspicion_mode),
            suspicion_top_ratio=self._conf_value("suspicion_top_ratio", 0.3),
            min_ball_conf=self._conf_value("min_ball_conf", 0.3),
            beta=self._conf_value("beta", 0.5),
            min_weight=self._conf_value("min_weight", 0.3),
        )
        self.soft_targets = soft_result.targets.detach()
        self.sample_weights = soft_result.weights.detach()
        self.gb_assignment = assignment
        self.gb_diagnostics = {
            **gb_result.summary,
            **soft_result.diagnostics,
            "random_control": int(self.random_control),
        }
        if self.conf.training['debug']:
            print("Reliable-GB diagnostics:", self.gb_diagnostics)

    @staticmethod
    def _weighted_soft_cross_entropy(logits, targets, weights):
        losses = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
        return (losses * weights).sum() / torch.clamp(weights.sum(), min=1e-12)

    def _append_diagnostics(self):
        if not self.gb_diagnostics:
            return
        os.makedirs("./runs", exist_ok=True)
        path = "./runs/reliablegb_gcn_diagnostics.csv"
        row = {
            "method": getattr(self, "method_name", self.conf.model.get("method", "reliablegb_gcn")),
            "dataset": self.dataset_name,
            "seed": getattr(self, "seed", ""),
            "noise_type": getattr(self, "noise_type", ""),
            "noise_rate": getattr(self, "noise_rate", ""),
            "train": self.result.get("train", ""),
            "valid": self.result.get("valid", ""),
            "test": self.result.get("test", ""),
            "total_time": self.total_time,
            "best_epoch": self.best_epoch,
            "best_phase": self.best_phase,
            **self.gb_diagnostics,
        }
        fieldnames = [
            "method", "dataset", "seed", "noise_type", "noise_rate",
            "train", "valid", "test", "total_time", "best_epoch", "best_phase", "random_control",
            "target_mode", "suspicion_mode",
            "num_balls", "mean_ball_size", "max_ball_size", "singleton_ratio", "mean_radius", "reliable_k",
            "alpha_mean", "alpha_max", "target_changed_ratio", "target_argmax_changed_ratio",
            "effective_coverage", "mean_q", "mean_train_coverage", "ncr", "cdr", "suspicion_auroc",
            "gated_ratio", "gated_noisy_precision", "gated_noisy_recall", "gated_clean_ratio",
            "weight_mean", "weight_min",
        ]
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        warmup_epochs = int(self._conf_value("warmup_epochs", self.conf.training.get("warmup_epochs", 30)))
        refresh_interval = int(self._conf_value("gb_refresh_interval", 20))
        n_epochs = int(self.conf.training['n_epochs'])
        if warmup_epochs <= 0:
            self._make_gb_targets()

        for epoch in range(n_epochs):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            features, adj = self.feats, self.adj
            use_soft_loss = self.soft_targets is not None
            phase = "gb" if use_soft_loss else "warmup"

            output = self.model(features, adj)
            if use_soft_loss:
                loss_train = self._weighted_soft_cross_entropy(
                    output[self.train_mask],
                    self.soft_targets[self.train_mask],
                    self.sample_weights[self.train_mask],
                )
            else:
                loss_train = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
            acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                    output[self.train_mask].detach().cpu().numpy())
            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.best_epoch = epoch + 1
                self.best_phase = phase
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            # First build at warmup, then PERIODIC rebuild so denoising tracks the representation.
            reached_warmup = epoch + 1 >= warmup_epochs
            if reached_warmup:
                first_build = self.soft_targets is None
                periodic = refresh_interval > 0 and ((epoch + 1 - warmup_epochs) % refresh_interval == 0)
                if first_build or periodic:
                    self._make_gb_targets()

            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Phase {:7s} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, phase, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        self._append_diagnostics()
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result


class reliablegb_gcn_reweight_Predictor(_ReliableGBGCNBase):
    random_control = False
    target_mode = "reweight"
    suspicion_mode = "gb"


class reliablegb_gcn_reweight_random_Predictor(_ReliableGBGCNBase):
    random_control = True
    target_mode = "reweight"
    suspicion_mode = "gb"


class reliablegb_gcn_reweight_noball_Predictor(_ReliableGBGCNBase):
    random_control = False
    target_mode = "reweight"
    suspicion_mode = "no_ball"
