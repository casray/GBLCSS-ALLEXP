import time
from copy import deepcopy

import nni
import torch.nn as nn
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN, MLP


def _as_index(mask, device):
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


def _entropy(probs):
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1)


def ffg2m_low_frequency_loss(student_logits, teacher_logits, edge_index, num_nodes):
    edge_index = edge_index.to(student_logits.device).long()
    if edge_index.numel() == 0:
        edge_index = edge_index.reshape(2, 0)
    else:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    nodes = torch.arange(int(num_nodes), device=student_logits.device)
    loop_index = torch.stack([nodes, nodes], dim=0)
    edge_with_loops = torch.cat([edge_index, loop_index], dim=1)
    src, dst = edge_with_loops[0], edge_with_loops[1]
    return F.kl_div(
        F.log_softmax(student_logits[src], dim=1),
        F.log_softmax(teacher_logits.detach()[dst], dim=1),
        reduction="batchmean",
        log_target=True,
    )


def ffg2m_high_frequency_loss(student_logits, teacher_logits, edge_index, tau=1.0):
    edge_index = edge_index.to(student_logits.device).long()
    if edge_index.numel() == 0:
        return student_logits.new_zeros(())
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] == 0:
        return student_logits.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    tau = max(float(tau), 1e-6)
    student_difference = (student_logits[src] - student_logits[dst]).abs()
    teacher_difference = (teacher_logits.detach()[src] - teacher_logits.detach()[dst]).abs()
    return F.kl_div(
        F.log_softmax(student_difference / tau, dim=1),
        F.log_softmax(teacher_difference / tau, dim=1),
        reduction="batchmean",
        log_target=True,
    )


class _G2MAdaptedBase(Predictor):
    strategy = "g2m_adapted"

    def method_init(self, conf, data):
        self.model = MLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        self.train_idx = _as_index(self.train_mask, self.device)
        self.best_epoch = -1
        self.run_status = "ok"
        self.run_error = ""
        self.g2m_diagnostics = {
            "g2m_strategy": self.strategy,
            "g2m_adapted_to_protocol": 1,
            "g2m_train_uses_graph": 1,
            "g2m_inference_uses_graph": 0,
            "g2m_uses_unlabeled_nodes": 1,
        }

    def _conf_value(self, key, default):
        return self.conf.model.get(key, default)

    def _teacher_model(self):
        return GCN(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=int(self._conf_value("teacher_hidden", self.conf.model["n_hidden"])),
            out_channels=self.n_classes,
            n_layers=int(self._conf_value("teacher_n_layer", self.conf.model["n_layer"])),
            dropout=float(self._conf_value("teacher_dropout", self.conf.model["dropout"])),
            norm_info=self.conf.model.get("norm_info"),
            act=self.conf.model.get("act", "F.relu"),
            input_layer=self.conf.model.get("input_layer", False),
            output_layer=self.conf.model.get("output_layer", False),
        ).to(self.device)

    def _train_teacher(self):
        teacher = self._teacher_model()
        optim = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", self.conf.training["lr"])),
            weight_decay=float(self._conf_value("teacher_weight_decay", self.conf.training["weight_decay"])),
        )
        teacher_epochs = int(self._conf_value("teacher_epochs", self.conf.training["n_epochs"]))
        best_val = -1.0
        best_epoch = -1
        best_state = None
        for epoch in range(teacher_epochs):
            teacher.train()
            optim.zero_grad()
            logits = teacher(self.feats, self.adj)
            loss = self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
            loss.backward()
            optim.step()

            teacher.eval()
            with torch.no_grad():
                logits = teacher(self.feats, self.adj)
                val_acc = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
            if val_acc >= best_val:
                best_val = float(val_acc)
                best_epoch = epoch + 1
                best_state = deepcopy(teacher.state_dict())
        if best_state is not None:
            teacher.load_state_dict(best_state)
        teacher.eval()
        with torch.no_grad():
            best_logits = teacher(self.feats, self.adj)
            best_test = self.metric(
                self.clean_label[self.test_mask].cpu().numpy(),
                best_logits[self.test_mask].detach().cpu().numpy(),
            )
        self.g2m_diagnostics.update({
            "teacher_epochs": int(teacher_epochs),
            "teacher_best_epoch": int(best_epoch),
            "teacher_best_val": float(best_val),
            "teacher_best_test": float(best_test),
        })
        return teacher

    def _teacher_probs(self, teacher):
        teacher.eval()
        temperature = float(self._conf_value("distill_temperature", 1.0))
        with torch.no_grad():
            logits = teacher(self.feats, self.adj)
            self.teacher_logits = logits.detach()
            probs = F.softmax(logits / max(temperature, 1e-6), dim=1).detach()
        self.g2m_diagnostics["distill_temperature"] = float(temperature)
        return probs, temperature

    def _distill_weights(self, teacher):
        return torch.ones(self.n_nodes, device=self.device)

    def _extra_student_loss(self, output, teacher_probs, temperature):
        return output.new_zeros(())

    def _student_loss(self, output, teacher_probs, weights, temperature):
        log_prob = F.log_softmax(output / max(temperature, 1e-6), dim=1)
        per_node = -(teacher_probs * log_prob).sum(dim=1) * (temperature ** 2)
        weights = weights.to(output.device, dtype=output.dtype).detach().clamp_min(0.0)
        if float(weights.sum().detach().cpu()) <= 1e-12:
            weights = torch.ones_like(weights)
        weights = weights / weights.mean().clamp_min(1e-12)
        kd_loss = (per_node * weights).mean()
        ce_weight = float(self._conf_value("lambda_ce", 0.0))
        ce_loss = output.new_zeros(())
        if ce_weight > 0:
            ce_loss = self.loss_fn(output[self.train_mask], self.noisy_label[self.train_mask])
        return (
            float(self._conf_value("lambda_kd", 1.0)) * kd_loss
            + ce_weight * ce_loss
            + self._extra_student_loss(output, teacher_probs, temperature)
        )

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, None)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        teacher = self._train_teacher()
        teacher_probs, temperature = self._teacher_probs(teacher)
        weights = self._distill_weights(teacher)
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output = self.model(self.feats, None)
            loss_train = self._student_loss(output, teacher_probs, weights, temperature)
            acc_train = self.metric(
                self.noisy_label[self.train_mask].cpu().numpy(),
                output[self.train_mask].detach().cpu().numpy(),
            )
            loss_train.backward()
            self.optim.step()

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
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
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
        self.g2m_diagnostics["student_best_epoch"] = int(self.best_epoch)
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, None, label, mask)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        return self.evaluate(self.clean_label, mask)


class ffg2m_adapted_Predictor(_G2MAdaptedBase):
    strategy = "ffg2m_adapted"

    def _edge_index_for_frequency(self):
        edge_index = self.edge_index.to(self.device).long()
        if edge_index.numel() == 0:
            return edge_index.reshape(2, 0)
        keep = edge_index[0] != edge_index[1]
        return edge_index[:, keep]

    def _ffg2m_frequency_loss(self, student_logits, teacher_probs):
        edge_index = self._edge_index_for_frequency()
        if edge_index.shape[1] == 0:
            return student_logits.new_zeros(())
        src, dst = edge_index[0], edge_index[1]
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = teacher_probs.detach()
        student_low = (student_probs[src] * student_probs[dst]).sum(dim=1)
        teacher_low = (teacher_probs[src] * teacher_probs[dst]).sum(dim=1)
        student_high = (student_probs[src] - student_probs[dst]).pow(2).sum(dim=1).sqrt()
        teacher_high = (teacher_probs[src] - teacher_probs[dst]).pow(2).sum(dim=1).sqrt()
        low_loss = F.mse_loss(student_low, teacher_low)
        high_loss = F.mse_loss(student_high, teacher_high)
        self.g2m_diagnostics.update({
            "ffg2m_last_low_loss": float(low_loss.detach().cpu()),
            "ffg2m_last_high_loss": float(high_loss.detach().cpu()),
        })
        return (
            float(self._conf_value("lambda_low", 0.5)) * low_loss
            + float(self._conf_value("lambda_high", 0.5)) * high_loss
        )

    def _extra_student_loss(self, output, teacher_probs, temperature):
        return self._ffg2m_frequency_loss(output, teacher_probs)


class ffg2m_reimplemented_Predictor(_G2MAdaptedBase):
    strategy = "ffg2m_reimplemented"

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.g2m_diagnostics.update(
            {
                "g2m_paper_reimplementation": 1,
                "g2m_official_code_port": 0,
                "ffg2m_lambda": float(self._conf_value("ffg2m_lambda", 0.4)),
                "ffg2m_tau": float(self._conf_value("ffg2m_tau", 1.5)),
            }
        )

    def _student_loss(self, output, teacher_probs, weights, temperature):
        if not hasattr(self, "teacher_logits"):
            raise RuntimeError("FF-G2M teacher logits must be prepared before student training")
        label_loss = self.loss_fn(
            output[self.train_mask],
            self.noisy_label[self.train_mask],
        )
        low_loss = ffg2m_low_frequency_loss(
            output,
            self.teacher_logits,
            self.edge_index,
            self.n_nodes,
        )
        high_loss = ffg2m_high_frequency_loss(
            output,
            self.teacher_logits,
            self.edge_index,
            tau=self._conf_value("ffg2m_tau", 1.5),
        )
        label_weight = float(self._conf_value("ffg2m_lambda", 0.4))
        self.g2m_diagnostics.update(
            {
                "ffg2m_last_label_loss": float(label_loss.detach().cpu()),
                "ffg2m_last_low_loss": float(low_loss.detach().cpu()),
                "ffg2m_last_high_loss": float(high_loss.detach().cpu()),
            }
        )
        return label_weight * label_loss + (1.0 - label_weight) * (low_loss + high_loss)


class krd_adapted_Predictor(_G2MAdaptedBase):
    strategy = "krd_adapted"

    def _estimate_teacher_reliability(self, teacher):
        teacher.eval()
        perturbations = max(1, int(self._conf_value("reliability_perturbations", 4)))
        noise_std = float(self._conf_value("reliability_noise_std", 0.01))
        with torch.no_grad():
            base_probs = F.softmax(teacher(self.feats, self.adj), dim=1)
            base_entropy = _entropy(base_probs)
            diffs = []
            dense_feats = self.feats.to_dense() if getattr(self.feats, "is_sparse", False) else self.feats
            for _ in range(perturbations):
                perturbed = dense_feats + torch.randn_like(dense_feats) * noise_std
                probs = F.softmax(teacher(perturbed, self.adj), dim=1)
                diffs.append((_entropy(probs) - base_entropy).abs())
            instability = torch.stack(diffs, dim=0).mean(dim=0)
            reliability = 1.0 / (instability + 1e-6)
            min_value = reliability.min()
            max_value = reliability.max()
            if float((max_value - min_value).detach().cpu()) <= 1e-12:
                normalized = torch.ones_like(reliability)
            else:
                normalized = (reliability - min_value) / (max_value - min_value)
        return normalized.detach()

    def _distill_weights(self, teacher):
        reliability = self._estimate_teacher_reliability(teacher)
        fraction = float(self._conf_value("reliable_fraction", 0.5))
        fraction = min(max(fraction, 1.0 / max(1, self.n_nodes)), 1.0)
        k = max(1, int(round(float(self.n_nodes) * fraction)))
        selected = torch.topk(reliability, k, largest=True).indices
        weights = torch.zeros_like(reliability)
        weights[selected] = reliability[selected].clamp_min(1e-6)
        self.g2m_diagnostics.update({
            "krd_reliable_fraction": float(fraction),
            "krd_selected_nodes": int(k),
            "krd_reliability_mean": float(reliability.mean().detach().cpu()),
            "krd_reliability_min": float(reliability.min().detach().cpu()),
            "krd_reliability_max": float(reliability.max().detach().cpu()),
        })
        return weights


class _MLPWithHidden(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout=0.5, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.dropout = float(dropout)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if int(n_layers) <= 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(int(n_layers) - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward_with_hidden(self, x):
        hidden = None
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            hidden = x
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lins[-1](x)
        if hidden is None:
            hidden = logits
        return logits, hidden

    def forward(self, x, adj=None):
        logits, _ = self.forward_with_hidden(x)
        return logits


class _VQGraphTeacher(nn.Module):
    def __init__(self, base_gcn, hidden_channels, codebook_size, token_temperature=1.0):
        super().__init__()
        self.base_gcn = base_gcn
        self.codebook = nn.Parameter(torch.randn(int(codebook_size), int(hidden_channels)) * 0.02)
        self.token_temperature = float(token_temperature)

    def forward(self, feats, adj):
        logits, hidden = self.base_gcn.forward_with_hidden(feats, adj)
        assignments = self.soft_assignments(hidden)
        codebook = F.normalize(self.codebook, p=2, dim=1)
        quantized = assignments @ codebook
        return logits, hidden, assignments, quantized

    def soft_assignments(self, hidden):
        hidden = F.normalize(hidden, p=2, dim=1)
        codebook = F.normalize(self.codebook, p=2, dim=1)
        logits = hidden @ codebook.t()
        return F.softmax(logits / max(self.token_temperature, 1e-6), dim=1)


def _edge_index_from_sparse_adj(adj, device):
    if getattr(adj, "is_sparse", False):
        edge_index = adj.coalesce().indices()
    else:
        edge_index = adj
    edge_index = edge_index.to(device).long()
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    return edge_index[:, edge_index[0] != edge_index[1]]


class vqgraph_adapted_Predictor(_G2MAdaptedBase):
    strategy = "vqgraph_adapted"

    def method_init(self, conf, data):
        self.model = _MLPWithHidden(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        self.train_idx = _as_index(self.train_mask, self.device)
        self.best_epoch = -1
        self.run_status = "ok"
        self.run_error = ""
        self.teacher_token_assignments = None
        self.teacher_codebook = None
        self.g2m_diagnostics = {
            "g2m_strategy": self.strategy,
            "g2m_adapted_to_protocol": 1,
            "g2m_official_reproduction": 0,
            "g2m_train_uses_graph": 1,
            "g2m_inference_uses_graph": 0,
            "g2m_uses_unlabeled_nodes": 1,
        }

    def _teacher_model(self):
        hidden = int(self._conf_value("teacher_hidden", self.conf.model["n_hidden"]))
        base = GCN(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=hidden,
            out_channels=self.n_classes,
            n_layers=int(self._conf_value("teacher_n_layer", self.conf.model["n_layer"])),
            dropout=float(self._conf_value("teacher_dropout", self.conf.model["dropout"])),
            norm_info=self.conf.model.get("norm_info"),
            act=self.conf.model.get("act", "F.relu"),
            input_layer=self.conf.model.get("input_layer", False),
            output_layer=self.conf.model.get("output_layer", False),
        )
        return _VQGraphTeacher(
            base_gcn=base,
            hidden_channels=hidden,
            codebook_size=int(self._conf_value("codebook_size", 128)),
            token_temperature=float(self._conf_value("token_temperature", 1.0)),
        ).to(self.device)

    def _vq_edge_loss(self, quantized):
        edge_index = _edge_index_from_sparse_adj(self.adj, self.device)
        if edge_index.shape[1] == 0:
            return quantized.new_zeros(())
        max_edges = int(self._conf_value("vq_edge_sample", 4096))
        if edge_index.shape[1] > max_edges:
            perm = torch.randperm(edge_index.shape[1], device=self.device)[:max_edges]
            edge_index = edge_index[:, perm]
        src, dst = edge_index[0], edge_index[1]
        q = F.normalize(quantized, p=2, dim=1)
        pos_logits = (q[src] * q[dst]).sum(dim=1)
        neg_src = torch.randint(0, self.n_nodes, (src.numel(),), device=self.device)
        neg_dst = torch.randint(0, self.n_nodes, (src.numel(),), device=self.device)
        neg_logits = (q[neg_src] * q[neg_dst]).sum(dim=1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def _train_teacher(self):
        teacher = self._teacher_model()
        optim = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", self.conf.training["lr"])),
            weight_decay=float(self._conf_value("teacher_weight_decay", self.conf.training["weight_decay"])),
        )
        teacher_epochs = int(self._conf_value("teacher_epochs", self.conf.training["n_epochs"]))
        lambda_node = float(self._conf_value("lambda_node", 0.001))
        lambda_edge = float(self._conf_value("lambda_edge", 0.03))
        best_val = -1.0
        best_test = -1.0
        best_epoch = -1
        best_state = None
        for epoch in range(teacher_epochs):
            teacher.train()
            optim.zero_grad()
            logits, hidden, assignments, quantized = teacher(self.feats, self.adj)
            ce_loss = self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
            node_loss = F.mse_loss(F.normalize(hidden, p=2, dim=1), quantized)
            edge_loss = self._vq_edge_loss(quantized)
            loss = ce_loss + lambda_node * node_loss + lambda_edge * edge_loss
            loss.backward()
            optim.step()

            teacher.eval()
            with torch.no_grad():
                logits, _, _, _ = teacher(self.feats, self.adj)
                val_acc = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
                test_acc = self.metric(
                    self.clean_label[self.test_mask].cpu().numpy(),
                    logits[self.test_mask].detach().cpu().numpy(),
                )
            if val_acc >= best_val:
                best_val = float(val_acc)
                best_test = float(test_acc)
                best_epoch = epoch + 1
                best_state = deepcopy(teacher.state_dict())
        if best_state is not None:
            teacher.load_state_dict(best_state)
        teacher.eval()
        with torch.no_grad():
            _, _, assignments, _ = teacher(self.feats, self.adj)
            self.teacher_token_assignments = assignments.detach()
            self.teacher_codebook = F.normalize(teacher.codebook.detach(), p=2, dim=1)
        self.g2m_diagnostics.update({
            "teacher_epochs": int(teacher_epochs),
            "teacher_best_epoch": int(best_epoch),
            "teacher_best_val": float(best_val),
            "teacher_best_test": float(best_test),
            "vqgraph_codebook_size": int(self.teacher_codebook.shape[0]),
            "vqgraph_lambda_node": float(lambda_node),
            "vqgraph_lambda_edge": float(lambda_edge),
        })
        return teacher

    def _teacher_probs(self, teacher):
        teacher.eval()
        temperature = float(self._conf_value("distill_temperature", 1.0))
        with torch.no_grad():
            logits, _, assignments, _ = teacher(self.feats, self.adj)
            probs = F.softmax(logits / max(temperature, 1e-6), dim=1).detach()
            self.teacher_token_assignments = assignments.detach()
            self.teacher_codebook = F.normalize(teacher.codebook.detach(), p=2, dim=1)
        self.g2m_diagnostics["distill_temperature"] = float(temperature)
        return probs, temperature

    def _student_token_loss(self, hidden):
        if self.teacher_token_assignments is None or self.teacher_codebook is None:
            return hidden.new_zeros(())
        hidden = F.normalize(hidden, p=2, dim=1)
        logits = hidden @ self.teacher_codebook.to(hidden.device).t()
        log_probs = F.log_softmax(logits / max(float(self._conf_value("token_temperature", 1.0)), 1e-6), dim=1)
        target = self.teacher_token_assignments.to(hidden.device).detach()
        return F.kl_div(log_probs, target, reduction="batchmean")

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, None)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        teacher = self._train_teacher()
        teacher_probs, temperature = self._teacher_probs(teacher)
        weights = self._distill_weights(teacher)
        token_weight = float(self._conf_value("lambda_token", 1.0))
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output, hidden = self.model.forward_with_hidden(self.feats)
            loss_train = self._student_loss(output, teacher_probs, weights, temperature)
            loss_train = loss_train + token_weight * self._student_token_loss(hidden)
            acc_train = self.metric(
                self.noisy_label[self.train_mask].cpu().numpy(),
                output[self.train_mask].detach().cpu().numpy(),
            )
            loss_train.backward()
            self.optim.step()

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
                    "Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1,
                        time.time() - t0,
                        loss_train.item(),
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
        self.g2m_diagnostics["student_best_epoch"] = int(self.best_epoch)
        return self.result
