import time
from copy import deepcopy

import nni
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor


def graphmlp_ncontrast(similarity, adjacency_label, tau=1.0):
    exponentiated = torch.exp(float(tau) * similarity)
    denominator = exponentiated.sum(dim=1).clamp_min(1e-12)
    numerator = (exponentiated * adjacency_label).sum(dim=1)
    return -torch.log(numerator / denominator + 1e-8).mean()


def graphmlp_batch_indices(train_idx, num_nodes, batch_size, generator=None):
    train_idx = torch.as_tensor(train_idx, dtype=torch.long).flatten().cpu()
    train_idx = torch.tensor(list(dict.fromkeys(train_idx.tolist())), dtype=torch.long)
    target_size = min(int(num_nodes), max(int(batch_size), int(train_idx.numel())))
    train_lookup = torch.zeros(int(num_nodes), dtype=torch.bool)
    train_lookup[train_idx] = True
    candidates = torch.nonzero(~train_lookup, as_tuple=False).flatten()
    needed = target_size - train_idx.numel()
    if needed > 0:
        permutation = torch.randperm(candidates.numel(), generator=generator)[:needed]
        sampled = candidates[permutation]
        return torch.cat([train_idx, sampled], dim=0)
    return train_idx


class GraphMLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward_with_hidden(self, features):
        hidden = self.fc1(features)
        hidden = F.gelu(hidden)
        hidden = self.layernorm(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return self.classifier(hidden), hidden

    def forward(self, features, adj=None):
        logits, _ = self.forward_with_hidden(features)
        return logits


class graphmlp_reimplemented_Predictor(Predictor):
    strategy = "graphmlp_reimplemented"

    def method_init(self, conf, data):
        self.model = GraphMLPNetwork(
            input_dim=conf.model["n_feat"],
            hidden_dim=conf.model["n_hidden"],
            num_classes=conf.model["n_classes"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        self.train_idx = torch.as_tensor(self.train_mask, dtype=torch.long, device=self.device)
        self.best_epoch = -1
        self._r_hop_adjacency = self._build_r_hop_adjacency(
            order=int(self.conf.model.get("graphmlp_order", 2))
        )
        self.graphfree_diagnostics = {
            "graphfree_strategy": self.strategy,
            "graphfree_paper_reimplementation": 1,
            "graphfree_official_code_port": 0,
            "graphfree_train_uses_graph": 1,
            "graphfree_inference_uses_graph": 0,
            "graphmlp_alpha": float(self.conf.model.get("graphmlp_alpha", 2.0)),
            "graphmlp_tau": float(self.conf.model.get("graphmlp_tau", 1.0)),
            "graphmlp_order": int(self.conf.model.get("graphmlp_order", 2)),
        }

    def _build_r_hop_adjacency(self, order):
        import scipy.sparse as sp

        adjacency = self.adj.coalesce().detach().cpu()
        indices = adjacency.indices().numpy()
        values = adjacency.values().numpy()
        matrix = sp.coo_matrix(
            (values, (indices[0], indices[1])),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()
        result = matrix.copy()
        for _ in range(max(1, int(order)) - 1):
            result = result @ matrix
        return result.tocsr()

    def _batch_indices(self, epoch):
        generator = torch.Generator().manual_seed(int(getattr(self, "seed", 0)) * 100003 + int(epoch))
        return graphmlp_batch_indices(
            self.train_idx.detach().cpu(),
            self.n_nodes,
            self.conf.model.get("graphmlp_batch_size", 2000),
            generator=generator,
        )

    def _ncontrast_loss(self, hidden, batch_idx):
        batch_cpu = batch_idx.detach().cpu().numpy()
        adjacency_label = self._r_hop_adjacency[batch_cpu][:, batch_cpu].toarray()
        adjacency_label = torch.as_tensor(
            adjacency_label,
            device=self.device,
            dtype=hidden.dtype,
        )
        normalized = F.normalize(hidden[batch_idx.to(self.device)], p=2, dim=1)
        similarity = normalized @ normalized.t()
        similarity.fill_diagonal_(0.0)
        return graphmlp_ncontrast(
            similarity,
            adjacency_label,
            tau=self.conf.model.get("graphmlp_tau", 1.0),
        )

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, None)
        loss, acc = None, None
        if label is not None and mask is not None:
            idx = torch.as_tensor(mask, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(output[idx], label[idx])
            acc = self.metric(label[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        return output, loss, acc

    def evaluate(self, label, mask):
        self.model.eval()
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, None, label, mask)
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        return self.evaluate(self.clean_label, mask)

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output, hidden = self.model.forward_with_hidden(self.feats)
            label_loss = F.cross_entropy(output[self.train_idx], self.noisy_label[self.train_idx])
            batch_idx = self._batch_indices(epoch).to(self.device)
            contrast_loss = self._ncontrast_loss(hidden, batch_idx)
            loss_train = label_loss + float(self.conf.model.get("graphmlp_alpha", 2.0)) * contrast_loss
            acc_train = self.metric(
                self.noisy_label[self.train_idx].cpu().numpy(),
                output[self.train_idx].detach().cpu().numpy(),
            )
            loss_train.backward()
            self.optim.step()

            self.graphfree_diagnostics.update(
                {
                    "graphmlp_last_label_loss": float(label_loss.detach().cpu()),
                    "graphmlp_last_ncontrast_loss": float(contrast_loss.detach().cpu()),
                    "graphmlp_effective_batch_size": int(batch_idx.numel()),
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
        return self.result
