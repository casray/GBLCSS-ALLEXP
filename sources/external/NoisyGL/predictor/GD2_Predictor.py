import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.GD2 import (
    GD2Model,
    build_graph_views,
    gmm_low_component_probability,
    prediction_discrepancy,
    select_purified_nodes,
    structural_classification_loss,
)


class gd2_Predictor(Predictor):
    """Paper reimplementation of GD² (NeurIPS 2025)."""

    def method_init(self, conf, data):
        self.model = GD2Model(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            n_classes=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=conf.training["lr"],
            weight_decay=conf.training["weight_decay"],
        )
        self.clean_threshold = float(conf.model["tau"])
        self.pseudo_threshold = float(conf.model["gamma"])
        self.train_idx = torch.as_tensor(
            self.train_mask, dtype=torch.long, device=self.device)
        self.ego_edges, self.structure_edges, self.mixed_edges = build_graph_views(
            self.adj.indices(), self.n_nodes)
        self.clean_probability = torch.ones(
            len(self.train_mask), dtype=torch.float, device=self.device)

    def _training_logits(self):
        ego = self.model.ego_logits(self.feats, self.ego_edges)
        structure = self.model.graph_logits(self.feats, self.structure_edges)
        mixed = self.model.graph_logits(self.feats, self.mixed_edges)
        return ego, structure, mixed

    def _select_clean_and_purified(self, ego_logits, structure_logits, mixed_logits, epoch):
        discrepancy = prediction_discrepancy(
            F.softmax(ego_logits[self.train_idx], dim=1),
            F.softmax(structure_logits[self.train_idx], dim=1),
        )
        probability = gmm_low_component_probability(
            discrepancy, int(getattr(self, "seed", 0)) + int(epoch))
        if torch.isfinite(probability).all():
            self.clean_probability = probability
        clean_local = self.clean_probability > self.clean_threshold
        if not bool(clean_local.any()):
            clean_local = torch.ones_like(clean_local, dtype=torch.bool)
        clean_idx = self.train_idx[clean_local]
        noisy_idx = self.train_idx[~clean_local]
        purified_idx, pseudo_label = select_purified_nodes(
            F.softmax(mixed_logits[noisy_idx], dim=1),
            noisy_idx,
            self.pseudo_threshold,
        )
        return clean_idx, purified_idx, pseudo_label

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            self.model.train()
            self.optim.zero_grad()
            ego_logits, structure_logits, mixed_logits = self._training_logits()
            clean_idx, purified_idx, pseudo_label = self._select_clean_and_purified(
                ego_logits, structure_logits, mixed_logits, epoch)

            loss_ego = F.cross_entropy(
                ego_logits[self.train_idx], self.noisy_label[self.train_idx])
            loss_structure = structural_classification_loss(
                structure_logits,
                clean_idx,
                self.noisy_label,
                purified_idx,
                pseudo_label,
            )
            loss_mixed = F.cross_entropy(
                mixed_logits[clean_idx], self.noisy_label[clean_idx])
            loss = loss_ego + loss_structure + loss_mixed
            loss.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.clean_label, self.val_mask)
            acc_train = self.metric(
                self.noisy_label[self.train_idx].cpu().numpy(),
                mixed_logits[self.train_idx].detach().cpu().numpy(),
            )
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                self.total_time = time.time() - self.start_time
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.graph_logits(self.feats, self.mixed_edges)
            loss = F.cross_entropy(logits[mask], label[mask])
            acc = self.metric(
                label[mask].cpu().numpy(), logits[mask].detach().cpu().numpy())
        return loss, acc
