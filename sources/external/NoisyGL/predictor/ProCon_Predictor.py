import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.module.ProCon import (
    ProConEncoder,
    ProjectionPredictor,
    build_peer_index_bounded,
    build_peer_index_exact,
    copy_frozen,
    ema_update,
    gmm_clean_probability,
    make_graph_view,
    negative_cosine_similarity,
    update_prototypes_ema,
)


class procon_Predictor(Predictor):
    """Paper reimplementation of ProCon (IJCAI 2025)."""

    def method_init(self, conf, data):
        hidden = int(conf.model["n_hidden"])
        self.model = ProConEncoder(
            in_channels=conf.model["n_feat"],
            hidden_channels=hidden,
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.target_encoder = copy_frozen(self.model).to(self.device)
        self.classifier = nn.Linear(hidden, conf.model["n_classes"]).to(self.device)
        self.ssl_predictor = ProjectionPredictor(hidden).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.classifier.parameters())
            + list(self.ssl_predictor.parameters()),
            lr=conf.training["lr"],
            weight_decay=conf.training["weight_decay"],
        )
        self.peer_k = int(conf.model["peer_k"])
        self.peer_alpha = float(conf.model["alpha"])
        self.clean_threshold = float(conf.model["clean_threshold"])
        self.prototype_momentum = float(conf.model["prototype_momentum"])
        self.target_momentum = float(conf.model["target_momentum"])
        self.lambda_supervised = float(conf.model["lambda_supervised"])
        self.feature_drop = float(conf.model["feature_drop"])
        self.edge_drop = float(conf.model["edge_drop"])
        self.edge_index = self.adj.indices()
        self.train_idx = torch.as_tensor(
            self.train_mask, dtype=torch.long, device=self.device)
        seed = int(getattr(self, "seed", torch.initial_seed() % (2 ** 31)))
        exact_max_nodes = int(conf.model["exact_max_nodes"])
        if self.n_nodes <= exact_max_nodes:
            self.peer_index = build_peer_index_exact(
                self.feats, self.edge_index, self.peer_k, self.peer_alpha)
            self.peer_mode = "exact"
        else:
            self.peer_index = build_peer_index_bounded(
                self.feats,
                self.edge_index,
                self.peer_k,
                self.peer_alpha,
                int(conf.model["candidate_limit"]),
                seed,
            )
            self.peer_mode = "bounded"
        self.prototypes = None
        self.best_classifier_weights = None
        self.clean_probability = torch.ones(
            len(self.train_idx), dtype=torch.float, device=self.device)

    def _initialize_prototypes(self, embeddings):
        prototypes = embeddings.new_zeros((self.n_classes, embeddings.shape[1]))
        for class_id in range(self.n_classes):
            mask = self.noisy_label[self.train_idx] == class_id
            if bool(mask.any()):
                prototypes[class_id] = embeddings[self.train_idx[mask]].mean(dim=0)
        self.prototypes = prototypes.detach()

    def _clean_selection(self, embeddings, epoch):
        if self.prototypes is None:
            self._initialize_prototypes(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)
        pseudo_label = (normalized_embeddings @ normalized_prototypes.t()).argmax(dim=1)
        peer_label = pseudo_label[self.peer_index]
        noisy_train = self.noisy_label[self.train_idx]
        consistency = (
            peer_label[self.train_idx] == noisy_train.unsqueeze(1)
        ).float().mean(dim=1)
        probability = gmm_clean_probability(
            consistency, int(getattr(self, "seed", 0)) + int(epoch))
        if torch.isfinite(probability).all():
            self.clean_probability = probability
        selected = self.clean_probability > self.clean_threshold
        if not bool(selected.any()):
            selected = torch.ones_like(selected, dtype=torch.bool)
        return selected

    def _ssl_loss(self):
        features_one, edges_one = make_graph_view(
            self.feats, self.edge_index, self.feature_drop, self.edge_drop)
        features_two, edges_two = make_graph_view(
            self.feats, self.edge_index, self.feature_drop, self.edge_drop)
        online_one = self.ssl_predictor(self.model(features_one, edges_one))
        online_two = self.ssl_predictor(self.model(features_two, edges_two))
        with torch.no_grad():
            target_one = self.target_encoder(features_one, edges_one)
            target_two = self.target_encoder(features_two, edges_two)
        return 0.5 * (
            negative_cosine_similarity(online_one, target_two)
            + negative_cosine_similarity(online_two, target_one)
        )

    def _forward_logits(self):
        embeddings = self.model(self.feats, self.edge_index)
        return embeddings, self.classifier(embeddings)

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            self.model.train()
            self.classifier.train()
            self.ssl_predictor.train()
            self.optim.zero_grad()

            embeddings, logits = self._forward_logits()
            selected = self._clean_selection(embeddings.detach(), epoch)
            selected_idx = self.train_idx[selected]
            supervised_loss = F.cross_entropy(
                logits[selected_idx], self.noisy_label[selected_idx])
            ssl_loss = self._ssl_loss()
            loss = (
                self.lambda_supervised * supervised_loss
                + (1.0 - self.lambda_supervised) * ssl_loss
            )
            loss.backward()
            self.optim.step()
            ema_update(self.target_encoder, self.model, self.target_momentum)

            with torch.no_grad():
                self.prototypes = update_prototypes_ema(
                    self.prototypes,
                    embeddings[self.train_idx],
                    self.noisy_label[self.train_idx],
                    self.clean_probability * selected.float(),
                    self.prototype_momentum,
                    self.n_classes,
                )
            loss_val, acc_val = self.evaluate(self.clean_label, self.val_mask)
            acc_train = self.metric(
                self.noisy_label[self.train_idx].cpu().numpy(),
                logits[self.train_idx].detach().cpu().numpy(),
            )
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                self.total_time = time.time() - self.start_time
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.weights = deepcopy(self.model.state_dict())
                self.best_classifier_weights = deepcopy(self.classifier.state_dict())
            elif flag_earlystop:
                break

        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            _, logits = self._forward_logits()
            loss = F.cross_entropy(logits[mask], label[mask])
            acc = self.metric(
                label[mask].cpu().numpy(), logits[mask].detach().cpu().numpy())
        return loss, acc

    def test(self, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        if self.best_classifier_weights is not None:
            self.classifier.load_state_dict(self.best_classifier_weights)
        return self.evaluate(self.clean_label, mask)
