import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.MLPNoiseBaselines_Predictor import mlp_forward_with_hidden
from predictor.module.GNNs import MLP


def _as_index(mask, device):
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


class _GraphFreeFamilyBase(Predictor):
    strategy = "graphfree"

    def method_init(self, conf, data):
        self.model = MLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        self.train_idx = _as_index(self.train_mask, self.device)
        self.edge_index_graphfree = self._clean_edge_index()
        self.best_epoch = -1
        self.graphfree_diagnostics = {
            "graphfree_train_uses_edges": 1,
            "graphfree_inference_uses_graph": 0,
            "graphfree_strategy": self.strategy,
        }
        self._init_extra_modules(conf)
        self.optim = torch.optim.Adam(
            self._optim_parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )

    def _conf_value(self, key, default):
        return self.conf.model.get(key, default)

    def _init_extra_modules(self, conf):
        return None

    def _optim_parameters(self):
        return self.model.parameters()

    def _clean_edge_index(self):
        edge_index = self.edge_index.to(self.device).long()
        if edge_index.numel() == 0:
            return edge_index.reshape(2, 0)
        keep = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, keep]
        if edge_index.shape[1] == 0:
            fallback_src = torch.arange(self.n_nodes, device=self.device)
            fallback_dst = torch.roll(fallback_src, shifts=-1)
            edge_index = torch.stack([fallback_src, fallback_dst], dim=0)
        return edge_index

    def _sample_edges(self, batch_size, edge_index=None):
        edge_index = self.edge_index_graphfree if edge_index is None else edge_index
        if edge_index.shape[1] == 0:
            raise ValueError("graph-free family methods require at least one edge for training-time regularization")
        batch_size = max(1, int(batch_size))
        sampled = torch.randint(edge_index.shape[1], (batch_size,), device=self.device)
        return edge_index[0, sampled], edge_index[1, sampled]

    def _sample_negative_nodes(self, batch_size):
        return torch.randint(int(self.n_nodes), (int(batch_size),), device=self.device)

    def _forward_with_hidden(self, features):
        return mlp_forward_with_hidden(self.model, features, None)

    def _extra_loss(self, features, logits, hidden, epoch):
        return logits.new_zeros(())

    def get_prediction(self, features, adj, label=None, mask=None):
        output, hidden = self._forward_with_hidden(features)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            idx = _as_index(mask, self.device)
            loss = self.loss_fn(output[idx], label[idx])
            acc = self.metric(label[idx].cpu().numpy(), output[idx].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self._set_extra_train_mode(True)
            self.optim.zero_grad()
            features = self.feats
            output, hidden = self._forward_with_hidden(features)
            loss_train = self.loss_fn(output[self.train_idx], self.noisy_label[self.train_idx])
            loss_train = loss_train + self._extra_loss(features, output, hidden, epoch)
            acc_train = self.metric(
                self.noisy_label[self.train_idx].cpu().numpy(),
                output[self.train_idx].detach().cpu().numpy(),
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
                self.weights = self._state_dict_snapshot()
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
        if self.conf.training["debug"]:
            print("Optimization Finished!")
            print("Time(s): {:.4f}".format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def _set_extra_train_mode(self, training: bool):
        return None

    def _state_dict_snapshot(self):
        return {"model": deepcopy(self.model.state_dict())}

    def _load_snapshot(self):
        if self.weights is not None:
            self.model.load_state_dict(self.weights["model"])

    def evaluate(self, label, mask):
        self.model.eval()
        self._set_extra_train_mode(False)
        with torch.no_grad():
            _, loss, acc = self.get_prediction(self.feats, None, label, mask)
        return loss, acc

    def test(self, mask):
        self._load_snapshot()
        return self.evaluate(self.clean_label, mask)


class gsdn_Predictor(_GraphFreeFamilyBase):
    strategy = "gsdn"

    def _init_extra_modules(self, conf):
        self.aux_model = MLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
        ).to(self.device)
        init = float(self._conf_value("mixup_init", 0.5))
        init = min(max(init, 1e-4), 1.0 - 1e-4)
        self.mixup_logit = torch.nn.Parameter(torch.logit(torch.tensor(init, device=self.device)))
        self._train_source_lookup = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        self._train_source_lookup[self.train_idx] = True
        self.graphfree_diagnostics.update({
            "gsdn_lambda_feature": float(self._conf_value("lambda_feature", 0.8)),
            "gsdn_lambda_label": float(self._conf_value("lambda_label", 0.8)),
            "gsdn_lambda_negative": float(self._conf_value("lambda_negative", 0.1)),
            "gsdn_graph_batch_size": int(self._conf_value("graph_batch_size", 512)),
        })

    def _optim_parameters(self):
        return list(self.model.parameters()) + list(self.aux_model.parameters()) + [self.mixup_logit]

    def _set_extra_train_mode(self, training: bool):
        self.aux_model.train(training)

    def _state_dict_snapshot(self):
        return {
            "model": deepcopy(self.model.state_dict()),
            "aux_model": deepcopy(self.aux_model.state_dict()),
            "mixup_logit": self.mixup_logit.detach().clone(),
        }

    def _load_snapshot(self):
        if self.weights is not None:
            self.model.load_state_dict(self.weights["model"])
            self.aux_model.load_state_dict(self.weights["aux_model"])
            with torch.no_grad():
                self.mixup_logit.copy_(self.weights["mixup_logit"])

    def _mixup_coeff(self):
        return torch.sigmoid(self.mixup_logit).clamp(1e-6, 1.0 - 1e-6)

    def _label_edge_indices(self):
        src, dst = self.edge_index_graphfree
        keep = self._train_source_lookup[src]
        return src[keep], dst[keep]

    def _extra_loss(self, features, logits, hidden, epoch):
        batch_size = int(self._conf_value("graph_batch_size", 512))
        src, dst = self._sample_edges(batch_size)
        neg = self._sample_negative_nodes(src.shape[0])
        coeff = self._mixup_coeff()
        mixed_x = coeff * features[src] + (1.0 - coeff) * features[dst]
        mixed_logits = self.model(mixed_x, None)
        aux_logits = self.aux_model(features, None)
        with torch.no_grad():
            soft_src = F.softmax(aux_logits[src], dim=1)
            soft_dst = F.softmax(aux_logits[dst], dim=1)
            mixed_target = coeff * soft_src + (1.0 - coeff) * soft_dst
        feature_loss = F.kl_div(
            F.log_softmax(mixed_logits, dim=1),
            mixed_target,
            reduction="batchmean",
        )

        main_probs = F.softmax(logits[src], dim=1)
        neg_probs = F.softmax(aux_logits[neg], dim=1)
        negative_loss = (main_probs * neg_probs).sum(dim=1).mean()

        label_src, label_dst = self._label_edge_indices()
        if label_src.numel() == 0:
            label_loss = logits.new_zeros(())
        else:
            sample_idx = torch.randint(label_src.shape[0], (batch_size,), device=self.device)
            label_loss = F.cross_entropy(
                aux_logits[label_dst[sample_idx]],
                self.noisy_label[label_src[sample_idx]],
            )

        self.graphfree_diagnostics.update({
            "gsdn_last_feature_loss": float(feature_loss.detach().cpu()),
            "gsdn_last_label_loss": float(label_loss.detach().cpu()),
            "gsdn_last_negative_loss": float(negative_loss.detach().cpu()),
            "gsdn_mixup_coeff": float(coeff.detach().cpu()),
        })
        return (
            float(self._conf_value("lambda_feature", 0.8)) * feature_loss
            + float(self._conf_value("lambda_label", 0.8)) * label_loss
            + float(self._conf_value("lambda_negative", 0.1)) * negative_loss
        )


class graphmlp_Predictor(_GraphFreeFamilyBase):
    strategy = "graphmlp"

    def _init_extra_modules(self, conf):
        self.graphfree_diagnostics.update({
            "graphmlp_lambda_contrast": float(self._conf_value("lambda_contrast", 0.5)),
            "graphmlp_tau": float(self._conf_value("tau", 0.5)),
            "graphmlp_graph_batch_size": int(self._conf_value("graph_batch_size", 512)),
        })

    def _graph_contrast_loss(self, hidden):
        batch_size = int(self._conf_value("graph_batch_size", 512))
        src, dst = self._sample_edges(batch_size)
        neg = self._sample_negative_nodes(src.shape[0])
        z = F.normalize(hidden, p=2, dim=1)
        tau = max(float(self._conf_value("tau", 0.5)), 1e-6)
        pos = (z[src] * z[dst]).sum(dim=1) / tau
        neg_sim = (z[src] * z[neg]).sum(dim=1) / tau
        logits = torch.stack([pos, neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def _extra_loss(self, features, logits, hidden, epoch):
        contrast = self._graph_contrast_loss(hidden)
        self.graphfree_diagnostics["graphmlp_last_contrast_loss"] = float(contrast.detach().cpu())
        return float(self._conf_value("lambda_contrast", 0.5)) * contrast
