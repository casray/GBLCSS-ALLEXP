import importlib.util
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from torch_geometric.nn import GCNConv

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
from predictor.module.vqgraph_neighbor_sampling import (
    CpuNeighborSampler,
    DGLBothNormGraphConv,
    bidirected_with_self_loops,
    local_dense_adjacency,
)


KRD_SOURCE_COMMIT = "32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21"
VQGRAPH_SOURCE_COMMIT = "60aa1c99782d542c316a37f0394eba934bcf97b0"


def _load_official_vqgraph_quantizer():
    module_name = "vqgraph_official_vq"
    if module_name in sys.modules:
        return sys.modules[module_name].VectorQuantize
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party_reference"
        / "VQGraph"
        / "vq.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the official VQGraph quantizer from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.VectorQuantize


def _as_index(mask, device):
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


def _edge_index(adj, device):
    if getattr(adj, "is_sparse", False):
        edge_index = adj.coalesce().indices()
    else:
        edge_index = adj
    return edge_index.to(device=device, dtype=torch.long)


def _add_self_pairs(pairs, num_nodes, device):
    nodes = torch.arange(int(num_nodes), device=device)
    loops = torch.stack([nodes, nodes], dim=0)
    return torch.cat([pairs, loops], dim=1)


def krd_edge_kl(student_logits, teacher_logits, pairs, tau):
    tau = max(float(tau), 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits[pairs[0]] / tau, dim=1),
        F.log_softmax(teacher_logits.detach()[pairs[1]] / tau, dim=1),
        reduction="batchmean",
        log_target=True,
    )


def sample_krd_pairs(nonself_pairs, reliability, num_nodes):
    reliability = reliability.to(nonself_pairs.device).clamp(0.0, 1.0)
    keep = torch.bernoulli(reliability[nonself_pairs[1]]).bool()
    sampled = nonself_pairs[:, keep]
    return _add_self_pairs(sampled, num_nodes, nonself_pairs.device)


def vqgraph_token_kl(projected_student, codebook, teacher_similarities, temperature):
    temperature = max(float(temperature), 1e-6)
    student_scores = -torch.cdist(projected_student, codebook, p=2) / temperature
    teacher_probs = F.softmax(teacher_similarities.detach() / temperature, dim=-1)
    return F.kl_div(
        F.log_softmax(student_scores, dim=-1),
        teacher_probs,
        reduction="batchmean",
    ) * temperature**2


class _OfficialMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        dropout,
        norm_type="none",
        token_projection_dim=None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.norm_type = str(norm_type)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        n_layers = int(n_layers)
        if n_layers <= 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
            if self.norm_type == "batch":
                self.norms.extend(nn.BatchNorm1d(hidden_channels) for _ in range(n_layers - 1))
            elif self.norm_type == "layer":
                self.norms.extend(nn.LayerNorm(hidden_channels) for _ in range(n_layers - 1))
        self.token_projection = None
        if token_projection_dim is not None and n_layers > 1:
            self.token_projection = nn.Linear(hidden_channels, int(token_projection_dim))

    def forward_components(self, features):
        hidden = None
        projected = None
        x = features
        if getattr(x, "is_sparse", False):
            x = x.to_dense()
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if layer_idx < len(self.layers) - 1:
                hidden = x
                if self.norm_type != "none":
                    x = self.norms[layer_idx](x)
                x = F.relu(x)
                x = self.dropout(x)
                if self.token_projection is not None:
                    projected = self.token_projection(x)
        if hidden is None:
            hidden = x
        return x, hidden, projected

    def forward(self, features, adj=None):
        logits, _, _ = self.forward_components(features)
        return logits


class _FaithfulG2MBase(Predictor):
    strategy = "faithful_protocol_port"

    def _conf_value(self, key, default):
        return self.conf.model.get(key, default)

    def get_prediction(self, features, adj, label=None, mask=None):
        output = self.model(features, None)
        loss, acc = None, None
        if label is not None and mask is not None:
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
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

    def _finish_student(self):
        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        return self.result


class _KRDReliability:
    def __init__(self, init_power=1.0, momentum=0.99, bins=50, noise_level=1.0):
        self.power = float(init_power)
        self.momentum = float(momentum)
        self.bins = int(bins)
        self.noise_level = float(noise_level)
        self.delta_entropy = None
        self.update_failures = 0

    @staticmethod
    def _entropy(probs):
        probs = probs.clamp_min(1e-12)
        return -(probs * probs.log()).sum(dim=1)

    def initialize(self, teacher, features, adj):
        teacher.eval()
        dense_features = features.to_dense() if getattr(features, "is_sparse", False) else features
        with torch.no_grad():
            base = F.softmax(teacher(dense_features, adj), dim=1)
            perturbed = dense_features + torch.randn_like(dense_features) * self.noise_level
            noisy = F.softmax(teacher(perturbed, adj), dim=1)
            delta = (self._entropy(noisy) - self._entropy(base)).abs()
            maximum = delta.max()
            if float(maximum.detach().cpu()) <= 1e-12:
                self.delta_entropy = torch.zeros_like(delta)
            else:
                self.delta_entropy = delta / maximum

    @staticmethod
    def _curve(x, power):
        return 1.0 - np.power(x, power)

    def update(self, teacher_logits, student_logits):
        if self.delta_entropy is None:
            raise RuntimeError("KRD reliability must be initialized before update")
        teacher_pred = teacher_logits.argmax(dim=1).detach().cpu().numpy()
        student_pred = student_logits.argmax(dim=1).detach().cpu().numpy()
        delta = self.delta_entropy.detach().cpu().numpy()
        agree = delta[teacher_pred == student_pred]
        disagree = delta[teacher_pred != student_pred]
        hist_agree, bins = np.histogram(agree, bins=self.bins, range=(0.0, 1.0))
        hist_disagree, _ = np.histogram(disagree, bins=self.bins, range=(0.0, 1.0))
        probability = hist_agree / (hist_agree + hist_disagree + 1e-6)
        spread = float(probability.max() - probability.min())
        if spread <= 1e-12:
            self.update_failures += 1
            return
        probability = (probability - probability.min()) / spread
        centers = bins[:-1] + 0.5 / self.bins
        try:
            fitted, _ = curve_fit(
                self._curve,
                centers,
                probability,
                p0=[self.power],
                maxfev=10000,
            )
            update_power = float(fitted[0])
            if np.isfinite(update_power):
                self.power = self.momentum * self.power + (1.0 - self.momentum) * update_power
            else:
                self.update_failures += 1
        except (RuntimeError, ValueError, FloatingPointError):
            self.update_failures += 1

    def probabilities(self):
        if self.delta_entropy is None:
            raise RuntimeError("KRD reliability must be initialized before prediction")
        return (1.0 - self.delta_entropy.pow(self.power)).clamp(0.0, 1.0)


class krd_faithful_Predictor(_FaithfulG2MBase):
    strategy = "krd_faithful"

    def method_init(self, conf, data):
        self.model = _OfficialMLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
            norm_type=conf.model.get("norm_type", "none"),
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=float(conf.training["lr"]),
            weight_decay=float(conf.training["weight_decay"]),
        )
        self.best_epoch = -1
        self.run_status = "ok"
        self.run_error = ""
        self.g2m_diagnostics = {
            "g2m_strategy": self.strategy,
            "g2m_source_commit": KRD_SOURCE_COMMIT,
            "g2m_official_core_faithful_port": 1,
            "g2m_official_default_protocol": 0,
            "g2m_checkpoint_selection": "clean_validation",
            "g2m_train_uses_graph": 1,
            "g2m_inference_uses_graph": 0,
            "g2m_uses_unlabeled_nodes": 1,
            "g2m_parameter_source": self._conf_value(
                "parameter_source",
                "unspecified",
            ),
            "g2m_parameter_mapping_dataset": self._conf_value(
                "parameter_mapping_dataset",
                "",
            ),
        }

    def _teacher_model(self):
        return GCN(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=int(self._conf_value("teacher_hidden", self.conf.model["n_hidden"])),
            out_channels=self.n_classes,
            n_layers=int(self._conf_value("teacher_n_layer", self.conf.model["n_layer"])),
            dropout=float(self._conf_value("teacher_dropout", 0.5)),
            norm_info=None,
            act="F.relu",
            input_layer=False,
            output_layer=False,
        ).to(self.device)

    def _train_teacher(self):
        teacher = self._teacher_model()
        optimizer = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", self.conf.training["lr"])),
            weight_decay=float(self._conf_value("teacher_weight_decay", self.conf.training["weight_decay"])),
        )
        max_epochs = int(self._conf_value("teacher_epochs", self.conf.training["n_epochs"]))
        patience = int(self._conf_value("teacher_patience", 50))
        best_val = -1.0
        best_state = None
        best_epoch = -1
        stale = 0
        for epoch in range(1, max_epochs + 1):
            teacher.train()
            optimizer.zero_grad()
            logits = teacher(self.feats, self.adj)
            loss = self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
            loss.backward()
            optimizer.step()
            teacher.eval()
            with torch.no_grad():
                logits = teacher(self.feats, self.adj)
                val = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
            if val >= best_val:
                best_val = float(val)
                best_state = deepcopy(teacher.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        teacher.load_state_dict(best_state)
        teacher.eval()
        with torch.no_grad():
            logits = teacher(self.feats, self.adj)
            test = self.metric(
                self.clean_label[self.test_mask].cpu().numpy(),
                logits[self.test_mask].detach().cpu().numpy(),
            )
        self.g2m_diagnostics.update(
            {
                "teacher_best_epoch": int(best_epoch),
                "teacher_best_val": float(best_val),
                "teacher_best_test": float(test),
                "teacher_last_loss": float(loss.detach().cpu()),
            }
        )
        return teacher, logits.detach()

    def _official_pairs(self):
        source_destination = _edge_index(self.adj, self.device)
        keep = source_destination[0] != source_destination[1]
        source_destination = source_destination[:, keep]
        # Official KRD indexes the student target first and its teacher
        # knowledge point second. PyG stores source first and destination second.
        nonself = torch.stack([source_destination[1], source_destination[0]], dim=0)
        return nonself, _add_self_pairs(nonself, self.n_nodes, self.device)

    def train(self):
        teacher, teacher_logits = self._train_teacher()
        reliability = _KRDReliability(
            init_power=self._conf_value("reliability_init_power", 1.0),
            momentum=self._conf_value("reliability_momentum", 0.99),
            bins=self._conf_value("reliability_bins", 50),
            noise_level=self._conf_value("reliability_noise_level", 1.0),
        )
        reliability.initialize(teacher, self.feats, self.adj)
        nonself_pairs, pairs = self._official_pairs()
        update_start = int(self._conf_value("reliability_update_start", 50))
        label_weight = float(self._conf_value("distill_lambda", 0.5))
        tau = float(self._conf_value("distill_temperature", 0.8))
        patience = int(self.conf.training.get("patience") or 50)
        best_val = -1.0
        best_state = None
        stale = 0
        updates = 0
        sampled_pair_sum = 0
        previous_logits = None
        for epoch in range(1, int(self.conf.training["n_epochs"]) + 1):
            if epoch > update_start and previous_logits is not None:
                reliability.update(teacher_logits, previous_logits)
                pairs = sample_krd_pairs(nonself_pairs, reliability.probabilities(), self.n_nodes)
                updates += 1
            sampled_pair_sum += int(pairs.shape[1])
            self.model.train()
            self.optim.zero_grad()
            logits = self.model(self.feats, None)
            label_loss = self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
            distill_loss = krd_edge_kl(logits, teacher_logits, pairs, tau)
            loss = label_weight * label_loss + (1.0 - label_weight) * distill_loss
            loss.backward()
            self.optim.step()
            previous_logits = logits.detach()
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.feats, None)
                val = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    val_logits[self.val_mask].detach().cpu().numpy(),
                )
            if val >= best_val:
                best_val = float(val)
                best_state = deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.result["valid"] = float(val)
                self.result["train"] = self.metric(
                    self.noisy_label[self.train_mask].cpu().numpy(),
                    logits[self.train_mask].detach().cpu().numpy(),
                )
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.weights = best_state
        self.total_time = time.time() - self.start_time
        self.g2m_diagnostics.update(
            {
                "student_best_epoch": int(self.best_epoch),
                "krd_reliability_updates": int(updates),
                "krd_final_power": float(reliability.power),
                "krd_update_failures": int(reliability.update_failures),
                "krd_mean_sampled_pairs": float(sampled_pair_sum / max(1, epoch)),
                "krd_last_label_loss": float(label_loss.detach().cpu()),
                "krd_last_distill_loss": float(distill_loss.detach().cpu()),
            }
        )
        return self._finish_student()


class _VQGraphTeacher(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        codebook_size,
        vq_decay,
        commitment_weight,
        lambda_node,
        lambda_edge,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, hidden_channels)
        self.decoder_edge = nn.Linear(in_channels, in_channels)
        self.decoder_node = nn.Linear(in_channels, in_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        quantizer_cls = _load_official_vqgraph_quantizer()
        self.quantizer = quantizer_cls(
            dim=in_channels,
            codebook_size=codebook_size,
            decay=vq_decay,
            commitment_weight=commitment_weight,
            use_cosine_sim=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.lambda_node = float(lambda_node)
        self.lambda_edge = float(lambda_edge)

    def forward(self, features, edge_index, dense_adjacency):
        if getattr(features, "is_sparse", False):
            features = features.to_dense()
        hidden = F.relu(self.conv1(features, edge_index))
        hidden = self.dropout(hidden)
        quantized, indices, commitment, similarities, codebook = self.quantizer(hidden)
        similarities = torch.squeeze(similarities)
        codebook = torch.squeeze(codebook, dim=0)
        quantized_edge = self.decoder_edge(quantized)
        quantized_node = self.decoder_node(quantized)
        feature_loss = self.lambda_node * F.mse_loss(hidden, quantized_node)
        reconstructed = quantized_edge @ quantized_edge.t()
        span = reconstructed.max() - reconstructed.min()
        if float(span.detach().cpu()) > 1e-12:
            reconstructed = (reconstructed - reconstructed.min()) / span
        else:
            reconstructed = torch.zeros_like(reconstructed)
        edge_loss = self.lambda_edge * torch.sqrt(F.mse_loss(dense_adjacency, reconstructed))
        second_hidden = F.relu(self.conv2(quantized_edge, edge_index))
        logits = self.classifier(second_hidden)
        auxiliary = feature_loss + edge_loss + commitment
        return logits, auxiliary, similarities, codebook, indices, feature_loss, edge_loss, commitment


class _VQGraphSampledTeacher(nn.Module):
    """Official VQGraph SAGE teacher semantics on a sampled local graph."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        codebook_size,
        vq_decay,
        commitment_weight,
        lambda_node,
        lambda_edge,
    ):
        super().__init__()
        self.conv1 = DGLBothNormGraphConv(in_channels, in_channels, activation=F.relu)
        self.conv2 = DGLBothNormGraphConv(in_channels, hidden_channels, activation=F.relu)
        self.decoder_edge = nn.Linear(in_channels, in_channels)
        self.decoder_node = nn.Linear(in_channels, in_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        quantizer_cls = _load_official_vqgraph_quantizer()
        self.quantizer = quantizer_cls(
            dim=in_channels,
            codebook_size=codebook_size,
            decay=vq_decay,
            commitment_weight=commitment_weight,
            use_cosine_sim=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.lambda_node = float(lambda_node)
        self.lambda_edge = float(lambda_edge)

    def forward(self, features, edge_index, dense_adjacency, *, output_count):
        if getattr(features, "is_sparse", False):
            features = features.to_dense()
        hidden = self.dropout(self.conv1(features, edge_index))
        quantized, indices, commitment, similarities, codebook = self.quantizer(hidden)
        similarities = torch.squeeze(similarities)
        codebook = torch.squeeze(codebook, dim=0)
        quantized_edge = self.decoder_edge(quantized)
        quantized_node = self.decoder_node(quantized)
        feature_loss = self.lambda_node * F.mse_loss(hidden, quantized_node)
        reconstructed = quantized_edge @ quantized_edge.t()
        span = reconstructed.max() - reconstructed.min()
        reconstructed = torch.where(
            span > 1e-12,
            (reconstructed - reconstructed.min()) / span.clamp_min(1e-12),
            torch.zeros_like(reconstructed),
        )
        edge_loss = self.lambda_edge * torch.sqrt(F.mse_loss(dense_adjacency, reconstructed))
        second_hidden = self.conv2(quantized_edge, edge_index)
        logits = self.classifier(second_hidden[: int(output_count)])
        auxiliary = feature_loss + edge_loss + commitment
        return logits, auxiliary, similarities, codebook, indices, feature_loss, edge_loss, commitment


class vqgraph_faithful_Predictor(_FaithfulG2MBase):
    strategy = "vqgraph_faithful"

    def method_init(self, conf, data):
        self.model = _OfficialMLP(
            in_channels=conf.model["n_feat"],
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
            norm_type=conf.model.get("norm_type", "none"),
            token_projection_dim=conf.model["n_feat"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=float(conf.training["lr"]),
            weight_decay=float(conf.training["weight_decay"]),
        )
        self.best_epoch = -1
        self.run_status = "ok"
        self.run_error = ""
        self.g2m_diagnostics = {
            "g2m_strategy": self.strategy,
            "g2m_source_commit": VQGRAPH_SOURCE_COMMIT,
            "g2m_official_core_faithful_port": 1,
            "g2m_official_default_protocol": 0,
            "g2m_checkpoint_selection": "clean_validation",
            "g2m_train_uses_graph": 1,
            "g2m_inference_uses_graph": 0,
            "g2m_uses_unlabeled_nodes": 1,
            "vqgraph_teacher_architecture": self._conf_value(
                "teacher_architecture",
                "unspecified",
            ),
            "vqgraph_quantizer_implementation": "vendored_official_VectorQuantize",
            "g2m_parameter_source": self._conf_value(
                "parameter_source",
                "unspecified",
            ),
            "g2m_parameter_mapping_dataset": self._conf_value(
                "parameter_mapping_dataset",
                "",
            ),
        }

    def _teacher_model(self):
        return _VQGraphTeacher(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=int(self._conf_value("teacher_hidden", 64)),
            out_channels=self.n_classes,
            dropout=float(self._conf_value("teacher_dropout", 0.8)),
            codebook_size=int(self._conf_value("codebook_size", 5000)),
            vq_decay=float(self._conf_value("vq_decay", 0.8)),
            commitment_weight=float(self._conf_value("vq_commitment_weight", 0.25)),
            lambda_node=float(self._conf_value("vq_lambda_node", 0.001)),
            lambda_edge=float(self._conf_value("vq_lambda_edge", 0.03)),
        ).to(self.device)

    def _graph_inputs(self):
        edges = _edge_index(self.adj, self.device)
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        edges = torch.unique(edges.t(), dim=0).t().contiguous()
        edges = _add_self_pairs(edges, self.n_nodes, self.device)
        edges = torch.unique(edges.t(), dim=0).t().contiguous()
        dense = torch.zeros((self.n_nodes, self.n_nodes), device=self.device)
        dense[edges[0], edges[1]] = 1.0
        return edges, dense

    def _train_teacher(self):
        teacher = self._teacher_model()
        optimizer = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", 0.0001)),
            weight_decay=float(self._conf_value("teacher_weight_decay", 0.001)),
        )
        edges, dense = self._graph_inputs()
        max_epochs = int(self._conf_value("teacher_epochs", 100))
        patience = int(self._conf_value("teacher_patience", 50))
        best_val = -1.0
        best_state = None
        best_epoch = -1
        stale = 0
        for epoch in range(1, max_epochs + 1):
            teacher.train()
            optimizer.zero_grad()
            outputs = teacher(self.feats, edges, dense)
            logits, auxiliary = outputs[0], outputs[1]
            classification = self.loss_fn(logits[self.train_mask], self.noisy_label[self.train_mask])
            loss = classification + auxiliary
            loss.backward()
            optimizer.step()
            teacher.eval()
            with torch.no_grad():
                eval_outputs = teacher(self.feats, edges, dense)
                val_logits = eval_outputs[0]
                val = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    val_logits[self.val_mask].detach().cpu().numpy(),
                )
            if val >= best_val:
                best_val = float(val)
                best_state = deepcopy(teacher.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        teacher.load_state_dict(best_state)
        teacher.eval()
        with torch.no_grad():
            outputs = teacher(self.feats, edges, dense)
            logits, auxiliary, similarities, codebook, indices = outputs[:5]
            test = self.metric(
                self.clean_label[self.test_mask].cpu().numpy(),
                logits[self.test_mask].detach().cpu().numpy(),
            )
        self.g2m_diagnostics.update(
            {
                "teacher_best_epoch": int(best_epoch),
                "teacher_best_val": float(best_val),
                "teacher_best_test": float(test),
                "teacher_last_loss": float(loss.detach().cpu()),
                "vqgraph_codebook_size": int(codebook.shape[0]),
                "vqgraph_active_codes": int(torch.unique(indices).numel()),
                "vqgraph_last_auxiliary_loss": float(auxiliary.detach().cpu()),
                "vqgraph_last_feature_loss": float(outputs[5].detach().cpu()),
                "vqgraph_last_edge_loss": float(outputs[6].detach().cpu()),
                "vqgraph_last_commitment_loss": float(outputs[7].detach().cpu()),
            }
        )
        return F.log_softmax(logits, dim=1).detach(), codebook.detach(), similarities.detach()

    @staticmethod
    def _official_batches(length, batch_size, device):
        num_batches = max(1, int(length) // int(batch_size))
        order = torch.randperm(int(length), device=device)[: num_batches * int(batch_size)]
        if num_batches == 1:
            return [order]
        return list(order.view(num_batches, int(batch_size)))

    def _token_step(self, indices, codebook, teacher_similarities, temperature, weight):
        total = 0.0
        for batch in indices:
            self.model.train()
            self.optim.zero_grad()
            _, _, projected = self.model.forward_components(self.feats[batch])
            loss = vqgraph_token_kl(projected, codebook, teacher_similarities[batch], temperature)
            weighted = float(weight) * loss
            weighted.backward()
            self.optim.step()
            total += float(weighted.detach().cpu())
        return total / max(1, len(indices))

    def _label_step(self, node_indices, targets, weight, *, log_target):
        total = 0.0
        batch_size = int(self._conf_value("batch_size", 512))
        batches = self._official_batches(len(node_indices), batch_size, self.device)
        for local_batch in batches:
            batch = node_indices[local_batch]
            self.model.train()
            self.optim.zero_grad()
            logits = self.model(self.feats[batch], None)
            if log_target:
                loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    targets[batch],
                    reduction="batchmean",
                    log_target=True,
                )
            else:
                loss = self.loss_fn(logits, targets[batch])
            weighted = float(weight) * loss
            weighted.backward()
            self.optim.step()
            total += float(weighted.detach().cpu())
        return total / max(1, len(batches))

    def train(self):
        teacher_log_probs, codebook, teacher_similarities = self._train_teacher()
        all_nodes = torch.arange(self.n_nodes, device=self.device)
        train_nodes = _as_index(self.train_mask, self.device)
        temperature = float(self._conf_value("token_temperature", 4.0))
        token_weight = float(self._conf_value("lambda_soft_tokens", 1e-8))
        label_weight = float(self._conf_value("lambda_soft_labels", 0.0))
        batch_size = int(self._conf_value("batch_size", 512))
        patience = int(self.conf.training.get("patience") or 50)
        best_val = -1.0
        best_state = None
        stale = 0
        for epoch in range(1, int(self.conf.training["n_epochs"]) + 1):
            token_batches = self._official_batches(self.n_nodes, batch_size, self.device)
            token_loss = self._token_step(
                token_batches,
                codebook,
                teacher_similarities,
                temperature,
                token_weight,
            )
            hard_loss = self._label_step(
                train_nodes,
                self.noisy_label,
                label_weight,
                log_target=False,
            )
            soft_loss = self._label_step(
                all_nodes,
                teacher_log_probs,
                1.0 - label_weight,
                log_target=True,
            )
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.feats, None)
                val = self.metric(
                    self.noisy_label[self.val_mask].cpu().numpy(),
                    logits[self.val_mask].detach().cpu().numpy(),
                )
            if val >= best_val:
                best_val = float(val)
                best_state = deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.result["valid"] = float(val)
                self.result["train"] = self.metric(
                    self.noisy_label[self.train_mask].cpu().numpy(),
                    logits[self.train_mask].detach().cpu().numpy(),
                )
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.weights = best_state
        self.total_time = time.time() - self.start_time
        self.g2m_diagnostics.update(
            {
                "student_best_epoch": int(self.best_epoch),
                "vqgraph_last_token_loss": float(token_loss),
                "vqgraph_last_hard_label_loss": float(hard_loss),
                "vqgraph_last_soft_label_loss": float(soft_loss),
                "vqgraph_student_batch_size": int(batch_size),
            }
        )
        return self._finish_student()


class vqgraph_sampled_faithful_Predictor(vqgraph_faithful_Predictor):
    strategy = "vqgraph_sampled_faithful"

    def method_init(self, conf, data):
        super().method_init(conf, data)
        self.g2m_diagnostics.update(
            {
                "g2m_official_core_faithful_port": 0,
                "g2m_faithful_protocol_reimplementation": 1,
                "vqgraph_sampling_backend": "pytorch_cpu_csr",
                "vqgraph_training_backend": "pytorch_rocm",
                "vqgraph_sampling_device": "cpu",
                "vqgraph_training_device": str(self.device),
            }
        )

    def _teacher_model(self):
        return _VQGraphSampledTeacher(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=int(self._conf_value("teacher_hidden", 64)),
            out_channels=self.n_classes,
            dropout=float(self._conf_value("teacher_dropout", 0.0)),
            codebook_size=int(self._conf_value("codebook_size", 5000)),
            vq_decay=float(self._conf_value("vq_decay", 0.8)),
            commitment_weight=float(self._conf_value("vq_commitment_weight", 0.25)),
            lambda_node=float(self._conf_value("vq_lambda_node", 0.001)),
            lambda_edge=float(self._conf_value("vq_lambda_edge", 0.03)),
        ).to(self.device)

    @staticmethod
    def _cpu_batches(indices, batch_size, *, shuffle, seed):
        values = torch.as_tensor(indices, dtype=torch.long, device="cpu").flatten()
        if shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            values = values[torch.randperm(values.numel(), generator=generator)]
        return list(values.split(int(batch_size)))

    def _sampling_inputs(self):
        edges = _edge_index(self.adj, "cpu")
        return bidirected_with_self_loops(edges, self.n_nodes)

    def _sampled_forward(self, teacher, sample):
        global_nodes = sample.global_nodes.to(self.device)
        features = self.feats[global_nodes]
        edges = sample.edge_index.to(self.device)
        dense = local_dense_adjacency(
            edges,
            sample.num_nodes,
            device=self.device,
            dtype=features.dtype,
        )
        outputs = teacher(
            features,
            edges,
            dense,
            output_count=sample.output_count,
        )
        return outputs, global_nodes

    def _evaluate_sampled_teacher(self, teacher, sampler, target_nodes, batch_size):
        target_nodes = torch.as_tensor(target_nodes, dtype=torch.long, device="cpu").flatten()
        logits_all = torch.empty(
            (target_nodes.numel(), self.n_classes),
            device=self.device,
        )
        similarities_all = torch.empty(
            (target_nodes.numel(), int(self._conf_value("codebook_size", 5000))),
            device=self.device,
        )
        indices_all = torch.empty(target_nodes.numel(), dtype=torch.long, device=self.device)
        output_offset = 0
        last_outputs = None
        max_local_nodes = 0
        for batch_id, batch in enumerate(
            self._cpu_batches(target_nodes, batch_size, shuffle=False, seed=0)
        ):
            sample = sampler.sample(batch, epoch=0, batch_id=batch_id)
            outputs, _ = self._sampled_forward(teacher, sample)
            count = int(batch.numel())
            logits_all[output_offset : output_offset + count] = outputs[0]
            similarities_all[output_offset : output_offset + count] = outputs[2][:count]
            indices_all[output_offset : output_offset + count] = outputs[4].reshape(-1)[:count]
            output_offset += count
            max_local_nodes = max(max_local_nodes, sample.num_nodes)
            last_outputs = outputs
        return logits_all, similarities_all, indices_all, last_outputs, max_local_nodes

    def _train_teacher(self):
        if self.device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("vqgraph_sampled_faithful requires a ROCm GPU training device")
        teacher = self._teacher_model()
        if any(parameter.device.type != "cuda" for parameter in teacher.parameters()):
            raise RuntimeError("VQGraph teacher parameters are not on the ROCm GPU")
        if any(parameter.device.type != "cuda" for parameter in self.model.parameters()):
            raise RuntimeError("VQGraph student parameters are not on the ROCm GPU")

        optimizer = torch.optim.Adam(
            teacher.parameters(),
            lr=float(self._conf_value("teacher_lr", 0.01)),
            weight_decay=float(self._conf_value("teacher_weight_decay", 0.0005)),
        )
        graph_edges = self._sampling_inputs()
        fanouts = tuple(int(value) for value in self._conf_value("teacher_fanouts", [5, 5]))
        sampling_seed = int(getattr(self, "seed", 0))
        train_sampler = CpuNeighborSampler(
            graph_edges,
            self.n_nodes,
            fanouts=fanouts,
            seed=sampling_seed,
        )
        eval_sampler = CpuNeighborSampler(
            graph_edges,
            self.n_nodes,
            fanouts=(-1,),
            seed=sampling_seed,
        )
        batch_size = int(self._conf_value("teacher_batch_size", 512))
        train_nodes = _as_index(self.train_mask, "cpu")
        val_nodes = _as_index(self.val_mask, "cpu")
        max_epochs = int(self._conf_value("teacher_epochs", 100))
        patience = int(self._conf_value("teacher_patience", 50))
        best_val = -1.0
        best_state = None
        best_epoch = -1
        stale = 0
        max_train_local_nodes = 0
        max_eval_local_nodes = 0
        last_train_loss = None
        torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, max_epochs + 1):
            teacher.train()
            epoch_loss = 0.0
            batches = self._cpu_batches(
                train_nodes,
                batch_size,
                shuffle=True,
                seed=sampling_seed + epoch,
            )
            for batch_id, batch in enumerate(batches):
                sample = train_sampler.sample(batch, epoch=epoch, batch_id=batch_id)
                optimizer.zero_grad()
                outputs, global_nodes = self._sampled_forward(teacher, sample)
                labels = self.noisy_label[global_nodes[: sample.output_count]]
                loss = self.loss_fn(outputs[0], labels) + outputs[1]
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                max_train_local_nodes = max(max_train_local_nodes, sample.num_nodes)
            last_train_loss = epoch_loss / max(1, len(batches))

            teacher.eval()
            with torch.no_grad():
                val_logits, _, _, _, eval_local_nodes = self._evaluate_sampled_teacher(
                    teacher,
                    eval_sampler,
                    val_nodes,
                    batch_size,
                )
                val_labels = self.noisy_label[val_nodes.to(self.device)]
                val = self.metric(
                    val_labels.detach().cpu().numpy(),
                    val_logits.detach().cpu().numpy(),
                )
            max_eval_local_nodes = max(max_eval_local_nodes, eval_local_nodes)
            if self.conf.training.get("debug") and (epoch == 1 or epoch % 10 == 0):
                print(
                    "VQGRAPH_SAMPLED_TEACHER "
                    f"epoch={epoch} train_loss={last_train_loss:.6f} "
                    f"val={float(val):.6f} max_local_nodes={max_train_local_nodes}"
                )
            if val >= best_val:
                best_val = float(val)
                best_state = deepcopy(teacher.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_state is None:
            raise RuntimeError("VQGraph sampled teacher did not produce a checkpoint")
        teacher.load_state_dict(best_state)
        teacher.eval()
        all_nodes = torch.arange(self.n_nodes, dtype=torch.long)
        with torch.no_grad():
            logits, similarities, indices, outputs, eval_local_nodes = self._evaluate_sampled_teacher(
                teacher,
                eval_sampler,
                all_nodes,
                batch_size,
            )
        max_eval_local_nodes = max(max_eval_local_nodes, eval_local_nodes)
        test = self.metric(
            self.clean_label[self.test_mask].detach().cpu().numpy(),
            logits[_as_index(self.test_mask, self.device)].detach().cpu().numpy(),
        )
        codebook = outputs[3]
        self.g2m_diagnostics.update(
            {
                "teacher_best_epoch": int(best_epoch),
                "teacher_best_val": float(best_val),
                "teacher_best_test": float(test),
                "teacher_last_loss": float(last_train_loss),
                "vqgraph_codebook_size": int(codebook.shape[0]),
                "vqgraph_active_codes": int(torch.unique(indices).numel()),
                "vqgraph_last_auxiliary_loss": float(outputs[1].detach().cpu()),
                "vqgraph_last_feature_loss": float(outputs[5].detach().cpu()),
                "vqgraph_last_edge_loss": float(outputs[6].detach().cpu()),
                "vqgraph_last_commitment_loss": float(outputs[7].detach().cpu()),
                "vqgraph_teacher_batch_size": int(batch_size),
                "vqgraph_teacher_fanouts": ",".join(str(value) for value in fanouts),
                "vqgraph_max_train_local_nodes": int(max_train_local_nodes),
                "vqgraph_max_eval_local_nodes": int(max_eval_local_nodes),
                "vqgraph_max_local_dense_elements": int(
                    max(max_train_local_nodes, max_eval_local_nodes) ** 2
                ),
                "vqgraph_peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "vqgraph_teacher_parameter_device": str(next(teacher.parameters()).device),
                "vqgraph_student_parameter_device": str(next(self.model.parameters()).device),
            }
        )
        return F.log_softmax(logits, dim=1).detach(), codebook.detach(), similarities.detach()
