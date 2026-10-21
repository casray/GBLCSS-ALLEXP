from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.data import load_dataset
from noisy_label_gb_grace.upstream_grace import load_grace_config
from noisy_label_gb_grace.utils import set_seed


UPSTREAM_REFS = {
    "grace": "upstream/GRACE/config.yaml fixed tau",
    "grace_uniformity": "upstream/GRACE/config.yaml fixed tau plus label-free encoder uniformity regularizer",
    "grace_learnable_tau": "upstream/GRACE/config.yaml with learnable log_tau init 0.5",
    "bgrl": "upstream/bgrl official transductive code; no Cora/CiteSeer config, uses official Amazon-Photos transductive augmentation/optimizer pattern",
    "graphmae": "upstream/GraphMAE/configs.yml dataset defaults",
    "ccassg": "upstream/CCA-SSG README dataset commands",
}

GRACE_CONFIG_FALLBACKS = {
    "citeseerfull": "CiteSeer",
    "citeseer-full": "CiteSeer",
    "amazonpho": "PubMed",
    "amazon-photo": "PubMed",
    "amazonphoto": "PubMed",
    "amazoncom": "PubMed",
    "amazon-computers": "PubMed",
    "amazoncomputers": "PubMed",
    "corafull": "DBLP",
    "coraml": "Cora",
    "cora-ml": "Cora",
    "cora_ml": "Cora",
    "attribwiki": "Cora",
    "attributedwiki": "Cora",
    "attributed-wiki": "Cora",
    "coauthorcs": "DBLP",
    "coauthor-cs": "DBLP",
    "coauthor_cs": "DBLP",
    "coauthorph": "DBLP",
    "coauthor-physics": "DBLP",
    "coauthorphysics": "DBLP",
    "lastfmasia": "PubMed",
    "lastfm-asia": "PubMed",
    "wikics": "PubMed",
    "wiki-cs": "PubMed",
}


def activation_module(name: str):
    name = str(name).lower()
    if name == "prelu":
        return nn.PReLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation: {name}")


def format_float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def method_output_name(method: str, uniformity_weight: float | None = None) -> str:
    if method == "grace_uniformity":
        if uniformity_weight is None:
            raise ValueError("grace_uniformity requires uniformity_weight")
        return f"grace_unif{format_float_token(float(uniformity_weight))}"
    return method


def uniformity_loss(z: torch.Tensor, t: float = 2.0, max_nodes: int | None = None) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("uniformity_loss expects a 2-D tensor")
    if max_nodes is not None and int(max_nodes) > 0 and int(z.shape[0]) > int(max_nodes):
        idx = torch.linspace(0, int(z.shape[0]) - 1, steps=int(max_nodes), device=z.device).round().long()
        z = z[idx]
    if int(z.shape[0]) < 2:
        return z.sum() * 0.0
    z = F.normalize(z, p=2, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-float(t) * sq_pdist).mean().clamp_min(1e-12))


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, activation_name: str = "relu"):
        super().__init__()
        num_layers = max(1, int(num_layers))
        self.activation = activation_module(activation_name)
        dims = [int(in_dim)]
        if num_layers == 1:
            dims.append(int(out_dim))
        else:
            dims.extend([int(hidden_dim)] * (num_layers - 1))
            dims.append(int(out_dim))
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.activation(x)
        return x


def gcn_encoder(in_dim: int, hidden_dim: int, num_layers: int, activation_name: str = "relu") -> GCNEncoder:
    return GCNEncoder(in_dim, hidden_dim, hidden_dim, num_layers, activation_name)


class LearnableTauGraceModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, proj_dim: int, init_tau: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(int(hidden_dim), int(proj_dim))
        self.fc2 = nn.Linear(int(proj_dim), int(hidden_dim))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    @property
    def tau(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau), min=0.05, max=2.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    def _semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        refl = torch.exp(torch.mm(z1, z1.t()) / self.tau)
        between = torch.exp(torch.mm(z1, z2.t()) / self.tau)
        return -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))

    def loss_on_views(
        self,
        x1,
        edge_index1,
        x2,
        edge_index2,
        uniformity_weight: float = 0.0,
        uniformity_t: float = 2.0,
        uniformity_max_nodes: int | None = None,
    ) -> torch.Tensor:
        h1 = self.encoder(x1, edge_index1)
        h2 = self.encoder(x2, edge_index2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        loss = 0.5 * (self._semi_loss(z1, z2) + self._semi_loss(z2, z1)).mean()
        if float(uniformity_weight) > 0.0:
            loss = loss + float(uniformity_weight) * 0.5 * (
                uniformity_loss(h1, t=uniformity_t, max_nodes=uniformity_max_nodes)
                + uniformity_loss(h2, t=uniformity_t, max_nodes=uniformity_max_nodes)
            )
        return loss


class FixedTauGraceModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, proj_dim: int, tau: float):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(int(hidden_dim), int(proj_dim))
        self.fc2 = nn.Linear(int(proj_dim), int(hidden_dim))
        self.tau = float(tau)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    def _semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        refl = torch.exp(torch.mm(z1, z1.t()) / self.tau)
        between = torch.exp(torch.mm(z1, z2.t()) / self.tau)
        return -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))

    def _chunked_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        def chunk_loss(anchor: torch.Tensor, start: int, stop: int) -> torch.Tensor:
            refl = torch.exp(torch.mm(anchor, z1.t()) / self.tau)
            between = torch.exp(torch.mm(anchor, z2.t()) / self.tau)
            local = torch.arange(stop - start, device=anchor.device)
            positive = between[local, torch.arange(start, stop, device=anchor.device)]
            self_sim = refl[local, torch.arange(start, stop, device=anchor.device)]
            return -torch.log(positive / (refl.sum(1) + between.sum(1) - self_sim))

        losses = []
        for start in range(0, int(z1.shape[0]), int(batch_size)):
            stop = min(start + int(batch_size), int(z1.shape[0]))
            anchor = z1[start:stop]
            if torch.is_grad_enabled():
                losses.append(checkpoint(lambda value, s=start, e=stop: chunk_loss(value, s, e), anchor, use_reentrant=False))
            else:
                losses.append(chunk_loss(anchor, start, stop))
        return torch.cat(losses, dim=0)

    def loss_on_views(
        self,
        x1,
        edge_index1,
        x2,
        edge_index2,
        uniformity_weight: float = 0.0,
        uniformity_t: float = 2.0,
        uniformity_max_nodes: int | None = None,
        contrastive_batch_size: int = 0,
    ) -> torch.Tensor:
        h1 = self.encoder(x1, edge_index1)
        h2 = self.encoder(x2, edge_index2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        if int(contrastive_batch_size) > 0:
            loss = 0.5 * (
                self._chunked_semi_loss(z1, z2, int(contrastive_batch_size))
                + self._chunked_semi_loss(z2, z1, int(contrastive_batch_size))
            ).mean()
        else:
            loss = 0.5 * (self._semi_loss(z1, z2) + self._semi_loss(z2, z1)).mean()
        if float(uniformity_weight) > 0.0:
            loss = loss + float(uniformity_weight) * 0.5 * (
                uniformity_loss(h1, t=uniformity_t, max_nodes=uniformity_max_nodes)
                + uniformity_loss(h2, t=uniformity_t, max_nodes=uniformity_max_nodes)
            )
        return loss


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.PReLU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BGRLModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, predictor_hidden_dim: int = 512):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.predictor = MLP(hidden_dim, predictor_hidden_dim, hidden_dim)

    def forward_online(self, x, edge_index):
        return self.predictor(self.online_encoder(x, edge_index))

    @torch.no_grad()
    def forward_target(self, x, edge_index):
        return self.target_encoder(x, edge_index)

    def loss_on_views(self, x1, edge_index1, x2, edge_index2) -> torch.Tensor:
        q1 = self.forward_online(x1, edge_index1)
        q2 = self.forward_online(x2, edge_index2)
        y1 = self.forward_target(x1, edge_index1)
        y2 = self.forward_target(x2, edge_index2)
        return 2.0 - F.cosine_similarity(q1, y2.detach(), dim=-1).mean() - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()

    @torch.no_grad()
    def update_target_network(self, momentum: float = 0.99) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(float(momentum)).add_(online.data, alpha=1.0 - float(momentum))

    def encode(self, x, edge_index) -> torch.Tensor:
        return self.online_encoder(x, edge_index)


class CCASSGModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 512, n_layers: int = 2, use_mlp: bool = False):
        super().__init__()
        if use_mlp:
            self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        else:
            self.backbone = GCNEncoder(in_dim, hidden_dim, out_dim, n_layers, "relu")

    def encode(self, x, edge_index) -> torch.Tensor:
        return self.backbone(x, edge_index) if not isinstance(self.backbone, nn.Sequential) else self.backbone(x)

    def _normalized(self, x, edge_index):
        h = self.encode(x, edge_index)
        return (h - h.mean(dim=0)) / (h.std(dim=0) + 1e-9)

    def loss_on_views(self, x1, edge_index1, x2, edge_index2, lambd: float = 1e-3) -> torch.Tensor:
        z1 = self._normalized(x1, edge_index1)
        z2 = self._normalized(x2, edge_index2)
        n = max(1, int(z1.shape[0]))
        c = torch.mm(z1.t(), z2) / n
        c1 = torch.mm(z1.t(), z1) / n
        c2 = torch.mm(z2.t(), z2) / n
        ident = torch.eye(c.shape[0], device=c.device)
        loss_inv = -torch.diagonal(c).sum()
        loss_dec = (ident - c1).pow(2).sum() + (ident - c2).pow(2).sum()
        return loss_inv + float(lambd) * loss_dec


class GraphMAEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 2, activation_name: str = "prelu"):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, hidden_dim, num_layers, activation_name)
        self.decoder = nn.Linear(hidden_dim, in_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, int(in_dim)))

    def encode(self, x, edge_index) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def _mask_input(self, x: torch.Tensor, mask_rate: float, replace_rate: float):
        n_nodes = int(x.shape[0])
        n_mask = max(1, int(mask_rate * n_nodes))
        perm = torch.randperm(n_nodes, device=x.device)
        mask_idx = perm[:n_mask]
        keep_x = x.clone()
        if replace_rate > 0:
            n_replace = int(replace_rate * n_mask)
            replace_idx = mask_idx[:n_replace]
            token_idx = mask_idx[n_replace:]
            random_idx = torch.randint(0, n_nodes, (n_replace,), device=x.device)
            keep_x[replace_idx] = x[random_idx]
        else:
            token_idx = mask_idx
        keep_x[token_idx] = 0.0
        keep_x[token_idx] += self.mask_token
        return keep_x, mask_idx

    @staticmethod
    def sce_loss(pred: torch.Tensor, target: torch.Tensor, alpha_l: float = 3.0) -> torch.Tensor:
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).pow(float(alpha_l)).mean()

    def loss(self, x, edge_index, mask_rate: float = 0.5, replace_rate: float = 0.0, alpha_l: float = 3.0) -> torch.Tensor:
        masked_x, mask_idx = self._mask_input(x, mask_rate, replace_rate)
        h = self.encoder(masked_x, edge_index)
        recon = self.decoder(h)
        return self.sce_loss(recon[mask_idx], x[mask_idx], alpha_l)


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return x
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < float(drop_prob)
    out = x.clone()
    out[:, drop_mask] = 0
    return out


def drop_edges(edge_index: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return edge_index
    return dropout_edge(edge_index, p=float(drop_prob))[0]


def ssl_defaults(method: str, dataset: str) -> dict:
    dataset_key = dataset.lower()
    if method in {"grace", "grace_uniformity", "grace_learnable_tau"}:
        config_dataset = dataset
        fallback_note = ""
        try:
            cfg = load_grace_config(dataset)
        except KeyError:
            if dataset_key not in GRACE_CONFIG_FALLBACKS:
                raise
            config_dataset = GRACE_CONFIG_FALLBACKS[dataset_key]
            cfg = load_grace_config(config_dataset)
            fallback_note = f"official GRACE config missing for {dataset}; preregistered fallback uses {config_dataset} unchanged"
        defaults = {
            "epochs": int(cfg["num_epochs"]),
            "lr": float(cfg["learning_rate"]),
            "weight_decay": float(cfg["weight_decay"]),
            "hidden_dim": int(cfg["num_hidden"]),
            "proj_dim": int(cfg["num_proj_hidden"]),
            "num_layers": int(cfg["num_layers"]),
            "activation": str(cfg["activation"]),
            "drop_edge_1": float(cfg["drop_edge_rate_1"]),
            "drop_edge_2": float(cfg["drop_edge_rate_2"]),
            "drop_feat_1": float(cfg["drop_feature_rate_1"]),
            "drop_feat_2": float(cfg["drop_feature_rate_2"]),
            "tau": float(cfg["tau"]),
            "uniformity_weight": 0.0,
            "uniformity_t": 2.0,
            "uniformity_max_nodes": 2048,
            "contrastive_batch_size": 1024 if dataset_key in {"coauthorph", "coauthor-physics", "coauthorphysics"} else 0,
            "config_source_dataset": str(config_dataset),
            "fallback_note": fallback_note,
        }
        if method == "grace_learnable_tau":
            defaults["init_tau"] = 0.5
        return defaults
    if method == "ccassg":
        defaults = {
            "cora": {"epochs": 50, "n_layers": 2, "lambd": 1e-3, "dfr": 0.1, "der": 0.4},
            "citeseer": {"epochs": 20, "n_layers": 1, "lambd": 5e-4, "dfr": 0.0, "der": 0.4},
            "pubmed": {"epochs": 100, "n_layers": 2, "lambd": 1e-3, "dfr": 0.3, "der": 0.5},
        }
        cfg = defaults.get(dataset_key, defaults["cora"]).copy()
        cfg.update({"lr": 1e-3, "weight_decay": 0.0, "hidden_dim": 512, "out_dim": 512})
        return cfg
    if method == "graphmae":
        all_cfg = yaml.safe_load((ROOT / "upstream" / "GraphMAE" / "configs.yml").read_text(encoding="utf-8"))
        cfg = dict(all_cfg[dataset_key])
        return {
            "epochs": int(cfg["max_epoch"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "hidden_dim": int(cfg["num_hidden"]),
            "num_layers": int(cfg["num_layers"]),
            "activation": str(cfg["activation"]),
            "mask_rate": float(cfg["mask_rate"]),
            "replace_rate": float(cfg.get("replace_rate", 0.0)),
            "alpha_l": float(cfg["alpha_l"]),
            "drop_edge": float(cfg.get("drop_edge_rate", 0.0)),
        }
    if method == "bgrl":
        try:
            base = load_grace_config(dataset)
        except KeyError:
            base = load_grace_config(GRACE_CONFIG_FALLBACKS.get(dataset_key, "PubMed"))
        return {
            "epochs": 10000,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "mm": 0.99,
            "hidden_dim": int(base["num_hidden"]),
            "num_layers": int(base["num_layers"]),
            "activation": str(base["activation"]),
            "predictor_hidden_dim": 512,
            "drop_edge_1": 0.4,
            "drop_feat_1": 0.1,
            "drop_edge_2": 0.1,
            "drop_feat_2": 0.2,
            "source_note": "BGRL official repo has no Cora/CiteSeer config; reused official Amazon-Photos transductive defaults for augmentation/lr/mm.",
        }
    raise ValueError(f"unsupported method: {method}")


def build_model(method: str, data, dataset_name: str):
    cfg = ssl_defaults(method, dataset_name)
    in_dim = int(data.x.shape[1])
    if method in {"grace", "grace_uniformity"}:
        encoder = gcn_encoder(in_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["activation"])
        model = FixedTauGraceModel(encoder, cfg["hidden_dim"], cfg["proj_dim"], cfg["tau"])
    elif method == "grace_learnable_tau":
        encoder = gcn_encoder(in_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["activation"])
        model = LearnableTauGraceModel(encoder, cfg["hidden_dim"], cfg["proj_dim"], cfg["init_tau"])
    elif method == "bgrl":
        encoder = gcn_encoder(in_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["activation"])
        model = BGRLModel(encoder, cfg["hidden_dim"], cfg["predictor_hidden_dim"])
    elif method == "ccassg":
        model = CCASSGModel(in_dim, cfg["hidden_dim"], cfg["out_dim"], cfg["n_layers"])
    elif method == "graphmae":
        model = GraphMAEModel(in_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["activation"])
    else:
        raise ValueError(f"unsupported method: {method}")
    return model, cfg


def train_one_epoch(method: str, model: nn.Module, optimizer, data, cfg: dict, epoch: int) -> float:
    model.train()
    optimizer.zero_grad()
    x = data.x
    edge_index = data.edge_index
    if method in {"grace", "grace_uniformity", "grace_learnable_tau"}:
        x1 = drop_feature(x, cfg["drop_feat_1"])
        x2 = drop_feature(x, cfg["drop_feat_2"])
        e1 = drop_edges(edge_index, cfg["drop_edge_1"])
        e2 = drop_edges(edge_index, cfg["drop_edge_2"])
        if method == "grace_uniformity":
            loss = model.loss_on_views(
                x1,
                e1,
                x2,
                e2,
                uniformity_weight=float(cfg.get("uniformity_weight", 0.0)),
                uniformity_t=float(cfg.get("uniformity_t", 2.0)),
                uniformity_max_nodes=int(cfg.get("uniformity_max_nodes", 2048)),
            )
        else:
            loss = model.loss_on_views(
                x1,
                e1,
                x2,
                e2,
                contrastive_batch_size=int(cfg.get("contrastive_batch_size", 0)),
            )
    elif method == "bgrl":
        x1 = drop_feature(x, cfg["drop_feat_1"])
        x2 = drop_feature(x, cfg["drop_feat_2"])
        e1 = drop_edges(edge_index, cfg["drop_edge_1"])
        e2 = drop_edges(edge_index, cfg["drop_edge_2"])
        loss = model.loss_on_views(x1, e1, x2, e2)
    elif method == "ccassg":
        x1 = drop_feature(x, cfg["dfr"])
        x2 = drop_feature(x, cfg["dfr"])
        e1 = drop_edges(edge_index, cfg["der"])
        e2 = drop_edges(edge_index, cfg["der"])
        loss = model.loss_on_views(x1, e1, x2, e2, cfg["lambd"])
    elif method == "graphmae":
        e = drop_edges(edge_index, cfg["drop_edge"])
        loss = model.loss(x, e, cfg["mask_rate"], cfg["replace_rate"], cfg["alpha_l"])
    else:
        raise ValueError(f"unsupported method: {method}")
    loss.backward()
    optimizer.step()
    if method == "bgrl":
        model.update_target_network(cfg["mm"])
    return float(loss.item())


@torch.no_grad()
def extract_h(method: str, model: nn.Module, data) -> torch.Tensor:
    model.eval()
    if method in {"grace", "grace_uniformity", "grace_learnable_tau"}:
        h = model(data.x, data.edge_index)
    elif method == "bgrl":
        h = model.encode(data.x, data.edge_index)
    elif method == "ccassg":
        h = model.encode(data.x, data.edge_index)
    elif method == "graphmae":
        h = model.encode(data.x, data.edge_index)
    else:
        raise ValueError(f"unsupported method: {method}")
    return F.normalize(h.detach().cpu().float(), p=2, dim=1)


def output_dir(dataset: str, method: str, uniformity_weight: float | None = None) -> Path:
    return ROOT / "checkpoints" / "embeddings_v2" / dataset / method_output_name(method, uniformity_weight)


def pretrain(args, dataset_name: str, method: str, uniformity_weight: float | None = None) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root) if args.data_root else None
    dataset, data = load_dataset(dataset_name, data_root=data_root, device=device)
    model, cfg = build_model(method, data, dataset_name)
    model = model.to(device)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if method == "grace_uniformity":
        cfg["uniformity_weight"] = float(uniformity_weight)
        cfg["uniformity_t"] = 2.0
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    losses = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        loss = train_one_epoch(method, model, optimizer, data, cfg, epoch)
        losses.append({"epoch": epoch, "loss": loss})
        if args.log_every > 0 and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(cfg["epochs"])):
            if method == "grace_learnable_tau":
                extra = f", tau={float(model.tau.item()):.4f}"
            elif method in {"grace", "grace_uniformity"}:
                extra = f", tau={float(model.tau):.4f}"
                if method == "grace_uniformity":
                    extra += f", uniformity_weight={float(cfg['uniformity_weight']):.4f}"
            else:
                extra = ""
            print(f"SSL_PRETRAIN dataset={dataset_name} method={method} epoch={epoch}/{cfg['epochs']} loss={loss:.4f}{extra}")
    h = extract_h(method, model, data)
    output_method = method_output_name(method, uniformity_weight)
    out_dir = output_dir(dataset_name, method, uniformity_weight)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(h, out_dir / "H.pt")
    metadata = {
        "dataset": dataset_name,
        "method": method,
        "output_method": output_method,
        "seed": int(args.seed),
        "embedding_shape": list(h.shape),
        "config": cfg,
        "source_reference": UPSTREAM_REFS[method],
        "upstream_commits": args.upstream_commits,
        "label_free": True,
        "used_labels_or_masks_in_training": False,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out_dir / "losses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(losses)
    return {
        "dataset": dataset_name,
        "method": output_method,
        "base_method": method,
        "uniformity_weight": float(uniformity_weight) if uniformity_weight is not None else "",
        "seed": int(args.seed),
        "embedding_path": str(out_dir / "H.pt"),
        "embedding_dim": int(h.shape[1]),
        "epochs": int(cfg["epochs"]),
        "final_loss": float(losses[-1]["loss"]) if losses else math.nan,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain label-free SSL candidates and export H.pt embeddings.")
    parser.add_argument("--datasets", default="Cora,CiteSeer")
    parser.add_argument("--methods", default="grace_learnable_tau,bgrl,graphmae,ccassg")
    parser.add_argument("--uniformity_weights", default="0.1,0.5,1.0", help="Comma-separated weights for method=grace_uniformity.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_root", default=None, help="Optional PyG dataset root; leave unset to use upstream repo data.")
    parser.add_argument("--epochs", type=int, default=None, help="Debug override only; leave unset for preregistered defaults.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--summary", default=str(ROOT / "checkpoints" / "embeddings_v2" / "ssl_pretrain_summary.csv"))
    parser.add_argument(
        "--upstream_commits",
        default="GRACE:b3b5ac3,bgrl:60f9f19,GraphMAE:b14f080,CCA-SSG:ce5b2c0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    uniformity_weights = [float(item.strip()) for item in args.uniformity_weights.split(",") if item.strip()]
    rows = []
    for dataset_name in datasets:
        for method in methods:
            if method == "grace_uniformity":
                for uniformity_weight in uniformity_weights:
                    rows.append(pretrain(args, dataset_name, method, uniformity_weight=uniformity_weight))
            else:
                rows.append(pretrain(args, dataset_name, method))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "base_method",
                "uniformity_weight",
                "seed",
                "embedding_path",
                "embedding_dim",
                "epochs",
                "final_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {summary_path}")


if __name__ == "__main__":
    main()
