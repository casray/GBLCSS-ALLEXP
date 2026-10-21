from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from yaml import SafeLoader

from .utils import UPSTREAM_GRACE


def _load_upstream_module(name: str, upstream_dir: Path = UPSTREAM_GRACE):
    module_path = upstream_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"grace_upstream_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream GRACE module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_grace_config(dataset_name: str, config_path: Path | None = None) -> dict:
    path = config_path or UPSTREAM_GRACE / "config.yaml"
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=SafeLoader)
    return dict(config[dataset_name])


def build_grace_model(dataset, config: dict, device, upstream_dir: Path = UPSTREAM_GRACE):
    model_module = _load_upstream_module("model", upstream_dir)
    activation = {"relu": F.relu, "prelu": nn.PReLU()}[config["activation"]]
    base_model = {"GCNConv": GCNConv}[config["base_model"]]
    encoder = model_module.Encoder(
        dataset.num_features,
        config["num_hidden"],
        activation,
        base_model=base_model,
        k=config["num_layers"],
    ).to(device)
    return model_module.Model(
        encoder,
        config["num_hidden"],
        config["num_proj_hidden"],
        config["tau"],
    ).to(device)


def train_grace_epoch(model, optimizer, x, edge_index, config: dict) -> float:
    model_module = _load_upstream_module("model")
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=config["drop_edge_rate_1"])[0]
    edge_index_2 = dropout_adj(edge_index, p=config["drop_edge_rate_2"])[0]
    x_1 = model_module.drop_feature(x, config["drop_feature_rate_1"])
    x_2 = model_module.drop_feature(x, config["drop_feature_rate_2"])
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def extract_encoder_h(model, x, edge_index, normalize: bool = True) -> torch.Tensor:
    model.eval()
    h = model(x, edge_index).detach()
    if normalize:
        h = F.normalize(h, p=2, dim=-1)
    return h

