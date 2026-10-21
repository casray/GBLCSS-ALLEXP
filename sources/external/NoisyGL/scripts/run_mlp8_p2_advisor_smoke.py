from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import MLP
from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import (
    build_soft_targets_from_features,
    load_frozen_embeddings,
)
from utils.functional import accuracy
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "mlp8_p2_advisor_smoke_20260702"
DEFAULT_EMBEDDING_ROOT = REPO_ROOT.parent.parent / "gb_graph_aware_minival_6_1" / "checkpoints" / "embeddings_v2"


class GraceEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, num_layers: int = 2):
        super().__init__()
        if int(num_layers) < 2:
            raise ValueError("GRACE encoder requires num_layers >= 2")
        self.activation = activation
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, 2 * out_channels))
        for _ in range(1, int(num_layers) - 1):
            self.convs.append(GCNConv(2 * out_channels, 2 * out_channels))
        self.convs.append(GCNConv(2 * out_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x


class GraceModel(nn.Module):
    def __init__(self, encoder: GraceEncoder, num_hidden: int, num_proj_hidden: int, tau: float):
        super().__init__()
        self.encoder = encoder
        self.tau = float(tau)
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.mm(F.normalize(z1), F.normalize(z2).t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda value: torch.exp(value / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        return 0.5 * (self.semi_loss(h1, h2) + self.semi_loss(h2, h1)).mean()


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _dense_features(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if getattr(x, "is_sparse", False) else x


def _edge_index_from_data(data) -> torch.Tensor:
    adj = data.adj.coalesce() if getattr(data.adj, "is_sparse", False) else data.adj
    return adj.indices() if getattr(adj, "is_sparse", False) else adj


def _mask_index(mask, device: str) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return torch.where(mask.detach().cpu())[0].to(device=device, dtype=torch.long)
        return mask.to(device=device, dtype=torch.long)
    tensor = torch.as_tensor(mask)
    if tensor.dtype == torch.bool:
        return torch.where(tensor)[0].to(device=device, dtype=torch.long)
    return tensor.to(device=device, dtype=torch.long)


def _mask_bool_cpu(mask, n_nodes: int) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().cpu()
    else:
        tensor = torch.as_tensor(mask)
    if tensor.dtype == torch.bool and int(tensor.numel()) == int(n_nodes):
        return tensor.bool()
    output = torch.zeros(int(n_nodes), dtype=torch.bool)
    output[tensor.long()] = True
    return output


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < float(drop_prob)
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def load_grace_config(dataset: str) -> dict:
    config_path = Path(r"C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\config.yaml")
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)[canonical_dataset_name(dataset)]


def canonical_dataset_name(dataset: str) -> str:
    key = str(dataset).lower()
    return {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}.get(key, dataset)


def build_grace_model(dataset, config: dict, device: str) -> tuple[GraceModel, nn.Module]:
    activation = {"relu": F.relu, "prelu": nn.PReLU().to(device)}[config["activation"]]
    encoder = GraceEncoder(
        in_channels=dataset.dim_feats,
        out_channels=int(config["num_hidden"]),
        activation=activation,
        num_layers=int(config["num_layers"]),
    ).to(device)
    model = GraceModel(
        encoder=encoder,
        num_hidden=int(config["num_hidden"]),
        num_proj_hidden=int(config["num_proj_hidden"]),
        tau=float(config["tau"]),
    ).to(device)
    classifier = nn.Linear(int(config["num_hidden"]), int(dataset.n_classes)).to(device)
    return model, classifier


def train_grace_ce_embedding(
    data,
    noisy_label: torch.Tensor,
    dataset_name: str,
    seed: int,
    device: str,
    ce_lambda: float,
    epochs_override: int | None = None,
) -> tuple[torch.Tensor, dict]:
    setup_seed(seed)
    config = load_grace_config(dataset_name)
    if epochs_override is not None:
        config["num_epochs"] = int(epochs_override)
    model, classifier = build_grace_model(data, config, device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = _dense_features(data.feats).float().to(device)
    edge_index = _edge_index_from_data(data).to(device)
    y = noisy_label.long().to(device)
    train_idx = _mask_index(data.train_masks, device)
    losses = []
    ce_losses = []
    contrast_losses = []
    started_at = time.perf_counter()
    for _ in range(int(config["num_epochs"])):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        edge_index_1 = dropout_adj(edge_index, p=float(config["drop_edge_rate_1"]))[0]
        edge_index_2 = dropout_adj(edge_index, p=float(config["drop_edge_rate_2"]))[0]
        z1 = model(drop_feature(x, float(config["drop_feature_rate_1"])), edge_index_1)
        z2 = model(drop_feature(x, float(config["drop_feature_rate_2"])), edge_index_2)
        contrast_loss = model.loss(z1, z2)
        logits = classifier(0.5 * (z1 + z2))
        ce_loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss = contrast_loss + float(ce_lambda) * ce_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        ce_losses.append(float(ce_loss.detach().item()))
        contrast_losses.append(float(contrast_loss.detach().item()))
    model.eval()
    with torch.no_grad():
        embedding = model(x, edge_index).detach().cpu()
    diagnostics = {
        "pretrain_epochs": int(config["num_epochs"]),
        "ce_lambda": float(ce_lambda),
        "pretrain_loss_last": losses[-1],
        "pretrain_contrast_loss_last": contrast_losses[-1],
        "pretrain_ce_loss_last": ce_losses[-1],
        "pretrain_wall_time_seconds": float(time.perf_counter() - started_at),
    }
    return embedding, diagnostics


def soft_target_diagnostics_from_embedding(
    embeddings: torch.Tensor,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask: torch.Tensor,
    n_classes: int,
    seed: int,
    dataset_name: str,
) -> tuple[torch.Tensor, dict]:
    conf = load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset_name.lower())
    result = build_soft_targets_from_features(
        ball_features=embeddings,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=train_mask,
        n_classes=n_classes,
        max_balls=conf.gb["max_balls"],
        min_ball_size=conf.gb["min_ball_size"],
        split_improvement=conf.gb["split_improvement"],
        min_train_per_ball=conf.gb["min_train_per_ball"],
        alpha_baseline=conf.gb["alpha_baseline"],
        target_mode=conf.gb["target_mode"],
        target_beta=conf.gb["target_beta"],
        random_state=seed,
        random_control=False,
    )
    return result.soft_target, result.diagnostics


def train_mlp_with_soft_targets(
    data,
    soft_target: torch.Tensor,
    noisy_label: torch.Tensor,
    seed: int,
    device: str,
) -> tuple[MLP, dict]:
    setup_seed(seed)
    conf = load_conf(None, "frozengrace_gb_mlp_softdist_beta05", data.name.lower())
    x = _dense_features(data.feats).float().to(device)
    adj = data.adj.to(device)
    soft_target = soft_target.float().to(device)
    noisy_label = noisy_label.long().to(device)
    clean_label = data.labels.long().to(device)
    train_idx = _mask_index(data.train_masks, device)
    val_idx = _mask_index(data.val_masks, device)
    test_idx = _mask_index(data.test_masks, device)
    model = MLP(
        in_channels=int(data.dim_feats),
        hidden_channels=int(conf.model["n_hidden"]),
        out_channels=int(data.n_classes),
        n_layers=int(conf.model["n_layer"]),
        dropout=float(conf.model["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(conf.training["lr"]),
        weight_decay=float(conf.training["weight_decay"]),
    )
    best_state = None
    best_val = -1.0
    best_train = -1.0
    best_epoch = -1
    started_at = time.perf_counter()
    for epoch in range(int(conf.training["n_epochs"])):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        log_prob = F.log_softmax(logits[train_idx], dim=1)
        loss = -(soft_target[train_idx] * log_prob).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            train_acc = accuracy(soft_target[train_idx].argmax(dim=1).cpu().numpy(), logits[train_idx].detach().cpu().numpy())
            val_acc = accuracy(noisy_label[val_idx].cpu().numpy(), logits[val_idx].detach().cpu().numpy())
        if val_acc >= best_val:
            best_val = float(val_acc)
            best_train = float(train_acc)
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        test_acc = accuracy(clean_label[test_idx].cpu().numpy(), logits[test_idx].detach().cpu().numpy())
    result = {
        "train": float(best_train),
        "valid": float(best_val),
        "test": float(test_acc),
        "best_epoch": int(best_epoch),
        "wall_time_seconds": float(time.perf_counter() - started_at),
    }
    return model, result


def extract_mlp_hidden(model: MLP, features: torch.Tensor) -> torch.Tensor:
    x = _dense_features(features).float().to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        hidden = None
        for i, layer in enumerate(model.lins[:-1]):
            x = layer(x)
            x = F.relu(x, inplace=False)
            if model.use_bn:
                x = model.bns[i](x)
            hidden = x
        if hidden is None:
            hidden = model.lins[-1](x)
    return hidden.detach().cpu()


def load_dataset_with_noise(dataset: str, seed: int, args):
    setup_seed(seed)
    data = _load_dataset(
        dataset.lower(),
        device=args.device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    return data, noisy_label, modified_mask


def run_p2a(args) -> list[dict]:
    dataset = "cora"
    seed = int(args.seed)
    data, noisy_label, modified_mask = load_dataset_with_noise(dataset, seed, args)
    anchor_path = embedding_path_for(args.embedding_root, dataset, "grace")
    anchor_h = load_frozen_embeddings(anchor_path, expected_nodes=int(data.n_nodes))
    candidate_h, pretrain_diag = train_grace_ce_embedding(
        data=data,
        noisy_label=noisy_label,
        dataset_name=dataset,
        seed=seed,
        device=args.device,
        ce_lambda=args.p2a_ce_lambda,
        epochs_override=args.p2a_epochs,
    )
    embed_dir = Path(args.output_dir) / "embeddings" / "Cora" / "grace_ce_l0p1"
    embed_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = embed_dir / "H.pt"
    torch.save(candidate_h, candidate_path)

    rows = []
    for variant, embeddings, embedding_path in [
        ("anchor_grace", anchor_h, anchor_path),
        ("grace_ce_l0p1", candidate_h, candidate_path),
    ]:
        soft_target, geom = soft_target_diagnostics_from_embedding(
            embeddings=embeddings,
            noisy_label=noisy_label,
            clean_label=data.labels,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            seed=seed,
            dataset_name=dataset,
        )
        _, downstream = train_mlp_with_soft_targets(
            data=data,
            soft_target=soft_target,
            noisy_label=noisy_label,
            seed=seed,
            device=args.device,
        )
        pretrain_fields = (
            {f"pretrain_{key}": value for key, value in pretrain_diag.items()}
            if variant == "grace_ce_l0p1"
            else {}
        )
        row = {
            "phase": "p2a",
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "embedding_path": str(embedding_path),
            "modified_train_count": int((
                _mask_bool_cpu(modified_mask, data.n_nodes) & _mask_bool_cpu(data.train_masks, data.n_nodes)
            ).sum().item()),
            **pretrain_fields,
            "num_balls": geom.get("num_balls", ""),
            "mean_ball_size": geom.get("mean_ball_size", ""),
            "true_all_purity_weighted": geom.get("true_all_purity_weighted", ""),
            "true_train_purity_weighted": geom.get("true_train_purity_weighted", ""),
            "noisy_train_purity_weighted": geom.get("noisy_train_purity_weighted", ""),
            "mean_alpha_train": geom.get("mean_alpha_train", ""),
            "downstream_test": downstream["test"],
            "downstream_valid": downstream["valid"],
            "downstream_best_epoch": downstream["best_epoch"],
        }
        rows.append(row)
    return rows


def run_p2b(args) -> list[dict]:
    rows = []
    for dataset in _parse_str_list(args.p2b_datasets):
        seed = int(args.seed)
        data, noisy_label, modified_mask = load_dataset_with_noise(dataset, seed, args)
        anchor_path = embedding_path_for(args.embedding_root, dataset, "grace")
        anchor_h = load_frozen_embeddings(anchor_path, expected_nodes=int(data.n_nodes))
        f0_target, f0_geom = soft_target_diagnostics_from_embedding(
            embeddings=anchor_h,
            noisy_label=noisy_label,
            clean_label=data.labels,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            seed=seed,
            dataset_name=dataset,
        )
        f0_model, f0_result = train_mlp_with_soft_targets(data, f0_target, noisy_label, seed, args.device)
        z1 = extract_mlp_hidden(f0_model, data.feats)
        z1_target, z1_geom = soft_target_diagnostics_from_embedding(
            embeddings=z1,
            noisy_label=noisy_label,
            clean_label=data.labels,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            seed=seed,
            dataset_name=dataset,
        )
        _, f1_result = train_mlp_with_soft_targets(data, z1_target, noisy_label, seed, args.device)
        rows.extend([
            {
                "phase": "p2b",
                "dataset": dataset.lower(),
                "variant": "F0_graceH",
                "seed": seed,
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "modified_train_count": int((
                    _mask_bool_cpu(modified_mask, data.n_nodes) & _mask_bool_cpu(data.train_masks, data.n_nodes)
                ).sum().item()),
                "num_balls": f0_geom.get("num_balls", ""),
                "true_all_purity_weighted": f0_geom.get("true_all_purity_weighted", ""),
                "true_train_purity_weighted": f0_geom.get("true_train_purity_weighted", ""),
                "noisy_train_purity_weighted": f0_geom.get("noisy_train_purity_weighted", ""),
                "downstream_test": f0_result["test"],
                "downstream_valid": f0_result["valid"],
                "downstream_best_epoch": f0_result["best_epoch"],
            },
            {
                "phase": "p2b",
                "dataset": dataset.lower(),
                "variant": "F1_loop_Z1",
                "seed": seed,
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "modified_train_count": int((
                    _mask_bool_cpu(modified_mask, data.n_nodes) & _mask_bool_cpu(data.train_masks, data.n_nodes)
                ).sum().item()),
                "num_balls": z1_geom.get("num_balls", ""),
                "true_all_purity_weighted": z1_geom.get("true_all_purity_weighted", ""),
                "true_train_purity_weighted": z1_geom.get("true_train_purity_weighted", ""),
                "noisy_train_purity_weighted": z1_geom.get("noisy_train_purity_weighted", ""),
                "downstream_test": f1_result["test"],
                "downstream_valid": f1_result["valid"],
                "downstream_best_epoch": f1_result["best_epoch"],
                "loop_delta": float(f1_result["test"] - f0_result["test"]),
            },
        ])
    return rows


def p2a_gate(rows: list[dict]) -> dict:
    by_variant = {row["variant"]: row for row in rows if row["phase"] == "p2a"}
    anchor = by_variant.get("anchor_grace")
    candidate = by_variant.get("grace_ce_l0p1")
    if not anchor or not candidate:
        return {"p2a_gate": "missing"}
    num_balls_ok = float(candidate["num_balls"]) >= 0.5 * float(anchor["num_balls"])
    purity_ok = float(candidate["true_all_purity_weighted"]) >= float(anchor["true_all_purity_weighted"]) - 0.005
    downstream_ok = float(candidate["downstream_test"]) >= float(anchor["downstream_test"])
    return {
        "p2a_gate": "pass" if num_balls_ok and purity_ok and downstream_ok else "fail",
        "num_balls_ok": num_balls_ok,
        "purity_ok": purity_ok,
        "downstream_ok": downstream_ok,
        "candidate_minus_anchor_test": float(candidate["downstream_test"]) - float(anchor["downstream_test"]),
        "candidate_minus_anchor_purity": float(candidate["true_all_purity_weighted"]) - float(anchor["true_all_purity_weighted"]),
    }


def p2b_gate(rows: list[dict]) -> dict:
    deltas = [float(row.get("loop_delta", 0.0)) for row in rows if row["phase"] == "p2b" and row["variant"] == "F1_loop_Z1"]
    return {
        "p2b_gate": "pass" if len(deltas) == 2 and all(delta > 0 for delta in deltas) else "fail",
        "positive_count": sum(1 for delta in deltas if delta > 0),
        "n": len(deltas),
        "mean_loop_delta": float(sum(deltas) / max(1, len(deltas))),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_acc(value) -> str:
    return f"{100.0 * float(value):.2f}"


def write_report(path: Path, p2a_rows: list[dict], p2b_rows: list[dict], p2a_decision: dict, p2b_decision: dict) -> None:
    lines = [
        "# MLP-8 P2 Advisor Smoke Report",
        "",
        "## Material Passport",
        "- Artifact type: smoke experiment result",
        "- Scope: P2a supervised-CE-assisted GRACE stage-1 and P2b T=2 loop",
        "- Protocol: seed0, train-only uniform label noise, percent split 0.2/0.1/0.7",
        "- Claim limit: smoke only; no 5-seed significance claim",
        "",
        "## P2a: GRACE + 0.1 noisy CE",
        "",
        "| Variant | Num balls | True all purity | Downstream test |",
        "|---|---:|---:|---:|",
    ]
    for row in p2a_rows:
        lines.append(
            f"| {row['variant']} | {row['num_balls']} | "
            f"{float(row['true_all_purity_weighted']):.4f} | {fmt_acc(row['downstream_test'])} |"
        )
    lines.extend([
        "",
        f"P2a gate: **{p2a_decision.get('p2a_gate')}**",
        f"- candidate - anchor downstream: {100.0 * float(p2a_decision.get('candidate_minus_anchor_test', 0.0)):+.2f}pp",
        f"- candidate - anchor true_all_purity: {float(p2a_decision.get('candidate_minus_anchor_purity', 0.0)):+.4f}",
        "",
        "## P2b: F0 -> Z1 -> balls -> F1 loop",
        "",
        "| Dataset | Variant | Num balls | True all purity | Test | Loop delta |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in p2b_rows:
        delta = row.get("loop_delta", "")
        delta_text = "" if delta == "" else f"{100.0 * float(delta):+.2f}pp"
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['num_balls']} | "
            f"{float(row['true_all_purity_weighted']):.4f} | {fmt_acc(row['downstream_test'])} | {delta_text} |"
        )
    lines.extend([
        "",
        f"P2b gate: **{p2b_decision.get('p2b_gate')}**",
        f"- positive datasets: {p2b_decision.get('positive_count')}/{p2b_decision.get('n')}",
        f"- mean loop delta: {100.0 * float(p2b_decision.get('mean_loop_delta', 0.0)):+.2f}pp",
        "",
        "## Decision",
        "- P2 is a direct advisor-suggestion smoke, not a new mainline unless the gate passes.",
        "- If either gate fails, do not expand the corresponding direction without a new preregistered mechanism and control.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Run MLP-8 P2 advisor-suggestion smoke experiments.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding_root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--p2a_ce_lambda", type=float, default=0.1)
    parser.add_argument("--p2a_epochs", type=int, default=None)
    parser.add_argument("--p2b_datasets", default="cora,citeseer")
    parser.add_argument("--skip_p2a", action="store_true")
    parser.add_argument("--skip_p2b", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p2a_rows = [] if args.skip_p2a else run_p2a(args)
    p2b_rows = [] if args.skip_p2b else run_p2b(args)
    p2a_decision = p2a_gate(p2a_rows) if p2a_rows else {"p2a_gate": "skipped"}
    p2b_decision = p2b_gate(p2b_rows) if p2b_rows else {"p2b_gate": "skipped"}
    write_csv(output_dir / "p2a_rows.csv", p2a_rows)
    write_csv(output_dir / "p2b_rows.csv", p2b_rows)
    write_csv(output_dir / "p2_decisions.csv", [
        {"phase": "p2a", **p2a_decision},
        {"phase": "p2b", **p2b_decision},
    ])
    write_report(output_dir / "p2_report.md", p2a_rows, p2b_rows, p2a_decision, p2b_decision)
    (output_dir / "p2_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"P2A_GATE={p2a_decision.get('p2a_gate')}")
    print(f"P2B_GATE={p2b_decision.get('p2b_gate')}")
    print(f"WROTE {output_dir / 'p2_report.md'}")


if __name__ == "__main__":
    main()
