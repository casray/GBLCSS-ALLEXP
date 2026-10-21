from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import DATASET_DIR_ALIASES, embedding_path_for
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.tools import setup_seed


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


class AngularProjection(nn.Module):
    """Identity-initialized linear head for an angular ball-building space."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(int(dim), int(dim), bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.proj(embeddings.float())
        return F.normalize(projected, p=2, dim=1, eps=1e-12)


def _feature_dropout(embeddings: torch.Tensor, drop_prob: float) -> torch.Tensor:
    drop_prob = float(drop_prob)
    if drop_prob <= 0.0:
        return embeddings
    if drop_prob >= 1.0:
        raise ValueError("drop probability must be < 1.0")
    keep_prob = 1.0 - drop_prob
    mask = torch.empty(
        (1, embeddings.shape[1]),
        device=embeddings.device,
        dtype=embeddings.dtype,
    ).bernoulli_(keep_prob)
    if float(mask.sum().item()) == 0.0:
        mask[0, torch.randint(embeddings.shape[1], (1,), device=embeddings.device)] = 1.0
    return embeddings * mask / keep_prob


@torch.no_grad()
def build_knn_targets(embeddings: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = F.normalize(embeddings.detach().float(), p=2, dim=1, eps=1e-12)
    n_nodes = int(base.shape[0])
    k = max(0, min(int(k), n_nodes - 1))
    if k == 0:
        empty_idx = torch.empty((n_nodes, 0), dtype=torch.long, device=base.device)
        empty_weight = torch.empty((n_nodes, 0), dtype=base.dtype, device=base.device)
        return empty_idx, empty_weight

    sim = base @ base.t()
    sim.fill_diagonal_(-float("inf"))
    values, indices = torch.topk(sim, k=k, dim=1)
    weights = F.softmax(values, dim=1)
    return indices, weights


def _same_node_info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    logits = z1 @ z2.t() / float(tau)
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def _knn_soft_relation_loss(
    z: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_weights: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if int(knn_indices.shape[1]) == 0:
        return z.sum() * 0.0
    logits = z @ z.t() / float(tau)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, -1e9)
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, knn_indices.to(device=z.device))
    return -(knn_weights.to(device=z.device, dtype=z.dtype) * target_log_probs).sum(dim=1).mean()


def angular_projection_loss(
    model: AngularProjection,
    embeddings: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_weights: torch.Tensor,
    tau: float = 0.5,
    drop1: float = 0.1,
    drop2: float = 0.2,
    knn_weight: float = 0.05,
    anchor_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tau <= 0:
        raise ValueError("tau must be positive")
    base = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
    z1 = model(_feature_dropout(base, drop1))
    z2 = model(_feature_dropout(base, drop2))
    z = model(base)

    contrastive_loss = _same_node_info_nce(z1, z2, tau)
    knn_loss = _knn_soft_relation_loss(z, knn_indices, knn_weights, tau)
    anchor_loss = (1.0 - (z * base).sum(dim=1)).mean()
    loss = contrastive_loss + float(knn_weight) * knn_loss + float(anchor_weight) * anchor_loss
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "contrastive_loss": float(contrastive_loss.detach().cpu().item()),
        "knn_loss": float(knn_loss.detach().cpu().item()),
        "anchor_loss": float(anchor_loss.detach().cpu().item()),
    }
    return loss, metrics


def _dataset_output_dir(embedding_root: Path, source_path: Path, dataset: str) -> str:
    try:
        return source_path.resolve().relative_to(embedding_root.resolve()).parts[0]
    except ValueError:
        aliases = DATASET_DIR_ALIASES.get(str(dataset).lower(), [dataset])
        for alias in aliases:
            if (embedding_root / alias).exists():
                return alias
        return aliases[0]


def train_one_dataset(args, dataset: str) -> dict:
    source_path = embedding_path_for(args.embedding_root, dataset, args.source_ssl_method)
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing source embedding for dataset={dataset}, method={args.source_ssl_method}: {source_path}"
        )

    device = torch.device(args.device)
    embeddings = load_frozen_embeddings(source_path).to(device=device)
    embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    model = AngularProjection(dim=int(embeddings.shape[1])).to(device)
    knn_indices, knn_weights = build_knn_targets(embeddings, args.k)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    metrics = {}
    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = angular_projection_loss(
            model,
            embeddings,
            knn_indices,
            knn_weights,
            tau=float(args.tau),
            drop1=float(args.drop1),
            drop2=float(args.drop2),
            knn_weight=float(args.knn_weight),
            anchor_weight=float(args.anchor_weight),
        )
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite angular projection loss at epoch {epoch}: {loss.item()}")

    model.eval()
    with torch.no_grad():
        projected = model(embeddings).detach().cpu()

    embedding_root = Path(args.embedding_root)
    output_dir = embedding_root / _dataset_output_dir(embedding_root, source_path, dataset) / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "H.pt"
    torch.save(projected, output_path)

    return {
        "dataset": str(dataset).lower(),
        "source_ssl_method": args.source_ssl_method,
        "method_name": args.method_name,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "source_embedding_path": str(source_path),
        "output_embedding_path": str(output_path),
        "n_nodes": int(projected.shape[0]),
        "dim": int(projected.shape[1]),
        "final_loss": metrics.get("loss", ""),
        "final_contrastive_loss": metrics.get("contrastive_loss", ""),
        "final_knn_loss": metrics.get("knn_loss", ""),
        "final_anchor_loss": metrics.get("anchor_loss", ""),
        "k": int(args.k),
        "knn_weight": float(args.knn_weight),
        "anchor_weight": float(args.anchor_weight),
        "tau": float(args.tau),
        "drop1": float(args.drop1),
        "drop2": float(args.drop2),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen-H angular projection used only for ball construction.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--source_ssl_method", default="grace")
    parser.add_argument("--method_name", default="grace_angproj_knn10_anchor1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--knn_weight", type=float, default=0.05)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--drop1", type=float, default=0.1)
    parser.add_argument("--drop2", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary", default=str(REPO_ROOT / "runs" / "angular_projection_min_20260623" / "projection_summary.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(int(args.seed))
    rows = []
    for dataset in _parse_str_list(args.datasets):
        print(f"ANGPROJ dataset={dataset} source={args.source_ssl_method} method={args.method_name}")
        row = train_one_dataset(args, dataset)
        print(
            "RESULT "
            f"dataset={row['dataset']} loss={float(row['final_loss']):.6f} "
            f"output={row['output_embedding_path']}"
        )
        rows.append(row)
        write_csv(Path(args.summary), rows)
    print(f"WROTE {args.summary}")


if __name__ == "__main__":
    main()
