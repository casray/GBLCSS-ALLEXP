from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import Dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


DATASET_DIR_ALIASES = {
    "cora": ["cora", "Cora"],
    "citeseer": ["citeseer", "CiteSeer"],
    "pubmed": ["pubmed", "PubMed"],
    "amazonpho": ["amazonpho", "Amazon-Photo", "AmazonPhoto"],
    "coauthorcs": ["coauthorcs", "CoauthorCS", "Coauthor-CS"],
    "coauthorph": ["coauthorph", "CoauthorPhysics", "Coauthor-Physics"],
    "wikics": ["wikics", "WikiCS", "Wiki-CS"],
}

GRACE_RATIO_FALLBACK_DIRS = [
    "ratio_train0p2_val0p1_split0_pretrain0",
]


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _canonical_dataset_name(dataset: str) -> str:
    return str(dataset).strip().lower()


def embedding_path_for(root: str | Path, dataset: str, method: str) -> Path:
    root = Path(root)
    dataset_key = _canonical_dataset_name(dataset)
    candidates = [dataset]
    candidates.extend(DATASET_DIR_ALIASES.get(dataset_key, []))
    seen = set()
    for dataset_dir in candidates:
        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        path = root / dataset_dir / method / "H.pt"
        if path.exists():
            return path
        if str(method).lower() == "grace":
            for fallback_dir in GRACE_RATIO_FALLBACK_DIRS:
                fallback_path = root / dataset_dir / fallback_dir / "H.pt"
                if fallback_path.exists():
                    return fallback_path
    return root / dataset / method / "H.pt"


def _load_dataset(dataset_name, device, split_type, train_percent, val_percent, test_percent):
    dataset_key = _canonical_dataset_name(dataset_name)
    data_conf = load_conf(str(REPO_ROOT / "config" / "_dataset" / f"{dataset_key}.yaml"))
    if split_type is not None:
        data_conf.split["split_type"] = split_type
    if train_percent is not None:
        data_conf.split["train_percent"] = train_percent
    if val_percent is not None:
        data_conf.split["val_percent"] = val_percent
    if test_percent is not None:
        data_conf.split["test_percent"] = test_percent
    if data_conf.split["split_type"] == "percent":
        for key in [
            "train_size",
            "val_size",
            "test_size",
            "train_examples_per_class",
            "val_examples_per_class",
            "test_examples_per_class",
        ]:
            data_conf.split[key] = None
    return Dataset(
        dataset_key,
        path=str(REPO_ROOT / "data") + "/",
        feat_norm=data_conf.norm["feat_norm"],
        adj_norm=data_conf.norm["adj_norm"],
        train_size=data_conf.split["train_size"],
        val_size=data_conf.split["val_size"],
        test_size=data_conf.split["test_size"],
        train_percent=data_conf.split["train_percent"],
        val_percent=data_conf.split["val_percent"],
        test_percent=data_conf.split["test_percent"],
        train_examples_per_class=data_conf.split["train_examples_per_class"],
        val_examples_per_class=data_conf.split["val_examples_per_class"],
        test_examples_per_class=data_conf.split["test_examples_per_class"],
        add_self_loop=data_conf.modify["add_self_loop"],
        from_npz=data_conf.modify["from_npz_largest_component"],
        device=device,
        split_type=data_conf.split["split_type"],
    )


def _mask_to_bool_tensor(mask, n_nodes: int, device) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            out = mask.detach().bool().to(device)
        else:
            out = torch.zeros(int(n_nodes), dtype=torch.bool, device=device)
            out[mask.detach().long().to(device)] = True
        return out
    out = torch.zeros(int(n_nodes), dtype=torch.bool, device=device)
    out[torch.as_tensor(mask, dtype=torch.long, device=device)] = True
    return out


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    return float((pred == labels[mask]).float().mean().item())


def train_linear_probe(
    embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    eval_labels: torch.Tensor,
    train_mask,
    val_mask,
    test_mask,
    n_classes: int,
    seed: int,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: torch.device | str = torch.device("cpu"),
    normalize: bool = True,
) -> dict:
    setup_seed(seed)
    device = torch.device(device)
    x = embeddings.detach().float().to(device)
    if normalize:
        x = F.normalize(x, p=2, dim=1)
    train_labels = train_labels.detach().long().to(device)
    eval_labels = eval_labels.detach().long().to(device)
    n_nodes = int(x.shape[0])
    train_mask = _mask_to_bool_tensor(train_mask, n_nodes, device)
    val_mask = _mask_to_bool_tensor(val_mask, n_nodes, device)
    test_mask = _mask_to_bool_tensor(test_mask, n_nodes, device)
    model = LinearProbe(int(x.shape[1]), int(n_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_test = 0.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_mask], train_labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            val_acc = _accuracy(logits, eval_labels, val_mask)
            test_acc = _accuracy(logits, eval_labels, test_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if int(patience) > 0 and stale >= int(patience):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        train_acc = _accuracy(logits, eval_labels, train_mask)
        val_acc = _accuracy(logits, eval_labels, val_mask)
        test_acc = _accuracy(logits, eval_labels, test_mask)
    return {
        "best_val_acc": float(best_val),
        "test_acc_at_best_val": float(best_test),
        "best_epoch": int(best_epoch),
        "train_acc_at_best_state": float(train_acc),
        "val_acc_at_best_state": float(val_acc),
        "test_acc_at_best_state": float(test_acc),
    }


def _probe_one(args, dataset_name: str, method: str, seed: int) -> list[dict]:
    setup_seed(seed)
    data = _load_dataset(
        dataset_name=dataset_name,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    embedding_path = embedding_path_for(args.embedding_root, dataset_name, method)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding for dataset={dataset_name}, method={method}: {embedding_path}")
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.n_nodes))
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    rows = []
    for probe_label, train_labels in (("clean", data.labels), ("noisy", noisy_label)):
        result = train_linear_probe(
            embeddings=embeddings,
            train_labels=train_labels,
            eval_labels=data.labels,
            train_mask=data.train_masks,
            val_mask=data.val_masks,
            test_mask=data.test_masks,
            n_classes=data.n_classes,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            normalize=not args.no_normalize,
        )
        rows.append({
            "dataset": _canonical_dataset_name(dataset_name),
            "method": method,
            "seed": int(seed),
            "probe_label": probe_label,
            "embedding_path": str(embedding_path),
            "embedding_dim": int(embeddings.shape[1]),
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "split_type": args.split_type,
            "train_percent": float(args.train_percent),
            "val_percent": float(args.val_percent),
            "test_percent": float(args.test_percent),
            **result,
        })
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["probe_label"])].append(row)
    out_rows = []
    for (dataset, method, probe_label), group in sorted(groups.items()):
        values = [float(row["test_acc_at_best_val"]) for row in group]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        out_rows.append({
            "dataset": dataset,
            "method": method,
            "probe_label": probe_label,
            "num_seeds": len(group),
            "test_acc_mean": mean,
            "test_acc_std": math.sqrt(variance),
            "best_val_mean": sum(float(row["best_val_acc"]) for row in group) / len(group),
            "embedding_dim": int(group[0]["embedding_dim"]),
        })
    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, raw_rows: list[dict], aggregate: list[dict], grace_method: str) -> None:
    by_key = {(row["dataset"], row["method"], row["probe_label"]): row for row in aggregate}
    lines = [
        "# SSL Linear Probe Report",
        "",
        "## Protocol",
        "",
        "- Split: percent 0.2/0.1/0.7, seeds 0-2.",
        "- Probe: one linear layer on frozen H, selected by clean validation accuracy.",
        "- Clean probe trains on clean train labels; noisy probe trains on noisy train labels and evaluates on clean val/test.",
        "- SSL embeddings are treated as label-free upstream artifacts; labels are used only inside this probe.",
        "",
        "## Aggregate",
        "",
        "| Dataset | Method | Probe label | Test acc mean | Test acc std | Delta vs GRACE | Pass >=1pp |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        grace = by_key.get((row["dataset"], grace_method, row["probe_label"]))
        delta = float(row["test_acc_mean"]) - float(grace["test_acc_mean"]) if grace else 0.0
        passes = int(row["method"] != grace_method and delta >= 0.01)
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['probe_label']} | "
            f"{float(row['test_acc_mean']):.4f} | {float(row['test_acc_std']):.4f} | {delta:.4f} | {passes} |"
        )
    lines.extend([
        "",
        "## Research Record",
        "",
        "- Hypothesis: stronger label-free SSL H improves both classifier input and ball geometry.",
        "- Granular-ball role: unchanged downstream consumer of H; no ball mechanism is modified in Phase 8.",
        "- Code change: added `scripts/linear_probe_embeddings.py` for clean/noisy frozen-H probing.",
        "- Config change: none.",
        "- Success standard: any candidate exceeds GRACE by >=1pp in clean or noisy probe on either dataset.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run clean/noisy linear probes on frozen SSL embeddings.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--methods", default="grace")
    parser.add_argument("--grace_method", default="grace")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "ssl_probe_20260612" / "probe_summary.csv"))
    parser.add_argument("--aggregate_output", default=None)
    parser.add_argument("--report_output", default=None)
    args = parser.parse_args()
    args.datasets = _parse_str_list(args.datasets)
    args.methods = _parse_str_list(args.methods)
    args.seeds = _parse_int_list(args.seeds)

    rows = []
    for dataset_name in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                print(f"PROBE dataset={dataset_name} method={method} seed={seed}")
                rows.extend(_probe_one(args, dataset_name, method, seed))

    output_path = Path(args.output)
    aggregate_path = Path(args.aggregate_output) if args.aggregate_output else output_path.with_name("probe_aggregate.csv")
    report_path = Path(args.report_output) if args.report_output else output_path.with_name("probe_report.md")
    aggregate = aggregate_rows(rows)
    write_csv(output_path, rows)
    write_csv(aggregate_path, aggregate)
    write_report(report_path, rows, aggregate, args.grace_method)
    print(f"WROTE {output_path}")
    print(f"WROTE {aggregate_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
