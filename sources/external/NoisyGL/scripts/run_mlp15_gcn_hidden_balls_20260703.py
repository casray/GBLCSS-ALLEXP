from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import GCN
from scripts.run_ssl_swap_downstream import run_one
from scripts.run_standard_downstream import _split_fingerprint
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.functional import accuracy
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp15_gcn_hidden_balls_20260703"
DEFAULT_GRACE_ROOT = REPO_ROOT.parent.parent / "gb_graph_aware_minival_6_1" / "checkpoints" / "embeddings_v2"
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MLP_CE = "mlp_ce"


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _ratio_token(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _dense_features(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if getattr(x, "is_sparse", False) else x


def _mask_index(mask, device: str) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach()
    else:
        tensor = torch.as_tensor(mask)
    if tensor.dtype == torch.bool:
        return torch.where(tensor.cpu())[0].to(device=device, dtype=torch.long)
    return tensor.to(device=device, dtype=torch.long)


def _train_gcn_hidden(
    *,
    dataset: str,
    train_percent: float,
    val_percent: float,
    test_percent: float,
    seed: int,
    noise_type: str,
    noise_rate: float,
    device: str,
    output_root: Path,
) -> dict:
    setup_seed(seed)
    data = _load_dataset(
        dataset,
        device=device,
        split_type="percent",
        train_percent=train_percent,
        val_percent=val_percent,
        test_percent=test_percent,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=noise_type,
        noise_rate=noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    conf = load_conf(None, "gcn", dataset.lower())
    conf.model["n_feat"] = int(data.dim_feats)
    conf.model["n_classes"] = int(data.n_classes)
    x = _dense_features(data.feats).float().to(device)
    adj = data.adj.to(device)
    labels = noisy_label.long().to(device)
    clean = data.labels.long().to(device)
    train_idx = _mask_index(data.train_masks, device)
    val_idx = _mask_index(data.val_masks, device)
    test_idx = _mask_index(data.test_masks, device)
    model = GCN(
        in_channels=int(data.dim_feats),
        hidden_channels=int(conf.model["n_hidden"]),
        out_channels=int(data.n_classes),
        n_layers=int(conf.model["n_layer"]),
        dropout=float(conf.model["dropout"]),
        norm_info=conf.model["norm_info"],
        act=conf.model["act"],
        input_layer=conf.model["input_layer"],
        output_layer=conf.model["output_layer"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(conf.training["lr"]),
        weight_decay=float(conf.training["weight_decay"]),
    )
    best_state = None
    best_val = -1.0
    best_epoch = -1
    best_train = -1.0
    started = time.perf_counter()
    for epoch in range(int(conf.training["n_epochs"])):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            train_acc = accuracy(labels[train_idx].detach().cpu().numpy(), logits[train_idx].detach().cpu().numpy())
            val_acc = accuracy(labels[val_idx].detach().cpu().numpy(), logits[val_idx].detach().cpu().numpy())
        if float(val_acc) >= best_val:
            best_val = float(val_acc)
            best_train = float(train_acc)
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, hidden = model.forward_with_hidden(x, adj)
        test_acc = accuracy(clean[test_idx].detach().cpu().numpy(), logits[test_idx].detach().cpu().numpy())
    method = f"gcn_hidden_train{_ratio_token(train_percent)}"
    dataset_dir = {"cora": "Cora", "citeseer": "CiteSeer"}.get(dataset.lower(), dataset)
    embed_dir = output_root / "embeddings" / dataset_dir / method
    embed_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embed_dir / "H.pt"
    torch.save(hidden.detach().cpu(), embedding_path)
    return {
        "dataset": dataset.lower(),
        "seed": int(seed),
        "train_percent": float(train_percent),
        "val_percent": float(val_percent),
        "test_percent": float(test_percent),
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "method": method,
        "embedding_path": str(embedding_path),
        "gcn_train": float(best_train),
        "gcn_valid": float(best_val),
        "gcn_test": float(test_acc),
        "gcn_best_epoch": int(best_epoch),
        "hidden_dim": int(hidden.shape[1]),
        "modified_train_count": int(torch.as_tensor(modified_mask).bool().sum().item()),
        "wall_time_seconds": float(time.perf_counter() - started),
    }


def _downstream_args(
    *,
    embedding_root: Path,
    dataset: str,
    train_percent: float,
    val_percent: float,
    test_percent: float,
    seed: int,
    noise_type: str,
    noise_rate: float,
    device: str,
    output_path: Path,
    ssl_method: str,
    base_methods: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_root=str(embedding_root),
        ball_embedding_root=None,
        datasets=dataset,
        ssl_methods=ssl_method,
        feature_ssl_method=None,
        base_methods=base_methods,
        seeds=str(seed),
        noise_type=noise_type,
        noise_rate=float(noise_rate),
        noise_scope="train",
        device=device,
        split_type="percent",
        train_percent=float(train_percent),
        val_percent=float(val_percent),
        test_percent=float(test_percent),
        gb_max_balls=None,
        gb_min_ball_size=None,
        gb_split_improvement=None,
        gb_keep_ratio=None,
        gb_lambda_u=None,
        config_overrides={},
        debug=False,
        output=str(output_path),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _method_label(row: dict) -> str:
    method = row.get("method", "")
    ssl = row.get("ssl_method", "")
    if method == OURS and ssl == "grace":
        return "GRACE anchor"
    if method == OURS and str(ssl).startswith("gcn_hidden"):
        return "GCN-hidden real"
    if method == SHUFFLED and str(ssl).startswith("gcn_hidden"):
        return "GCN-hidden shuffled"
    if method == MLP_CE:
        return "MLP+CE"
    return method


def _summarize(output_dir: Path, rows: list[dict], gcn_rows: list[dict]) -> tuple[list[dict], list[dict], str]:
    summary: list[dict] = []
    deltas: list[dict] = []
    grouped: dict[tuple[str, float], dict[str, dict]] = {}
    for row in rows:
        key = (row["dataset"], float(row["train_percent"]))
        grouped.setdefault(key, {})[_method_label(row)] = row
        summary.append({
            "dataset": row["dataset"],
            "train_percent": float(row["train_percent"]),
            "method_label": _method_label(row),
            "test": float(row["test"]),
            "valid": float(row["valid"]),
            "num_balls": row.get("num_balls", ""),
            "noisy_train_purity_weighted": row.get("noisy_train_purity_weighted", ""),
            "true_train_purity_weighted": row.get("true_train_purity_weighted", ""),
            "alpha_positive_train_count": row.get("alpha_positive_train_count", ""),
        })
    for grow in gcn_rows:
        summary.append({
            "dataset": grow["dataset"],
            "train_percent": float(grow["train_percent"]),
            "method_label": "GCN+CE reference",
            "test": float(grow["gcn_test"]),
            "valid": float(grow["gcn_valid"]),
            "num_balls": "",
            "noisy_train_purity_weighted": "",
            "true_train_purity_weighted": "",
            "alpha_positive_train_count": "",
        })
    for key, values in sorted(grouped.items()):
        real = values.get("GCN-hidden real")
        if not real:
            continue
        for baseline in ["GRACE anchor", "GCN-hidden shuffled", "MLP+CE"]:
            base = values.get(baseline)
            if base:
                deltas.append({
                    "dataset": key[0],
                    "train_percent": key[1],
                    "comparison": f"gcn_hidden_real_minus_{baseline}",
                    "delta": float(real["test"]) - float(base["test"]),
                    "real_test": float(real["test"]),
                    "baseline_test": float(base["test"]),
                })
    lines = [
        "# MLP-15 GCN-Hidden Balls Report",
        "",
        "## Protocol",
        "",
        "- Scope: why-GRACE diagnostic; use noisy-label GCN hidden embeddings as ball features, then train the normal SoftDist-MLP student.",
        "- Datasets: Cora, CiteSeer.",
        "- Train ratios: 0.05, 0.10, 0.20; validation ratio fixed at 0.10.",
        "- Noise: train-only uniform 0.4; seed 0.",
        "- Arms: frozen-GRACE anchor, GCN-hidden real, GCN-hidden shuffled, MLP+CE, and GCN+CE reference.",
        "",
        "## Results",
        "",
        "| Dataset | train% | Method | Test | Valid | Balls | True train purity |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: (item["dataset"], item["train_percent"], item["method_label"])):
        balls = row["num_balls"]
        purity = row["true_train_purity_weighted"]
        lines.append(
            f"| {row['dataset']} | {100.0 * float(row['train_percent']):.0f} | {row['method_label']} | "
            f"{100.0 * float(row['test']):.2f} | {100.0 * float(row['valid']):.2f} | "
            f"{balls} | {purity} |"
        )
    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "| Dataset | train% | Comparison | Delta |",
        "|---|---:|---|---:|",
    ])
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {100.0 * float(row['train_percent']):.0f} | {row['comparison']} | "
            f"{100.0 * float(row['delta']):+.2f}pp |"
        )
    pass_grace = sum(1 for row in deltas if row["comparison"].endswith("GRACE anchor") and row["delta"] > 0.005)
    pass_shuffle = sum(1 for row in deltas if row["comparison"].endswith("GCN-hidden shuffled") and row["delta"] > 0.0)
    lines.extend([
        "",
        "## Decision",
        "",
        f"- GCN-hidden real beats GRACE anchor by >0.5pp in {pass_grace}/6 settings.",
        f"- GCN-hidden real beats its shuffled control in {pass_shuffle}/6 settings.",
        "- This is a seed0 diagnostic only. Expand only if it beats GRACE anchor in at least 4/6 settings and beats shuffled in at least 5/6 settings.",
    ])
    report = "\n".join(lines) + "\n"
    (output_dir / "mlp15_report.md").write_text(report, encoding="utf-8")
    return summary, deltas, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use noisy-label GCN hidden embeddings as SoftDist-MLP ball features.")
    parser.add_argument("--output_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--grace_root", default=str(DEFAULT_GRACE_ROOT))
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--train_percents", default="0.05,0.10,0.20")
    parser.add_argument("--val_percent", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _parse_str_list(args.datasets)
    train_percents = _parse_float_list(args.train_percents)
    all_rows: list[dict] = []
    gcn_rows: list[dict] = []
    for dataset in datasets:
        for train_percent in train_percents:
            test_percent = 1.0 - float(train_percent) - float(args.val_percent)
            if test_percent <= 0:
                raise ValueError(f"invalid train/val split: train={train_percent}, val={args.val_percent}")
            print(f"GCN_HIDDEN dataset={dataset} train_percent={train_percent} seed={args.seed}")
            gcn_row = _train_gcn_hidden(
                dataset=dataset,
                train_percent=float(train_percent),
                val_percent=float(args.val_percent),
                test_percent=float(test_percent),
                seed=int(args.seed),
                noise_type=args.noise_type,
                noise_rate=float(args.noise_rate),
                device=args.device,
                output_root=output_dir,
            )
            gcn_rows.append(gcn_row)
            token = _ratio_token(train_percent)
            method = f"gcn_hidden_train{token}"
            setting_prefix = f"{dataset}_train{token}"
            for label, embedding_root, ssl_method, methods in [
                ("anchor", Path(args.grace_root), "grace", OURS),
                ("gcn_hidden", output_dir / "embeddings", method, f"{OURS},{SHUFFLED}"),
                ("mlp_ce", Path(args.grace_root), "grace", MLP_CE),
            ]:
                out_path = output_dir / f"{setting_prefix}_{label}.csv"
                ds_args = _downstream_args(
                    embedding_root=embedding_root,
                    dataset=dataset,
                    train_percent=float(train_percent),
                    val_percent=float(args.val_percent),
                    test_percent=float(test_percent),
                    seed=int(args.seed),
                    noise_type=args.noise_type,
                    noise_rate=float(args.noise_rate),
                    device=args.device,
                    output_path=out_path,
                    ssl_method=ssl_method,
                    base_methods=methods,
                )
                for base_method in _parse_str_list(methods):
                    row = run_one(ds_args, dataset, ssl_method, base_method, int(args.seed))
                    row["train_percent"] = float(train_percent)
                    row["val_percent"] = float(args.val_percent)
                    row["test_percent"] = float(test_percent)
                    row["diagnostic_arm"] = label
                    all_rows.append(row)
                    _write_csv(out_path, _load_csv(out_path) + [row])
                    print(
                        f"RESULT dataset={dataset} train={train_percent:.2f} arm={label} "
                        f"method={base_method} test={float(row['test']):.4f}"
                    )
    _write_csv(output_dir / "gcn_hidden_pretrain.csv", gcn_rows)
    _write_csv(output_dir / "mlp15_raw.csv", all_rows)
    summary, deltas, _report = _summarize(output_dir, all_rows, gcn_rows)
    _write_csv(output_dir / "mlp15_summary.csv", summary)
    _write_csv(output_dir / "mlp15_paired_deltas.csv", deltas)
    (output_dir / "mlp15_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output_dir / 'mlp15_report.md'}")


if __name__ == "__main__":
    main()
