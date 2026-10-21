from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import GCN
from utils.dataloader import Dataset
from utils.frozen_grace_targets import (
    build_frozen_grace_soft_targets,
    diagnostics_for_early_arbitration,
    load_frozen_embeddings,
)
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


def _conf_value(conf, key, default):
    if isinstance(conf, dict):
        value = conf.get(key, default)
    else:
        value = getattr(conf, key, default)
    return default if value is None else value


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_dataset(dataset_name, device, split_type, train_percent, val_percent, test_percent):
    data_conf = load_conf(str(REPO_ROOT / "config" / "_dataset" / f"{dataset_name}.yaml"))
    if split_type is not None:
        data_conf.split["split_type"] = split_type
    if train_percent is not None:
        data_conf.split["train_percent"] = train_percent
    if val_percent is not None:
        data_conf.split["val_percent"] = val_percent
    if test_percent is not None:
        data_conf.split["test_percent"] = test_percent
    return Dataset(
        dataset_name,
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


def _prepare_warmup_features(features, embeddings, warmup_features, device):
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    features = features.float().to(device)
    warmup_features = str(warmup_features)
    if warmup_features == "x":
        return features
    if warmup_features != "xh":
        raise ValueError("warmup_features must be 'xh' or 'x'")
    embeddings = embeddings.float().to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return torch.cat([features, embeddings], dim=1)


def _warmup_early_probs(data, noisy_label, model_conf, embeddings, args):
    features = _prepare_warmup_features(data.feats, embeddings, args.warmup_features, data.feats.device)
    model = GCN(
        in_channels=int(features.shape[1]),
        hidden_channels=model_conf.model["n_hidden"],
        out_channels=data.n_classes,
        n_layers=model_conf.model["n_layer"],
        dropout=model_conf.model["dropout"],
        norm_info=model_conf.model["norm_info"],
        act=model_conf.model["act"],
        input_layer=model_conf.model["input_layer"],
        output_layer=model_conf.model["output_layer"],
    ).to(data.feats.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_conf.training["lr"],
        weight_decay=model_conf.training["weight_decay"],
    )
    snapshots = []
    train_mask = data.train_masks
    for epoch in range(1, int(args.early_warmup_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.adj)
        loss = F.cross_entropy(logits[train_mask], noisy_label[train_mask])
        loss.backward()
        optimizer.step()
        if int(args.early_window_start) <= epoch <= int(args.early_window_end):
            model.eval()
            with torch.no_grad():
                snapshots.append(F.softmax(model(features, data.adj), dim=1).detach().cpu())
    if not snapshots:
        raise ValueError("early window produced no snapshots")
    return torch.stack(snapshots, dim=0).mean(dim=0).to(data.feats.device)


def _diagnose_one(args, dataset_name, seed):
    setup_seed(seed)
    data = _load_dataset(
        dataset_name=dataset_name,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    model_conf = load_conf(None, args.method_config, dataset_name)
    gb_conf = getattr(model_conf, "gb", {})
    embedding_path = _conf_value(gb_conf, "embedding_path", None)
    if embedding_path is None:
        raise ValueError(f"gb.embedding_path missing in {args.method_config}/{dataset_name}")
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.n_nodes))
    early_probs = _warmup_early_probs(data, noisy_label, model_conf, embeddings, args)
    node_idx = torch.arange(int(noisy_label.shape[0]), device=noisy_label.device)
    early_noisy_conf = early_probs[node_idx, noisy_label.long()]
    soft_result = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=_conf_value(gb_conf, "max_balls", 64),
        min_ball_size=_conf_value(gb_conf, "min_ball_size", 32),
        split_improvement=_conf_value(gb_conf, "split_improvement", 0.05),
        min_train_per_ball=_conf_value(gb_conf, "min_train_per_ball", 2),
        alpha_baseline=_conf_value(gb_conf, "alpha_baseline", "multiclass"),
        target_mode=_conf_value(gb_conf, "target_mode", "distribution"),
        random_state=seed,
        random_control=False,
        ball_mode="geometric",
    )
    rows = []
    for tau in args.tau_values:
        diag = diagnostics_for_early_arbitration(
            noisy_label=noisy_label,
            clean_label=data.labels,
            alpha=soft_result.alpha,
            pseudo_label=soft_result.pseudo_label,
            train_mask=data.train_masks,
            early_noisy_conf=early_noisy_conf,
            tau_protect=tau,
        )
        changed = (soft_result.alpha > 0) & (soft_result.pseudo_label != noisy_label)
        changed_conf = early_noisy_conf[changed]
        rows.append({
            "dataset": dataset_name,
            "seed": int(seed),
            "warmup_features": str(args.warmup_features),
            "tau_protect": float(tau),
            "early_warmup_epochs": int(args.early_warmup_epochs),
            "early_window_start": int(args.early_window_start),
            "early_window_end": int(args.early_window_end),
            "changed_mean_early_noisy_conf": float(changed_conf.mean().item()) if changed_conf.numel() else 0.0,
            **diag,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Diagnose early-learning protection for SoftDist corrections.")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--tau_values", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--method_config", default="frozengrace_gb_gcn_softdist_xh")
    parser.add_argument("--warmup_features", default="xh", choices=["xh", "x"])
    parser.add_argument("--early_warmup_epochs", type=int, default=10)
    parser.add_argument("--early_window_start", type=int, default=3)
    parser.add_argument("--early_window_end", type=int, default=10)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "early_arbitration_diag_20260611" / "early_arbitration_diag_raw_20260611.csv"))
    args = parser.parse_args()
    args.datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.seeds = _parse_int_list(args.seeds)
    args.tau_values = _parse_float_list(args.tau_values)

    rows = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"EARLY_DIAG {dataset_name} seed={seed}")
            rows.extend(_diagnose_one(args, dataset_name, seed))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "seed", "tau_protect", "early_warmup_epochs", "early_window_start", "early_window_end",
        "warmup_features",
        "changed_mean_early_noisy_conf",
        "baseline_changed_train_count", "protected_count", "protected_ratio_changed",
        "protection_precision", "baseline_ncr", "protected_ncr", "ncr_relative_loss",
        "baseline_cdr", "protected_cdr", "cdr_relative_reduction",
        "baseline_ncr_count", "protected_ncr_count", "baseline_cdr_count", "protected_cdr_count",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
