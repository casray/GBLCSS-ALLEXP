from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import build_frozen_grace_soft_targets
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _gb_value(model_conf, key, default):
    gb_conf = getattr(model_conf, "gb", {})
    if isinstance(gb_conf, dict):
        value = gb_conf.get(key, default)
    else:
        value = getattr(gb_conf, key, default)
    return default if value is None else value


def _noise_stats(labels, noisy_label, modified_mask, train_mask) -> dict:
    n_nodes = int(labels.shape[0])
    train_idx = np.asarray(train_mask, dtype=np.int64)
    modified = np.asarray(modified_mask, dtype=np.int64)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    return {
        "modified_count": int(modified.size),
        "actual_noise_rate": float(modified.size / max(1, n_nodes)),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
    }


def diagnose_one(args, dataset_name: str, ssl_method: str, base_method: str, seed: int) -> dict:
    setup_seed(seed)
    dataset_key = dataset_name.lower()
    data = _load_dataset(
        dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    embedding_path = embedding_path_for(args.embedding_root, dataset_name, ssl_method)
    if not embedding_path.exists():
        raise FileNotFoundError(f"missing H for ssl_method={ssl_method}, dataset={dataset_name}: {embedding_path}")
    model_conf = load_conf(None, base_method, dataset_key)
    result = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=_gb_value(model_conf, "max_balls", 64),
        min_ball_size=_gb_value(model_conf, "min_ball_size", 32),
        split_improvement=_gb_value(model_conf, "split_improvement", 0.05),
        min_train_per_ball=_gb_value(model_conf, "min_train_per_ball", 2),
        alpha_baseline=_gb_value(model_conf, "alpha_baseline", "multiclass"),
        target_mode=_gb_value(model_conf, "target_mode", "distribution"),
        random_state=seed,
        random_control=False,
    )
    return {
        "dataset": dataset_key,
        "ssl_method": ssl_method,
        "base_method": base_method,
        "seed": int(seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "embedding_path": str(embedding_path),
        **_noise_stats(data.labels, noisy_label, modified_mask, data.train_masks),
        **result.diagnostics,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Diagnose frozen-embedding granular-ball soft targets.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="pubmed")
    parser.add_argument("--ssl_methods", default="grace")
    parser.add_argument("--base_method", default="frozengrace_gb_gcn_softdist_xh")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "phase10_pubmed_20260612" / "phase10_pubmed_ball_diagnostics.csv"))
    args = parser.parse_args()

    rows = []
    for dataset_name in _parse_str_list(args.datasets):
        for ssl_method in _parse_str_list(args.ssl_methods):
            for seed in _parse_int_list(args.seeds):
                print(f"BALL_DIAG dataset={dataset_name} ssl={ssl_method} seed={seed}")
                row = diagnose_one(args, dataset_name, ssl_method, args.base_method, seed)
                print(
                    "RESULT "
                    f"dataset={row['dataset']} ssl={ssl_method} seed={seed} "
                    f"num_balls={row.get('num_balls')} "
                    f"purity={float(row.get('true_all_purity_weighted', 0.0)):.4f} "
                    f"ncr={float(row.get('soft_pseudo_ncr', 0.0)):.4f} "
                    f"cdr={float(row.get('soft_pseudo_cdr', 0.0)):.4f}"
                )
                rows.append(row)
                write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
