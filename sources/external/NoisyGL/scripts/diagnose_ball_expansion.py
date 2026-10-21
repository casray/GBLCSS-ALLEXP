from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import Dataset
from utils.frozen_grace_targets import (
    diagnose_ball_expansion_candidates,
    load_frozen_embeddings,
)
from utils.labelnoise import label_process
from utils.ordinary_gb import (
    build_ordinary_granular_balls,
    build_random_size_matched_assignment,
)
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


def _diagnose_one_dataset_seed(args, dataset_name, seed):
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
        raise ValueError(f"gb.embedding_path missing in method config {args.method_config}/{dataset_name}")
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.n_nodes))
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=_conf_value(gb_conf, "max_balls", 64),
        min_ball_size=_conf_value(gb_conf, "min_ball_size", 32),
        split_improvement=_conf_value(gb_conf, "split_improvement", 0.05),
        random_state=seed,
    )
    assignments = {
        "real": gb_result.assignment,
        "shuffled": build_random_size_matched_assignment(gb_result.assignment, random_state=seed + 17),
    }

    rows = []
    for assignment_name, assignment in assignments.items():
        for min_train in args.min_train_values:
            for margin in args.margin_values:
                result = diagnose_ball_expansion_candidates(
                    noisy_label=noisy_label,
                    clean_label=data.labels,
                    assignment=assignment,
                    train_mask=data.train_masks,
                    n_classes=data.n_classes,
                    min_train_per_ball=min_train,
                    margin_threshold=margin,
                )
                rows.append({
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "assignment": assignment_name,
                    "min_train_per_ball": int(min_train),
                    "margin_threshold": float(margin),
                    "num_balls": int(gb_result.summary["num_balls"]),
                    "mean_ball_size": float(gb_result.summary["mean_ball_size"]),
                    "max_ball_size": int(gb_result.summary["max_ball_size"]),
                    **result.diagnostics,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Diagnose train-vote ball expansion precision without training.")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--min_train_values", default="5,8,10")
    parser.add_argument("--margin_values", default="0.5,0.6,0.7")
    parser.add_argument("--method_config", default="frozengrace_gb_gcn_softdist")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "ball_expansion_diag_20260610" / "ball_expansion_diag_raw_20260610.csv"))
    args = parser.parse_args()
    args.datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.seeds = _parse_int_list(args.seeds)
    args.min_train_values = _parse_int_list(args.min_train_values)
    args.margin_values = _parse_float_list(args.margin_values)

    rows = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"DIAG {dataset_name} seed={seed}")
            rows.extend(_diagnose_one_dataset_seed(args, dataset_name, seed))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "seed", "assignment", "min_train_per_ball", "margin_threshold",
        "num_balls", "mean_ball_size", "max_ball_size",
        "eligible_balls", "selected_balls", "candidate_count", "candidate_correct_count",
        "coverage_unlabeled", "coverage_total", "precision",
        "mean_selected_margin", "mean_selected_train_votes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
