from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from scripts.run_standard_downstream import _noise_fingerprint, _noise_stats, _split_fingerprint
from utils.frozen_grace_targets import build_soft_targets_from_features, load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import resolve_effective_max_balls
from utils.tools import setup_seed


DEFAULT_DATASETS = (
    "cora",
    "citeseer",
    "corafull",
    "coauthorcs",
    "amazonpho",
    "citeseerfull",
    "coauthorph",
    "pubmed",
    "amazoncom",
    "coraml",
    "attribwiki",
)


def _ball_sizes(assignment) -> list[int]:
    _, counts = np.unique(np.asarray(assignment, dtype=np.int64), return_counts=True)
    return sorted(int(value) for value in counts.tolist())


def compare_geometry_variants(
    embeddings: torch.Tensor,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    dataset: str,
    seed: int,
    configured_max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
) -> list[dict[str, object]]:
    n_nodes = int(embeddings.shape[0])
    variants = (
        ("fixed", "fixed", False),
        ("sqrt_nodes", "sqrt_nodes", False),
        ("sqrt_nodes_shuffled", "sqrt_nodes", True),
    )
    built = {}
    rows = []
    for variant, policy, random_control in variants:
        effective = resolve_effective_max_balls(
            n_nodes=n_nodes,
            configured_max_balls=configured_max_balls,
            policy=policy,
        )
        started = time.perf_counter()
        result = build_soft_targets_from_features(
            ball_features=embeddings,
            noisy_label=noisy_label,
            clean_label=clean_label,
            train_mask=train_mask,
            n_classes=n_classes,
            max_balls=effective,
            min_ball_size=min_ball_size,
            split_improvement=split_improvement,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            random_state=seed,
            random_control=random_control,
        )
        elapsed = time.perf_counter() - started
        built[variant] = result
        row = {
            "dataset": str(dataset).lower(),
            "seed": int(seed),
            "variant": variant,
            "status": "ok",
            "max_balls_policy": policy,
            "configured_max_balls": int(configured_max_balls),
            "effective_max_balls": int(effective),
            "max_balls_cap_hit": int(int(result.diagnostics.get("num_balls", 0)) >= effective),
            "ball_build_time_seconds": float(elapsed),
            **result.diagnostics,
        }
        num_balls = max(1, int(result.diagnostics.get("num_balls", 0)))
        train_array = np.asarray(train_mask)
        train_count = int(train_array.sum()) if train_array.dtype == np.bool_ else int(train_array.size)
        row["mean_train_per_ball"] = float(train_count / num_balls)
        rows.append(row)

    fixed = built["fixed"]
    adaptive = built["sqrt_nodes"]
    shuffled = built["sqrt_nodes_shuffled"]
    adaptive_row = next(row for row in rows if row["variant"] == "sqrt_nodes")
    adaptive_row.update(
        {
            "assignment_equal_fixed": int(np.array_equal(adaptive.assignment, fixed.assignment)),
            "soft_target_equal_fixed": int(torch.equal(adaptive.soft_target, fixed.soft_target)),
            "alpha_equal_fixed": int(torch.equal(adaptive.alpha, fixed.alpha)),
        }
    )
    shuffled_row = next(row for row in rows if row["variant"] == "sqrt_nodes_shuffled")
    shuffled_row["size_multiset_equal_adaptive"] = int(
        _ball_sizes(shuffled.assignment) == _ball_sizes(adaptive.assignment)
    )
    shuffled_row["size_multiset_equal_source"] = int(
        shuffled.diagnostics.get("random_control_size_matched", 0)
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP-24 fixed vs sqrt-node ball geometry diagnostic.")
    parser.add_argument("--embedding_root", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", choices=["train"], default="train")
    parser.add_argument("--split_type", choices=["percent"], default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        setup_seed(args.seed)
        data = _load_dataset(
            dataset,
            device="cpu",
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
            random_seed=args.seed,
            eligible_mask=data.train_masks,
            debug=False,
        )
        embedding_path = embedding_path_for(args.embedding_root, dataset, "grace")
        if not embedding_path.exists():
            raise FileNotFoundError(f"missing GRACE embedding for {dataset}: {embedding_path}")
        embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.labels.shape[0]))
        dataset_rows = compare_geometry_variants(
            embeddings=embeddings,
            noisy_label=noisy_label.cpu(),
            clean_label=data.labels.cpu(),
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            dataset=dataset,
            seed=args.seed,
        )
        common = {
            "embedding_path": str(embedding_path),
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "noise_scope": args.noise_scope,
            "train_percent": float(args.train_percent),
            "val_percent": float(args.val_percent),
            "test_percent": float(args.test_percent),
            "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
            "noise_hash": _noise_fingerprint(data.labels, noisy_label),
            **_noise_stats(
                data.labels,
                noisy_label,
                modified_mask,
                data.train_masks,
                data.val_masks,
                data.test_masks,
            ),
        }
        for row in dataset_rows:
            row.update(common)
        rows.extend(dataset_rows)
        print(f"GEOMETRY_DONE dataset={dataset} rows={len(dataset_rows)}")
    write_csv(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
