from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _build_noisy_labels, _load_dataset
from utils.frozen_grace_targets import (
    _targets_from_node_distributions,
    apply_ball_soft_targets,
    load_frozen_embeddings,
    prepare_ball_features,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
METHODS = ("loo_gb", "exact_knn", "loo_kmeans")


def structure_storage_bytes(
    *,
    method: str,
    n_nodes: int,
    n_train: int,
    dim: int,
    groups: int,
    k: int,
) -> int:
    """Storage for the reusable grouping structure, excluding common targets."""
    int_bytes = np.dtype(np.int64).itemsize
    float_bytes = np.dtype(np.float32).itemsize
    if method == "exact_knn":
        return int(n_train) * int(k) * int_bytes
    if method in {"loo_gb", "loo_kmeans"}:
        return (
            int(n_nodes) * int_bytes
            + int(groups) * int(dim) * float_bytes
        )
    raise ValueError(f"unsupported method: {method}")


class PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.002):
        self.process = psutil.Process()
        self.interval_seconds = float(interval_seconds)
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, int(self.process.memory_info().rss))
            time.sleep(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, int(self.process.memory_info().rss))

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak - self.baseline)


def _train_indices(train_mask) -> np.ndarray:
    if isinstance(train_mask, torch.Tensor):
        train_mask = train_mask.detach().cpu().numpy()
    values = np.asarray(train_mask)
    if values.dtype == np.bool_:
        return np.flatnonzero(values).astype(np.int64)
    return values.astype(np.int64).reshape(-1)


def _reference_groups(embeddings: torch.Tensor, train_idx: np.ndarray, seed: int):
    started = time.perf_counter()
    result = build_ordinary_granular_balls(
        embeddings,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=seed,
    )
    elapsed = time.perf_counter() - started
    assignment = np.asarray(result.assignment, dtype=np.int64)
    counts = [
        int(np.sum(assignment[train_idx] == group))
        for group in sorted(np.unique(assignment).tolist())
    ]
    nonzero = [count for count in counts if count > 0]
    matched_k = max(1, int(round(float(np.mean(nonzero)))))
    return result, matched_k, elapsed


def _knn_distributions(
    neighbor_pos: np.ndarray,
    noisy_label: torch.Tensor,
    train_idx: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    labels = noisy_label.detach().cpu().long().numpy()
    distributions = np.zeros(
        (int(train_idx.size), int(n_classes)),
        dtype=np.float64,
    )
    for row_id, positions in enumerate(neighbor_pos):
        counts = np.bincount(
            labels[train_idx[positions]],
            minlength=int(n_classes),
        ).astype(np.float64)
        distributions[row_id] = counts / max(1.0, float(counts.sum()))
    return distributions


def _time_target_rebuild(
    *,
    method: str,
    repeats: int,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    train_idx: np.ndarray,
    n_classes: int,
    assignment: np.ndarray | None,
    neighbor_pos: np.ndarray | None,
) -> float:
    started = time.perf_counter()
    for _ in range(int(repeats)):
        if method in {"loo_gb", "loo_kmeans"}:
            apply_ball_soft_targets(
                noisy_label=noisy_label,
                assignment=assignment,
                train_mask=train_mask,
                n_classes=n_classes,
                min_train_per_ball=2,
                alpha_baseline="multiclass",
                target_mode="distribution",
                target_beta=0.5,
                leave_one_out=True,
                tie_mode="distribution_only",
            )
        else:
            distributions = _knn_distributions(
                neighbor_pos,
                noisy_label,
                train_idx,
                n_classes,
            )
            _targets_from_node_distributions(
                noisy_label=noisy_label,
                clean_label=clean_label,
                train_idx=train_idx,
                distributions=distributions,
                n_classes=n_classes,
                alpha_baseline="multiclass",
                target_mode="distribution",
                target_beta=0.5,
                tie_mode="distribution_only",
            )
    return (time.perf_counter() - started) / max(1, int(repeats))


def benchmark_worker(dataset: str, method: str, repeats: int) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")

    seed = 0
    setup_seed(seed)
    data = _load_dataset(
        dataset,
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, _ = _build_noisy_labels(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        seed=seed,
        noise_scope="train",
        train_mask=data.train_masks,
        val_mask=data.val_masks,
        debug=False,
    )
    config = load_conf(None, "kmeans_mlp_softdist_beta05_loo", dataset)
    embedding_path = Path(config.gb["embedding_path"])
    embeddings = prepare_ball_features(
        load_frozen_embeddings(
            embedding_path,
            expected_nodes=int(data.labels.shape[0]),
        )
    ).cpu()
    train_idx = _train_indices(data.train_masks)
    n_nodes, dim = map(int, embeddings.shape)
    n_train = int(train_idx.size)
    clean_label = data.labels.detach().cpu().long()
    noisy_label = noisy_label.detach().cpu().long()

    gc.collect()
    assignment = None
    neighbor_pos = None
    matching_seconds = 0.0
    groups = 0
    matched_k = 0

    with PeakRssSampler() as memory:
        if method == "loo_gb":
            reference, matched_k, structure_seconds = _reference_groups(
                embeddings,
                train_idx,
                seed,
            )
            assignment = np.asarray(reference.assignment, dtype=np.int64)
            groups = int(reference.summary["num_balls"])
        elif method == "exact_knn":
            reference, matched_k, matching_seconds = _reference_groups(
                embeddings,
                train_idx,
                seed,
            )
            groups = int(reference.summary["num_balls"])
            train_features = embeddings[
                torch.as_tensor(train_idx, dtype=torch.long)
            ]
            started = time.perf_counter()
            similarity = train_features @ train_features.T
            similarity.fill_diagonal_(float("-inf"))
            matched_k = min(matched_k, max(0, n_train - 1))
            neighbor_pos = torch.topk(
                similarity,
                k=matched_k,
                dim=1,
                largest=True,
            ).indices.cpu().numpy()
            structure_seconds = time.perf_counter() - started
            del similarity
        else:
            reference, matched_k, matching_seconds = _reference_groups(
                embeddings,
                train_idx,
                seed,
            )
            groups = int(reference.summary["num_balls"])
            started = time.perf_counter()
            assignment = KMeans(
                n_clusters=groups,
                random_state=seed,
                n_init=10,
            ).fit_predict(embeddings.numpy()).astype(np.int64)
            structure_seconds = time.perf_counter() - started

    rebuild_seconds = _time_target_rebuild(
        method=method,
        repeats=repeats,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=data.train_masks,
        train_idx=train_idx,
        n_classes=int(data.n_classes),
        assignment=assignment,
        neighbor_pos=neighbor_pos,
    )
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "n_nodes": n_nodes,
        "n_train": n_train,
        "embedding_dim": dim,
        "groups": groups,
        "matched_k": matched_k,
        "matching_seconds": matching_seconds,
        "structure_seconds": structure_seconds,
        "end_to_end_structure_seconds": matching_seconds + structure_seconds,
        "peak_rss_delta_bytes": memory.delta_bytes,
        "structure_storage_bytes": structure_storage_bytes(
            method=method,
            n_nodes=n_nodes,
            n_train=n_train,
            dim=dim,
            groups=groups,
            k=matched_k,
        ),
        "target_rebuild_seconds": rebuild_seconds,
        "target_rebuild_repeats": int(repeats),
        "approximate_knn_available": 0,
    }


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_controller(output: Path, repeats: int):
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--dataset",
                dataset,
                "--method",
                method,
                "--repeats",
                str(int(repeats)),
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            rows.append(payload)
            print(
                f"BENCH dataset={dataset} method={method} "
                f"time={payload['end_to_end_structure_seconds']:.4f}s "
                f"peak={payload['peak_rss_delta_bytes'] / 2**20:.2f}MiB"
            )
    _write_csv(output, rows)
    print(f"WROTE {output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "mlp45_loo_grouping_pareto_20260730"
        / "grouping_cost_benchmark.csv",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.worker:
        if args.dataset is None or args.method is None:
            raise ValueError("--dataset and --method are required in worker mode")
        print(json.dumps(
            benchmark_worker(args.dataset, args.method, args.repeats),
            sort_keys=True,
        ))
        return
    run_controller(args.output, args.repeats)


if __name__ == "__main__":
    main()
