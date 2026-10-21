from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
from gblcss.data import load_public_dataset, set_seed
from gblcss.experiment import load_config
from gblcss.granular_ball import build_granular_balls
from gblcss.representation import load_frozen_representation


def structure_storage_bytes(method, n_nodes, n_train, dimension, groups, k):
    """Retained group indices and float32 centers; shared embeddings are excluded."""
    if method == 'exact_knn':
        return n_train * k * np.dtype(np.int64).itemsize
    return n_nodes * np.dtype(np.int64).itemsize + groups * dimension * np.dtype(np.float32).itemsize


def benchmark(h, train_index, method, config, seed):
    set_seed(seed)
    h = F.normalize(h.detach().cpu().float(), dim=1)
    train_index = torch.as_tensor(train_index).long()
    started = perf_counter()
    reference = build_granular_balls(h, max_balls=config['max_balls'], min_ball_size=config['min_ball_size'],
                                    split_improvement=config['split_improvement'], random_state=seed)
    reference_seconds = perf_counter() - started
    groups = int(reference.summary['num_balls'])
    counts = np.bincount(reference.assignment[train_index.numpy()], minlength=groups)
    k = min(max(1, round(float(counts[counts > 0].mean()))), max(0, len(train_index)-1))
    matching_seconds = 0.0
    construction_seconds = reference_seconds
    if method != 'granular_ball':
        matching_seconds = reference_seconds
        started = perf_counter()
        if method == 'exact_knn':
            x = h[train_index]
            similarity = x @ x.T
            similarity.fill_diagonal_(float('-inf'))
            neighbors = torch.topk(similarity, k, dim=1).indices
            assert neighbors.shape == (len(train_index), k)
        elif method == 'kmeans':
            KMeans(n_clusters=groups, n_init=10, random_state=seed).fit_predict(h.numpy())
        else:
            raise ValueError(f'unknown grouping rule: {method}')
        construction_seconds = matching_seconds + perf_counter() - started
    return dict(method=method, seed=seed, n_nodes=len(h), n_train=len(train_index), groups=groups,
                matched_k=k, matching_seconds=matching_seconds, construction_seconds=construction_seconds,
                storage_bytes=structure_storage_bytes(method, len(h), len(train_index), h.shape[1], groups, k))


def main():
    parser = argparse.ArgumentParser(description='Measure grouping construction time and retained storage for Figure 5.')
    parser.add_argument('--datasets', default='cora,citeseer,corafull,coauthorcs,amazonphoto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rerun_grouping_cost.csv')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = load_config(ROOT / 'configs/main/gblcss.yaml')['method']
    rows = []
    for key in args.datasets.split(','):
        key = key.strip().lower()
        dataset = load_config(ROOT / 'configs/datasets' / f'{key}.yaml')
        data = load_public_dataset(dataset['name'], args.data_root, args.seed)
        h = load_frozen_representation(ROOT / dataset['representation'], data.num_nodes)
        for method in ['granular_ball', 'exact_knn', 'kmeans']:
            row = dict(dataset=key, **benchmark(h, data.train_index, method, config, args.seed))
            rows.append(row)
            print(f"{key} {method}: {row['construction_seconds']:.3f}s, {row['storage_bytes']} bytes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
