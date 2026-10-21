"""Audit MGBCC's original cross-view ball rule under correspondence noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parent


def partition(z: np.ndarray, p: int) -> np.ndarray:
    # Matches the author's one-shot batch KMeans and fixed random_state.
    k = max(len(z) // p, 1)
    k = min(k, len(np.unique(z, axis=0)))
    return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(z)


def ball_classes(parts: np.ndarray, truth: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    hist = np.zeros((parts.max() + 1, classes), dtype=np.int32)
    np.add.at(hist, (parts, truth), 1)
    modes = hist.argmax(axis=1)
    ties = (hist == hist.max(axis=1, keepdims=True)).sum(axis=1) > 1
    return modes, ties


def noise_permutation(n: int, eta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100_003)
    perm = np.arange(n)
    selected = rng.choice(n, size=int(np.floor(eta * n)), replace=False)
    perm[selected] = rng.permutation(selected)
    assert np.unique(perm).size == n
    return perm


def diagnose(z0: np.ndarray, z1: np.ndarray, y: np.ndarray,
             seed: int, eta: float, p: int, batch_size: int) -> dict:
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    perm = noise_permutation(n, eta, seed)
    totals = {key: 0 for key in ["ball_pairs", "edges", "wrong_edges", "untied_edges",
                                   "wrong_untied_edges", "ties", "balls0", "balls1",
                                   "covered_balls0", "covered_balls1", "covered_samples0",
                                   "covered_samples1", "edge_support", "wrong_edge_support"]}
    counts = {"c0": 0, "c1": 0, "c2": 0, "c3plus": 0}
    sizes0: list[int] = []
    sizes1: list[int] = []
    for rows in batches:
        source = perm[rows]
        a = partition(z0[rows], p)
        b = partition(z1[source], p)
        ka, kb = a.max() + 1, b.max() + 1
        table = np.zeros((ka, kb), dtype=np.int32)
        np.add.at(table, (a, b), 1)
        na = np.bincount(a, minlength=ka)
        nb = np.bincount(b, minlength=kb)
        sizes0.extend(na.tolist())
        sizes1.extend(nb.tolist())
        score = table / np.minimum(na[:, None], nb[None, :])
        edge = score > 0.1
        m0, t0 = ball_classes(a, y[rows], int(y.max()) + 1)
        m1, t1 = ball_classes(b, y[source], int(y.max()) + 1)
        wrong = m0[:, None] != m1[None, :]
        untied = ~(t0[:, None] | t1[None, :])
        support = table * edge
        totals["ball_pairs"] += ka * kb
        totals["edges"] += int(edge.sum())
        totals["wrong_edges"] += int((edge & wrong).sum())
        totals["untied_edges"] += int((edge & untied).sum())
        totals["wrong_untied_edges"] += int((edge & untied & wrong).sum())
        totals["ties"] += int((edge & ~untied).sum())
        totals["balls0"] += ka
        totals["balls1"] += kb
        totals["covered_balls0"] += int(edge.any(axis=1).sum())
        totals["covered_balls1"] += int(edge.any(axis=0).sum())
        totals["covered_samples0"] += int(na[edge.any(axis=1)].sum())
        totals["covered_samples1"] += int(nb[edge.any(axis=0)].sum())
        totals["edge_support"] += int(support.sum())
        totals["wrong_edge_support"] += int((support * wrong).sum())
        counts["c0"] += int((table == 0).sum())
        counts["c1"] += int((table == 1).sum())
        counts["c2"] += int((table == 2).sum())
        counts["c3plus"] += int((table >= 3).sum())
    output = {"seed": seed, "eta": eta, "p": p, "batch_size": batch_size,
              "actual_moved": float(np.mean(perm != np.arange(n))),
              "actual_cross_class": float(np.mean(y[perm] != y)),
              **totals, **counts,
              "wrong_edge_rate": totals["wrong_edges"] / totals["edges"] if totals["edges"] else None,
              "wrong_untied_rate": totals["wrong_untied_edges"] / totals["untied_edges"] if totals["untied_edges"] else None,
              "tie_edge_rate": totals["ties"] / totals["edges"] if totals["edges"] else None,
              "wrong_support_rate": totals["wrong_edge_support"] / totals["edge_support"] if totals["edge_support"] else None,
              "sample_coverage0": totals["covered_samples0"] / n,
              "sample_coverage1": totals["covered_samples1"] / n,
              "ball_size0_mean": float(np.mean(sizes0)), "ball_size1_mean": float(np.mean(sizes1)),
              "ball_size0_singleton": float(np.mean(np.asarray(sizes0) == 1)),
              "ball_size1_singleton": float(np.mean(np.asarray(sizes1) == 1))}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--etas", type=float, nargs="+", default=[0, .2, .4, .6])
    parser.add_argument("--ps", type=int, nargs="+", default=[2, 5, 10, 30])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    raw = np.load(ROOT / "data" / "Caltech101-20.npz")
    z = np.load(ROOT / "data" / "Caltech101-20_reconstruction_z.npz")
    y = raw["labels"].astype(np.int64)
    out = ROOT / "stage1_r0_caltech.csv"
    rows = []
    for seed in args.seeds:
        for eta in args.etas:
            for p in args.ps:
                row = diagnose(z["view3"], z["view4"], y, seed, eta, p, args.batch_size)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ["seed", "eta", "p", "edges", "wrong_edge_rate", "wrong_untied_rate", "sample_coverage0"]}), flush=True)
                with out.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
    (ROOT / "stage1_r0_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
