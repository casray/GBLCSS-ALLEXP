"""Rank the same ball pairs with overlap statistics and a soft-neighbor control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

from diagnose_r0 import ball_classes, noise_permutation, partition


ROOT = Path(__file__).resolve().parent
RULES = ("R0", "R1", "R2", "R3", "R4", "R6")


def soft_membership(z: np.ndarray, parts: np.ndarray, k: int = 5) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=min(k, len(z))).fit(z).kneighbors(z, return_distance=False)
    # Each row distributes one unit of weight across nearby ball assignments.
    return np.eye(parts.max() + 1, dtype=np.float64)[parts[neighbors]].mean(axis=1)


def compare(z0: np.ndarray, z1: np.ndarray, y: np.ndarray, seed: int,
            eta: float, p: int, batch_size: int = 128) -> list[dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    perm = noise_permutation(len(y), eta, seed)
    chunks: dict[str, list[np.ndarray]] = {r: [] for r in RULES}
    truths, batch_ids, r0_masks = [], [], []
    ball_a, ball_b = [], []
    offset_a = offset_b = 0
    for bid, start in enumerate(range(0, len(y), batch_size)):
        rows = order[start:start + batch_size]
        source = perm[rows]
        v0, v1 = z0[rows], z1[source]
        a, b = partition(v0, p), partition(v1, p)
        ka, kb, n = a.max() + 1, b.max() + 1, len(rows)
        table = np.zeros((ka, kb), dtype=np.int32)
        np.add.at(table, (a, b), 1)
        na = np.bincount(a, minlength=ka)
        nb = np.bincount(b, minlength=kb)
        expected = na[:, None] * nb[None, :] / n
        variance = na[:, None] * (nb[None, :] / n) * (1 - nb[None, :] / n) * ((n - na[:, None]) / max(n - 1, 1))
        r0 = table / np.minimum(na[:, None], nb[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(table > 0, np.log(table / expected), -1e9)
            r3 = np.where(variance > 0, (table - expected) / np.sqrt(variance), -1e9)
        pvalue = hypergeom.sf(table - 1, n, nb[None, :], na[:, None])
        # R6: per-view kNN spreads each sample's ball membership before co-occurrence.
        soft0, soft1 = soft_membership(v0, a), soft_membership(v1, b)
        overlap = soft0.T @ soft1
        denominator = np.minimum(soft0.sum(axis=0)[:, None], soft1.sum(axis=0)[None, :])
        r6 = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
        scores = {"R0": r0, "R1": table - expected, "R2": r2,
                  "R3": r3, "R4": -np.log10(np.maximum(pvalue, 1e-300)), "R6": r6}
        for name in RULES:
            chunks[name].append(scores[name].reshape(-1))
        m0, _ = ball_classes(a, y[rows], int(y.max()) + 1)
        m1, _ = ball_classes(b, y[source], int(y.max()) + 1)
        truths.append((m0[:, None] == m1[None, :]).reshape(-1))
        r0_masks.append((r0 > .1).reshape(-1))
        batch_ids.append(np.full(ka * kb, bid, dtype=np.int32))
        ball_a.append(np.broadcast_to(np.arange(ka)[:, None] + offset_a, (ka, kb)).reshape(-1))
        ball_b.append(np.broadcast_to(np.arange(kb)[None, :] + offset_b, (ka, kb)).reshape(-1))
        offset_a += ka
        offset_b += kb
    truth = np.concatenate(truths)
    baseline = np.concatenate(r0_masks)
    k = int(baseline.sum())
    ba, bb = np.concatenate(ball_a), np.concatenate(ball_b)
    out = []
    for name in RULES:
        score = np.concatenate(chunks[name])
        if name == "R0":
            select = baseline
        else:
            select = np.zeros(len(score), dtype=bool)
            # A deterministic global edge budget; never use true labels in selection.
            select[np.argsort(-score, kind="stable")[:k]] = True
        out.append({"seed": seed, "eta": eta, "p": p, "rule": name,
                    "pairs": len(truth), "matched_edges": k,
                    "precision_at_r0_edges": float(truth[select].mean()),
                    "delta_vs_r0_pp": float(100 * (truth[select].mean() - truth[baseline].mean())),
                    "average_precision_all_pairs": float(average_precision_score(truth, score)),
                    "ball_coverage0": float(np.unique(ba[select]).size / offset_a),
                    "ball_coverage1": float(np.unique(bb[select]).size / offset_b),
                    "selected_zero_overlap": int(((np.concatenate(chunks["R0"]) == 0) & select).sum())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--etas", type=float, nargs="+", default=[.4, .6])
    parser.add_argument("--ps", type=int, nargs="+", default=[2, 5, 10, 30])
    args = parser.parse_args()
    raw = np.load(ROOT / "data" / "Caltech101-20.npz")
    z = np.load(ROOT / "data" / "Caltech101-20_reconstruction_z.npz")
    y = raw["labels"].astype(np.int64)
    path = ROOT / "stage1_rules_caltech.csv"
    rows = []
    for seed in args.seeds:
        for eta in args.etas:
            for p in args.ps:
                new = compare(z["view3"], z["view4"], y, seed, eta, p)
                rows.extend(new)
                print(json.dumps({"seed": seed, "eta": eta, "p": p,
                                  "deltas_pp": {r["rule"]: round(r["delta_vs_r0_pp"], 2) for r in new}}), flush=True)
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
    (ROOT / "stage1_rules_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
