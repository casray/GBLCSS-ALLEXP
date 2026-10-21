from __future__ import annotations

import math

import numpy as np
from scipy import stats


def paired_delta(scores_a: list[float], scores_b: list[float]):
    """Return paired delta statistics for two score lists."""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("score lists must be non-empty and have the same length")
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    delta = a - b
    if len(delta) > 1 and float(delta.std(ddof=1)) > 0:
        t_value, p_value = stats.ttest_rel(a, b)
    else:
        t_value = math.inf if float(delta.mean()) > 0 else (-math.inf if float(delta.mean()) < 0 else 0.0)
        p_value = 0.0 if math.isinf(t_value) else 1.0
    return {
        "mean_delta": float(delta.mean()),
        "t_value": float(t_value),
        "p_value": float(p_value),
        "sig_label": "SIG" if abs(float(t_value)) > 2.78 else "ns",
        "wins": int((a > b).sum()),
    }


def format_smoke_report(results: dict, gb_stats: dict, dataset: str, noise: str, seed: int):
    """Format a seed0 smoke report."""
    lines = [
        f"# Smoke Report: {dataset} {noise} seed{seed}",
        "",
        "## Test Accuracy",
    ]
    rows = sorted(results.items(), key=lambda item: item[1]["test_acc_at_best_val"], reverse=True)
    for method, metrics in rows:
        lines.append(f"- {method}: {metrics['test_acc_at_best_val']:.4f}")
    ce = results.get("ce", {}).get("test_acc_at_best_val", float("nan"))
    oracle = results.get("clean_oracle", {}).get("test_acc_at_best_val", float("nan"))
    gb = results.get("sel_gb", {}).get("test_acc_at_best_val", float("nan"))
    topk = results.get("sel_topk", {}).get("test_acc_at_best_val", float("nan"))
    kmeans = results.get("sel_kmeans", {}).get("test_acc_at_best_val", float("nan"))
    shuffled = results.get("sel_shuffled", {}).get("test_acc_at_best_val", float("nan"))
    lines.extend(
        [
            "",
            "## Deltas",
            f"- clean_oracle - ce: {oracle - ce:.4f}",
            f"- sel_gb - sel_topk: {gb - topk:.4f}",
            f"- sel_gb - sel_kmeans: {gb - kmeans:.4f}",
            f"- sel_gb - sel_shuffled: {gb - shuffled:.4f}",
            "",
            "## GB Stats",
            f"- num_balls: {gb_stats.get('num_balls')}",
            f"- mean_size: {gb_stats.get('mean_size')}",
            f"- min_size: {gb_stats.get('min_size')}",
            f"- max_size: {gb_stats.get('max_size')}",
            f"- mean_purity: {gb_stats.get('mean_purity')}",
        ]
    )
    return "\n".join(lines)

