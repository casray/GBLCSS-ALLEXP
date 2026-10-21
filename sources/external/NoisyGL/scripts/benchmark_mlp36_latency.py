from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_standard_downstream as runner
from utils.tools import load_conf


METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "frozengrace_gb_mlp_softdist_beta05": "LCSS",
    "gcn": "GCN",
}


def _synchronize(features: torch.Tensor) -> None:
    if features.is_cuda:
        torch.cuda.synchronize(features.device)


def benchmark_forward(model, features, adj, warmup: int = 20, repeats: int = 100) -> dict:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(features, adj)
        _synchronize(features)
        if features.is_cuda:
            torch.cuda.reset_peak_memory_stats(features.device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(features, adj)
        _synchronize(features)
        elapsed = time.perf_counter() - started
    latency_ms = 1000.0 * elapsed / repeats
    peak_mb = (
        float(torch.cuda.max_memory_allocated(features.device) / (1024 ** 2))
        if features.is_cuda
        else float("nan")
    )
    return {
        "inference_repeats": int(repeats),
        "fullgraph_latency_ms": float(latency_ms),
        "per_node_latency_us": float(1000.0 * latency_ms / max(1, features.shape[0])),
        "inference_peak_allocated_mb": peak_mb,
    }


def run_benchmark(
    dataset_name: str,
    method: str,
    device: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict:
    runner.setup_seed(seed)
    data = runner._load_dataset(
        dataset_name,
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    data.noisy_label, modified_mask = runner.label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    conf = load_conf(None, method, dataset_name)
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes
    conf.training["debug"] = False

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    predictor_cls = runner.predictor_class_for(method)
    predictor = predictor_cls(conf, data, device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = "uniform"
    predictor.noise_rate = 0.4
    predictor.seed = seed
    result = predictor.train()
    _synchronize(predictor.feats)
    downstream_wall = time.perf_counter() - started
    train_peak_mb = (
        float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 2))
        if str(device).startswith("cuda")
        else float("nan")
    )
    inference = benchmark_forward(
        predictor.model,
        predictor.feats,
        predictor.adj,
        warmup=warmup,
        repeats=repeats,
    )
    diagnostics = getattr(predictor, "gb_diagnostics", {})
    return {
        "dataset": dataset_name,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "nodes": int(data.labels.shape[0]),
        "adjacency_nnz": int(data.adj._nnz()) if getattr(data.adj, "is_sparse", False) else "",
        "test": float(result["test"]),
        "parameter_count": runner._parameter_count(predictor),
        "downstream_wall_seconds": float(downstream_wall),
        "model_train_seconds": float(getattr(predictor, "total_time", float("nan"))),
        "ball_build_seconds": float(diagnostics.get("ball_build_time_seconds", 0.0) or 0.0),
        "train_peak_allocated_mb": train_peak_mb,
        **inference,
    }


def write_report(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# MLP-36 latency and memory benchmark",
        "",
        "Protocol: uniform 0.4, train-only, percent 0.2/0.1/0.7, seed 0, one ROCm GPU.",
        "GRACE pretraining is excluded because H is reused across all downstream noise settings.",
        "Latency is full-batch transductive inference averaged over repeated forward passes.",
        "",
        "| Dataset | Method | Downstream wall (s) | Ball build (s) | Inference (ms/full graph) | us/node | Peak train MB | Peak inference MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.method_label} | {row.downstream_wall_seconds:.3f} | "
            f"{row.ball_build_seconds:.3f} | {row.fullgraph_latency_ms:.4f} | "
            f"{row.per_node_latency_us:.4f} | {row.train_peak_allocated_mb:.1f} | "
            f"{row.inference_peak_allocated_mb:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cora,citeseer,corafull")
    parser.add_argument(
        "--methods",
        default="mlp_ce,frozengrace_gb_mlp_softdist_beta05,gcn",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/mlp36_review_defense_20260715"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets.split(","):
        for method in args.methods.split(","):
            rows.append(
                run_benchmark(
                    dataset.strip(),
                    method.strip(),
                    args.device,
                    args.seed,
                    args.warmup,
                    args.repeats,
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "latency_memory_seed0.csv", index=False)
    write_report(frame, output_dir / "latency_memory_report.md")


if __name__ == "__main__":
    main()
