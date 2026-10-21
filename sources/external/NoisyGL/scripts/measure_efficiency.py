from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import (
    _load_dataset,
    _noise_stats,
    _set_embedding_path,
    predictor_class_for as xh_predictor_class_for,
)
from scripts.run_standard_downstream import predictor_class_for as standard_predictor_class_for
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


FROZEN_METHODS = {
    "mlp_ce": ("mlp_ce", False),
    "softdist_mlp": ("frozengrace_gb_mlp_softdist", True),
    "softdist_gcn": ("frozengrace_gb_gcn_softdist", True),
    "gcn_xh_ce": ("gcn_xh_ce", True),
    "softdist_xh": ("frozengrace_gb_gcn_softdist_xh", True),
}

STANDARD_METHODS = {"gcn", "nrgnn", "unionnet"}
SUPPORTED_METHODS = (
    "mlp_ce",
    "softdist_mlp",
    "gcn",
    "softdist_gcn",
    "gcn_xh_ce",
    "softdist_xh",
    "nrgnn",
)


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def count_unique_trainable_params(optimizer) -> int:
    seen = set()
    total = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not getattr(param, "requires_grad", False):
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            total += int(param.numel())
    return total


def _uses_cuda(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def cuda_synchronize(device: torch.device) -> None:
    if _uses_cuda(device):
        torch.cuda.synchronize(device)


def reset_gpu_peak_memory(device: torch.device) -> None:
    if _uses_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def prepare_gpu_measurement(device: torch.device) -> None:
    cuda_synchronize(device)
    reset_gpu_peak_memory(device)
    cuda_synchronize(device)


def gpu_memory_metrics(device: torch.device) -> dict[str, int]:
    if not _uses_cuda(device):
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    cuda_synchronize(device)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def install_ball_target_timer(predictor, device: torch.device):
    elapsed = [0.0]
    original = getattr(predictor, "_make_corrected_labels", None)
    if original is None:
        return lambda: 0.0

    def timed_make_corrected_labels(*args, **kwargs):
        cuda_synchronize(device)
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            cuda_synchronize(device)
            elapsed[0] += time.perf_counter() - started

    predictor._make_corrected_labels = timed_make_corrected_labels
    return lambda: float(elapsed[0])


def build_predictor(args, method: str):
    setup_seed(args.seed)
    dataset_key = args.dataset.lower()
    data = _load_dataset(
        dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    data.noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=args.seed,
        debug=False,
    )

    embedding_path = ""
    if method in FROZEN_METHODS:
        base_method, needs_embedding = FROZEN_METHODS[method]
        model_conf = load_conf(None, base_method, dataset_key)
        if needs_embedding:
            embedding_path_obj = embedding_path_for(
                args.embedding_root, args.dataset, args.ssl_method
            )
            if not embedding_path_obj.exists():
                raise FileNotFoundError(
                    f"missing H for method={method}: {embedding_path_obj}"
                )
            embedding_path = str(embedding_path_obj)
            _set_embedding_path(model_conf, embedding_path)
        predictor_cls = xh_predictor_class_for(base_method)
    elif method in STANDARD_METHODS:
        base_method = method
        model_conf = load_conf(None, method, dataset_key)
        predictor_cls = standard_predictor_class_for(method)
    else:
        raise ValueError(f"unsupported method for efficiency measurement: {method}")

    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)

    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = args.seed
    return predictor, data, modified_mask, base_method, embedding_path


def measure_one(args, method: str) -> dict:
    device = torch.device(args.device)
    prepare_gpu_measurement(device)
    total_started = time.perf_counter()
    started = time.perf_counter()
    predictor, data, modified_mask, base_method, embedding_path = build_predictor(args, method)
    cuda_synchronize(device)
    init_elapsed = time.perf_counter() - started
    params = count_unique_trainable_params(predictor.optim)
    ball_target_elapsed = install_ball_target_timer(predictor, device)
    row = {
        "dataset": args.dataset.lower(),
        "method": method,
        "base_method": base_method,
        "seed": int(args.seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "embedding_path": embedding_path,
        "trainable_params": int(params),
        "init_time_sec": float(init_elapsed),
        "ball_target_time_sec": "",
        "classifier_train_time_sec": "",
        "downstream_total_time_sec": "",
        "peak_allocated_bytes": "",
        "peak_reserved_bytes": "",
        "wall_time_sec": "",
        "predictor_total_time_sec": "",
        "status": "dry_run" if args.dry_run else "ok",
        "error": "",
        **_noise_stats(data.labels, data.noisy_label, modified_mask, data.train_masks),
        "train": "",
        "valid": "",
        "test": "",
    }
    if args.dry_run:
        row.update(gpu_memory_metrics(device))
        return row

    cuda_synchronize(device)
    train_started = time.perf_counter()
    result = predictor.train()
    cuda_synchronize(device)
    train_elapsed = time.perf_counter() - train_started
    ball_elapsed = ball_target_elapsed()
    memory = gpu_memory_metrics(device)
    row.update(
        {
            "ball_target_time_sec": float(ball_elapsed),
            "classifier_train_time_sec": float(max(0.0, train_elapsed - ball_elapsed)),
            "downstream_total_time_sec": float(time.perf_counter() - total_started),
            "wall_time_sec": float(train_elapsed),
            "predictor_total_time_sec": float(getattr(predictor, "total_time", -1)),
            **memory,
            "train": float(result.get("train", 0.0)),
            "valid": float(result.get("valid", 0.0)),
            "test": float(result.get("test", 0.0)),
        }
    )
    return row


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-run efficiency and trainable parameter counts.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "efficiency_20260612" / "efficiency_table.csv"))
    args = parser.parse_args()

    rows = []
    for method in _parse_str_list(args.methods):
        print(f"EFFICIENCY dataset={args.dataset} method={method} seed={args.seed} dry_run={args.dry_run}")
        try:
            row = measure_one(args, method)
            print(
                f"RESULT method={method} status={row['status']} "
                f"params={row['trainable_params']} wall={row['wall_time_sec']}"
            )
        except Exception as exc:
            row = {
                "dataset": args.dataset.lower(),
                "method": method,
                "seed": int(args.seed),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAILED method={method} error={row['error']}")
        rows.append(row)
        write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
