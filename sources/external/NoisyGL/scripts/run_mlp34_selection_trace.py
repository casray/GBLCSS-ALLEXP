from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.MLP34SelectionTrace_Predictor import (
    lcss_small_loss_shuffled_trace_Predictor,
    lcss_small_loss_trace_Predictor,
    mlp_small_loss_trace_Predictor,
)
from scripts.run_ssl_swap_downstream import _load_dataset
from scripts.run_standard_downstream import (
    _config_id,
    _noise_fingerprint,
    _noise_stats,
    _parameter_count,
    _split_fingerprint,
)
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


TRACE_METHOD_SPECS = {
    "small_loss": ("mlp_small_loss", mlp_small_loss_trace_Predictor),
    "lcss_soft_small_loss": (
        "frozengrace_gb_mlp_softdist_beta05_small_loss",
        lcss_small_loss_trace_Predictor,
    ),
    "lcss_soft_small_loss_shuffled": (
        "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled",
        lcss_small_loss_shuffled_trace_Predictor,
    ),
}


def _parse_list(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_noise_settings(value):
    settings = []
    for item in _parse_list(value):
        noise_type, noise_rate = item.split(":", 1)
        settings.append((noise_type.strip(), float(noise_rate)))
    return settings


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(args, dataset_name, trace_method, seed, noise_type, noise_rate):
    setup_seed(seed)
    data = _load_dataset(
        dataset_name,
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
        noise_type=noise_type,
        noise_rate=noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    data.noisy_label = noisy_label
    base_method, predictor_cls = TRACE_METHOD_SPECS[trace_method]
    conf = load_conf(None, base_method, dataset_name)
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes
    conf.training["debug"] = False
    split_hash = _split_fingerprint(data.train_masks, data.val_masks, data.test_masks)
    noise_hash = _noise_fingerprint(data.labels, data.noisy_label)
    predictor = predictor_cls(conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = base_method
    predictor.trace_method = trace_method
    predictor.noise_type = noise_type
    predictor.noise_rate = noise_rate
    predictor.seed = seed
    predictor.split_hash = split_hash
    predictor.noise_hash = noise_hash
    trace_name = (
        f"{dataset_name}_{noise_type}{str(noise_rate).replace('.', 'p')}_"
        f"seed{seed}_{trace_method}.npz"
    )
    predictor.trace_output_path = Path(args.trace_dir) / trace_name
    started = time.perf_counter()
    result = predictor.train()
    wall_time = time.perf_counter() - started
    status = getattr(predictor, "run_status", "ok")
    diagnostics = {}
    diagnostics.update(getattr(predictor, "noise_baseline_diagnostics", {}))
    diagnostics.update(getattr(predictor, "gb_diagnostics", {}))
    return {
        "dataset": dataset_name,
        "method": trace_method,
        "base_method": base_method,
        "seed": int(seed),
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "noise_scope": "train",
        "split_type": args.split_type,
        "train_percent": float(args.train_percent),
        "val_percent": float(args.val_percent),
        "test_percent": float(args.test_percent),
        "split_hash": split_hash,
        "noise_hash": noise_hash,
        "config_id": _config_id(base_method, dataset_name, conf),
        **_noise_stats(
            data.labels,
            data.noisy_label,
            modified_mask,
            data.train_masks,
            data.val_masks,
            data.test_masks,
        ),
        "wall_time_seconds": float(wall_time),
        "parameter_count": _parameter_count(predictor),
        "status": status,
        "error": getattr(predictor, "run_error", ""),
        "train": float(result["train"]),
        "valid": float(result["valid"]),
        "test": float(result["test"]),
        **diagnostics,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run MLP-34 selection-mask traces")
    parser.add_argument("--datasets", default="cora,citeseer,corafull,amazonpho,pubmed")
    parser.add_argument(
        "--methods",
        default="small_loss,lcss_soft_small_loss,lcss_soft_small_loss_shuffled",
    )
    parser.add_argument(
        "--noise_settings",
        default="uniform:0.6,pair:0.4,instance:0.4",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    datasets = _parse_list(args.datasets)
    methods = _parse_list(args.methods)
    unsupported = [method for method in methods if method not in TRACE_METHOD_SPECS]
    if unsupported:
        raise ValueError(f"unsupported trace methods: {unsupported}")
    noise_settings = _parse_noise_settings(args.noise_settings)
    seeds = _parse_list(args.seeds, int)
    output = Path(args.output)
    rows = []
    for noise_type, noise_rate in noise_settings:
        for dataset in datasets:
            for seed in seeds:
                for method in methods:
                    print(
                        f"TRACE_RUN dataset={dataset} noise={noise_type}:{noise_rate} "
                        f"seed={seed} method={method}"
                    )
                    try:
                        row = run_one(
                            args,
                            dataset,
                            method,
                            seed,
                            noise_type,
                            noise_rate,
                        )
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        row = {
                            "dataset": dataset,
                            "method": method,
                            "seed": int(seed),
                            "noise_type": noise_type,
                            "noise_rate": float(noise_rate),
                            "noise_scope": "train",
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    rows.append(row)
                    _write_csv(output, rows)
                    print(
                        f"TRACE_RESULT status={row['status']} test={row.get('test', 'NA')} "
                        f"path={row.get('trace_output_path', 'NA')}"
                    )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
