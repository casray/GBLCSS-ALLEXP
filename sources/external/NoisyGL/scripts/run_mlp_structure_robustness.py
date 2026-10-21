from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_standard_downstream import (
    _config_id,
    _noise_fingerprint,
    _noise_stats,
    _parameter_count,
    _parse_int_list,
    _parse_str_list,
    _split_fingerprint,
    _load_dataset,
)
from scripts.run_ssl_swap_downstream import _set_embedding_path, _set_gb_value
from utils.labelnoise import label_process
from utils.structure_noise import perturb_test_adj
from utils.tools import load_conf, setup_seed


PREDICTOR_MODULES = {
    "gcn": "predictor.GCN_Predictor",
    "gin": "predictor.GIN_Predictor",
    "frozengrace_gb_mlp_softdist": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh": "predictor.FrozenGraceGB_GCN_Predictor",
}


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def predictor_class_for(method: str):
    module = importlib.import_module(PREDICTOR_MODULES[method])
    return getattr(module, f"{method}_Predictor")


def _needs_embedding(method: str) -> bool:
    return method.startswith("frozengrace_") or method.startswith("featuregb_")


def _evaluate_with_adj(predictor, adj, mask) -> tuple[float, float]:
    original_adj = predictor.adj
    predictor.adj = adj
    try:
        loss, acc = predictor.test(mask)
    finally:
        predictor.adj = original_adj
    return float(loss.item() if hasattr(loss, "item") else loss), float(acc)


def _run_train_once(args, dataset_name: str, method: str, seed: int) -> tuple[object, object, dict, float]:
    setup_seed(seed)
    dataset_key = dataset_name.lower()
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
        noise_type=args.label_noise_type,
        noise_rate=args.label_noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    model_conf = load_conf(None, method, dataset_key)
    if _needs_embedding(method):
        embedding_path = embedding_path_for(args.embedding_root, dataset_key, args.ssl_method)
        if not embedding_path.exists():
            raise FileNotFoundError(f"missing H for dataset={dataset_key}, ssl={args.ssl_method}: {embedding_path}")
        _set_embedding_path(model_conf, str(embedding_path))
        _set_gb_value(model_conf, "ball_embedding_path", str(embedding_path))
    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)

    predictor_cls = predictor_class_for(method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = args.label_noise_type
    predictor.noise_rate = args.label_noise_rate
    predictor.seed = seed
    started_at = time.perf_counter()
    train_result = predictor.train()
    wall_time = time.perf_counter() - started_at

    meta = {
        "dataset": dataset_key,
        "method": method,
        "seed": int(seed),
        "label_noise_type": args.label_noise_type,
        "label_noise_rate": float(args.label_noise_rate),
        "noise_scope": args.noise_scope,
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, data.noisy_label),
        "config_id": _config_id(method, dataset_key, model_conf),
        **_noise_stats(
            data.labels,
            data.noisy_label,
            modified_mask,
            data.train_masks,
            data.val_masks,
            data.test_masks,
        ),
        "train_wall_time_seconds": float(wall_time),
        "parameter_count": _parameter_count(predictor),
        "train": float(train_result.get("train", 0.0)),
        "valid": float(train_result.get("valid", 0.0)),
    }
    return data, predictor, meta, wall_time


def run_one(args, dataset_name: str, method: str, seed: int) -> list[dict]:
    data, predictor, meta, _ = _run_train_once(args, dataset_name, method, seed)
    rows = []
    clean_loss, clean_acc = _evaluate_with_adj(predictor, data.adj, data.test_masks)
    rows.append(
        {
            **meta,
            "structure_mode": "clean",
            "structure_p": 0.0,
            "structure_seed": int(seed),
            "status": "ok",
            "error": "",
            "test_loss": clean_loss,
            "test": clean_acc,
            "clean_reference_test": clean_acc,
            "delta_from_clean": 0.0,
        }
    )
    for mode in _parse_str_list(args.structure_modes):
        for p in _parse_float_list(args.structure_ps):
            perturbed = perturb_test_adj(
                data.adj,
                mode=mode,
                p=p,
                seed=seed,
                test_mask=data.test_masks,
            )
            loss, acc = _evaluate_with_adj(predictor, perturbed, data.test_masks)
            rows.append(
                {
                    **meta,
                    "structure_mode": mode,
                    "structure_p": float(p),
                    "structure_seed": int(seed),
                    "status": "ok",
                    "error": "",
                    "test_loss": loss,
                    "test": acc,
                    "clean_reference_test": clean_acc,
                    "delta_from_clean": float(acc - clean_acc),
                }
            )
    return rows


def error_row_for(dataset_name: str, method: str, seed: int, exc: Exception) -> dict:
    return {
        "dataset": str(dataset_name).lower(),
        "method": method,
        "seed": int(seed),
        "label_noise_type": "",
        "label_noise_rate": "",
        "structure_mode": "",
        "structure_p": "",
        "structure_seed": int(seed),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "test": "",
        "delta_from_clean": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate graph-less MLP robustness under inference-time structure noise.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--datasets", default="cora,citeseer,coauthorcs")
    parser.add_argument("--methods", default="gcn,gin,frozengrace_gb_mlp_softdist")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--label_noise_type", default="clean")
    parser.add_argument("--label_noise_rate", type=float, default=0.0)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--structure_modes", default="edge_drop,isolated")
    parser.add_argument("--structure_ps", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "mlp_dualrobust_20260701" / "structure_summary.csv"))
    return parser


def main():
    args = build_parser().parse_args()
    datasets = _parse_str_list(args.datasets)
    methods = _parse_str_list(args.methods)
    seeds = _parse_int_list(args.seeds)
    unsupported = [method for method in methods if method not in PREDICTOR_MODULES]
    if unsupported:
        raise ValueError(f"unsupported structure methods: {unsupported}")

    rows = []
    for dataset_name in datasets:
        for method in methods:
            for seed in seeds:
                print(f"STRUCTURE_RUN dataset={dataset_name} method={method} seed={seed}")
                try:
                    new_rows = run_one(args, dataset_name, method, seed)
                    for row in new_rows:
                        print(
                            "STRUCTURE_RESULT dataset={dataset} method={method} seed={seed} "
                            "mode={mode} p={p} test={test:.4f} delta={delta:.4f}".format(
                                dataset=row["dataset"],
                                method=method,
                                seed=seed,
                                mode=row["structure_mode"],
                                p=row["structure_p"],
                                test=float(row["test"]),
                                delta=float(row["delta_from_clean"]),
                            )
                        )
                    rows.extend(new_rows)
                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    row = error_row_for(dataset_name, method, seed, exc)
                    rows.append(row)
                    print(f"STRUCTURE_FAILED dataset={dataset_name} method={method} seed={seed} error={row['error']}")
                write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
