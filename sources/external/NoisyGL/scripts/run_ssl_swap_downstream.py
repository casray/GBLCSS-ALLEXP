from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_standard_downstream import (
    _apply_config_overrides,
    _config_id,
    _noise_fingerprint,
    _noise_stats,
    _parameter_count,
    _split_fingerprint,
)
from utils.dataloader import Dataset
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


PREDICTOR_MODULE = "predictor.FrozenGraceGB_GCN_Predictor"
DIAG_KEYS = (
    "tie_mode",
    "tie_ball_count",
    "tie_active_ball_count",
    "tie_affected_node_count",
    "tie_affected_alpha_sum",
    "ball_assignment_hash",
)
PREDICTOR_MODULES = {
    "mlp_small_loss": "predictor.MLPNoiseBaselines_Predictor",
    "linear_small_loss": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_coteaching": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_jocor": "predictor.MLPNoiseBaselines_Predictor",
    "linear_jocor": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_ct": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_gmm_ct": "predictor.MLPNoiseBaselines_Predictor",
    "linear_ct": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_elr": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_sce": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_logitclip": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_anl": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_anl_ce": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_fine": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_fine_official": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_dividemix_lite": "predictor.MLPNoiseBaselines_Predictor",
    "linear_dividemix_lite": "predictor.MLPNoiseBaselines_Predictor",
    "gsdn": "predictor.GraphFreeFamily_Predictor",
    "graphmlp": "predictor.GraphFreeFamily_Predictor",
    "graphmlp_reimplemented": "predictor.GraphMLPReimplemented_Predictor",
    "ffg2m_reimplemented": "predictor.G2MBaselines_Predictor",
    "mlp_xh_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_small_loss_xh": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_jocor_xh": "predictor.MLPNoiseBaselines_Predictor",
    "frozengrace_gb_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta1": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta1_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_hard": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_gmmresp_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "knn_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "kmeans_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_expand": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_expand_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "knn_mlp_softdist_beta05_expand": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_gcn": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_gcn_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_knn_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_knn_k5_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "gbselect": "predictor.GBSelect_Predictor",
    "gbselect_gcn_only": "predictor.GBSelect_Predictor",
    "gbselect_shuffled": "predictor.GBSelect_Predictor",
}


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_dataset(dataset_name, device, split_type, train_percent, val_percent, test_percent):
    dataset_key = dataset_name.lower()
    data_conf = load_conf(str(REPO_ROOT / "config" / "_dataset" / f"{dataset_key}.yaml"))
    if split_type is not None:
        data_conf.split["split_type"] = split_type
    if train_percent is not None:
        data_conf.split["train_percent"] = train_percent
    if val_percent is not None:
        data_conf.split["val_percent"] = val_percent
    if test_percent is not None:
        data_conf.split["test_percent"] = test_percent
    if data_conf.split["split_type"] == "percent":
        for key in [
            "train_size",
            "val_size",
            "test_size",
            "train_examples_per_class",
            "val_examples_per_class",
            "test_examples_per_class",
        ]:
            data_conf.split[key] = None
    return Dataset(
        dataset_key,
        path=str(REPO_ROOT / "data") + "/",
        feat_norm=data_conf.norm["feat_norm"],
        adj_norm=data_conf.norm["adj_norm"],
        train_size=data_conf.split["train_size"],
        val_size=data_conf.split["val_size"],
        test_size=data_conf.split["test_size"],
        train_percent=data_conf.split["train_percent"],
        val_percent=data_conf.split["val_percent"],
        test_percent=data_conf.split["test_percent"],
        train_examples_per_class=data_conf.split["train_examples_per_class"],
        val_examples_per_class=data_conf.split["val_examples_per_class"],
        test_examples_per_class=data_conf.split["test_examples_per_class"],
        add_self_loop=data_conf.modify["add_self_loop"],
        from_npz=data_conf.modify["from_npz_largest_component"],
        device=device,
        split_type=data_conf.split["split_type"],
    )


def predictor_class_for(base_method: str):
    module = importlib.import_module(PREDICTOR_MODULES.get(base_method, PREDICTOR_MODULE))
    return getattr(module, f"{base_method}_Predictor")


def _set_embedding_path(model_conf, embedding_path: str) -> None:
    gb_conf = getattr(model_conf, "gb", None)
    if gb_conf is None:
        model_conf.gb = {"embedding_path": embedding_path}
    elif isinstance(gb_conf, dict):
        gb_conf["embedding_path"] = embedding_path
    else:
        setattr(gb_conf, "embedding_path", embedding_path)


def _set_gb_value(model_conf, key: str, value) -> None:
    gb_conf = getattr(model_conf, "gb", None)
    if gb_conf is None:
        model_conf.gb = {key: value}
    elif isinstance(gb_conf, dict):
        gb_conf[key] = value
    else:
        setattr(gb_conf, key, value)


def resolve_embedding_paths(args, dataset_name: str, ssl_method: str) -> tuple[Path, Path]:
    feature_ssl_method = getattr(args, "feature_ssl_method", None) or ssl_method
    ball_embedding_root = getattr(args, "ball_embedding_root", None) or args.embedding_root
    feature_embedding_path = embedding_path_for(args.embedding_root, dataset_name, feature_ssl_method)
    ball_embedding_path = embedding_path_for(ball_embedding_root, dataset_name, ssl_method)
    return feature_embedding_path, ball_embedding_path


def run_one(args, dataset_name: str, ssl_method: str, base_method: str, seed: int) -> dict:
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
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    feature_embedding_path, ball_embedding_path = resolve_embedding_paths(args, dataset_name, ssl_method)
    feature_ssl_method = getattr(args, "feature_ssl_method", None) or ssl_method
    if not feature_embedding_path.exists():
        raise FileNotFoundError(
            f"missing feature H for feature_ssl_method={feature_ssl_method}, dataset={dataset_name}: "
            f"{feature_embedding_path}"
        )
    if not ball_embedding_path.exists():
        raise FileNotFoundError(f"missing ball H for ssl_method={ssl_method}, dataset={dataset_name}: {ball_embedding_path}")
    model_conf = load_conf(None, base_method, dataset_key)
    _set_embedding_path(model_conf, str(feature_embedding_path))
    _set_gb_value(model_conf, "ball_embedding_path", str(ball_embedding_path))
    if args.gb_max_balls is not None:
        _set_gb_value(model_conf, "max_balls", int(args.gb_max_balls))
    if args.gb_max_balls_policy is not None:
        _set_gb_value(model_conf, "max_balls_policy", str(args.gb_max_balls_policy))
    if args.gb_min_ball_size is not None:
        _set_gb_value(model_conf, "min_ball_size", int(args.gb_min_ball_size))
    if args.gb_split_improvement is not None:
        _set_gb_value(model_conf, "split_improvement", float(args.gb_split_improvement))
    if args.gb_keep_ratio is not None:
        _set_gb_value(model_conf, "keep_ratio", float(args.gb_keep_ratio))
    if args.gb_lambda_u is not None:
        _set_gb_value(model_conf, "lambda_u", float(args.gb_lambda_u))
    _apply_config_overrides(model_conf, getattr(args, "config_overrides", {}))
    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)
    predictor_cls = predictor_class_for(base_method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = f"{base_method}__{ssl_method}"
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = seed
    started_at = time.perf_counter()
    result = predictor.train()
    wall_time = time.perf_counter() - started_at
    gb_diag = getattr(predictor, "gb_diagnostics", {}) or {}
    diag_keys = [
        *DIAG_KEYS,
        "soft_target_builder",
        "target_builder",
        "matched_k",
        "matched_num_balls",
        "matched_reference_num_balls",
        "matched_mean_train_per_ball",
        "matched_zero_train_balls",
        "kmeans_n_clusters",
        "num_balls",
        "max_balls_policy",
        "configured_max_balls",
        "effective_max_balls",
        "max_balls_cap_hit",
        "ball_build_time_seconds",
        "mean_ball_size",
        "mean_train_per_ball",
        "noisy_train_purity_weighted",
        "true_train_purity_weighted",
        "alpha_positive_train_count",
        "mean_alpha_train",
        "lambda_u",
        "expansion_builder",
        "expansion_target_beta",
        "expansion_min_train_per_ball",
        "expansion_margin_threshold",
        "expansion_matched_k",
        "expansion_candidate_count",
        "expansion_coverage_unlabeled",
        "expansion_coverage_total",
        "expansion_mean_selected_margin",
        "expansion_mean_selected_train_votes",
        "teacher_epochs",
        "teacher_best_epoch",
        "teacher_best_val",
        "teacher_best_test",
        "teacher_use_xh",
        "teacher_input_dim",
        "student_input_dim",
        "teacher_target",
        "distill_temperature",
        "distill_student_best_epoch",
    ]
    return {
        "dataset": dataset_key,
        "method": base_method,
        "ssl_method": ssl_method,
        "base_method": base_method,
        "seed": int(seed),
        "embedding_path": str(ball_embedding_path),
        "feature_ssl_method": feature_ssl_method,
        "feature_embedding_path": str(feature_embedding_path),
        "ball_embedding_path": str(ball_embedding_path),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "split_type": args.split_type,
        "train_percent": float(args.train_percent),
        "val_percent": float(args.val_percent),
        "test_percent": float(args.test_percent),
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, data.noisy_label),
        "config_id": _config_id(base_method, dataset_key, model_conf),
        "gb_max_balls": "" if args.gb_max_balls is None else int(args.gb_max_balls),
        "gb_max_balls_policy": "" if args.gb_max_balls_policy is None else str(args.gb_max_balls_policy),
        "gb_min_ball_size": "" if args.gb_min_ball_size is None else int(args.gb_min_ball_size),
        "gb_split_improvement": "" if args.gb_split_improvement is None else float(args.gb_split_improvement),
        "gb_keep_ratio": "" if args.gb_keep_ratio is None else float(args.gb_keep_ratio),
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
        "strategy": getattr(predictor, "strategy", ""),
        "logit_clip_tau": model_conf.model.get("logit_clip_tau", ""),
        "anl_active_weight": model_conf.model.get("anl_active_weight", ""),
        "anl_passive_weight": model_conf.model.get("anl_passive_weight", ""),
        "status": "ok",
        "error": "",
        "train": float(result.get("train", 0.0)),
        "valid": float(result.get("valid", 0.0)),
        "test": float(result.get("test", 0.0)),
        **{key: gb_diag.get(key, "") for key in diag_keys},
    }


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


def build_parser():
    parser = argparse.ArgumentParser(description="Run downstream seed smoke while optionally separating XH and ball embeddings.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--ball_embedding_root", default=None)
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--ssl_methods", default="grace,ccassg,graphmae")
    parser.add_argument("--feature_ssl_method", default=None)
    parser.add_argument("--base_methods", default="frozengrace_gb_gcn_softdist_xh,gcn_xh_ce,frozengrace_gb_gcn_softdist_xh_shuffled")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", required=True, choices=["train", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--gb_max_balls", type=int, default=None)
    parser.add_argument("--gb_max_balls_policy", choices=["fixed", "sqrt_nodes"], default=None)
    parser.add_argument("--gb_min_ball_size", type=int, default=None)
    parser.add_argument("--gb_split_improvement", type=float, default=None)
    parser.add_argument("--gb_keep_ratio", type=float, default=None)
    parser.add_argument("--gb_lambda_u", type=float, default=None)
    parser.add_argument("--config_overrides", default="{}")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "ssl_swap_seed0_20260612" / "ssl_swap_seed0_summary.csv"))
    return parser


def main():
    args = build_parser().parse_args()
    args.config_overrides = json.loads(args.config_overrides)
    if not isinstance(args.config_overrides, dict):
        raise ValueError("--config_overrides must decode to a JSON object")
    datasets = _parse_str_list(args.datasets)
    ssl_methods = _parse_str_list(args.ssl_methods)
    base_methods = _parse_str_list(args.base_methods)
    seeds = _parse_int_list(args.seeds)

    rows = []
    for dataset_name in datasets:
        for ssl_method in ssl_methods:
            for base_method in base_methods:
                for seed in seeds:
                    print(f"SSL_SWAP dataset={dataset_name} ssl={ssl_method} base={base_method} seed={seed}")
                    row = run_one(args, dataset_name, ssl_method, base_method, seed)
                    print(f"RESULT dataset={row['dataset']} ssl={ssl_method} base={base_method} seed={seed} test={row['test']:.4f}")
                    rows.append(row)
                    write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
