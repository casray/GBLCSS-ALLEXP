from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import Dataset
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


PREDICTOR_MODULES = {
    "cgnn": "predictor.CGNN_Predictor",
    "crgnn": "predictor.CRGNN_Predictor",
    "frozengrace_gb_gcn_softdist_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_gcn_softdist_xh_ellipsoid": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_gcn_softdist_xh_gmmresp": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_geaf_gcn_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_geaf_gcn_xh_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "gcn": "predictor.GCN_Predictor",
    "gin": "predictor.GIN_Predictor",
    "gcn_xh_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "linear_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "lp_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_xh_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "grace_h_linear_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "grace_h_mlp_ce": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_small_loss": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_small_loss_xh": "predictor.MLPNoiseBaselines_Predictor",
    "linear_small_loss": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_coteaching": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_jocor": "predictor.MLPNoiseBaselines_Predictor",
    "mlp_jocor_xh": "predictor.MLPNoiseBaselines_Predictor",
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
    "ffg2m_adapted": "predictor.G2MBaselines_Predictor",
    "ffg2m_reimplemented": "predictor.G2MBaselines_Predictor",
    "krd_adapted": "predictor.G2MBaselines_Predictor",
    "vqgraph_adapted": "predictor.G2MBaselines_Predictor",
    "frozengrace_gb_mlp_softdist": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_loo": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_loo_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_uniform_smoothing": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_prior_smoothing": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_logitclip": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss_loo": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss_loo": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_soft_jocor": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_soft_jocor_shuffled": "predictor.LCSSRobust_Predictor",
    "frozengrace_gb_linear_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_linear_softdist_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta1": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta1_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_hard": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_gmmresp": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_gmmresp_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_gmmresp_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "knn_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "kmeans_mlp_softdist_beta05": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_expand": "predictor.FrozenGraceGB_GCN_Predictor",
    "frozengrace_gb_mlp_softdist_beta05_expand_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "knn_mlp_softdist_beta05_expand": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_gcn": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_gcn_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh_loo": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh_loo_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_knn_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_knn_k5_xh": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_xh_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "mlp_distill_softdist_beta05_xh": "predictor.LCSSRobustDistill_Predictor",
    "mlp_distill_knn_beta05_xh": "predictor.LCSSRobustDistill_Predictor",
    "mlp_distill_softdist_beta05_small_loss_xh": "predictor.LCSSRobustDistill_Predictor",
    "mlp_distill_softdist_beta05_small_loss_xh_shuffled": "predictor.LCSSRobustDistill_Predictor",
    "mlp_distill_gcn_xh_lcss_small_loss": "predictor.LCSSRobustStudentDistill_Predictor",
    "mlp_distill_gcn_xh_lcss_small_loss_shuffled": "predictor.LCSSRobustStudentDistill_Predictor",
    "featuregb_mlp_softdist": "predictor.FrozenGraceGB_GCN_Predictor",
    "featuregb_mlp_softdist_shuffled": "predictor.FrozenGraceGB_GCN_Predictor",
    "nrgnn": "predictor.NRGNN_Predictor",
    "nrgnn_xh": "predictor.NRGNN_Predictor",
    "procon": "predictor.ProCon_Predictor",
    "r2lp": "predictor.R2LP_Predictor",
    "rncgln": "predictor.RNCGLN_Predictor",
    "rtgnn": "predictor.RTGNN_Predictor",
    "clnode": "predictor.CLNode_Predictor",
    "tss": "predictor.TSS_Predictor",
    "tfr": "predictor.TFR_Predictor",
    "tfr_gcn": "predictor.TFR_Predictor",
    "unionnet": "predictor.UnionNET_Predictor",
    "pignn": "predictor.PIGNN_Predictor",
    "gd2": "predictor.GD2_Predictor",
    "gsdn_faithful_core": "predictor.GSDNFaithful_Predictor",
    "postel_mlp_noipl_rm": "predictor.NearestWorkSoftTargets_Predictor",
    "als_mlp_rm": "predictor.NearestWorkSoftTargets_Predictor",
    "sals_mlp_adapted": "predictor.NearestWorkSoftTargets_Predictor",
}

METHOD_CAPABILITIES = {
    "gsdn_faithful_core": {
        "requires_embedding": False,
        "training_uses_graph": True,
        "inference_uses_graph": False,
        "comparison_tier": "same_resource_x_only",
        "implementation_status": (
            "equation-grounded common-protocol adaptation"
        ),
    },
    "postel_mlp_noipl_rm": {
        "requires_embedding": False,
        "training_uses_graph": True,
        "inference_uses_graph": False,
        "comparison_tier": "same_resource_x_only",
        "implementation_status": "official_code_faithful_target_project_mlp",
    },
    "als_mlp_rm": {
        "requires_embedding": False,
        "training_uses_graph": True,
        "inference_uses_graph": False,
        "comparison_tier": "same_resource_x_only",
        "implementation_status": "mechanism_faithful_project_mlp_adaptation",
    },
    "sals_mlp_adapted": {
        "requires_embedding": False,
        "training_uses_graph": True,
        "inference_uses_graph": False,
        "comparison_tier": "same_resource_x_only",
        "implementation_status": "mechanism_faithful_project_mlp_adaptation",
    },
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


def predictor_class_for(method: str):
    module = importlib.import_module(PREDICTOR_MODULES[method])
    return getattr(module, f"{method}_Predictor")


def _as_int_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _hash_arrays(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = _as_int_array(array)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _split_fingerprint(train_mask, val_mask, test_mask) -> str:
    return _hash_arrays(train_mask, val_mask, test_mask)


def _noise_fingerprint(labels, noisy_label) -> str:
    return _hash_arrays(labels, noisy_label)


def _config_id(method: str, dataset_name: str, conf) -> str:
    payload = {
        "dataset": dataset_name,
        "method": method,
        "config": vars(conf),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _apply_config_overrides(conf, overrides: dict) -> None:
    for dotted_key, value in overrides.items():
        parts = str(dotted_key).split(".")
        if len(parts) != 2:
            raise KeyError(f"config override must use section.key syntax: {dotted_key}")
        section_name, key = parts
        if not hasattr(conf, section_name):
            raise KeyError(f"unknown config section: {section_name}")
        section = getattr(conf, section_name)
        if not isinstance(section, dict) or key not in section:
            raise KeyError(f"unknown config key: {dotted_key}")
        section[key] = value


def _parameter_count(predictor) -> int:
    seen = set()
    total = 0
    for value in vars(predictor).values():
        if isinstance(value, torch.nn.Module):
            parameters = value.parameters()
        elif isinstance(value, torch.nn.Parameter):
            parameters = (value,)
        else:
            continue
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            pointer = parameter.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += parameter.numel()
    return int(total)


def _result_float(result: dict, key: str, status: str):
    value = result.get(key, "")
    if status != "ok" or value == "":
        return ""
    return float(value)


def _format_metric_for_log(value) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"


def _noise_stats(labels, noisy_label, modified_mask, train_mask, val_mask, test_mask) -> dict:
    n_nodes = int(labels.shape[0])
    train_idx = _as_int_array(train_mask)
    val_idx = _as_int_array(val_mask)
    test_idx = _as_int_array(test_mask)
    modified = _as_int_array(modified_mask)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    modified_val = np.intersect1d(val_idx, modified, assume_unique=False)
    modified_test = np.intersect1d(test_idx, modified, assume_unique=False)
    return {
        "modified_count": int(modified.size),
        "actual_noise_rate": float(modified.size / max(1, n_nodes)),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        "modified_val_count": int(modified_val.size),
        "modified_test_count": int(modified_test.size),
    }


def run_one(args, dataset_name: str, method: str, seed: int) -> dict:
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
    model_conf = load_conf(None, method, dataset_key)
    _apply_config_overrides(model_conf, getattr(args, "config_overrides", {}))
    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)
    started_at = time.perf_counter()
    predictor_cls = predictor_class_for(method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = seed
    result = predictor.train()
    wall_time = time.perf_counter() - started_at
    run_status = getattr(predictor, "run_status", "ok")
    run_error = getattr(predictor, "run_error", "")
    diagnostics = {}
    diagnostics.update(getattr(predictor, "gb_diagnostics", {}))
    diagnostics.update(getattr(predictor, "g2m_diagnostics", {}))
    diagnostics.update(getattr(predictor, "noise_baseline_diagnostics", {}))
    diagnostics.update(getattr(predictor, "graphfree_diagnostics", {}))
    diagnostics.update(getattr(predictor, "method_diagnostics", {}))
    capability = METHOD_CAPABILITIES.get(method, {})
    return {
        "dataset": dataset_key,
        "method": method,
        "seed": int(seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
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
        "wall_time_seconds": float(wall_time),
        "parameter_count": _parameter_count(predictor),
        "strategy": getattr(predictor, "strategy", ""),
        "requires_embedding": capability.get("requires_embedding", ""),
        "training_uses_graph": capability.get("training_uses_graph", ""),
        "inference_uses_graph": capability.get("inference_uses_graph", ""),
        "comparison_tier": capability.get("comparison_tier", ""),
        "implementation_status": capability.get(
            "implementation_status",
            "",
        ),
        "graph_batch_size": model_conf.model.get("graph_batch_size", ""),
        "lambda_feature": model_conf.model.get("lambda_feature", ""),
        "lambda_label": model_conf.model.get("lambda_label", ""),
        "lambda_negative": model_conf.model.get("lambda_negative", ""),
        "lambda_contrast": model_conf.model.get("lambda_contrast", ""),
        "tau": model_conf.model.get("tau", ""),
        "logit_clip_tau": model_conf.model.get("logit_clip_tau", ""),
        "anl_active_weight": model_conf.model.get("anl_active_weight", ""),
        "anl_passive_weight": model_conf.model.get("anl_passive_weight", ""),
        "status": run_status,
        "error": run_error,
        "train": _result_float(result, "train", run_status),
        "valid": _result_float(result, "valid", run_status),
        "test": _result_float(result, "test", run_status),
        **diagnostics,
    }


def error_row_for(
        dataset_name: str,
        method: str,
        seed: int,
        noise_type: str,
        noise_rate: float,
        exc: Exception,
        noise_scope: str = "train") -> dict:
    return {
        "dataset": str(dataset_name).lower(),
        "method": method,
        "seed": int(seed),
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "noise_scope": noise_scope,
        "split_hash": "",
        "noise_hash": "",
        "config_id": "",
        "modified_count": "",
        "actual_noise_rate": "",
        "modified_train_count": "",
        "actual_train_noise_rate": "",
        "modified_val_count": "",
        "modified_test_count": "",
        "wall_time_seconds": "",
        "parameter_count": "",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "train": "",
        "valid": "",
        "test": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Run standard NoisyGL predictors with explicit split overrides and CSV output.")
    parser.add_argument("--datasets", default="pubmed")
    parser.add_argument("--methods", default="gcn,nrgnn")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", required=True, choices=["train", "all"])
    parser.add_argument("--config_overrides", default="{}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "phase10_pubmed_20260612" / "standard_summary.csv"))
    return parser


def main():
    args = build_parser().parse_args()
    args.config_overrides = json.loads(args.config_overrides)
    if not isinstance(args.config_overrides, dict):
        raise ValueError("--config_overrides must decode to a JSON object")

    datasets = _parse_str_list(args.datasets)
    methods = _parse_str_list(args.methods)
    seeds = _parse_int_list(args.seeds)

    unsupported = [method for method in methods if method not in PREDICTOR_MODULES]
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}")

    rows = []
    for dataset_name in datasets:
        for method in methods:
            for seed in seeds:
                print(f"STANDARD_RUN dataset={dataset_name} method={method} seed={seed}")
                try:
                    row = run_one(args, dataset_name, method, seed)
                    print(
                        f"RESULT dataset={row['dataset']} method={method} seed={seed} "
                        f"status={row['status']} test={_format_metric_for_log(row['test'])}"
                    )
                except Exception as exc:
                    if args.stop_on_error:
                        raise
                    row = error_row_for(
                        dataset_name,
                        method,
                        seed,
                        args.noise_type,
                        args.noise_rate,
                        exc,
                        args.noise_scope,
                    )
                    print(f"FAILED dataset={row['dataset']} method={method} seed={seed} error={row['error']}")
                rows.append(row)
                write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
