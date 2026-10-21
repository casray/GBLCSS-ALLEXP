from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_ellipsoid_softdist import _gb_value, _parse_int_list, _parse_str_list, _resolve_embedding_path
from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import (
    _mask_to_index_array,
    apply_ball_ellipsoid_pca_shrinkage_soft_targets,
    apply_ball_soft_targets,
    diagnostics_for_soft_targets,
    load_frozen_embeddings,
    partition_purity_diagnostics,
    prepare_ball_features,
)
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls, build_random_size_matched_assignment
from utils.tools import load_conf, setup_seed


CURRENT_METHOD = "frozengrace_gb_mlp_softdist_beta05"
CANDIDATE_METHOD = "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05"
CANDIDATE_SHUFFLED_METHOD = "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05_shuffled"
DEFAULT_DATASETS = "cora,citeseer,pubmed,coauthorcs,amazonpho"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "mlp18_ellipsoid_feasibility_20260706"


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(float(value))


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


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


def row_key(row: dict) -> tuple[str, int, str, str]:
    return (
        str(row["dataset"]).lower(),
        _as_int(row, "seed"),
        str(row["candidate_id"]),
        str(row["method"]),
    )


def geometry_key(row: dict) -> tuple[str, int, int, int, float]:
    return (
        str(row["dataset"]).lower(),
        _as_int(row, "seed"),
        _as_int(row, "max_balls"),
        _as_int(row, "min_ball_size"),
        _as_float(row, "split_improvement"),
    )


def geometry_candidate_id(max_balls: int, min_ball_size: int, split_improvement: float) -> str:
    return f"mb{int(max_balls)}_min{int(min_ball_size)}_split{float(split_improvement):g}"


def target_candidate_id(
    max_balls: int,
    min_ball_size: int,
    split_improvement: float,
    feature_space: str,
    covariance: str,
    ridge_scale: float,
    floor: float,
    temperature: float,
    beta: float,
) -> str:
    return (
        f"{geometry_candidate_id(max_balls, min_ball_size, split_improvement)}"
        f"__fs{feature_space}_cov{covariance}_ridge{ridge_scale:g}"
        f"_floor{floor:g}_temp{temperature:g}_beta{beta:g}"
    )


def geometry_passes(row: dict) -> bool:
    if _as_int(row, "num_balls") < 16:
        return False
    if _as_float(row, "mean_train_per_ball") < 6.0:
        return False
    if _as_float(row, "zero_train_ball_ratio") > 0.20:
        return False
    if _as_float(row, "ellipsoid_fallback_ball_ratio") > 0.10:
        return False
    return (
        _as_float(row, "ellipsoid_effective_support_ratio_mean", 1.0) <= 0.95
        or _as_float(row, "target_l1_vs_current") >= 0.02
    )


def evaluate_offline_candidate_gate(row: dict) -> bool:
    if _as_int(row, "positive_seed_count") < 4:
        return False
    if not (_as_float(row, "mass_delta_per_train") >= 0.001 or _as_float(row, "hard_net") >= 0.015):
        return False
    if _as_float(row, "target_l1_vs_current") < 0.02:
        return False
    if _as_float(row, "ellipsoid_fallback_ball_ratio") > 0.10:
        return False
    return True


def candidate_score(row: dict) -> float:
    return (
        1000.0 * _as_float(row, "mass_delta_per_train")
        + 2.0 * _as_float(row, "hard_net")
        + 0.5 * _as_float(row, "target_l1_vs_current")
        - _as_float(row, "ellipsoid_fallback_ball_ratio")
    )


def select_top_candidates(rows: list[dict], per_dataset: int = 2) -> list[dict]:
    selected = []
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        if evaluate_offline_candidate_gate(row):
            row = dict(row)
            row["score"] = candidate_score(row)
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset in sorted(by_dataset):
        group = sorted(by_dataset[dataset], key=lambda item: (-_as_float(item, "score"), str(item["candidate_id"])))
        selected.extend(group[: int(per_dataset)])
    return selected


def _dataset_context(args, dataset_name: str, seed: int):
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
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    conf = load_conf(None, CURRENT_METHOD, dataset_key)
    embedding_path = _gb_value(conf, "ball_embedding_path", _gb_value(conf, "embedding_path", None))
    if embedding_path is None:
        raise ValueError(f"missing GRACE embedding path for {dataset_key}")
    embeddings = prepare_ball_features(
        load_frozen_embeddings(_resolve_embedding_path(embedding_path), expected_nodes=int(noisy_label.shape[0]))
    )
    train_idx = _mask_to_index_array(data.train_masks)
    modified = _mask_to_index_array(modified_mask)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    base_fields = {
        "dataset": dataset_key,
        "seed": int(seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "train_percent": float(args.train_percent),
        "val_percent": float(args.val_percent),
        "test_percent": float(args.test_percent),
        "modified_count": int(modified.size),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
    }
    return data, noisy_label, embeddings, base_fields


def _train_count_summary(assignment: np.ndarray, train_mask) -> dict:
    train_idx = _mask_to_index_array(train_mask)
    counts = []
    for ball_id in sorted(np.unique(assignment).tolist()):
        counts.append(int(np.sum(assignment[train_idx] == ball_id)))
    if counts:
        nonzero = [count for count in counts if count > 0]
        mean_train = float(np.mean(nonzero)) if nonzero else 0.0
        zero_ratio = float(np.mean(np.asarray(counts) == 0))
    else:
        mean_train = 0.0
        zero_ratio = 1.0
    return {
        "mean_train_per_ball": mean_train,
        "zero_train_ball_ratio": zero_ratio,
    }


def _current_targets(noisy_label, assignment, train_mask, n_classes, min_train_per_ball: int, beta: float):
    return apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=beta,
    )


def _target_delta_fields(current_diag: dict, candidate_diag: dict, shuffled_diag: dict, n_train: int) -> dict:
    n_train = max(1, int(n_train))
    ncr_delta = _as_float(candidate_diag, "soft_pseudo_ncr") - _as_float(current_diag, "soft_pseudo_ncr")
    cdr_delta = _as_float(candidate_diag, "soft_pseudo_cdr") - _as_float(current_diag, "soft_pseudo_cdr")
    hard_net = 0.4 * ncr_delta - 0.6 * cdr_delta
    shuffled_ncr_delta = _as_float(candidate_diag, "soft_pseudo_ncr") - _as_float(shuffled_diag, "soft_pseudo_ncr")
    shuffled_cdr_delta = _as_float(candidate_diag, "soft_pseudo_cdr") - _as_float(shuffled_diag, "soft_pseudo_cdr")
    return {
        "mass_delta_per_train": (
            _as_float(candidate_diag, "soft_clean_mass_delta") - _as_float(current_diag, "soft_clean_mass_delta")
        ) / float(n_train),
        "ncr_delta": ncr_delta,
        "cdr_delta": cdr_delta,
        "hard_net": hard_net,
        "real_minus_shuffled_mass_delta": (
            _as_float(candidate_diag, "soft_clean_mass_delta") - _as_float(shuffled_diag, "soft_clean_mass_delta")
        ) / float(n_train),
        "real_minus_shuffled_hard_net": 0.4 * shuffled_ncr_delta - 0.6 * shuffled_cdr_delta,
    }


def _diagnostics_for_assignment(
    data,
    noisy_label,
    embeddings,
    assignment: np.ndarray,
    base_fields: dict,
    max_balls: int,
    min_ball_size: int,
    split_improvement: float,
    feature_space: str,
    covariance: str,
    ridge_scale: float,
    floor: float,
    temperature: float,
    beta: float,
    method: str,
    candidate_id: str,
) -> dict:
    train_idx = _mask_to_index_array(data.train_masks)
    min_train_per_ball = 2
    current_soft, current_alpha, current_pseudo = _current_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=min_train_per_ball,
        beta=0.5,
    )
    candidate_soft, candidate_alpha, candidate_pseudo, candidate_weight_diag = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        ball_features=embeddings,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=beta,
        ellipsoid_feature_space=feature_space,
        ellipsoid_covariance_estimator=covariance,
        ellipsoid_ridge_scale=ridge_scale,
        ellipsoid_weight_floor=floor,
        ellipsoid_distance_temperature=temperature,
    )
    shuffled_assignment = build_random_size_matched_assignment(assignment, random_state=int(base_fields["seed"]) + 17)
    shuffled_soft, shuffled_alpha, shuffled_pseudo, shuffled_weight_diag = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
        noisy_label=noisy_label,
        assignment=shuffled_assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        ball_features=embeddings,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=beta,
        ellipsoid_feature_space=feature_space,
        ellipsoid_covariance_estimator=covariance,
        ellipsoid_ridge_scale=ridge_scale,
        ellipsoid_weight_floor=floor,
        ellipsoid_distance_temperature=temperature,
    )
    current_diag = diagnostics_for_soft_targets(
        noisy_label, data.labels, current_soft, current_alpha, current_pseudo, data.train_masks
    )
    candidate_diag = {
        **candidate_weight_diag,
        **diagnostics_for_soft_targets(
            noisy_label, data.labels, candidate_soft, candidate_alpha, candidate_pseudo, data.train_masks
        ),
    }
    shuffled_diag = {
        **shuffled_weight_diag,
        **diagnostics_for_soft_targets(
            noisy_label, data.labels, shuffled_soft, shuffled_alpha, shuffled_pseudo, data.train_masks
        ),
    }
    target_l1 = float(torch.abs(candidate_soft[torch.as_tensor(train_idx)] - current_soft[torch.as_tensor(train_idx)]).mean().item())
    row = {
        **base_fields,
        "method": method,
        "candidate_id": candidate_id,
        "max_balls": int(max_balls),
        "min_ball_size": int(min_ball_size),
        "split_improvement": float(split_improvement),
        "ellipsoid_feature_space": str(feature_space),
        "ellipsoid_covariance_estimator": str(covariance),
        "ellipsoid_ridge_scale": float(ridge_scale),
        "ellipsoid_weight_floor": float(floor),
        "ellipsoid_distance_temperature": float(temperature),
        "target_beta": float(beta),
        "target_l1_vs_current": target_l1,
        **candidate_diag,
        **_target_delta_fields(current_diag, candidate_diag, shuffled_diag, n_train=len(train_idx)),
    }
    row.update(partition_purity_diagnostics(noisy_label, data.labels, assignment, data.train_masks, data.n_classes, min_train_per_ball))
    return row


def diagnose_geometry_setting(args, dataset: str, seed: int, max_balls: int, min_ball_size: int, split: float) -> dict:
    data, noisy_label, embeddings, base_fields = _dataset_context(args, dataset, seed)
    return diagnose_geometry_setting_from_context(
        data=data,
        noisy_label=noisy_label,
        embeddings=embeddings,
        base_fields=base_fields,
        seed=seed,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split=split,
    )


def diagnose_geometry_setting_from_context(
    data,
    noisy_label,
    embeddings,
    base_fields: dict,
    seed: int,
    max_balls: int,
    min_ball_size: int,
    split: float,
) -> dict:
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=int(max_balls),
        min_ball_size=int(min_ball_size),
        split_improvement=float(split),
        random_state=int(seed),
    )
    row = _diagnostics_for_assignment(
        data=data,
        noisy_label=noisy_label,
        embeddings=embeddings,
        assignment=np.asarray(gb_result.assignment, dtype=np.int64),
        base_fields=base_fields,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split,
        feature_space="fullH",
        covariance="ridge",
        ridge_scale=0.05,
        floor=0.1,
        temperature=1.0,
        beta=0.5,
        method="geometry_probe",
        candidate_id=geometry_candidate_id(max_balls, min_ball_size, split),
    )
    row.update(gb_result.summary)
    row.update(_train_count_summary(gb_result.assignment, data.train_masks))
    row["geometry_pass"] = int(geometry_passes(row))
    return row


def aggregate_geometry(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int, float], list[dict]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            _as_int(row, "max_balls"),
            _as_int(row, "min_ball_size"),
            _as_float(row, "split_improvement"),
        )
        groups.setdefault(key, []).append(row)
    out = []
    metrics = [
        "num_balls",
        "mean_train_per_ball",
        "zero_train_ball_ratio",
        "ellipsoid_fallback_ball_ratio",
        "ellipsoid_effective_support_ratio_mean",
        "target_l1_vs_current",
        "mass_delta_per_train",
        "hard_net",
        "real_minus_shuffled_mass_delta",
    ]
    for (dataset, max_balls, min_ball_size, split), group in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "max_balls": int(max_balls),
            "min_ball_size": int(min_ball_size),
            "split_improvement": float(split),
            "candidate_id": geometry_candidate_id(max_balls, min_ball_size, split),
            "n": int(len(group)),
        }
        for metric in metrics:
            values = [_as_float(row, metric) for row in group if row.get(metric, "") not in ("", None)]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        item["geometry_pass_seed_count"] = int(sum(1 for row in group if int(row.get("geometry_pass", 0)) == 1))
        item["geometry_pass"] = bool(item["geometry_pass_seed_count"] >= min(4, len(group)))
        out.append(item)
    return out


def select_geometry_settings(aggregate: list[dict], per_dataset: int) -> list[dict]:
    selected = []
    by_dataset: dict[str, list[dict]] = {}
    for row in aggregate:
        if bool(row.get("geometry_pass", False)):
            score = (
                10.0 * _as_float(row, "target_l1_vs_current_mean")
                + _as_float(row, "real_minus_shuffled_mass_delta_mean")
                - _as_float(row, "ellipsoid_fallback_ball_ratio_mean")
            )
            item = dict(row)
            item["geometry_score"] = score
            by_dataset.setdefault(str(row["dataset"]), []).append(item)
    for dataset in sorted(by_dataset):
        group = sorted(by_dataset[dataset], key=lambda item: (-_as_float(item, "geometry_score"), str(item["candidate_id"])))
        selected.extend(group[: int(per_dataset)])
    return selected


def iter_target_param_grid(args, geometry_rows: list[dict]):
    for geo in geometry_rows:
        for feature_space, covariance, ridge, floor, temp, beta in itertools.product(
            _parse_str_list(args.feature_spaces),
            _parse_str_list(args.covariances),
            _parse_float_list(args.ridge_scales),
            _parse_float_list(args.weight_floors),
            _parse_float_list(args.distance_temperatures),
            _parse_float_list(args.target_betas),
        ):
            yield {
                "dataset": str(geo["dataset"]),
                "max_balls": _as_int(geo, "max_balls"),
                "min_ball_size": _as_int(geo, "min_ball_size"),
                "split_improvement": _as_float(geo, "split_improvement"),
                "ellipsoid_feature_space": str(feature_space),
                "ellipsoid_covariance_estimator": str(covariance),
                "ellipsoid_ridge_scale": float(ridge),
                "ellipsoid_weight_floor": float(floor),
                "ellipsoid_distance_temperature": float(temp),
                "target_beta": float(beta),
                "candidate_id": target_candidate_id(
                    _as_int(geo, "max_balls"),
                    _as_int(geo, "min_ball_size"),
                    _as_float(geo, "split_improvement"),
                    str(feature_space),
                    str(covariance),
                    float(ridge),
                    float(floor),
                    float(temp),
                    float(beta),
                ),
            }


def diagnose_target_candidate(args, candidate: dict, seed: int) -> dict:
    data, noisy_label, embeddings, base_fields = _dataset_context(args, str(candidate["dataset"]), seed)
    return diagnose_target_candidate_from_context(
        data=data,
        noisy_label=noisy_label,
        embeddings=embeddings,
        base_fields=base_fields,
        candidate=candidate,
        seed=seed,
    )


def diagnose_target_candidate_from_context(
    data,
    noisy_label,
    embeddings,
    base_fields: dict,
    candidate: dict,
    seed: int,
) -> dict:
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=_as_int(candidate, "max_balls"),
        min_ball_size=_as_int(candidate, "min_ball_size"),
        split_improvement=_as_float(candidate, "split_improvement"),
        random_state=int(seed),
    )
    row = _diagnostics_for_assignment(
        data=data,
        noisy_label=noisy_label,
        embeddings=embeddings,
        assignment=np.asarray(gb_result.assignment, dtype=np.int64),
        base_fields=base_fields,
        max_balls=_as_int(candidate, "max_balls"),
        min_ball_size=_as_int(candidate, "min_ball_size"),
        split_improvement=_as_float(candidate, "split_improvement"),
        feature_space=str(candidate["ellipsoid_feature_space"]),
        covariance=str(candidate["ellipsoid_covariance_estimator"]),
        ridge_scale=_as_float(candidate, "ellipsoid_ridge_scale"),
        floor=_as_float(candidate, "ellipsoid_weight_floor"),
        temperature=_as_float(candidate, "ellipsoid_distance_temperature"),
        beta=_as_float(candidate, "target_beta"),
        method=CANDIDATE_METHOD,
        candidate_id=str(candidate["candidate_id"]),
    )
    row.update(gb_result.summary)
    row.update(_train_count_summary(gb_result.assignment, data.train_masks))
    return row


def aggregate_target_candidates(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["candidate_id"])), []).append(row)
    out = []
    metrics = [
        "mass_delta_per_train",
        "hard_net",
        "target_l1_vs_current",
        "ellipsoid_fallback_ball_ratio",
        "ellipsoid_covariance_failure_ratio",
        "real_minus_shuffled_mass_delta",
        "real_minus_shuffled_hard_net",
        "num_balls",
        "mean_train_per_ball",
        "zero_train_ball_ratio",
        "ellipsoid_effective_support_ratio_mean",
        "soft_clean_mass_delta",
        "soft_pseudo_ncr",
        "soft_pseudo_cdr",
    ]
    for (dataset, candidate_id), group in sorted(groups.items()):
        first = group[0]
        item = {
            "dataset": dataset,
            "candidate_id": candidate_id,
            "n": int(len(group)),
            "positive_seed_count": int(sum(1 for row in group if _as_float(row, "real_minus_shuffled_mass_delta") > 0.0)),
            "max_balls": _as_int(first, "max_balls"),
            "min_ball_size": _as_int(first, "min_ball_size"),
            "split_improvement": _as_float(first, "split_improvement"),
            "ellipsoid_feature_space": str(first["ellipsoid_feature_space"]),
            "ellipsoid_covariance_estimator": str(first["ellipsoid_covariance_estimator"]),
            "ellipsoid_ridge_scale": _as_float(first, "ellipsoid_ridge_scale"),
            "ellipsoid_weight_floor": _as_float(first, "ellipsoid_weight_floor"),
            "ellipsoid_distance_temperature": _as_float(first, "ellipsoid_distance_temperature"),
            "target_beta": _as_float(first, "target_beta"),
        }
        for metric in metrics:
            values = [_as_float(row, metric) for row in group if row.get(metric, "") not in ("", None)]
            if values:
                item[metric] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        item["gate_pass"] = evaluate_offline_candidate_gate(item)
        item["score"] = candidate_score(item)
        out.append(item)
    return out


def write_report(path: Path, geometry_selected: list[dict], candidate_gate: list[dict], top_candidates: list[dict]) -> None:
    lines = [
        "# MLP-18 ellipsoid feasibility diagnostic report",
        "",
        "Scope: offline diagnostic only. Target construction uses frozen GRACE H, noisy train labels, and train mask. Clean labels are diagnostic-only.",
        "",
        "## Selected geometry settings",
        "",
    ]
    if geometry_selected:
        for row in geometry_selected:
            lines.append(
                "- {dataset}: {candidate_id}; geometry_score={score:.4f}; target_l1={l1:.4f}; fallback={fallback:.4f}".format(
                    dataset=row["dataset"],
                    candidate_id=row["candidate_id"],
                    score=_as_float(row, "geometry_score"),
                    l1=_as_float(row, "target_l1_vs_current_mean"),
                    fallback=_as_float(row, "ellipsoid_fallback_ball_ratio_mean"),
                )
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Offline candidate gate", ""])
    passed = [row for row in candidate_gate if bool(row.get("gate_pass", False))]
    lines.append(f"Passed candidates: {len(passed)} / {len(candidate_gate)}")
    for row in sorted(passed, key=lambda item: (str(item["dataset"]), -_as_float(item, "score")))[:20]:
        lines.append(
            "- {dataset}: {candidate_id}; score={score:.4f}; positive_seeds={seeds}; "
            "mass/train={mass:.5f}; hard_net={hard:.4f}; l1={l1:.4f}; fallback={fallback:.4f}".format(
                dataset=row["dataset"],
                candidate_id=row["candidate_id"],
                score=_as_float(row, "score"),
                seeds=_as_int(row, "positive_seed_count"),
                mass=_as_float(row, "mass_delta_per_train"),
                hard=_as_float(row, "hard_net"),
                l1=_as_float(row, "target_l1_vs_current"),
                fallback=_as_float(row, "ellipsoid_fallback_ball_ratio"),
            )
        )
    lines.extend(["", "## Selected downstream candidates", ""])
    if top_candidates:
        for row in top_candidates:
            lines.append(f"- {row['dataset']}: {row['candidate_id']} score={_as_float(row, 'score'):.4f}")
    else:
        lines.append("- None. Stop before downstream unless gates are intentionally revised.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="MLP-18 ellipsoid PCA/shrinkage feasibility diagnostic.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max_balls", default="32,64,128,256")
    parser.add_argument("--min_ball_sizes", default="16,32,64")
    parser.add_argument("--split_improvements", default="0.05,0.02,0.01,0.005,0.001")
    parser.add_argument("--feature_spaces", default="fullH,pca16,pca32,pca64,pca128")
    parser.add_argument("--covariances", default="ridge,oas,ledoitwolf")
    parser.add_argument("--ridge_scales", default="0.005,0.01,0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--weight_floors", default="0.0,0.02,0.05,0.1,0.2")
    parser.add_argument("--distance_temperatures", default="0.5,1.0,2.0")
    parser.add_argument("--target_betas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--top_geometry_per_dataset", type=int, default=20)
    parser.add_argument("--top_candidates_per_dataset", type=int, default=2)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--geometry_only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_cache = {}

    def context_for(dataset_name: str, seed_value: int):
        key = (str(dataset_name).lower(), int(seed_value))
        if key not in context_cache:
            context_cache[key] = _dataset_context(args, dataset_name, seed_value)
        return context_cache[key]

    geometry_raw_path = output_dir / "offline_geometry_raw.csv"
    geometry_rows = _read_csv(geometry_raw_path) if args.resume else []
    seen_geometry = {geometry_key(row) for row in geometry_rows}
    for dataset, seed, max_balls, min_ball_size, split in itertools.product(
        _parse_str_list(args.datasets),
        _parse_int_list(args.seeds),
        _parse_int_list(args.max_balls),
        _parse_int_list(args.min_ball_sizes),
        _parse_float_list(args.split_improvements),
    ):
        key = (dataset.lower(), int(seed), int(max_balls), int(min_ball_size), float(split))
        if key in seen_geometry:
            continue
        print(f"MLP18_GEOMETRY dataset={dataset} seed={seed} max_balls={max_balls} min={min_ball_size} split={split}", flush=True)
        data, noisy_label, embeddings, base_fields = context_for(dataset, seed)
        row = diagnose_geometry_setting_from_context(
            data=data,
            noisy_label=noisy_label,
            embeddings=embeddings,
            base_fields=base_fields,
            seed=seed,
            max_balls=max_balls,
            min_ball_size=min_ball_size,
            split=split,
        )
        geometry_rows.append(row)
        seen_geometry.add(key)
        write_csv(geometry_raw_path, geometry_rows)

    geometry_aggregate = aggregate_geometry(geometry_rows)
    geometry_selected = select_geometry_settings(geometry_aggregate, per_dataset=args.top_geometry_per_dataset)
    write_csv(output_dir / "offline_geometry_aggregate.csv", geometry_aggregate)
    write_csv(output_dir / "offline_geometry_gate.csv", geometry_selected)
    if args.geometry_only:
        write_report(output_dir / "mlp18_report.md", geometry_selected, [], [])
        return

    candidate_specs = list(iter_target_param_grid(args, geometry_selected))
    spec_path = output_dir / "offline_candidate_specs.json"
    spec_path.write_text(json.dumps(candidate_specs, indent=2, sort_keys=True), encoding="utf-8")

    target_raw_path = output_dir / "offline_target_raw.csv"
    target_rows = _read_csv(target_raw_path) if args.resume else []
    seen_target = {row_key(row) for row in target_rows}
    for candidate in candidate_specs:
        for seed in _parse_int_list(args.seeds):
            candidate_row = {
                "dataset": str(candidate["dataset"]).lower(),
                "seed": int(seed),
                "candidate_id": str(candidate["candidate_id"]),
                "method": CANDIDATE_METHOD,
            }
            key = row_key(candidate_row)
            if key in seen_target:
                continue
            print(f"MLP18_TARGET dataset={candidate['dataset']} seed={seed} candidate={candidate['candidate_id']}", flush=True)
            data, noisy_label, embeddings, base_fields = context_for(str(candidate["dataset"]), seed)
            row = diagnose_target_candidate_from_context(
                data=data,
                noisy_label=noisy_label,
                embeddings=embeddings,
                base_fields=base_fields,
                candidate=candidate,
                seed=seed,
            )
            target_rows.append(row)
            seen_target.add(key)
            write_csv(target_raw_path, target_rows)

    candidate_gate = aggregate_target_candidates(target_rows)
    top_candidates = select_top_candidates(candidate_gate, per_dataset=args.top_candidates_per_dataset)
    write_csv(output_dir / "offline_candidate_gate.csv", candidate_gate)
    write_csv(output_dir / "downstream_seed0_candidates.csv", top_candidates)
    write_report(output_dir / "mlp18_report.md", geometry_selected, candidate_gate, top_candidates)
    print(f"MLP18 output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
