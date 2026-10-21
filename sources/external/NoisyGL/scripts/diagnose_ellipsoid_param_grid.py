from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_ellipsoid_softdist import (  # noqa: E402
    BASE_METHOD,
    DEFAULT_METHODS,
    ELLIPSOID_METHOD,
    ELLIPSOID_SHUFFLED_METHOD,
    METRICS,
    _as_float,
    _as_index_array,
    _gb_value,
    _parse_int_list,
    _parse_str_list,
    _resolve_embedding_path,
    write_csv,
)
from scripts.run_standard_downstream import _load_dataset  # noqa: E402
from utils.frozen_grace_targets import (  # noqa: E402
    _soft_coverage_diagnostics,
    apply_ball_ellipsoid_soft_targets,
    apply_ball_soft_targets,
    build_frozen_grace_soft_targets,
    diagnostics_for_soft_targets,
    load_frozen_embeddings,
    partition_purity_diagnostics,
    prepare_ball_features,
)
from utils.labelnoise import label_process  # noqa: E402
from utils.ordinary_gb import (  # noqa: E402
    build_ordinary_granular_balls,
    build_purity_adaptive_granular_balls,
    build_random_size_matched_assignment,
)
from utils.tools import load_conf, setup_seed  # noqa: E402


BASE_SHUFFLED_METHOD = "frozengrace_gb_gcn_softdist_xh_shuffled"
DEFAULT_DATASETS = "dblp,pubmed,amazonpho,amazoncom"
DEFAULT_MAX_BALLS = "64,128,256,512"
DEFAULT_SPLIT_IMPROVEMENTS = "0.05,0.02,0.01,0.005,0.001"
GRID_OUTPUT_DIR = REPO_ROOT / "runs" / "ellipsoid_param_grid_20260628"
GRID_METRICS = list(
    dict.fromkeys(
        [
            "num_balls",
            "mean_ball_size",
            "max_ball_size",
            "true_train_purity_weighted",
            "noisy_train_purity_weighted",
            *METRICS,
        ]
    )
)


def _as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(float(value))


def _setting_key(row: dict) -> tuple[str, int, float]:
    return (
        str(row["dataset"]),
        _as_int(row, "max_balls"),
        _as_float(row, "split_improvement"),
    )


def _row_key(row: dict) -> tuple[str, int, int, float, str]:
    dataset, max_balls, split_improvement = _setting_key(row)
    return (
        dataset,
        _as_int(row, "seed"),
        max_balls,
        split_improvement,
        str(row["method"]),
    )


def _completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("status", "ok")).lower() == "ok"]


def _mean_for(rows: list[dict], metric: str, default: float = 0.0) -> float:
    values = [_as_float(row, metric) for row in rows if row.get(metric, "") not in ("", None)]
    if not values:
        return float(default)
    return float(np.mean(values))


def aggregate_grid_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int, float], list[dict]] = {}
    for row in _completed_rows(rows):
        dataset, max_balls, split_improvement = _setting_key(row)
        key = (dataset, str(row["method"]), max_balls, split_improvement)
        groups.setdefault(key, []).append(row)

    out = []
    for (dataset, method, max_balls, split_improvement), group in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "method": method,
            "max_balls": int(max_balls),
            "split_improvement": float(split_improvement),
            "n": int(len(group)),
        }
        for metric in GRID_METRICS:
            values = [_as_float(row, metric) for row in group if row.get(metric, "") not in ("", None)]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def paired_grid_delta_rows(
    rows: list[dict],
    reference_method: str = BASE_METHOD,
    candidate_method: str = ELLIPSOID_METHOD,
) -> list[dict]:
    by_key = {
        (
            str(row["dataset"]),
            _as_int(row, "seed"),
            _as_int(row, "max_balls"),
            _as_float(row, "split_improvement"),
            str(row["method"]),
        ): row
        for row in _completed_rows(rows)
    }
    candidate_keys = sorted(
        key
        for key in by_key
        if key[4] == candidate_method
        and (key[0], key[1], key[2], key[3], reference_method) in by_key
    )
    out = []
    for dataset, seed, max_balls, split_improvement, _ in candidate_keys:
        reference = by_key[(dataset, seed, max_balls, split_improvement, reference_method)]
        candidate = by_key[(dataset, seed, max_balls, split_improvement, candidate_method)]
        item = {
            "dataset": dataset,
            "seed": int(seed),
            "max_balls": int(max_balls),
            "split_improvement": float(split_improvement),
            "comparison": f"{candidate_method}_minus_{reference_method}",
            "candidate_method": candidate_method,
            "reference_method": reference_method,
        }
        for metric in GRID_METRICS:
            if metric in candidate or metric in reference:
                item[f"{metric}_delta"] = _as_float(candidate, metric) - _as_float(reference, metric)
        out.append(item)
    return out


def evaluate_grid_gates(
    rows: list[dict],
    reference_method: str = BASE_METHOD,
    candidate_method: str = ELLIPSOID_METHOD,
    shuffled_method: str = ELLIPSOID_SHUFFLED_METHOD,
    min_positive_seeds: int = 4,
) -> list[dict]:
    completed = _completed_rows(rows)
    deltas = paired_grid_delta_rows(completed, reference_method, candidate_method)
    shuffled_deltas = paired_grid_delta_rows(completed, shuffled_method, candidate_method)

    rows_by_setting: dict[tuple[str, int, float], list[dict]] = {}
    deltas_by_setting: dict[tuple[str, int, float], list[dict]] = {}
    shuffled_by_setting: dict[tuple[str, int, float], list[dict]] = {}
    for row in completed:
        rows_by_setting.setdefault(_setting_key(row), []).append(row)
    for row in deltas:
        deltas_by_setting.setdefault(_setting_key(row), []).append(row)
    for row in shuffled_deltas:
        shuffled_by_setting.setdefault(_setting_key(row), []).append(row)

    decisions = []
    for dataset, max_balls, split_improvement in sorted(rows_by_setting):
        setting_rows = rows_by_setting[(dataset, max_balls, split_improvement)]
        setting_deltas = deltas_by_setting.get((dataset, max_balls, split_improvement), [])
        setting_shuffled = shuffled_by_setting.get((dataset, max_balls, split_improvement), [])
        candidate_rows = [row for row in setting_rows if str(row["method"]) == candidate_method]
        common = {
            "dataset": dataset,
            "max_balls": int(max_balls),
            "split_improvement": float(split_improvement),
            "candidate_num_balls_mean": _mean_for(candidate_rows, "num_balls"),
            "candidate_mean_ball_size_mean": _mean_for(candidate_rows, "mean_ball_size"),
            "candidate_max_ball_size_mean": _mean_for(candidate_rows, "max_ball_size"),
            "candidate_true_train_purity_weighted_mean": _mean_for(
                candidate_rows, "true_train_purity_weighted"
            ),
            "candidate_noisy_train_purity_weighted_mean": _mean_for(
                candidate_rows, "noisy_train_purity_weighted"
            ),
            "candidate_effective_support_ratio_mean": _mean_for(
                candidate_rows, "ellipsoid_effective_support_ratio_mean"
            ),
        }
        if not setting_deltas:
            decisions.append(
                {
                    **common,
                    "passed": False,
                    "reason": "missing paired baseline rows",
                    "positive_seed_count": 0,
                    "beats_shuffled": False,
                    "fallback_ratio_mean": _mean_for(candidate_rows, "ellipsoid_fallback_ball_ratio", 1.0),
                    "fallback_ok": False,
                    "mass_delta_mean": 0.0,
                    "cdr_delta_mean": 0.0,
                    "ncr_delta_mean": 0.0,
                }
            )
            continue

        mass_delta_mean = _mean_for(setting_deltas, "soft_clean_mass_delta_delta")
        cdr_delta_mean = _mean_for(setting_deltas, "soft_pseudo_cdr_delta")
        ncr_delta_mean = _mean_for(setting_deltas, "soft_pseudo_ncr_delta")
        positive_seed_count = sum(
            1
            for row in setting_deltas
            if (
                _as_float(row, "soft_clean_mass_delta_delta") >= 0.01
                or (
                    _as_float(row, "soft_pseudo_cdr_delta") <= -0.02
                    and _as_float(row, "soft_pseudo_ncr_delta") >= -0.02
                )
            )
        )
        diagnostic_gain = (
            mass_delta_mean >= 0.01
            or (cdr_delta_mean <= -0.02 and ncr_delta_mean >= -0.02)
        )
        if setting_shuffled:
            shuffled_mass_delta_mean = _mean_for(setting_shuffled, "soft_clean_mass_delta_delta")
            shuffled_cdr_delta_mean = _mean_for(setting_shuffled, "soft_pseudo_cdr_delta")
            beats_shuffled = shuffled_mass_delta_mean > 0.0 and shuffled_cdr_delta_mean < 0.0
        else:
            shuffled_mass_delta_mean = 0.0
            shuffled_cdr_delta_mean = 0.0
            beats_shuffled = False
        fallback_mean = _mean_for(candidate_rows, "ellipsoid_fallback_ball_ratio", 1.0)
        fallback_ok = fallback_mean <= 0.05
        passed = (
            diagnostic_gain
            and positive_seed_count >= int(min_positive_seeds)
            and beats_shuffled
            and fallback_ok
        )
        reason_parts = []
        if not diagnostic_gain:
            reason_parts.append("no diagnostic gain")
        if positive_seed_count < int(min_positive_seeds):
            reason_parts.append("not enough positive seeds")
        if not beats_shuffled:
            reason_parts.append("not better than shuffled")
        if not fallback_ok:
            reason_parts.append("fallback too high")
        decisions.append(
            {
                **common,
                "passed": bool(passed),
                "reason": "pass" if passed else "; ".join(reason_parts),
                "mass_delta_mean": mass_delta_mean,
                "cdr_delta_mean": cdr_delta_mean,
                "ncr_delta_mean": ncr_delta_mean,
                "positive_seed_count": int(positive_seed_count),
                "beats_shuffled": bool(beats_shuffled),
                "ellipsoid_minus_shuffled_mass_delta_mean": shuffled_mass_delta_mean,
                "ellipsoid_minus_shuffled_cdr_delta_mean": shuffled_cdr_delta_mean,
                "fallback_ratio_mean": fallback_mean,
                "fallback_ok": bool(fallback_ok),
            }
        )
    return decisions


def _best_setting(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("passed", False)),
            _as_float(row, "mass_delta_mean"),
            -_as_float(row, "cdr_delta_mean"),
            _as_float(row, "ncr_delta_mean"),
            _as_float(row, "candidate_num_balls_mean"),
        ),
        reverse=True,
    )[0]


def classify_dataset_settings(decisions: list[dict], mass_sensitivity_threshold: float = 1.0) -> dict[str, dict]:
    by_dataset: dict[str, list[dict]] = {}
    for row in decisions:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    out = {}
    for dataset, group in sorted(by_dataset.items()):
        passed = [row for row in group if bool(row.get("passed", False))]
        best = _best_setting(group)
        if not passed:
            out[dataset] = {
                "classification": "no-offline-value",
                "reason": "no grid setting passed the offline gate",
                "best_setting": best,
            }
            continue

        collapsed_failures = [
            row
            for row in group
            if not bool(row.get("passed", False))
            and _as_float(row, "candidate_num_balls_mean") <= 1.5
        ]
        recovered_passes = [
            row for row in passed if _as_float(row, "candidate_num_balls_mean") >= 2.0
        ]
        if collapsed_failures and recovered_passes:
            out[dataset] = {
                "classification": "collapse-fixed",
                "reason": "a collapsed setting failed and a lower-threshold setting recovered usable balls",
                "best_setting": _best_setting(recovered_passes),
            }
            continue

        sensitive = False
        by_split: dict[float, list[dict]] = {}
        for row in group:
            by_split.setdefault(_as_float(row, "split_improvement"), []).append(row)
        for split_rows in by_split.values():
            if len({_as_int(row, "max_balls") for row in split_rows}) < 2:
                continue
            pass_values = {bool(row.get("passed", False)) for row in split_rows}
            mass_values = [_as_float(row, "mass_delta_mean") for row in split_rows]
            if len(pass_values) > 1 or (max(mass_values) - min(mass_values)) >= mass_sensitivity_threshold:
                sensitive = True
                break
        if sensitive:
            out[dataset] = {
                "classification": "ball-count-sensitive",
                "reason": "max_balls changed the gate decision or clean-mass delta materially",
                "best_setting": _best_setting(passed),
            }
        else:
            out[dataset] = {
                "classification": "stable-offline-value",
                "reason": "at least one setting passed without a strong collapse or max_balls signal",
                "best_setting": _best_setting(passed),
            }
    return out


def _load_grid_context(args, dataset_name: str, seed: int) -> tuple[str, object, object, np.ndarray, dict]:
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
    train_idx = _as_index_array(data.train_masks)
    modified = _as_index_array(modified_mask)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    base_fields = {
        "dataset": dataset_key,
        "seed": int(seed),
        "min_ball_size": int(args.min_ball_size),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "modified_count": int(modified.size),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        "status": "ok",
        "error": "",
    }
    return dataset_key, data, noisy_label, train_idx, base_fields


def _construction_key(model_conf, embedding_path: Path, max_balls: int, min_ball_size: int, split_improvement: float) -> tuple:
    ball_mode = str(_gb_value(model_conf, "ball_mode", "geometric"))
    return (
        str(embedding_path),
        ball_mode,
        int(max_balls),
        int(min_ball_size),
        float(split_improvement),
        float(_gb_value(model_conf, "purity_threshold", 0.6)),
        int(_gb_value(model_conf, "min_train_per_child", 4)),
        int(_gb_value(model_conf, "min_train_to_split", 16)),
        int(_gb_value(model_conf, "adaptive_min_ball_size", 8)),
        int(_gb_value(model_conf, "adaptive_max_balls", 128)),
        float(_gb_value(model_conf, "min_purity_gain", 0.05)),
        int(_gb_value(model_conf, "split_attempts", 1)),
    )


def _build_partition_from_conf(
    model_conf,
    embeddings,
    noisy_label,
    train_mask,
    n_classes: int,
    max_balls: int,
    min_ball_size: int,
    split_improvement: float,
    seed: int,
):
    ball_mode = str(_gb_value(model_conf, "ball_mode", "geometric"))
    if ball_mode == "adaptive":
        return build_purity_adaptive_granular_balls(
            embeddings,
            noisy_label=noisy_label,
            train_mask=train_mask,
            n_classes=n_classes,
            purity_threshold=_gb_value(model_conf, "purity_threshold", 0.6),
            min_train_per_child=_gb_value(model_conf, "min_train_per_child", 4),
            min_train_to_split=_gb_value(model_conf, "min_train_to_split", 16),
            min_ball_size=_gb_value(model_conf, "adaptive_min_ball_size", 8),
            max_balls=_gb_value(model_conf, "adaptive_max_balls", 128),
            min_purity_gain=_gb_value(model_conf, "min_purity_gain", 0.05),
            split_attempts=_gb_value(model_conf, "split_attempts", 1),
            random_state=seed,
        )
    if ball_mode == "geometric":
        return build_ordinary_granular_balls(
            embeddings,
            max_balls=int(max_balls),
            min_ball_size=int(min_ball_size),
            split_improvement=float(split_improvement),
            random_state=seed,
        )
    raise ValueError("ball_mode must be 'geometric' or 'adaptive'")


def _diagnostics_from_assignment(
    model_conf,
    embeddings,
    assignment: np.ndarray,
    partition_summary: dict,
    noisy_label,
    clean_label,
    train_mask,
    n_classes: int,
    random_control: bool,
) -> dict:
    min_train_per_ball = _gb_value(model_conf, "min_train_per_ball", 2)
    alpha_baseline = _gb_value(model_conf, "alpha_baseline", "multiclass")
    target_mode = _gb_value(model_conf, "target_mode", "distribution")
    softdist_weighting = str(_gb_value(model_conf, "softdist_weighting", "uniform"))
    if softdist_weighting == "uniform":
        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=n_classes,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
        )
        weighting_diagnostics = {
            **_soft_coverage_diagnostics(
                noisy_label, assignment, train_mask, n_classes, min_train_per_ball, alpha_baseline
            ),
            "softdist_weighting": "uniform",
        }
    elif softdist_weighting == "ellipsoid":
        soft_target, alpha, pseudo_label, weighting_diagnostics = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=n_classes,
            ball_features=embeddings,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            ellipsoid_ridge_scale=_gb_value(model_conf, "ellipsoid_ridge_scale", 0.05),
            ellipsoid_weight_floor=_gb_value(model_conf, "ellipsoid_weight_floor", 0.1),
        )
    else:
        raise ValueError(f"unsupported batched softdist_weighting={softdist_weighting}")

    ball_mode = str(_gb_value(model_conf, "ball_mode", "geometric"))
    return {
        **partition_summary,
        **weighting_diagnostics,
        **diagnostics_for_soft_targets(noisy_label, clean_label, soft_target, alpha, pseudo_label, train_mask),
        **partition_purity_diagnostics(
            noisy_label, clean_label, assignment, train_mask, n_classes, min_train_per_ball
        ),
        "ball_mode": ball_mode,
        "soft_alpha_baseline": str(alpha_baseline),
        "soft_target_mode": str(target_mode),
        "random_control": int(random_control),
        "ball_feature_source": "grace",
    }


def diagnose_grid_setting(
    args,
    dataset_name: str,
    seed: int,
    max_balls: int,
    split_improvement: float,
    methods: list[str],
) -> list[dict]:
    dataset_key, data, noisy_label, _train_idx, base_fields = _load_grid_context(args, dataset_name, seed)
    method_items = []
    for method in methods:
        model_conf = load_conf(None, method, dataset_key)
        embedding_path = _gb_value(model_conf, "ball_embedding_path", _gb_value(model_conf, "embedding_path", None))
        if embedding_path is None:
            raise ValueError(f"missing gb.embedding_path for method={method} dataset={dataset_key}")
        resolved_path = _resolve_embedding_path(embedding_path)
        method_items.append((method, model_conf, resolved_path))

    groups: dict[tuple, list[tuple[str, object, Path]]] = {}
    for method, model_conf, resolved_path in method_items:
        key = _construction_key(
            model_conf,
            resolved_path,
            max_balls=int(max_balls),
            min_ball_size=int(args.min_ball_size),
            split_improvement=float(split_improvement),
        )
        groups.setdefault(key, []).append((method, model_conf, resolved_path))

    rows = []
    embedding_cache = {}
    for _, group in groups.items():
        method0, model_conf0, resolved_path0 = group[0]
        try:
            if str(resolved_path0) not in embedding_cache:
                embedding_cache[str(resolved_path0)] = prepare_ball_features(
                    load_frozen_embeddings(resolved_path0, expected_nodes=int(noisy_label.shape[0]))
                )
            embeddings = embedding_cache[str(resolved_path0)]
            gb_result = _build_partition_from_conf(
                model_conf0,
                embeddings,
                noisy_label=noisy_label,
                train_mask=data.train_masks,
                n_classes=data.n_classes,
                max_balls=int(max_balls),
                min_ball_size=int(args.min_ball_size),
                split_improvement=float(split_improvement),
                seed=int(seed),
            )
        except Exception as exc:
            rows.extend(
                _error_row(args, dataset_key, method, seed, max_balls, split_improvement, exc)
                for method, _, _ in group
            )
            continue

        for method, model_conf, _resolved_path in group:
            try:
                random_control = method.endswith("_shuffled")
                assignment = gb_result.assignment
                if random_control:
                    assignment = build_random_size_matched_assignment(assignment, random_state=int(seed) + 17)
                diagnostics = _diagnostics_from_assignment(
                    model_conf=model_conf,
                    embeddings=embeddings,
                    assignment=assignment,
                    partition_summary=gb_result.summary,
                    noisy_label=noisy_label,
                    clean_label=data.labels,
                    train_mask=data.train_masks,
                    n_classes=data.n_classes,
                    random_control=random_control,
                )
                rows.append(
                    {
                        **base_fields,
                        "method": method,
                        "max_balls": int(max_balls),
                        "split_improvement": float(split_improvement),
                        **diagnostics,
                    }
                )
            except Exception as exc:
                rows.append(_error_row(args, dataset_key, method, seed, max_balls, split_improvement, exc))
    return rows


def diagnose_grid_one(
    args,
    dataset_name: str,
    method: str,
    seed: int,
    max_balls: int,
    split_improvement: float,
) -> dict:
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
    model_conf = load_conf(None, method, dataset_key)
    embedding_path = _gb_value(model_conf, "ball_embedding_path", _gb_value(model_conf, "embedding_path", None))
    if embedding_path is None:
        raise ValueError(f"missing gb.embedding_path for method={method} dataset={dataset_key}")
    result = build_frozen_grace_soft_targets(
        embedding_path=_resolve_embedding_path(embedding_path),
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=int(max_balls),
        min_ball_size=int(args.min_ball_size),
        split_improvement=float(split_improvement),
        min_train_per_ball=_gb_value(model_conf, "min_train_per_ball", 2),
        alpha_baseline=_gb_value(model_conf, "alpha_baseline", "multiclass"),
        target_mode=_gb_value(model_conf, "target_mode", "distribution"),
        softdist_weighting=_gb_value(model_conf, "softdist_weighting", "uniform"),
        ellipsoid_ridge_scale=_gb_value(model_conf, "ellipsoid_ridge_scale", 0.05),
        ellipsoid_weight_floor=_gb_value(model_conf, "ellipsoid_weight_floor", 0.1),
        random_state=seed,
        random_control=method.endswith("_shuffled"),
        ball_mode=_gb_value(model_conf, "ball_mode", "geometric"),
        purity_threshold=_gb_value(model_conf, "purity_threshold", 0.6),
        min_train_per_child=_gb_value(model_conf, "min_train_per_child", 4),
        min_train_to_split=_gb_value(model_conf, "min_train_to_split", 16),
        adaptive_min_ball_size=_gb_value(model_conf, "adaptive_min_ball_size", 8),
        adaptive_max_balls=_gb_value(model_conf, "adaptive_max_balls", 128),
        min_purity_gain=_gb_value(model_conf, "min_purity_gain", 0.05),
        split_attempts=_gb_value(model_conf, "split_attempts", 1),
    )
    train_idx = _as_index_array(data.train_masks)
    modified = _as_index_array(modified_mask)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    return {
        "dataset": dataset_key,
        "method": method,
        "seed": int(seed),
        "max_balls": int(max_balls),
        "min_ball_size": int(args.min_ball_size),
        "split_improvement": float(split_improvement),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "modified_count": int(modified.size),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        "status": "ok",
        "error": "",
        **result.diagnostics,
    }


def _error_row(
    args,
    dataset: str,
    method: str,
    seed: int,
    max_balls: int,
    split_improvement: float,
    exc: Exception,
) -> dict:
    return {
        "dataset": dataset.lower(),
        "method": method,
        "seed": int(seed),
        "max_balls": int(max_balls),
        "min_ball_size": int(args.min_ball_size),
        "split_improvement": float(split_improvement),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_adaptive_report(path: Path, decisions: list[dict], aggregate: list[dict], raw_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = classify_dataset_settings(decisions)
    error_count = sum(1 for row in raw_rows if str(row.get("status", "ok")).lower() != "ok")
    lines = [
        "# Ellipsoid SoftDist adaptive parameter grid report",
        "",
        "Scope: offline target diagnostic only. Clean labels are used only for diagnosis and gate checks.",
        "This report is not downstream accuracy evidence and should not be used as a paper claim by itself.",
        "",
        f"Raw rows: {len(raw_rows)}",
        f"Error rows: {error_count}",
        "",
        "## Dataset conclusions",
        "",
    ]
    for dataset, item in classes.items():
        best = item.get("best_setting", {})
        lines.append(
            "- {dataset}: {cls}; best=max_balls={max_balls}, split_improvement={split}, "
            "passed={passed}, mass_delta={mass:.4f}, cdr_delta={cdr:.4f}, "
            "ncr_delta={ncr:.4f}, balls={balls:.1f}; {reason}".format(
                dataset=dataset,
                cls=item.get("classification"),
                max_balls=_as_int(best, "max_balls"),
                split=_as_float(best, "split_improvement"),
                passed=bool(best.get("passed", False)),
                mass=_as_float(best, "mass_delta_mean"),
                cdr=_as_float(best, "cdr_delta_mean"),
                ncr=_as_float(best, "ncr_delta_mean"),
                balls=_as_float(best, "candidate_num_balls_mean"),
                reason=item.get("reason"),
            )
        )

    lines.extend(["", "## Direct answers", ""])
    lines.extend(_direct_answer_lines(classes, decisions))

    lines.extend(["", "## Worth downstream", ""])
    passed_rows = [row for row in decisions if bool(row.get("passed", False))]
    if passed_rows:
        for row in sorted(passed_rows, key=lambda r: (str(r["dataset"]), -_as_float(r, "mass_delta_mean"))):
            lines.append(
                "- {dataset}: max_balls={max_balls}, split_improvement={split}, "
                "mass_delta={mass:.4f}, cdr_delta={cdr:.4f}, ncr_delta={ncr:.4f}, "
                "positive_seeds={seeds}, fallback={fallback:.4f}, balls={balls:.1f}".format(
                    dataset=row.get("dataset"),
                    max_balls=_as_int(row, "max_balls"),
                    split=_as_float(row, "split_improvement"),
                    mass=_as_float(row, "mass_delta_mean"),
                    cdr=_as_float(row, "cdr_delta_mean"),
                    ncr=_as_float(row, "ncr_delta_mean"),
                    seeds=_as_int(row, "positive_seed_count"),
                    fallback=_as_float(row, "fallback_ratio_mean"),
                    balls=_as_float(row, "candidate_num_balls_mean"),
                )
            )
    else:
        lines.append("- None. Current grid did not provide a usable offline target signal.")

    lines.extend(["", "## Gate table", ""])
    lines.append(
        "| dataset | max_balls | split | gate | mass_delta | cdr_delta | ncr_delta | "
        "positive_seeds | beats_shuffled | fallback | balls | reason |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|")
    for row in sorted(decisions, key=lambda r: (str(r["dataset"]), _as_int(r, "max_balls"), _as_float(r, "split_improvement"))):
        lines.append(
            "| {dataset} | {max_balls} | {split:.4g} | {gate} | {mass:.4f} | {cdr:.4f} | "
            "{ncr:.4f} | {seeds} | {beats} | {fallback:.4f} | {balls:.1f} | {reason} |".format(
                dataset=row.get("dataset"),
                max_balls=_as_int(row, "max_balls"),
                split=_as_float(row, "split_improvement"),
                gate="PASS" if row.get("passed") else "FAIL",
                mass=_as_float(row, "mass_delta_mean"),
                cdr=_as_float(row, "cdr_delta_mean"),
                ncr=_as_float(row, "ncr_delta_mean"),
                seeds=_as_int(row, "positive_seed_count"),
                beats=bool(row.get("beats_shuffled", False)),
                fallback=_as_float(row, "fallback_ratio_mean"),
                balls=_as_float(row, "candidate_num_balls_mean"),
                reason=str(row.get("reason", "")),
            )
        )

    lines.extend(["", "## Aggregate snapshot", ""])
    for row in sorted(aggregate, key=lambda r: (str(r["dataset"]), str(r["method"]), _as_int(r, "max_balls"), _as_float(r, "split_improvement"))):
        lines.append(
            "- {dataset} / {method} / max_balls={max_balls} / split={split}: n={n}, "
            "balls={balls:.1f}, clean_mass={mass:.4f}, cdr={cdr:.4f}, ncr={ncr:.4f}".format(
                dataset=row.get("dataset"),
                method=row.get("method"),
                max_balls=_as_int(row, "max_balls"),
                split=_as_float(row, "split_improvement"),
                n=_as_int(row, "n"),
                balls=_as_float(row, "num_balls_mean"),
                mass=_as_float(row, "soft_clean_mass_delta_mean"),
                cdr=_as_float(row, "soft_pseudo_cdr_mean"),
                ncr=_as_float(row, "soft_pseudo_ncr_mean"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _direct_answer_lines(classes: dict[str, dict], decisions: list[dict]) -> list[str]:
    by_dataset: dict[str, list[dict]] = {}
    for row in decisions:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    lines = []
    dblp = classes.get("dblp", {})
    if dblp.get("classification") == "collapse-fixed":
        best = dblp.get("best_setting", {})
        lines.append(
            "- DBLP split adaptation: yes. Lower split_improvement recovers balls and passes the gate; "
            f"best=max_balls={_as_int(best, 'max_balls')}, split={_as_float(best, 'split_improvement')}."
        )
    else:
        lines.append("- DBLP split adaptation: no clear offline fix in this grid.")

    pubmed = classes.get("pubmed", {})
    pubmed_best = pubmed.get("best_setting", {})
    if pubmed.get("classification") == "ball-count-sensitive":
        lines.append(
            "- PubMed max_balls impact: yes. Gate and clean-mass delta change with max_balls; "
            f"best=max_balls={_as_int(pubmed_best, 'max_balls')}, split={_as_float(pubmed_best, 'split_improvement')}."
        )
    else:
        lines.append("- PubMed max_balls impact: no strong max_balls sensitivity found by this gate.")

    for dataset in ["amazonpho", "amazoncom"]:
        group = by_dataset.get(dataset, [])
        passed = [row for row in group if bool(row.get("passed", False))]
        if not passed:
            lines.append(f"- {dataset} 64/256 cap: no usable offline signal in this grid.")
            continue
        best = _best_setting(passed)
        high_cap_rows = [row for row in group if _as_int(row, "max_balls") >= 256]
        high_cap_balls = max((_as_float(row, "candidate_num_balls_mean") for row in high_cap_rows), default=0.0)
        if _as_int(best, "max_balls") <= 64:
            lines.append(
                f"- {dataset} 64/256 cap: no evidence that the cap is too low; "
                f"best offline setting uses max_balls={_as_int(best, 'max_balls')} and higher caps are weaker."
            )
        elif high_cap_balls < 256.0:
            lines.append(
                f"- {dataset} 64/256 cap: no hard cap bottleneck; construction saturates below 256/512 "
                f"(max observed balls={high_cap_balls:.1f})."
            )
        else:
            lines.append(
                f"- {dataset} 64/256 cap: possible cap sensitivity; best=max_balls={_as_int(best, 'max_balls')}, "
                f"split={_as_float(best, 'split_improvement')}."
            )
    return lines


def build_parser():
    parser = argparse.ArgumentParser(description="Run Ellipsoid-SoftDist target diagnostics over GB parameter grid.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max_balls", default=DEFAULT_MAX_BALLS)
    parser.add_argument("--split_improvements", default=DEFAULT_SPLIT_IMPROVEMENTS)
    parser.add_argument("--min_ball_size", type=int, default=32)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output_dir", default=str(GRID_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "diagnostic_grid_raw.csv"
    raw_rows = read_csv_rows(raw_path) if args.resume else []
    seen_keys = {_row_key(row) for row in raw_rows}
    if args.resume and raw_rows:
        print(f"ELLIPSOID_PARAM_GRID resume_rows={len(raw_rows)}", flush=True)
    methods = _parse_str_list(args.methods)
    for dataset in _parse_str_list(args.datasets):
        for seed in _parse_int_list(args.seeds):
            for max_balls in _parse_int_list(args.max_balls):
                for split_improvement in [float(item) for item in _parse_str_list(args.split_improvements)]:
                    pending_methods = []
                    for method in methods:
                        key = (
                            dataset.lower(),
                            int(seed),
                            int(max_balls),
                            float(split_improvement),
                            method,
                        )
                        if key not in seen_keys:
                            pending_methods.append(method)
                    if not pending_methods:
                        continue
                    print(
                        "ELLIPSOID_PARAM_GRID "
                        f"dataset={dataset} seed={seed} max_balls={max_balls} "
                        f"split_improvement={split_improvement} methods={len(pending_methods)}",
                        flush=True,
                    )
                    try:
                        new_rows = diagnose_grid_setting(
                            args=args,
                            dataset_name=dataset,
                            seed=seed,
                            max_balls=max_balls,
                            split_improvement=split_improvement,
                            methods=pending_methods,
                        )
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        new_rows = [
                            _error_row(args, dataset, method, seed, max_balls, split_improvement, exc)
                            for method in pending_methods
                        ]
                    for row in new_rows:
                        key = _row_key(row)
                        if key in seen_keys:
                            continue
                        print(
                            "ELLIPSOID_PARAM_GRID "
                            f"row dataset={row.get('dataset')} seed={row.get('seed')} "
                            f"max_balls={row.get('max_balls')} split_improvement={row.get('split_improvement')} "
                            f"method={row.get('method')} status={row.get('status', 'ok')}",
                            flush=True,
                        )
                        raw_rows.append(row)
                        seen_keys.add(key)
                        write_csv(raw_path, raw_rows)

    aggregate = aggregate_grid_rows(raw_rows)
    paired = paired_grid_delta_rows(raw_rows, BASE_METHOD, ELLIPSOID_METHOD)
    paired.extend(paired_grid_delta_rows(raw_rows, ELLIPSOID_SHUFFLED_METHOD, ELLIPSOID_METHOD))
    decisions = evaluate_grid_gates(raw_rows)
    write_csv(output_dir / "diagnostic_grid_aggregate.csv", aggregate)
    write_csv(output_dir / "diagnostic_grid_paired_deltas.csv", paired)
    write_csv(output_dir / "diagnostic_grid_gate.csv", decisions)
    write_adaptive_report(output_dir / "adaptive_candidate_report.md", decisions, aggregate, raw_rows)
    print(f"ELLIPSOID_PARAM_GRID output_dir={output_dir}")


if __name__ == "__main__":
    main()
