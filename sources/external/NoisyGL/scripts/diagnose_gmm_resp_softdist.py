from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import build_frozen_grace_soft_targets
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


BASE_METHOD = "frozengrace_gb_gcn_softdist_xh"
BASE_SHUFFLED_METHOD = "frozengrace_gb_gcn_softdist_xh_shuffled"
ELLIPSOID_METHOD = "frozengrace_gb_gcn_softdist_xh_ellipsoid"
ELLIPSOID_SHUFFLED_METHOD = "frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled"
GMM_METHOD = "frozengrace_gb_gcn_softdist_xh_gmmresp"
GMM_SHUFFLED_METHOD = "frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled"
DEFAULT_METHODS = ",".join(
    [
        BASE_METHOD,
        BASE_SHUFFLED_METHOD,
        ELLIPSOID_METHOD,
        ELLIPSOID_SHUFFLED_METHOD,
        GMM_METHOD,
        GMM_SHUFFLED_METHOD,
    ]
)
METRICS = [
    "soft_clean_mass_delta",
    "soft_pseudo_ncr",
    "soft_pseudo_cdr",
    "mean_alpha_train",
    "soft_shift_train_mass",
    "gmm_fallback",
    "gmm_converged",
    "gmm_n_iter",
    "gmm_component_support_min",
    "gmm_component_support_median",
    "gmm_component_support_mean",
    "gmm_low_support_component_ratio",
    "gmm_max_resp_mean",
    "gmm_resp_entropy_mean",
    "gmm_noise_auc",
    "gmm_support_bin_lt2_ncr",
    "gmm_support_bin_lt2_cdr",
    "gmm_support_bin_2-3_ncr",
    "gmm_support_bin_2-3_cdr",
    "gmm_support_bin_4-7_ncr",
    "gmm_support_bin_4-7_cdr",
    "gmm_support_bin_8+_ncr",
    "gmm_support_bin_8+_cdr",
]


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _gb_value(model_conf, key, default):
    gb_conf = getattr(model_conf, "gb", {})
    if isinstance(gb_conf, dict):
        value = gb_conf.get(key, default)
    else:
        value = getattr(gb_conf, key, default)
    return default if value is None else value


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _resolve_embedding_path(path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _as_index_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def aggregate_diagnostic_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)

    out = []
    for (dataset, method), group in sorted(groups.items()):
        item = {
            "dataset": dataset,
            "method": method,
            "n": int(len(group)),
        }
        for metric in METRICS:
            values = [_as_float(row, metric) for row in group if row.get(metric, "") not in ("", None)]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def paired_delta_rows(
    rows: list[dict],
    reference_method: str = BASE_METHOD,
    candidate_method: str = GMM_METHOD,
) -> list[dict]:
    by_key = {
        (str(row["dataset"]), int(row["seed"]), str(row["method"])): row
        for row in rows
    }
    keys = sorted(
        (dataset, seed)
        for dataset, seed, method in by_key
        if method == candidate_method and (dataset, seed, reference_method) in by_key
    )
    out = []
    for dataset, seed in keys:
        reference = by_key[(dataset, seed, reference_method)]
        candidate = by_key[(dataset, seed, candidate_method)]
        item = {
            "dataset": dataset,
            "seed": int(seed),
            "comparison": f"{candidate_method}_minus_{reference_method}",
            "candidate_method": candidate_method,
            "reference_method": reference_method,
        }
        for metric in METRICS:
            if metric in candidate or metric in reference:
                item[f"{metric}_delta"] = _as_float(candidate, metric) - _as_float(reference, metric)
        out.append(item)
    return out


def evaluate_diagnostic_gate(
    rows: list[dict],
    min_passing_datasets: int = 2,
    min_positive_seeds: int = 4,
) -> tuple[bool, list[dict]]:
    gmm_vs_base = paired_delta_rows(rows, BASE_METHOD, GMM_METHOD)
    gmm_vs_shuffled = paired_delta_rows(rows, GMM_SHUFFLED_METHOD, GMM_METHOD)
    rows_by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_dataset.setdefault(str(row["dataset"]), []).append(row)

    decisions = []
    for dataset in sorted(rows_by_dataset):
        dataset_deltas = [row for row in gmm_vs_base if row["dataset"] == dataset]
        shuffled_deltas = [row for row in gmm_vs_shuffled if row["dataset"] == dataset]
        gmm_rows = [
            row for row in rows_by_dataset[dataset]
            if str(row["method"]) == GMM_METHOD
        ]
        if not dataset_deltas:
            decisions.append({"dataset": dataset, "passed": False, "reason": "missing paired baseline rows"})
            continue

        mass_delta_mean = float(np.mean([_as_float(row, "soft_clean_mass_delta_delta") for row in dataset_deltas]))
        cdr_delta_mean = float(np.mean([_as_float(row, "soft_pseudo_cdr_delta") for row in dataset_deltas]))
        ncr_delta_mean = float(np.mean([_as_float(row, "soft_pseudo_ncr_delta") for row in dataset_deltas]))
        positive_seed_count = sum(
            1
            for row in dataset_deltas
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
        if shuffled_deltas:
            shuffled_mass_delta_mean = float(np.mean([_as_float(row, "soft_clean_mass_delta_delta") for row in shuffled_deltas]))
            shuffled_cdr_delta_mean = float(np.mean([_as_float(row, "soft_pseudo_cdr_delta") for row in shuffled_deltas]))
            beats_shuffled = shuffled_mass_delta_mean > 0.0 and shuffled_cdr_delta_mean < 0.0
        else:
            shuffled_mass_delta_mean = 0.0
            shuffled_cdr_delta_mean = 0.0
            beats_shuffled = False
        converged_ratio = (
            float(np.mean([_as_float(row, "gmm_converged") for row in gmm_rows]))
            if gmm_rows
            else 0.0
        )
        low_support_mean = (
            float(np.mean([_as_float(row, "gmm_low_support_component_ratio", 1.0) for row in gmm_rows]))
            if gmm_rows
            else 1.0
        )
        fallback_mean = (
            float(np.mean([_as_float(row, "gmm_fallback") for row in gmm_rows]))
            if gmm_rows
            else 1.0
        )
        converged_ok = converged_ratio >= 0.95
        support_ok = low_support_mean <= 0.50
        passed = (
            diagnostic_gain
            and positive_seed_count >= int(min_positive_seeds)
            and beats_shuffled
            and converged_ok
            and support_ok
            and fallback_mean == 0.0
        )
        decisions.append(
            {
                "dataset": dataset,
                "passed": bool(passed),
                "mass_delta_mean": mass_delta_mean,
                "cdr_delta_mean": cdr_delta_mean,
                "ncr_delta_mean": ncr_delta_mean,
                "positive_seed_count": int(positive_seed_count),
                "beats_shuffled": bool(beats_shuffled),
                "gmm_minus_shuffled_mass_delta_mean": shuffled_mass_delta_mean,
                "gmm_minus_shuffled_cdr_delta_mean": shuffled_cdr_delta_mean,
                "converged_ratio": converged_ratio,
                "converged_ok": bool(converged_ok),
                "low_support_component_ratio_mean": low_support_mean,
                "support_ok": bool(support_ok),
                "fallback_mean": fallback_mean,
            }
        )

    gate_passed = sum(1 for row in decisions if row.get("passed")) >= int(min_passing_datasets)
    return bool(gate_passed), decisions


def diagnose_one(args, dataset_name: str, method: str, seed: int) -> dict:
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
        max_balls=_gb_value(model_conf, "max_balls", 64),
        min_ball_size=_gb_value(model_conf, "min_ball_size", 32),
        split_improvement=_gb_value(model_conf, "split_improvement", 0.05),
        min_train_per_ball=_gb_value(model_conf, "min_train_per_ball", 2),
        alpha_baseline=_gb_value(model_conf, "alpha_baseline", "multiclass"),
        target_mode=_gb_value(model_conf, "target_mode", "distribution"),
        softdist_weighting=_gb_value(model_conf, "softdist_weighting", "uniform"),
        ellipsoid_ridge_scale=_gb_value(model_conf, "ellipsoid_ridge_scale", 0.05),
        ellipsoid_weight_floor=_gb_value(model_conf, "ellipsoid_weight_floor", 0.1),
        gmm_n_components=_gb_value(model_conf, "gmm_n_components", None),
        gmm_covariance_type=_gb_value(model_conf, "gmm_covariance_type", "diag"),
        gmm_reg_covar=_gb_value(model_conf, "gmm_reg_covar", 1e-5),
        gmm_max_iter=_gb_value(model_conf, "gmm_max_iter", 100),
        gmm_n_init=_gb_value(model_conf, "gmm_n_init", 1),
        gmm_prior=_gb_value(model_conf, "gmm_prior", "jeffreys"),
        gmm_geometry_confidence=_gb_value(model_conf, "gmm_geometry_confidence", "max_resp"),
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
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "modified_count": int(modified.size),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        **result.diagnostics,
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


def write_report(path: Path, gate_passed: bool, decisions: list[dict], aggregate: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GMM-RespSoftDist 离线诊断报告",
        "",
        f"Gate decision: {'PROCEED' if gate_passed else 'STOP'}",
        "",
        "## 门槛判断",
        "",
    ]
    for row in decisions:
        lines.append(
            "- {dataset}: {decision}; clean_mass_delta={mass:.4f}; cdr_delta={cdr:.4f}; "
            "ncr_delta={ncr:.4f}; positive_seeds={seeds}; beats_shuffled={beats}; "
            "converged={converged:.3f}; low_support={low_support:.3f}; fallback={fallback:.3f}".format(
                dataset=row.get("dataset"),
                decision="PASS" if row.get("passed") else "FAIL",
                mass=float(row.get("mass_delta_mean", 0.0)),
                cdr=float(row.get("cdr_delta_mean", 0.0)),
                ncr=float(row.get("ncr_delta_mean", 0.0)),
                seeds=int(row.get("positive_seed_count", 0)),
                beats=row.get("beats_shuffled", False),
                converged=float(row.get("converged_ratio", 0.0)),
                low_support=float(row.get("low_support_component_ratio_mean", 0.0)),
                fallback=float(row.get("fallback_mean", 0.0)),
            )
        )
    lines.extend(["", "## 聚合结果", ""])
    for row in aggregate:
        lines.append(
            "- {dataset} / {method}: n={n}, clean_mass={mass:.4f}, cdr={cdr:.4f}, "
            "ncr={ncr:.4f}, gmm_auc={auc:.4f}".format(
                dataset=row.get("dataset"),
                method=row.get("method"),
                n=int(row.get("n", 0)),
                mass=float(row.get("soft_clean_mass_delta_mean", 0.0)),
                cdr=float(row.get("soft_pseudo_cdr_mean", 0.0)),
                ncr=float(row.get("soft_pseudo_ncr_mean", 0.0)),
                auc=float(row.get("gmm_noise_auc_mean", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Run static GMM-RespSoftDist target diagnostics.")
    parser.add_argument("--datasets", default="cora,citeseer,coauthorcs")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "gmm_resp_softdist_20260626"))
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_rows = []
    for dataset in _parse_str_list(args.datasets):
        for method in _parse_str_list(args.methods):
            for seed in _parse_int_list(args.seeds):
                print(f"GMM_RESP_DIAG dataset={dataset} method={method} seed={seed}", flush=True)
                row = diagnose_one(args, dataset, method, seed)
                raw_rows.append(row)
                write_csv(output_dir / "diagnostic_raw.csv", raw_rows)

    aggregate = aggregate_diagnostic_rows(raw_rows)
    paired = paired_delta_rows(raw_rows, BASE_METHOD, GMM_METHOD)
    paired.extend(paired_delta_rows(raw_rows, GMM_SHUFFLED_METHOD, GMM_METHOD))
    gate_passed, decisions = evaluate_diagnostic_gate(raw_rows)
    write_csv(output_dir / "diagnostic_aggregate.csv", aggregate)
    write_csv(output_dir / "paired_deltas.csv", paired)
    write_csv(output_dir / "diagnostic_gate.csv", decisions)
    write_report(output_dir / "diagnostic_report.md", gate_passed, decisions, aggregate)
    print(f"GMM_RESP_DIAG decision={'PROCEED' if gate_passed else 'STOP'}")
    print(f"GMM_RESP_DIAG output_dir={output_dir}")


if __name__ == "__main__":
    main()
