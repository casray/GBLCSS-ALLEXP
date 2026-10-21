from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mlp34_noise_rate_diagnostic import evaluate_rhohat_gate
from scripts.run_ssl_swap_downstream import _load_dataset
from scripts.run_standard_downstream import _split_fingerprint
from utils.tools import setup_seed


TRACE_METHODS = {
    "small_loss",
    "lcss_soft_small_loss",
    "lcss_soft_small_loss_shuffled",
}
TRACE_TO_ANCHOR = {
    "small_loss": "mlp_small_loss",
    "lcss_soft_small_loss": "frozengrace_gb_mlp_softdist_beta05_small_loss",
    "lcss_soft_small_loss_shuffled": "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled",
}


def jaccard_similarity(first, second) -> float:
    first = np.asarray(first, dtype=np.bool_)
    second = np.asarray(second, dtype=np.bool_)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def validate_trace_rows(rows, expected_datasets, expected_settings) -> None:
    frame = pd.DataFrame(rows)
    required = {
        "dataset", "method", "seed", "noise_type", "noise_rate", "noise_scope",
        "split_type", "train_percent", "val_percent", "test_percent", "split_hash",
        "noise_hash", "modified_val_count", "modified_test_count", "status",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing trace columns: {sorted(missing_columns)}")
    keys = ["dataset", "method", "seed", "noise_type", "noise_rate"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate trace run rows")
    if not (frame["status"] == "ok").all():
        raise ValueError("trace contains non-ok rows")
    if not (frame["noise_scope"] == "train").all():
        raise ValueError("trace contains non-train noise")
    if not (frame["split_type"] == "percent").all():
        raise ValueError("trace contains a non-percent split")
    for column, expected in [
        ("train_percent", 0.2), ("val_percent", 0.1), ("test_percent", 0.7)
    ]:
        if not np.allclose(frame[column].astype(float), expected):
            raise ValueError(f"trace contains the wrong {column}")
    if int(frame["modified_val_count"].sum()) or int(frame["modified_test_count"].sum()):
        raise ValueError("trace contains val/test label pollution")

    actual_keys = set(
        zip(
            frame["dataset"].astype(str),
            frame["noise_type"].astype(str),
            frame["noise_rate"].astype(float),
            frame["seed"].astype(int),
            frame["method"].astype(str),
        )
    )
    expected_keys = {
        (dataset, noise_type, float(noise_rate), 0, method)
        for dataset in expected_datasets
        for noise_type, noise_rate in expected_settings
        for method in TRACE_METHODS
    }
    if actual_keys != expected_keys:
        raise ValueError(
            f"trace completeness mismatch: missing={len(expected_keys-actual_keys)} "
            f"extra={len(actual_keys-expected_keys)}"
        )
    grouped = frame.groupby(["dataset", "noise_type", "noise_rate", "seed"])
    if (grouped["split_hash"].nunique() != 1).any() or (grouped["noise_hash"].nunique() != 1).any():
        raise ValueError("trace split/noise hash mismatch")


def _audit_trace_regression(trace, mlp31, mlp33, mlp29):
    audit_rows = []
    for row in trace.itertuples(index=False):
        anchor_method = TRACE_TO_ANCHOR[row.method]
        sources = [mlp31, mlp29] if row.noise_type == "uniform" else [mlp33]
        anchor = None
        for source in sources:
            if source.empty or "dataset" not in source.columns:
                continue
            candidates = source[
                (source["dataset"] == row.dataset)
                & (source["method"] == anchor_method)
                & (source["noise_type"] == row.noise_type)
                & np.isclose(source["noise_rate"].astype(float), float(row.noise_rate))
                & (source["seed"].astype(int) == int(row.seed))
            ]
            if len(candidates) > 1:
                raise ValueError("duplicate trace regression anchor")
            if len(candidates) == 1:
                anchor = candidates.iloc[0]
                break
        if anchor is None:
            raise ValueError(
                f"missing trace regression anchor for {row.dataset}/{row.method}/"
                f"{row.noise_type}:{row.noise_rate}"
            )
        metric_delta = float(row.test) - float(anchor["test"])
        protocol_match = bool(
            row.split_hash == anchor["split_hash"]
            and row.noise_hash == anchor["noise_hash"]
        )
        trace_selected = getattr(row, "robust_selected_count", float("nan"))
        anchor_selected = anchor.get("robust_selected_count", float("nan"))
        selected_count_match = bool(
            (pd.isna(trace_selected) and pd.isna(anchor_selected))
            or float(trace_selected) == float(anchor_selected)
        )
        exact = bool(abs(metric_delta) <= 1e-12 and protocol_match and selected_count_match)
        within_tolerance = bool(
            abs(metric_delta) <= 0.03 and protocol_match and selected_count_match
        )
        audit_rows.append(
            {
                "dataset": row.dataset,
                "noise_type": row.noise_type,
                "noise_rate": float(row.noise_rate),
                "method": row.method,
                "trace_test": float(row.test),
                "anchor_test": float(anchor["test"]),
                "metric_delta": metric_delta,
                "exact_match": exact,
                "within_rocm_tolerance": within_tolerance,
                "protocol_match": protocol_match,
                "selected_count_match": selected_count_match,
            }
        )
    if not all(row["within_rocm_tolerance"] for row in audit_rows):
        raise ValueError("trace regression mismatch")
    return audit_rows


def spearman_with_lodo(rows, x_key, y_key) -> dict:
    frame = pd.DataFrame(rows)[["dataset", x_key, y_key]].dropna()
    rho = float(spearmanr(frame[x_key], frame[y_key]).statistic)
    lodo = []
    for dataset in sorted(frame["dataset"].unique()):
        subset = frame[frame["dataset"] != dataset]
        if len(subset) >= 3:
            lodo.append(float(spearmanr(subset[x_key], subset[y_key]).statistic))
    sign = np.sign(rho)
    consistent = bool(lodo and sign != 0 and all(np.sign(value) == sign for value in lodo))
    return {
        "x": x_key,
        "y": y_key,
        "n": int(len(frame)),
        "rho": rho,
        "lodo_min": float(min(lodo)) if lodo else float("nan"),
        "lodo_max": float(max(lodo)) if lodo else float("nan"),
        "lodo_direction_consistent": consistent,
    }


def _saturation_direction_support(predictor, rho) -> bool:
    expected_sign = {
        "brier_gain": -1,
        "ncr": -1,
        "cdr": 1,
        "mean_margin": -1,
    }
    if predictor not in expected_sign or float(rho) == 0.0:
        return False
    return int(np.sign(float(rho))) == expected_sign[predictor]


def _population_std(values) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _phase_a_summary(args, output_dir: Path):
    pair_jocor = pd.read_csv(args.phase_a_pair)
    instance_jocor = pd.read_csv(args.phase_a_instance)
    mlp33 = pd.read_csv(args.mlp33_raw)
    mlp31 = pd.read_csv(args.mlp31_raw)
    u06_jocor = pd.read_csv(args.u06_jocor)
    combo_method = "frozengrace_gb_mlp_softdist_beta05_small_loss"
    rows = []
    seed_rows = []
    for noise_type, rate, jocor_source, combo_source in [
        ("pair", 0.4, pair_jocor, mlp33),
        ("instance", 0.4, instance_jocor, mlp33),
        ("uniform", 0.6, u06_jocor, mlp31),
    ]:
        jocor = jocor_source[
            (jocor_source["dataset"] == "pubmed")
            & (jocor_source["method"] == "mlp_jocor")
            & (jocor_source["noise_type"] == noise_type)
            & np.isclose(jocor_source["noise_rate"].astype(float), rate)
        ].copy()
        combo = combo_source[
            (combo_source["dataset"] == "pubmed")
            & (combo_source["method"] == combo_method)
            & (combo_source["noise_type"] == noise_type)
            & np.isclose(combo_source["noise_rate"].astype(float), rate)
        ].copy()
        if len(jocor) != 5 or len(combo) != 5:
            raise ValueError(f"Phase A requires five seeds for {noise_type}:{rate}")
        merged = combo.merge(jocor, on="seed", suffixes=("_combo", "_jocor"), validate="one_to_one")
        if not (
            (merged["split_hash_combo"] == merged["split_hash_jocor"]).all()
            and (merged["noise_hash_combo"] == merged["noise_hash_jocor"]).all()
        ):
            raise ValueError(f"Phase A hash mismatch for {noise_type}:{rate}")
        merged["delta"] = merged["test_combo"] - merged["test_jocor"]
        for _, member in merged.iterrows():
            seed_rows.append(
                {
                    "noise_type": noise_type,
                    "noise_rate": rate,
                    "seed": int(member["seed"]),
                    "combo": float(member["test_combo"]),
                    "jocor": float(member["test_jocor"]),
                    "delta": float(member["delta"]),
                }
            )
        mean_delta = float(merged["delta"].mean())
        positive = int((merged["delta"] > 0).sum())
        rows.append(
            {
                "noise_type": noise_type,
                "noise_rate": rate,
                "combo_mean": float(merged["test_combo"].mean()),
                "combo_std": _population_std(merged["test_combo"]),
                "jocor_mean": float(merged["test_jocor"].mean()),
                "jocor_std": _population_std(merged["test_jocor"]),
                "mean_delta": mean_delta,
                "positive_seeds": positive,
                "passes_one_pp_four_of_five": bool(mean_delta >= 0.01 and positive >= 4),
            }
        )
    strict = all(row["passes_one_pp_four_of_five"] for row in rows)
    conditional = bool(
        sum(row["passes_one_pp_four_of_five"] for row in rows) >= 2
        and min(row["mean_delta"] for row in rows) >= -0.005
    )
    pd.DataFrame(rows).to_csv(output_dir / "mlp34_phase_a_pubmed_jocor.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_dir / "mlp34_phase_a_pubmed_jocor_seed_deltas.csv", index=False)
    return rows, {"strong_boundary_elimination": strict, "conditional_boundary_elimination": conditional}


def _load_clean_train(dataset, split_hash):
    setup_seed(0)
    data = _load_dataset(dataset, "cpu", "percent", 0.2, 0.1, 0.7)
    observed_hash = _split_fingerprint(data.train_masks, data.val_masks, data.test_masks)
    if observed_hash != split_hash:
        raise ValueError(f"clean-label reload split hash mismatch for {dataset}")
    labels = data.labels.detach().cpu().long().numpy()
    return labels


def _trace_path(value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _target_statistics(targets, noisy, clean):
    n_classes = int(targets.shape[1])
    clean_onehot = np.eye(n_classes, dtype=np.float32)[clean]
    noisy_onehot = np.eye(n_classes, dtype=np.float32)[noisy]
    prediction = targets.argmax(axis=1)
    noisy_nodes = noisy != clean
    clean_nodes = ~noisy_nodes
    sorted_target = np.sort(targets, axis=1)
    margin = sorted_target[:, -1] - sorted_target[:, -2]
    return {
        "target_accuracy": float(np.mean(prediction == clean)),
        "target_brier": float(np.mean(np.sum((targets - clean_onehot) ** 2, axis=1))),
        "noisy_brier": float(np.mean(np.sum((noisy_onehot - clean_onehot) ** 2, axis=1))),
        "brier_gain": float(
            np.mean(np.sum((noisy_onehot - clean_onehot) ** 2, axis=1))
            - np.mean(np.sum((targets - clean_onehot) ** 2, axis=1))
        ),
        "ncr": float(np.mean(prediction[noisy_nodes] == clean[noisy_nodes])) if noisy_nodes.any() else 0.0,
        "cdr": float(np.mean(prediction[clean_nodes] != clean[clean_nodes])) if clean_nodes.any() else 0.0,
        "mean_margin": float(np.mean(margin)),
        "prediction": prediction,
        "correctly_repaired": noisy_nodes & (prediction == clean) & (prediction != noisy),
    }


def _trace_metrics(trace_frame, output_dir: Path):
    metric_rows = []
    pair_rows = []
    target_rows = []
    for (dataset, noise_type, noise_rate), group in trace_frame.groupby(
        ["dataset", "noise_type", "noise_rate"]
    ):
        by_method = {row.method: row for row in group.itertuples(index=False)}
        loaded = {method: np.load(_trace_path(row.trace_output_path)) for method, row in by_method.items()}
        metadata = {
            method: json.loads(data["metadata_json"].item()) for method, data in loaded.items()
        }
        for method, row in by_method.items():
            if metadata[method]["split_hash"] != row.split_hash or metadata[method]["noise_hash"] != row.noise_hash:
                raise ValueError("NPZ metadata hash mismatch")
        train_idx = loaded["lcss_soft_small_loss"]["train_indices"]
        noisy = loaded["lcss_soft_small_loss"]["noisy_train_labels"]
        clean_all = _load_clean_train(dataset, by_method["lcss_soft_small_loss"].split_hash)
        clean = clean_all[train_idx]
        real_stats = _target_statistics(
            loaded["lcss_soft_small_loss"]["train_targets"], noisy, clean
        )
        target_stats = {}
        for method, data in loaded.items():
            if not np.array_equal(train_idx, data["train_indices"]):
                raise ValueError("trace train-index mismatch")
            if not np.array_equal(noisy, data["noisy_train_labels"]):
                raise ValueError("trace noisy-label mismatch")
            target_stats[method] = _target_statistics(data["train_targets"], noisy, clean)
            target_rows.append(
                {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": float(noise_rate),
                    "method": method,
                    **{key: value for key, value in target_stats[method].items() if key not in {"prediction", "correctly_repaired"}},
                }
            )

        epoch_map = {
            method: {int(epoch): mask for epoch, mask in zip(data["epochs"], data["selected_masks"])}
            for method, data in loaded.items()
        }
        common_epochs = sorted(set.intersection(*(set(value) for value in epoch_map.values())))
        if common_epochs != [1, 10, 50, 100, 200]:
            raise ValueError(f"incomplete trace epochs for {dataset}/{noise_type}")
        previous = {method: None for method in TRACE_METHODS}
        for epoch in common_epochs:
            local_selected = {}
            for method in TRACE_METHODS:
                global_mask = epoch_map[method][epoch].astype(bool)
                selected = global_mask[train_idx]
                local_selected[method] = selected
                stats = target_stats[method]
                selected_accuracy = (
                    float(np.mean(stats["prediction"][selected] == clean[selected]))
                    if selected.any() else float("nan")
                )
                repaired = real_stats["correctly_repaired"]
                repaired_retention = (
                    float(np.mean(selected[repaired])) if repaired.any() else float("nan")
                )
                turnover = (
                    1.0 - jaccard_similarity(previous[method], global_mask)
                    if previous[method] is not None else float("nan")
                )
                previous[method] = global_mask
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": float(noise_rate),
                        "epoch": int(epoch),
                        "method": method,
                        "selected_count": int(selected.sum()),
                        "selected_target_accuracy": selected_accuracy,
                        "selected_harmful_target_rate": 1.0 - selected_accuracy,
                        "correctly_repaired_noisy_retention": repaired_retention,
                        "mask_turnover": turnover,
                    }
                )
            hard = local_selected["small_loss"]
            real = local_selected["lcss_soft_small_loss"]
            shuffled = local_selected["lcss_soft_small_loss_shuffled"]
            hard_stats = target_stats["small_loss"]
            shuffled_stats = target_stats["lcss_soft_small_loss_shuffled"]
            def selected_accuracy(stats, selected):
                return float(np.mean(stats["prediction"][selected] == clean[selected]))
            repaired = real_stats["correctly_repaired"]
            hard_retention = float(np.mean(hard[repaired])) if repaired.any() else float("nan")
            real_retention = float(np.mean(real[repaired])) if repaired.any() else float("nan")
            pair_rows.append(
                {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": float(noise_rate),
                    "epoch": int(epoch),
                    "jaccard_small_vs_real": jaccard_similarity(hard, real),
                    "jaccard_real_vs_shuffled": jaccard_similarity(real, shuffled),
                    "selected_target_accuracy_small": selected_accuracy(hard_stats, hard),
                    "selected_target_accuracy_real": selected_accuracy(real_stats, real),
                    "selected_target_accuracy_shuffled": selected_accuracy(shuffled_stats, shuffled),
                    "selected_target_accuracy_gain_vs_small": selected_accuracy(real_stats, real) - selected_accuracy(hard_stats, hard),
                    "selected_target_accuracy_gain_vs_shuffled": selected_accuracy(real_stats, real) - selected_accuracy(shuffled_stats, shuffled),
                    "correctly_repaired_retention_small": hard_retention,
                    "correctly_repaired_retention_real": real_retention,
                    "correctly_repaired_retention_gain": real_retention - hard_retention,
                }
            )
    pd.DataFrame(metric_rows).to_csv(output_dir / "mlp34_selection_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "mlp34_selection_pairwise.csv", index=False)
    pd.DataFrame(target_rows).to_csv(output_dir / "mlp34_target_diagnostics.csv", index=False)
    return metric_rows, pair_rows, target_rows


def _find_lcss_test(dataset, noise_type, noise_rate, mlp31, mlp33, mlp29):
    sources = [mlp31, mlp29] if noise_type == "uniform" else [mlp33]
    for source in sources:
        rows = source[
            (source["dataset"] == dataset)
            & (source["method"] == "frozengrace_gb_mlp_softdist_beta05")
            & (source["noise_type"] == noise_type)
            & np.isclose(source["noise_rate"].astype(float), noise_rate)
            & (source["seed"].astype(int) == 0)
        ]
        if len(rows) > 1:
            raise ValueError(f"duplicate LCSS anchor for {dataset}/{noise_type}:{noise_rate}")
        if len(rows) == 1:
            return float(rows.iloc[0]["test"])
    raise ValueError(f"missing unique LCSS anchor for {dataset}/{noise_type}:{noise_rate}")


def _saturation_analysis(trace_frame, target_rows, args, output_dir):
    target = pd.DataFrame(target_rows)
    target = target[target["method"] == "lcss_soft_small_loss"].copy()
    mlp31 = pd.read_csv(args.mlp31_raw)
    mlp33 = pd.read_csv(args.mlp33_raw)
    mlp29 = pd.read_csv(args.mlp29_raw)
    rows = []
    for row in target.itertuples(index=False):
        combo = trace_frame[
            (trace_frame["dataset"] == row.dataset)
            & (trace_frame["noise_type"] == row.noise_type)
            & np.isclose(trace_frame["noise_rate"].astype(float), row.noise_rate)
            & (trace_frame["method"] == "lcss_soft_small_loss")
        ]
        if len(combo) != 1:
            raise ValueError("missing unique combo trace result")
        lcss_test = _find_lcss_test(
            row.dataset, row.noise_type, row.noise_rate, mlp31, mlp33, mlp29
        )
        rows.append(
            {
                "dataset": row.dataset,
                "noise_type": row.noise_type,
                "noise_rate": float(row.noise_rate),
                "brier_gain": float(row.brier_gain),
                "ncr": float(row.ncr),
                "cdr": float(row.cdr),
                "mean_margin": float(row.mean_margin),
                "combo_minus_lcss": float(combo.iloc[0]["test"] - lcss_test),
            }
        )
    correlations = [
        spearman_with_lodo(rows, predictor, "combo_minus_lcss")
        for predictor in ["brier_gain", "ncr", "cdr", "mean_margin"]
    ]
    for row in correlations:
        row["saturation_direction_support"] = _saturation_direction_support(
            row["x"], row["rho"]
        )
    pd.DataFrame(rows).to_csv(output_dir / "mlp34_saturation_rows.csv", index=False)
    pd.DataFrame(correlations).to_csv(output_dir / "mlp34_saturation_correlations.csv", index=False)
    qualifies = [
        row for row in correlations
        if abs(row["rho"]) >= 0.5
        and row["lodo_direction_consistent"]
        and row["saturation_direction_support"]
    ]
    return rows, correlations, qualifies


def _mechanism_gate(pair_rows):
    final = pd.DataFrame(pair_rows)
    final = final[
        (final["epoch"] == 200)
        & (final["noise_type"].isin(["pair", "instance"]))
    ]
    accuracy_pass = final["selected_target_accuracy_gain_vs_small"] >= 0.03
    retention_pass = final["correctly_repaired_retention_gain"] >= 0.05
    shuffled_pass = final["selected_target_accuracy_gain_vs_shuffled"] > 0.0
    joint = accuracy_pass & retention_pass & shuffled_pass
    return {
        "n_settings": int(len(final)),
        "accuracy_pass_count": int(accuracy_pass.sum()),
        "retention_pass_count": int(retention_pass.sum()),
        "real_vs_shuffled_pass_count": int(shuffled_pass.sum()),
        "joint_pass_count": int(joint.sum()),
        "pass": bool(joint.sum() >= 6),
    }


def _format_percent(value):
    return f"{100.0 * float(value):.2f}%"


def _write_report(
    output_dir,
    phase_a,
    phase_a_gate,
    rho_rows,
    rho_gate,
    mechanism_gate,
    correlations,
    qualifies,
    trace_regression,
):
    lines = [
        "# MLP-34 Strong Baseline, Noise-Rate Estimation, and Selection Mechanism Report",
        "",
        "## Phase A: PubMed JoCoR five-seed comparison",
        "",
        "| Noise | Combination | JoCoR | Paired delta | Positive seeds | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in phase_a:
        lines.append(
            f"| {row['noise_type']} {row['noise_rate']} | "
            f"{_format_percent(row['combo_mean'])}±{_format_percent(row['combo_std'])} | "
            f"{_format_percent(row['jocor_mean'])}±{_format_percent(row['jocor_std'])} | "
            f"{100*row['mean_delta']:+.2f}pp | {row['positive_seeds']}/5 | "
            f"{int(row['passes_one_pp_four_of_five'])} |"
        )
    lines.extend(
        [
            "",
            f"Strong three-noise boundary elimination: {int(phase_a_gate['strong_boundary_elimination'])}.",
            f"Conditional boundary elimination: {int(phase_a_gate['conditional_boundary_elimination'])}.",
            "",
            "## Phase B: noisy-only LOO noise-rate estimate",
            "",
            "| Dataset | Estimate | Absolute error | Coverage | Effective sample size |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    rho_frame = pd.DataFrame(rho_rows)
    for dataset, group in rho_frame.groupby("dataset"):
        lines.append(
            f"| {dataset} | {group['weighted_estimate'].mean():.3f} | "
            f"{group['weighted_abs_error'].mean():.3f} | {group['coverage'].mean():.3f} | "
            f"{group['effective_sample_size'].mean():.1f} |"
        )
    lines.extend(
        [
            "",
            f"B1 gate: {int(rho_gate['pass'])}. B2 was {'run' if rho_gate['pass'] else 'skipped by preregistration'}.",
            "",
            "## Phase C: selection mechanism",
            "",
            f"Trace regression: {sum(row['exact_match'] for row in trace_regression)}/{len(trace_regression)} exact, "
            f"{sum(row['within_rocm_tolerance'] for row in trace_regression)}/{len(trace_regression)} protocol/selection matched within the observed ROCm band; "
            f"maximum absolute metric delta={100*max(abs(row['metric_delta']) for row in trace_regression):.2f}pp.",
            "",
            f"Mechanism gate: {json.dumps(mechanism_gate, ensure_ascii=False)}",
            "",
            "### Saturation correlations",
            "",
            "| Predictor | n | Spearman | LODO range | LODO consistent | Saturation direction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in correlations:
        lines.append(
            f"| {row['x']} | {row['n']} | {row['rho']:.3f} | "
            f"[{row['lodo_min']:.3f}, {row['lodo_max']:.3f}] | "
            f"{int(row['lodo_direction_consistent'])} | "
            f"{int(row['saturation_direction_support'])} |"
        )
    lines.extend(
        [
            "",
            f"Correlations meeting magnitude, LODO, and preregistered semantic direction: {len(qualifies)}.",
            "A single supporting diagnostic is insufficient for a general saturation claim when other diagnostics point in the opposite direction.",
            "",
            "## Decision",
            "",
            "- Phase A determines whether the PubMed claim may be conditional; the strict three-noise claim remains prohibited if any row misses the frozen gate.",
            "- Phase B does not tune a replacement estimator after failure; the known-noise-rate limitation remains.",
            "- Phase C is seed-0, clean-dependent post-hoc diagnosis only and is not five-seed performance evidence.",
        ]
    )
    (output_dir / "mlp34_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize MLP-34 experiments")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--phase_a_pair", required=True)
    parser.add_argument("--phase_a_instance", required=True)
    parser.add_argument("--rhohat", required=True)
    parser.add_argument("--trace_raw", required=True)
    parser.add_argument("--mlp31_raw", required=True)
    parser.add_argument("--mlp33_raw", required=True)
    parser.add_argument("--mlp29_raw", required=True)
    parser.add_argument("--u06_jocor", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_a, phase_a_gate = _phase_a_summary(args, output_dir)

    rho_frame = pd.read_csv(args.rhohat)
    rho_rows = rho_frame.to_dict("records")
    rho_gate = evaluate_rhohat_gate(rho_rows)
    (output_dir / "mlp34_rhohat_gate.json").write_text(
        json.dumps(rho_gate, indent=2), encoding="utf-8"
    )

    trace_frame = pd.read_csv(args.trace_raw)
    expected_datasets = {"cora", "citeseer", "corafull", "amazonpho", "pubmed"}
    expected_settings = {("uniform", 0.6), ("pair", 0.4), ("instance", 0.4)}
    validate_trace_rows(trace_frame.to_dict("records"), expected_datasets, expected_settings)
    trace_regression = _audit_trace_regression(
        trace_frame,
        pd.read_csv(args.mlp31_raw),
        pd.read_csv(args.mlp33_raw),
        pd.read_csv(args.mlp29_raw),
    )
    pd.DataFrame(trace_regression).to_csv(
        output_dir / "mlp34_trace_regression.csv", index=False
    )
    metric_rows, pair_rows, target_rows = _trace_metrics(trace_frame, output_dir)
    mechanism_gate = _mechanism_gate(pair_rows)
    (output_dir / "mlp34_mechanism_gate.json").write_text(
        json.dumps(mechanism_gate, indent=2), encoding="utf-8"
    )
    _, correlations, qualifies = _saturation_analysis(
        trace_frame, target_rows, args, output_dir
    )
    _write_report(
        output_dir,
        phase_a,
        phase_a_gate,
        rho_rows,
        rho_gate,
        mechanism_gate,
        correlations,
        qualifies,
        trace_regression,
    )
    completeness = {
        "phase_a_pair_rows": int(len(pd.read_csv(args.phase_a_pair))),
        "phase_a_instance_rows": int(len(pd.read_csv(args.phase_a_instance))),
        "rhohat_rows": int(len(rho_frame)),
        "trace_rows": int(len(trace_frame)),
        "trace_npz": int(len(list((output_dir / "traces").glob("*.npz")))),
        "trace_all_ok": bool((trace_frame["status"] == "ok").all()),
        "trace_val_test_pollution": int(
            trace_frame["modified_val_count"].sum() + trace_frame["modified_test_count"].sum()
        ),
    }
    (output_dir / "mlp34_completeness.json").write_text(
        json.dumps(completeness, indent=2), encoding="utf-8"
    )
    print(f"WROTE {output_dir / 'mlp34_report.md'}")


if __name__ == "__main__":
    main()
