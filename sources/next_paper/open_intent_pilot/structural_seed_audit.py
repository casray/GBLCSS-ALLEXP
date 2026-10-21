from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REQUIRED_NPZ_KEYS = {
    "train_features",
    "train_labels",
    "train_pred",
    "test_features",
    "test_open_labels",
    "test_logits",
}


def validate_config(config: dict) -> None:
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("config must define five distinct structural seeds")


def summarize_metric(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "positive_count": int(np.sum(values > 0.0)),
    }


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    observed = abs(float(deltas.mean()))
    means = []
    for mask in range(1 << deltas.size):
        signs = np.asarray(
            [1.0 if mask & (1 << idx) else -1.0 for idx in range(deltas.size)],
            dtype=np.float64,
        )
        means.append(abs(float(np.mean(signs * deltas))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    replicates: int,
    seed: int,
) -> list[float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        raise ValueError("deltas must not be empty")
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(int(seed))
    means = rng.choice(
        deltas,
        size=(int(replicates), deltas.size),
        replace=True,
    ).mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def prepare_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def summarize_dataset(
    rows: list[dict],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    if not rows:
        raise ValueError("rows must not be empty")
    dataset_names = {str(row["dataset"]) for row in rows}
    if len(dataset_names) != 1:
        raise ValueError("rows must contain exactly one dataset")
    result = {
        "dataset": next(iter(dataset_names)),
        "n_seeds": len(rows),
        "bootstrap_seed": int(bootstrap_seed),
    }
    metric_keys = [
        "full_real",
        "full_kmeans",
        "full_shuffle",
        "full_delta",
        "all_safe_real",
        "all_safe_kmeans",
        "all_safe_shuffle",
        "all_safe_delta",
    ]
    for key in metric_keys:
        summary = summarize_metric(rows, key)
        result[f"{key}_mean"] = summary["mean"]
        result[f"{key}_sample_std"] = summary["sample_std"]
        if key.endswith("_delta"):
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            result[f"{key}_positive_count"] = summary["positive_count"]
            result[f"{key}_bootstrap_ci95"] = paired_bootstrap_mean_ci(
                values,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            result[f"{key}_sign_flip_p"] = exact_sign_flip_p(values)
    result["gate_pass"] = bool(
        result["full_delta_mean"] > 0.0
        and result["full_delta_positive_count"] >= 4
        and result["all_safe_delta_mean"] > 0.0
        and result["all_safe_delta_positive_count"] >= 4
        and result["full_real_mean"] > result["full_shuffle_mean"]
        and result["all_safe_real_mean"] > result["all_safe_shuffle_mean"]
    )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    output_dir: Path,
    raw_rows: list[dict],
    summaries: list[dict],
    statistics: dict,
    resolved_config: dict,
    log_lines: list[str],
) -> None:
    output_dir = Path(output_dir)
    _write_csv(output_dir / "raw_seed_metrics.csv", raw_rows)
    csv_summaries = []
    for summary in summaries:
        csv_summaries.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                for key, value in summary.items()
            }
        )
    _write_csv(output_dir / "summary.csv", csv_summaries)
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.resolved.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _load_source_api(source_workspace: Path) -> dict:
    source_workspace = Path(source_workspace).resolve()
    project_dir = source_workspace / "开放意图粒球新方向0630"
    code_dir = project_dir / "code"
    external_dir = source_workspace / "external" / "NoisyGL"
    if not code_dir.is_dir():
        raise FileNotFoundError(f"open-intent code directory not found: {code_dir}")
    for path in (code_dir, external_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from oi_gb.gb_scores import fit_control_model_like, fit_gb_model, score_gb_model
    from oi_gb.metrics import binary_ood_metrics
    from oi_gb.posthoc import compute_core_posthoc_scores

    return {
        "fit_control_model_like": fit_control_model_like,
        "fit_gb_model": fit_gb_model,
        "score_gb_model": score_gb_model,
        "binary_ood_metrics": binary_ood_metrics,
        "compute_core_posthoc_scores": compute_core_posthoc_scores,
    }


def _zmean(component_scores: dict[str, np.ndarray]) -> np.ndarray:
    normalized = []
    for key in ("s_dist", "s_margin"):
        values = np.asarray(component_scores[key], dtype=np.float64)
        normalized.append((values - values.mean()) / max(float(values.std()), 1e-12))
    return np.mean(np.vstack(normalized), axis=0)


def _auroc(binary_ood_metrics, labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(binary_ood_metrics(labels, scores)["auroc"])


def evaluate_dataset(
    dataset: str,
    npz_path: Path,
    source_workspace: Path,
    config: dict,
) -> list[dict]:
    validate_config(config)
    api = _load_source_api(source_workspace)
    npz_path = Path(npz_path).resolve()
    with np.load(npz_path) as payload:
        missing = sorted(REQUIRED_NPZ_KEYS.difference(payload.files))
        if missing:
            raise KeyError(f"NPZ missing required keys: {missing}")
        train_features = np.asarray(payload["train_features"])
        train_labels = np.asarray(payload["train_labels"])
        train_pred = np.asarray(payload["train_pred"])
        test_features = np.asarray(payload["test_features"])
        test_open_labels = np.asarray(payload["test_open_labels"])
        test_logits = np.asarray(payload["test_logits"])

    y_unknown = (test_open_labels == -1).astype(np.int64)
    base_scores = api["compute_core_posthoc_scores"](
        train_features,
        train_labels,
        test_features,
        test_logits,
        random_state=0,
    )
    safe_masks = [
        base_scores[name] <= np.quantile(base_scores[name], 0.5)
        for name in ("MSP", "Prototype", "kNN")
    ]
    all_safe_mask = safe_masks[0] & safe_masks[1] & safe_masks[2]
    gb_config = config["gb"]
    rows = []
    for structural_seed in [int(seed) for seed in config["seeds"]]:
        real_model = api["fit_gb_model"](
            train_features,
            train_labels,
            train_pseudo_labels=train_pred,
            max_balls_per_class=int(gb_config["max_balls_per_class"]),
            min_ball_size=int(gb_config["min_ball_size"]),
            split_improvement=float(gb_config["split_improvement"]),
            random_state=structural_seed,
        )
        kmeans_seed = int(config["kmeans_seed_offset"]) + structural_seed
        shuffle_seed = int(config["shuffle_seed_offset"]) + structural_seed
        kmeans_model = api["fit_control_model_like"](
            real_model,
            "kmeans",
            train_pseudo_labels=train_pred,
            random_state=kmeans_seed,
        )
        shuffle_model = api["fit_control_model_like"](
            real_model,
            "random",
            train_pseudo_labels=train_pred,
            random_state=shuffle_seed,
        )
        real_scores = _zmean(api["score_gb_model"](real_model, test_features))
        kmeans_scores = _zmean(api["score_gb_model"](kmeans_model, test_features))
        shuffle_scores = _zmean(api["score_gb_model"](shuffle_model, test_features))
        full_real = _auroc(api["binary_ood_metrics"], y_unknown, real_scores)
        full_kmeans = _auroc(api["binary_ood_metrics"], y_unknown, kmeans_scores)
        full_shuffle = _auroc(api["binary_ood_metrics"], y_unknown, shuffle_scores)
        all_safe_labels = y_unknown[all_safe_mask]
        all_safe_real = _auroc(
            api["binary_ood_metrics"],
            all_safe_labels,
            real_scores[all_safe_mask],
        )
        all_safe_kmeans = _auroc(
            api["binary_ood_metrics"],
            all_safe_labels,
            kmeans_scores[all_safe_mask],
        )
        all_safe_shuffle = _auroc(
            api["binary_ood_metrics"],
            all_safe_labels,
            shuffle_scores[all_safe_mask],
        )
        rows.append(
            {
                "dataset": str(dataset),
                "structural_seed": structural_seed,
                "kmeans_seed": kmeans_seed,
                "shuffle_seed": shuffle_seed,
                "n_balls": int(real_model.labels.size),
                "n_test": int(y_unknown.size),
                "n_unknown": int(y_unknown.sum()),
                "n_all_safe": int(all_safe_mask.sum()),
                "n_all_safe_unknown": int(all_safe_labels.sum()),
                "full_real": full_real,
                "full_kmeans": full_kmeans,
                "full_shuffle": full_shuffle,
                "full_delta": full_real - full_kmeans,
                "all_safe_real": all_safe_real,
                "all_safe_kmeans": all_safe_kmeans,
                "all_safe_shuffle": all_safe_shuffle,
                "all_safe_delta": all_safe_real - all_safe_kmeans,
            }
        )
    return rows


def _file_fingerprint(path: Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest().upper(),
    }


def _resolve_npz_path(source_workspace: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = Path(source_workspace) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured NPZ not found: {path}")
    return path


def run_audit(
    source_workspace: Path,
    config_path: Path,
    output_dir: Path,
) -> dict:
    source_workspace = Path(source_workspace).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not config.get("datasets"):
        raise ValueError("config must define at least one dataset")
    prepare_output_dir(output_dir)
    resolved_config = json.loads(json.dumps(config))
    resolved_config["source_workspace"] = str(source_workspace)
    resolved_config["config_path"] = str(config_path)
    resolved_config["datasets"] = {}
    inputs = {}
    raw_rows = []
    summaries = []
    log_lines = [
        f"audit_date={config.get('audit_date', '')}",
        "randomness_scope=frozen-representation structural seeds",
        f"source_workspace={source_workspace}",
        f"seeds={','.join(str(seed) for seed in config['seeds'])}",
    ]
    try:
        bootstrap_config = config["bootstrap"]
        for dataset_index, (dataset, dataset_config) in enumerate(config["datasets"].items()):
            npz_path = _resolve_npz_path(source_workspace, dataset_config["npz"])
            fingerprint = _file_fingerprint(npz_path)
            inputs[dataset] = fingerprint
            resolved_config["datasets"][dataset] = {
                **dataset_config,
                **fingerprint,
            }
            dataset_rows = evaluate_dataset(
                dataset=dataset,
                npz_path=npz_path,
                source_workspace=source_workspace,
                config=config,
            )
            raw_rows.extend(dataset_rows)
            summary = summarize_dataset(
                dataset_rows,
                bootstrap_replicates=int(bootstrap_config["replicates"]),
                bootstrap_seed=int(bootstrap_config["seed"]) + dataset_index,
            )
            summaries.append(summary)
            for row in dataset_rows:
                log_lines.append(
                    "dataset={dataset} structural_seed={seed} balls={balls} "
                    "full_delta={full:+.8f} all_safe_delta={safe:+.8f}".format(
                        dataset=dataset,
                        seed=row["structural_seed"],
                        balls=row["n_balls"],
                        full=row["full_delta"],
                        safe=row["all_safe_delta"],
                    )
                )
            log_lines.append(
                f"dataset={dataset} gate={'PASS' if summary['gate_pass'] else 'FAIL'}"
            )
        gate_pass = bool(summaries and all(summary["gate_pass"] for summary in summaries))
        statistics = {
            "audit_date": config.get("audit_date"),
            "randomness_scope": "frozen-representation structural seeds",
            "model_seed_warning": (
                "Encoder/model and known-class split remain seed0; structural-seed "
                "mean/std must not be reported as model-seed uncertainty."
            ),
            "inputs": inputs,
            "bootstrap": {
                "type": "paired bootstrap over five structural-seed deltas",
                "replicates": int(bootstrap_config["replicates"]),
                "seed_base": int(bootstrap_config["seed"]),
            },
            "datasets": {summary["dataset"]: summary for summary in summaries},
            "gate_pass": gate_pass,
            "decision": "GO" if gate_pass else "NO-GO",
        }
        log_lines.append(f"overall_gate={'PASS' if gate_pass else 'FAIL'}")
        write_artifacts(
            output_dir=output_dir,
            raw_rows=raw_rows,
            summaries=summaries,
            statistics=statistics,
            resolved_config=resolved_config,
            log_lines=log_lines,
        )
        return statistics
    except Exception as exc:
        log_lines.append(f"ERROR {type(exc).__name__}: {exc}")
        (output_dir / "run.log").write_text(
            "\n".join(log_lines) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit five structural seeds on frozen open-intent representations."
    )
    parser.add_argument("--source-workspace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    statistics = run_audit(
        source_workspace=args.source_workspace,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
