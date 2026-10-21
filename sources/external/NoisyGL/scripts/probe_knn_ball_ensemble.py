from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import diagnose_label_noise_detection as phase39


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def percentile_rank_scores(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
    return ranks / float(scores.shape[0] - 1)


def _zscore(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    std = float(np.std(scores))
    if std <= 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - float(np.mean(scores))) / std


def rank_mean_ensemble(s_knn, s_ball) -> np.ndarray:
    return 0.5 * (percentile_rank_scores(s_knn) + percentile_rank_scores(s_ball))


def z_mean_ensemble(s_knn, s_ball) -> np.ndarray:
    return 0.5 * (_zscore(s_knn) + _zscore(s_ball))


def _load_seed_scores(args, dataset_name: str, seed: int) -> dict:
    dataset_key = phase39._dataset_key(dataset_name)
    phase39.setup_seed(seed)
    data = phase39._load_dataset(
        dataset_name=dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, _ = phase39.label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    gb_params = phase39.load_softdist_xh_gb_config(dataset_key, method=args.softdist_method)
    embeddings = phase39.load_frozen_embeddings(Path(gb_params["embedding_path"]), expected_nodes=int(data.n_nodes))
    gb = phase39.build_ordinary_granular_balls(
        embeddings=embeddings,
        max_balls=int(gb_params["max_balls"]),
        min_ball_size=int(gb_params["min_ball_size"]),
        split_improvement=float(gb_params["split_improvement"]),
        random_state=seed,
    )
    train_idx = phase39._mask_to_index_array(data.train_masks)
    noisy_train = phase39._as_noisy_numpy(noisy_label)[train_idx]
    clean_train = data.labels.detach().cpu().long().numpy()[train_idx]
    noisy_flag = (noisy_train != clean_train).astype(np.int64)
    args.min_train_per_ball = int(gb_params.get("min_train_per_ball", args.min_train_per_ball))
    score_map = phase39.build_detector_scores(data, noisy_label, embeddings, gb.assignment, train_idx, args, seed)
    s_knn = np.asarray(score_map["s_knn"][0], dtype=np.float64)
    s_ball = np.asarray(score_map["s_ball_loo"][0], dtype=np.float64)
    return {
        "dataset": dataset_key,
        "seed": int(seed),
        "train_idx": train_idx,
        "noisy_flag": noisy_flag,
        "s_knn": s_knn,
        "s_ball_loo": s_ball,
        "rank_mean_knn_ball": rank_mean_ensemble(s_knn, s_ball),
        "z_mean_knn_ball": z_mean_ensemble(s_knn, s_ball),
    }


def _fit_leave_one_seed_logreg(seed_payloads: list[dict]) -> dict[int, np.ndarray]:
    predictions = {}
    for held in seed_payloads:
        train_payloads = [item for item in seed_payloads if int(item["seed"]) != int(held["seed"])]
        if not train_payloads:
            predictions[int(held["seed"])] = np.full_like(held["s_knn"], 0.5, dtype=np.float64)
            continue
        x_train = np.vstack([
            np.column_stack([item["s_knn"], item["s_ball_loo"]])
            for item in train_payloads
        ])
        y_train = np.concatenate([item["noisy_flag"] for item in train_payloads])
        x_test = np.column_stack([held["s_knn"], held["s_ball_loo"]])
        if np.unique(y_train).size < 2:
            predictions[int(held["seed"])] = np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(x_train, y_train)
        predictions[int(held["seed"])] = model.predict_proba(x_test)[:, 1]
    return predictions


def _row_for_detector(payload: dict, detector: str, scores: np.ndarray) -> dict:
    labels = payload["noisy_flag"]
    metrics = phase39.detection_metrics(labels, scores)
    top_rows = phase39.top_fraction_metrics(labels, scores)
    baseline = phase39.detection_metrics(labels, payload["s_knn"])
    row = {
        "dataset": payload["dataset"],
        "seed": int(payload["seed"]),
        "detector": detector,
        **metrics,
        **top_rows,
        "delta_auroc_vs_knn": metrics["auroc"] - baseline["auroc"],
        "delta_auprc_lift_vs_knn": metrics["auprc_lift"] - baseline["auprc_lift"],
        "delta_precision_at_k_vs_knn": metrics["precision_at_k"] - baseline["precision_at_k"],
    }
    return row


def summarize_delta_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        base = {keys[idx]: key[idx] for idx in range(len(keys))}
        numeric = sorted({
            field for row in group for field, value in row.items()
            if field not in keys and isinstance(value, (int, float))
        })
        for field in numeric:
            vals = [float(row[field]) for row in group if row.get(field) is not None and math.isfinite(float(row[field]))]
            base[f"{field}_mean"] = float(sum(vals) / len(vals)) if vals else float("nan")
            if len(vals) <= 1:
                base[f"{field}_std"] = 0.0 if vals else float("nan")
            else:
                mean = base[f"{field}_mean"]
                base[f"{field}_std"] = float(math.sqrt(sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)))
            if field.startswith("delta_"):
                base[f"{field}_pos"] = int(sum(value > 0.0 for value in vals))
        base["num_seeds"] = len(group)
        out.append(base)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _gate_decision(summary: list[dict]) -> str:
    primary = [row for row in summary if row["detector"] in {"rank_mean_knn_ball", "z_mean_knn_ball"}]
    pass_rows = [
        row for row in primary
        if float(row.get("delta_auroc_vs_knn_mean", float("nan"))) >= 0.01
        and int(row.get("delta_auroc_vs_knn_pos", 0)) >= 4
        and float(row.get("delta_auprc_lift_vs_knn_mean", 0.0)) >= 0.0
        and float(row.get("delta_precision_at_k_vs_knn_mean", 0.0)) >= 0.0
    ]
    covered = {row["dataset"] for row in pass_rows}
    if len(covered) >= 2:
        return "PASS: deployable kNN+ball ensemble shows stable auxiliary value over kNN."
    return "FAIL: deployable kNN+ball ensemble does not show stable auxiliary value over kNN."


def write_report(path: Path, summary: list[dict], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = _gate_decision(summary)
    lines = [
        "# Phase 40A: kNN + ball Ensemble Probe",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{','.join(args.datasets)}`; seeds: `{','.join(map(str, args.seeds))}`.",
        f"- Noise: `{args.noise_type}` {args.noise_rate}; split: percent {args.train_percent}/{args.val_percent}/{args.test_percent}.",
        "- Primary deployable probes: `rank_mean_knn_ball`, `z_mean_knn_ball`.",
        "- Oracle diagnostic: `loo_logreg_knn_ball` uses labels from other seeds and is not deployable.",
        "",
        "## Summary",
        "",
        "| Dataset | Detector | AUROC | ΔAUROC vs kNN | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {detector} | {auroc:.4f} | {dauc:.4f} | {pos}/{n} | {lift:.4f} | {dlift:.4f} | {pk:.4f} | {dpk:.4f} |".format(
                dataset=row["dataset"],
                detector=row["detector"],
                auroc=float(row.get("auroc_mean", float("nan"))),
                dauc=float(row.get("delta_auroc_vs_knn_mean", float("nan"))),
                pos=int(row.get("delta_auroc_vs_knn_pos", 0)),
                n=int(row.get("num_seeds", 0)),
                lift=float(row.get("auprc_lift_mean", float("nan"))),
                dlift=float(row.get("delta_auprc_lift_vs_knn_mean", float("nan"))),
                pk=float(row.get("precision_at_k_mean", float("nan"))),
                dpk=float(row.get("delta_precision_at_k_vs_knn_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Gate Decision",
        "",
        decision,
        "",
        "Success requires a deployable ensemble to improve mean AUROC by at least +0.01 over `s_knn`, with at least 4/5 positive seeds, on at least 2/3 datasets, without AUPRC/P@K degradation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Probe whether kNN + ball improves over kNN alone.")
    parser.add_argument("--datasets", default="cora,citeseer,coauthorcs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--softdist_method", default=phase39.SOFTDIST_METHOD)
    parser.add_argument("--method_config", default="gcn_xh_ce")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--window_start", type=int, default=5)
    parser.add_argument("--window_end", type=int, default=10)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--ball_neighbor_k", type=int, default=10)
    parser.add_argument("--lp_steps", type=int, default=2)
    parser.add_argument("--output_dir", default="runs/knn_ball_ensemble_probe_20260624")
    args = parser.parse_args()
    args.datasets = parse_str_list(args.datasets)
    args.seeds = parse_int_list(args.seeds)

    raw_rows = []
    for dataset in args.datasets:
        print(f"PHASE40A dataset={dataset}")
        payloads = [_load_seed_scores(args, dataset, seed) for seed in args.seeds]
        oracle = _fit_leave_one_seed_logreg(payloads)
        for payload in payloads:
            detectors = {
                "s_knn": payload["s_knn"],
                "s_ball_loo": payload["s_ball_loo"],
                "rank_mean_knn_ball": payload["rank_mean_knn_ball"],
                "z_mean_knn_ball": payload["z_mean_knn_ball"],
                "loo_logreg_knn_ball": oracle[int(payload["seed"])],
            }
            for detector, scores in detectors.items():
                raw_rows.append(_row_for_detector(payload, detector, np.asarray(scores, dtype=np.float64)))

    summary = summarize_delta_rows(raw_rows, ["dataset", "detector"])
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "knn_ball_ensemble_raw.csv", raw_rows)
    write_csv(output_dir / "knn_ball_ensemble_summary.csv", summary)
    write_report(output_dir / "knn_ball_ensemble_report.md", summary, args)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
