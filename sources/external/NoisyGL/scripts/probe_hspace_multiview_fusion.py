from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import diagnose_label_noise_detection as phase39


SINGLE_DETECTORS = ["s_knn", "s_gcn_conf", "s_ball_loo", "s_graphnbr"]
FUSION_SPECS = {
    "F1_rank_knn_gcn": ["s_knn", "s_gcn_conf"],
    "F2_rank_knn_ball": ["s_knn", "s_ball_loo"],
    "F3_rank_gcn_ball": ["s_gcn_conf", "s_ball_loo"],
    "F4_rank_knn_gcn_ball": ["s_knn", "s_gcn_conf", "s_ball_loo"],
    "F5_rank_knn_gcn_ball_graphnbr": ["s_knn", "s_gcn_conf", "s_ball_loo", "s_graphnbr"],
}
ORACLE_SPECS = {
    "O1_logreg_knn_gcn": ["s_knn", "s_gcn_conf"],
    "O2_logreg_knn_gcn_ball": ["s_knn", "s_gcn_conf", "s_ball_loo"],
}


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


def rank_mean_many(score_map: dict[str, np.ndarray], detector_names: list[str]) -> np.ndarray:
    missing = [name for name in detector_names if name not in score_map]
    if missing:
        raise KeyError(f"Missing detector scores: {missing}")
    ranks = [percentile_rank_scores(score_map[name]) for name in detector_names]
    return np.mean(np.vstack(ranks), axis=0)


def _fit_leave_one_seed_logreg(seed_payloads: list[dict], feature_names: list[str]) -> dict[int, np.ndarray]:
    predictions = {}
    for held in seed_payloads:
        train_payloads = [item for item in seed_payloads if int(item["seed"]) != int(held["seed"])]
        x_test = np.column_stack([held[name] for name in feature_names])
        if not train_payloads:
            predictions[int(held["seed"])] = np.full(x_test.shape[0], 0.5, dtype=np.float64)
            continue
        x_train = np.vstack([
            np.column_stack([item[name] for name in feature_names])
            for item in train_payloads
        ])
        y_train = np.concatenate([item["noisy_flag"] for item in train_payloads])
        if np.unique(y_train).size < 2:
            predictions[int(held["seed"])] = np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(x_train, y_train)
        predictions[int(held["seed"])] = model.predict_proba(x_test_scaled)[:, 1]
    return predictions


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
    noisy_np = phase39._as_noisy_numpy(noisy_label)
    noisy_train = noisy_np[train_idx]
    clean_train = data.labels.detach().cpu().long().numpy()[train_idx]
    noisy_flag = (noisy_train != clean_train).astype(np.int64)
    args.min_train_per_ball = int(gb_params.get("min_train_per_ball", args.min_train_per_ball))
    score_map = phase39.build_detector_scores(data, noisy_label, embeddings, gb.assignment, train_idx, args, seed)
    payload = {
        "dataset": dataset_key,
        "seed": int(seed),
        "train_idx": train_idx,
        "noisy_label": noisy_train.astype(np.int64),
        "clean_label": clean_train.astype(np.int64),
        "noisy_flag": noisy_flag,
        "train_noise_rate": float(np.mean(noisy_flag)) if noisy_flag.size else float("nan"),
        "config_path": gb_params.get("config_path", ""),
        "embedding_path": gb_params.get("embedding_path", ""),
        "max_balls": int(gb_params["max_balls"]),
        "min_ball_size": int(gb_params["min_ball_size"]),
        "split_improvement": float(gb_params["split_improvement"]),
    }
    for detector in SINGLE_DETECTORS:
        payload[detector] = np.asarray(score_map[detector][0], dtype=np.float64)
        payload[f"{detector}_eligible_rate"] = float(np.mean(score_map[detector][1]))
    for detector, inputs in FUSION_SPECS.items():
        payload[detector] = rank_mean_many(payload, inputs)
    return payload


def _row_for_detector(payload: dict, detector: str, scores: np.ndarray, family: str) -> dict:
    metrics = phase39.detection_metrics(payload["noisy_flag"], scores)
    top_rows = phase39.top_fraction_metrics(payload["noisy_flag"], scores)
    return {
        "dataset": payload["dataset"],
        "seed": int(payload["seed"]),
        "detector": detector,
        "family": family,
        **metrics,
        **top_rows,
        "train_noise_rate": payload["train_noise_rate"],
        "s_graphnbr_eligible_rate": payload.get("s_graphnbr_eligible_rate", float("nan")),
    }


def add_best_single_deltas(rows: list[dict], primary_metric: str = "auroc") -> list[dict]:
    best_by_seed = {}
    for row in rows:
        if row.get("family") != "single":
            continue
        key = (row["dataset"], int(row["seed"]))
        current = best_by_seed.get(key)
        if current is None or float(row[primary_metric]) > float(current[primary_metric]):
            best_by_seed[key] = row
    out = []
    for row in rows:
        new_row = dict(row)
        best = best_by_seed.get((row["dataset"], int(row["seed"])))
        if best is not None:
            new_row["best_single_detector"] = best["detector"]
            for metric in ["auroc", "auprc_lift", "precision_at_k", "recall_at_k", "f1_at_k"]:
                if metric not in best or metric not in row:
                    continue
                new_row[f"best_single_{metric}"] = float(best[metric])
                new_row[f"delta_{metric}_vs_best_single"] = float(row[metric]) - float(best[metric])
        out.append(new_row)
    return out


def summarize_rows(rows: list[dict], keys: list[str]) -> list[dict]:
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
        best_names = [str(row.get("best_single_detector", "")) for row in group if row.get("best_single_detector")]
        if best_names:
            counts = {name: best_names.count(name) for name in sorted(set(best_names))}
            base["best_single_detector_counts"] = ";".join(f"{name}:{counts[name]}" for name in sorted(counts))
        base["num_seeds"] = len(group)
        out.append(base)
    return out


def gate_decision(summary: list[dict]) -> str:
    fusion_rows = [row for row in summary if row.get("family") == "fusion"]
    pass_rows = [
        row for row in fusion_rows
        if float(row.get("delta_auroc_vs_best_single_mean", float("nan"))) >= 0.01
        and int(row.get("delta_auroc_vs_best_single_pos", 0)) >= 4
        and float(row.get("delta_auprc_lift_vs_best_single_mean", 0.0)) >= 0.0
        and float(row.get("delta_precision_at_k_vs_best_single_mean", 0.0)) >= 0.0
    ]
    covered = {row["dataset"] for row in pass_rows}
    if len(covered) >= 2:
        return "PASS: rank fusion beats the per-seed best single detector on at least 2/3 datasets."
    return "FAIL: rank fusion does not show stable value over the per-seed best single detector."


def _format_float(value, digits=4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


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


def node_score_rows(payloads: list[dict]) -> list[dict]:
    rows = []
    node_fields = SINGLE_DETECTORS + list(FUSION_SPECS.keys())
    for payload in payloads:
        for pos, node_id in enumerate(payload["train_idx"].tolist()):
            row = {
                "dataset": payload["dataset"],
                "seed": int(payload["seed"]),
                "node_id": int(node_id),
                "noisy_label": int(payload["noisy_label"][pos]),
                "clean_label": int(payload["clean_label"][pos]),
                "noisy_flag": int(payload["noisy_flag"][pos]),
            }
            for detector in node_fields:
                row[detector] = float(payload[detector][pos])
            rows.append(row)
    return rows


def write_report(path: Path, summary: list[dict], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = gate_decision(summary)
    lines = [
        "# Phase 40a: H-space Multi-view Rank Fusion Diagnostic",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{','.join(args.datasets)}`; seeds: `{','.join(map(str, args.seeds))}`.",
        f"- Noise: `{args.noise_type}` {args.noise_rate}; split: percent {args.train_percent}/{args.val_percent}/{args.test_percent}.",
        "- Single detectors: `s_knn`, `s_gcn_conf`, `s_ball_loo`, `s_graphnbr`.",
        "- Primary fusion detectors: F1-F5 percentile-rank averages.",
        "- Oracle logreg rows use clean/noisy flags from other seeds and are diagnostic only.",
        "",
        "## Summary",
        "",
        "| Dataset | Family | Detector | AUROC | ΔAUROC vs best single | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K | best single counts |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {family} | {detector} | {auroc} | {dauc} | {pos}/{n} | {lift} | {dlift} | {pk} | {dpk} | {best} |".format(
                dataset=row["dataset"],
                family=row["family"],
                detector=row["detector"],
                auroc=_format_float(row.get("auroc_mean")),
                dauc=_format_float(row.get("delta_auroc_vs_best_single_mean")),
                pos=int(row.get("delta_auroc_vs_best_single_pos", 0)),
                n=int(row.get("num_seeds", 0)),
                lift=_format_float(row.get("auprc_lift_mean")),
                dlift=_format_float(row.get("delta_auprc_lift_vs_best_single_mean")),
                pk=_format_float(row.get("precision_at_k_mean")),
                dpk=_format_float(row.get("delta_precision_at_k_vs_best_single_mean")),
                best=row.get("best_single_detector_counts", ""),
            )
        )
    f4_rows = [row for row in summary if row["detector"] == "F4_rank_knn_gcn_ball"]
    f4_note = "; ".join(
        f"{row['dataset']}: ΔAUROC={_format_float(row.get('delta_auroc_vs_best_single_mean'))}, pos={int(row.get('delta_auroc_vs_best_single_pos', 0))}/{int(row.get('num_seeds', 0))}"
        for row in f4_rows
    )
    lines.extend([
        "",
        "## Gate Decision",
        "",
        decision,
        "",
        "Success requires a rank fusion detector to improve mean AUROC by at least +0.01 over the per-seed best single detector, with at least 4/5 positive seeds, on at least 2/3 datasets, without AUPRC/P@K degradation.",
        "",
        "## Key Check",
        "",
        f"`F4_rank_knn_gcn_ball` vs per-seed best single: {f4_note}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Probe whether H-space multi-view rank fusion beats the best single detector.")
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
    parser.add_argument("--output_dir", default="runs/hspace_multiview_fusion_20260624")
    parser.add_argument("--skip_oracle", action="store_true")
    args = parser.parse_args()
    args.datasets = parse_str_list(args.datasets)
    args.seeds = parse_int_list(args.seeds)

    raw_rows = []
    all_payloads = []
    for dataset in args.datasets:
        print(f"PHASE40a multiview dataset={dataset}")
        payloads = [_load_seed_scores(args, dataset, seed) for seed in args.seeds]
        all_payloads.extend(payloads)
        oracle_predictions = {}
        if not args.skip_oracle:
            for oracle_name, features in ORACLE_SPECS.items():
                oracle_predictions[oracle_name] = _fit_leave_one_seed_logreg(payloads, features)
        for payload in payloads:
            detectors = {}
            for name in SINGLE_DETECTORS:
                detectors[name] = (payload[name], "single")
            for name in FUSION_SPECS:
                detectors[name] = (payload[name], "fusion")
            for name, preds_by_seed in oracle_predictions.items():
                detectors[name] = (preds_by_seed[int(payload["seed"])], "oracle")
            for detector, (scores, family) in detectors.items():
                raw_rows.append(_row_for_detector(payload, detector, np.asarray(scores, dtype=np.float64), family))

    raw_rows = add_best_single_deltas(raw_rows, primary_metric="auroc")
    summary = summarize_rows(raw_rows, ["dataset", "family", "detector"])
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "hspace_multiview_fusion_raw.csv", raw_rows)
    write_csv(output_dir / "hspace_multiview_fusion_summary.csv", summary)
    write_csv(output_dir / "node_scores_raw.csv", node_score_rows(all_payloads))
    write_report(output_dir / "hspace_multiview_fusion_report.md", summary, args)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
