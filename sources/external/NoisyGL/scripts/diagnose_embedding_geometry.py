from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for, train_linear_probe
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import build_frozen_grace_soft_targets, load_frozen_embeddings
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / max(1, len(values)))


def _std(values) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def embedding_uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    if int(embeddings.shape[0]) < 2:
        return 0.0
    z = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1)
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return float(torch.log(torch.exp(-float(t) * sq_pdist).mean().clamp_min(1e-12)).item())


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["ssl_method"])].append(row)

    metrics = [
        "uniformity",
        "num_balls",
        "mean_ball_size",
        "true_all_purity_weighted",
        "true_train_purity_weighted",
        "noisy_train_purity_weighted",
        "soft_pseudo_ncr",
        "soft_pseudo_cdr",
        "mean_alpha_train",
        "clean_probe",
        "noisy_probe",
    ]
    summary = []
    for (dataset, method), group in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "ssl_method": method,
            "n": len(group),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in group if metric in item and item[metric] != ""]
            if vals:
                row[f"{metric}_mean"] = _mean(vals)
                row[f"{metric}_std"] = _std(vals)
        summary.append(row)
    return summary


def gate_candidates(
    summary_rows: list[dict],
    anchor_method: str = "grace",
    purity_tolerance: float = 0.005,
) -> list[dict]:
    anchors = {
        row["dataset"]: row
        for row in summary_rows
        if row["ssl_method"] == anchor_method
    }
    decisions = []
    for row in summary_rows:
        if row["ssl_method"] == anchor_method:
            continue
        anchor = anchors.get(row["dataset"])
        if anchor is None:
            continue
        min_balls = max(1.0, float(anchor.get("num_balls_mean", 0.0)) * 0.5)
        num_balls_ok = float(row.get("num_balls_mean", 0.0)) >= min_balls
        anchor_purity = float(anchor.get("true_all_purity_weighted_mean", 0.0))
        purity_ok = float(row.get("true_all_purity_weighted_mean", 0.0)) >= anchor_purity - float(purity_tolerance)
        decisions.append({
            "dataset": row["dataset"],
            "ssl_method": row["ssl_method"],
            "anchor_method": anchor_method,
            "candidate_num_balls_mean": row.get("num_balls_mean", ""),
            "anchor_num_balls_mean": anchor.get("num_balls_mean", ""),
            "candidate_true_all_purity_weighted_mean": row.get("true_all_purity_weighted_mean", ""),
            "anchor_true_all_purity_weighted_mean": anchor.get("true_all_purity_weighted_mean", ""),
            "candidate_clean_probe_mean": row.get("clean_probe_mean", ""),
            "anchor_clean_probe_mean": anchor.get("clean_probe_mean", ""),
            "purity_tolerance": float(purity_tolerance),
            "num_balls_ok": bool(num_balls_ok),
            "purity_ok": bool(purity_ok),
            "passes_geometry_gate": bool(num_balls_ok and purity_ok),
        })
    return decisions


def _gb_value(model_conf, key, default):
    gb_conf = getattr(model_conf, "gb", {})
    if isinstance(gb_conf, dict):
        value = gb_conf.get(key, default)
    else:
        value = getattr(gb_conf, key, default)
    return default if value is None else value


def diagnose_one(args, dataset_name: str, ssl_method: str, seed: int) -> dict:
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
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    embedding_path = embedding_path_for(args.embedding_root, dataset_name, ssl_method)
    if not embedding_path.exists():
        raise FileNotFoundError(f"missing H for ssl_method={ssl_method}, dataset={dataset_name}: {embedding_path}")
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.labels.shape[0]))
    model_conf = load_conf(None, args.base_method, dataset_key)
    ball_result = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
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
        random_state=seed,
        random_control=False,
    )
    row = {
        "dataset": dataset_key,
        "ssl_method": ssl_method,
        "seed": int(seed),
        "embedding_path": str(embedding_path),
        "noise_scope": args.noise_scope,
        "uniformity": embedding_uniformity(embeddings, t=args.uniformity_t),
        **ball_result.diagnostics,
    }
    if args.probe:
        clean_probe = train_linear_probe(
            embeddings,
            data.labels,
            data.labels,
            data.train_masks,
            data.val_masks,
            data.test_masks,
            data.n_classes,
            seed=seed,
            epochs=args.probe_epochs,
            device=args.device,
        )
        noisy_probe = train_linear_probe(
            embeddings,
            noisy_label,
            data.labels,
            data.train_masks,
            data.val_masks,
            data.test_masks,
            data.n_classes,
            seed=seed,
            epochs=args.probe_epochs,
            device=args.device,
        )
        row["clean_probe"] = clean_probe["test_acc_at_best_val"]
        row["noisy_probe"] = noisy_probe["test_acc_at_best_val"]
    return row


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


def write_report(path: Path, summary_rows: list[dict], decisions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 20a Embedding Geometry Diagnostic",
        "",
        "Gate uses only geometry: non-collapsed ball count and true weighted ball purity.",
        "Probe accuracy is reported as reference only and is not used for pass/fail.",
        "",
        "## Summary",
        "",
        "| Dataset | Method | Num balls | Uniformity | True all purity | Clean probe | Noisy probe |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {method} | {num_balls:.4f} | {uniformity:.4f} | {purity:.4f} | {clean_probe} | {noisy_probe} |".format(
                dataset=row["dataset"],
                method=row["ssl_method"],
                num_balls=float(row.get("num_balls_mean", 0.0)),
                uniformity=float(row.get("uniformity_mean", 0.0)),
                purity=float(row.get("true_all_purity_weighted_mean", 0.0)),
                clean_probe=_fmt_optional(row.get("clean_probe_mean", "")),
                noisy_probe=_fmt_optional(row.get("noisy_probe_mean", "")),
            )
        )
    lines.extend([
        "",
        "## Gate Decisions",
        "",
        "| Dataset | Method | Num balls OK | Purity OK | Pass |",
        "|---|---|---:|---:|---:|",
    ])
    for row in decisions:
        lines.append(
            f"| {row['dataset']} | {row['ssl_method']} | {int(row['num_balls_ok'])} | "
            f"{int(row['purity_ok'])} | {int(row['passes_geometry_gate'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional(value) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser(description="Diagnose embedding geometry before downstream SSL swap.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--ssl_methods", default="grace,grace_unif0p1,grace_unif0p5,grace_unif1")
    parser.add_argument("--base_method", default="frozengrace_gb_gcn_softdist_xh")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--uniformity_t", type=float, default=2.0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe_epochs", type=int, default=500)
    parser.add_argument("--purity_tolerance", type=float, default=0.005)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "phase20_custom_ssl_20260612"))
    args = parser.parse_args()

    rows = []
    for dataset_name in _parse_str_list(args.datasets):
        for ssl_method in _parse_str_list(args.ssl_methods):
            for seed in _parse_int_list(args.seeds):
                print(f"GEOMETRY_DIAG dataset={dataset_name} ssl={ssl_method} seed={seed}")
                row = diagnose_one(args, dataset_name, ssl_method, seed)
                print(
                    "RESULT "
                    f"dataset={row['dataset']} ssl={ssl_method} seed={seed} "
                    f"num_balls={row.get('num_balls')} "
                    f"uniformity={float(row.get('uniformity', 0.0)):.4f} "
                    f"purity={float(row.get('true_all_purity_weighted', 0.0)):.4f}"
                )
                rows.append(row)

    output_dir = Path(args.output_dir)
    summary = summarize_rows(rows)
    decisions = gate_candidates(summary, anchor_method="grace", purity_tolerance=args.purity_tolerance)
    write_csv(output_dir / "geometry_raw.csv", rows)
    write_csv(output_dir / "geometry_summary.csv", summary)
    write_csv(output_dir / "geometry_gate_decisions.csv", decisions)
    write_report(output_dir / "geometry_report.md", summary, decisions)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
