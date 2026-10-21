"""Run the validity-boundary matrix for local-consistency reliability selection.

This runner stays in the minival frozen-GRACE protocol. It extends the formal main-table runner with
label-free warning metrics and covers the full noise axis:
  Cora, CiteSeer x symmetric/pair x 0.2/0.4/0.6 x seeds 0..4.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls import build_dcgb_balls
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation.failure_signal import boundary_warning_metrics
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing.targets import build_topk_group_size
from noisy_label_gb_grace.utils import get_device, save_csv, save_json
from scripts.extract_embeddings import embedding_path
from scripts.run_deathmatch_common import evaluate_gates, run_seed, write_report
from scripts.run_maintable import METHODS, evaluate_main_gate
from scripts.summarize_boundary import (
    boundary_summary_rows,
    compute_boundary_correlations,
    evaluate_boundary_claims,
    write_boundary_report,
    _plot_outputs,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    ap.add_argument(
        "--noise-specs",
        nargs="+",
        default=["symmetric:0.2", "symmetric:0.4", "symmetric:0.6", "pair:0.2", "pair:0.4", "pair:0.6"],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="ratio")
    ap.add_argument("--train-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--ramp-epochs", type=int, default=20)
    ap.add_argument("--classifier-epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    ap.add_argument("--min-train-per-ball", type=int, default=3)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/boundary_20260608")
    return ap.parse_args()


def _parse_noise(spec: str) -> tuple[str, float]:
    kind, rate = spec.split(":", 1)
    if kind not in {"pair", "symmetric"}:
        raise ValueError(f"Unsupported noise spec: {spec}")
    return kind, float(rate)


def _mean(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _method_summary(rows: list[dict]) -> dict:
    out = {}
    for method in METHODS:
        vals = [float(r["test_acc_at_best_val"]) for r in rows if r["method"] == method]
        if vals:
            out[f"{method}_mean"] = _mean(vals)
            out[f"{method}_std"] = _std(vals)
    return out


def _per_seed(rows: list[dict]) -> dict[int, dict[str, float]]:
    table: dict[int, dict[str, float]] = {}
    for row in rows:
        table.setdefault(int(row["seed"]), {})[row["method"]] = float(row["test_acc_at_best_val"])
    return table


def _setting_slug(dataset: str, noise_type: str, noise_rate: float) -> str:
    return f"{dataset}_{noise_type}_{str(noise_rate).replace('.', 'p')}"


def _add_label_free_warning_metrics(args, rows: list[dict], seed: int):
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    payload = torch.load(
        embedding_path(args.dataset, args.split_seed, seed, args.split_mode, args.train_ratio, args.val_ratio),
        map_location="cpu",
    )
    h_norm = F.normalize(payload["H"].float(), p=2, dim=1)
    clean = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            args.dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index
    noisy, _, _ = load_or_create_noisy_labels(
        clean,
        masks["train_mask"],
        num_classes,
        dataset=args.dataset,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        split_seed=args.split_seed,
        noise_seed=seed,
        split_name=split_token,
    )
    dcgb = build_dcgb_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=seed,
        mode="emb_only",
    )
    topk = build_topk_group_size(
        dcgb["groups"],
        train_mask=masks["train_mask"],
        min_train_per_group=args.min_train_per_ball,
        match_train_nodes=True,
    )
    warning = boundary_warning_metrics(
        h_norm,
        noisy,
        masks["train_mask"],
        int(topk),
        num_classes,
        prefix="topk_",
    )
    for row in rows:
        if int(row["seed"]) == int(seed) and row["method"] == "sel_topk":
            row.update(warning)


def _write_setting_report(out_dir: Path, dataset: str, noise_type: str, noise_rate: float, rows: list[dict], main_gate: dict, common_gate: dict):
    summary = _method_summary(rows)
    setting_summary = boundary_summary_rows(rows)[0]
    lines = [
        f"# Boundary setting: {dataset} {noise_type} {noise_rate}",
        "",
        "## Mean +- Std Test",
        "",
        "| method | mean | std |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        if f"{method}_mean" in summary:
            lines.append(f"| {method} | {summary[f'{method}_mean']:.4f} | {summary[f'{method}_std']:.4f} |")
    lines.extend(
        [
            "",
            "## Boundary Diagnostics",
            "",
            f"- random clean rate: `{setting_summary['random_clean_rate']:.4f}`",
            f"- selection precision - random clean rate: `{setting_summary['selection_precision_minus_random_clean_rate']:.4f}`",
            f"- noise excluded recall: `{setting_summary['noise_excluded_recall_mean']:.4f}`",
            f"- topk neighborhood majority disagreement: `{setting_summary.get('topk_neighborhood_majority_self_disagreement', math.nan):.4f}`",
            f"- topk neighborhood majority confidence: `{setting_summary.get('topk_neighborhood_majority_confidence', math.nan):.4f}`",
            f"- topk pairflip neighborhood consistency: `{setting_summary.get('topk_pairflip_neighborhood_consistency', math.nan):.4f}`",
            "",
            "## Paired Deltas",
            "",
            "| contrast | delta pp | t | sig | wins |",
            "|---|---:|---:|---|---:|",
        ]
    )
    contrasts = {
        "sel_topk - coteaching": main_gate["gate_1_topk_beats_coteaching"],
        "sel_topk - ce": main_gate["diagnostic_topk_minus_ce"],
        "sel_dcgb - sel_topk": main_gate["dcgb_minus_topk"],
    }
    for name, stats in contrasts.items():
        lines.append(f"| {name} | {stats['mean_delta_pp']:.2f} | {stats['t_value']:.2f} | {stats['sig_label']} | {stats['wins']}/{stats['n']} |")
    lines.extend(["", "## Raw Gate JSON", "", "```json", json.dumps({"main_gate": main_gate, "deathmatch_gate": common_gate}, indent=2), "```"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    setting_rows = []
    for dataset in args.datasets:
        for noise_spec in args.noise_specs:
            noise_type, noise_rate = _parse_noise(noise_spec)
            slug = _setting_slug(dataset, noise_type, noise_rate)
            setting_dir = args.out_dir / slug
            setting_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== RUN {slug} ===")
            rows = []
            for seed in args.seeds:
                run_args = argparse.Namespace(**vars(args))
                run_args.dataset = dataset
                run_args.noise_type = noise_type
                run_args.noise_rate = noise_rate
                run_args.out_dir = setting_dir
                rows.extend(run_seed(run_args, seed, device))
                _add_label_free_warning_metrics(run_args, rows, seed)
            common_gate = evaluate_gates(rows, list(args.seeds), noise_type)
            write_report(setting_dir, run_args, rows, common_gate)
            main_gate = evaluate_main_gate(_per_seed(rows), list(args.seeds))
            save_json(setting_dir / "main_gate.json", main_gate)
            save_csv(setting_dir / "boundary_rows.csv", rows)
            _write_setting_report(setting_dir, dataset, noise_type, noise_rate, rows, main_gate, common_gate)

            setting_summary = boundary_summary_rows(rows)[0]
            summary = {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                **_method_summary(rows),
                **setting_summary,
                "topk_minus_coteaching_pp": main_gate["gate_1_topk_beats_coteaching"]["mean_delta_pp"],
                "topk_minus_coteaching_t": main_gate["gate_1_topk_beats_coteaching"]["t_value"],
                "topk_minus_coteaching_sig": main_gate["gate_1_topk_beats_coteaching"]["sig_label"],
                "topk_minus_coteaching_wins": main_gate["gate_1_topk_beats_coteaching"]["wins"],
                "topk_minus_ce_pp": main_gate["diagnostic_topk_minus_ce"]["mean_delta_pp"],
                "topk_minus_ce_t": main_gate["diagnostic_topk_minus_ce"]["t_value"],
                "topk_minus_ce_sig": main_gate["diagnostic_topk_minus_ce"]["sig_label"],
                "dcgb_minus_topk_pp": main_gate["dcgb_minus_topk"]["mean_delta_pp"],
                "dcgb_minus_topk_t": main_gate["dcgb_minus_topk"]["t_value"],
                "dcgb_minus_topk_sig": main_gate["dcgb_minus_topk"]["sig_label"],
            }
            setting_rows.append(summary)
            all_rows.extend(rows)

    save_csv(args.out_dir / "boundary_rows.csv", all_rows)
    save_csv(args.out_dir / "boundary_setting_summary.csv", setting_rows)
    correlations = compute_boundary_correlations(setting_rows)
    claims = evaluate_boundary_claims(setting_rows, correlations)
    save_json(args.out_dir / "boundary_correlations.json", correlations)
    save_json(args.out_dir / "boundary_claims.json", claims)
    save_json(args.out_dir / "boundary_config.json", vars(args))
    _plot_outputs(setting_rows, correlations, args.out_dir)
    write_boundary_report(args.out_dir, setting_rows, correlations, claims)
    print(f"saved {args.out_dir / 'boundary_setting_summary.csv'}")


if __name__ == "__main__":
    main()
