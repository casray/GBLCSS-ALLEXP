from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.balls import build_graph_aware_balls
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.diagnostics import GATED_DIAGNOSTIC_FIELDS, compute_gated_node_diagnostics, summarize_rows
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import train_noisy_linear_teacher
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, save_csv


def embedding_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    split_mode: str,
    train_ratio: float | None,
    val_ratio: float | None,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "embeddings" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "H.pt"


def result_rows_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    noise_seed: int,
    classifier_seed: int,
    split_mode: str,
    train_ratio: float | None,
    val_ratio: float | None,
    noise_type: str,
    noise_rate: float,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    rate_name = f"{noise_rate:g}".replace(".", "p")
    return (
        PROJECT_ROOT
        / "results"
        / "gate_runs"
        / dataset
        / f"{noise_type}_{rate_name}"
        / f"{split_token}_pretrain{pretrain_seed}_noise{noise_seed}_clf{classifier_seed}"
        / "result_rows.csv"
    )


def read_current_reported_ncr(path: Path, method: str, alpha: float) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != method:
                continue
            try:
                if abs(float(row.get("alpha_max", "nan")) - float(alpha)) > 1e-8:
                    continue
                value = row.get("noise_correction_rate", "")
                return float(value) if value not in {"", "nan"} else math.nan
            except ValueError:
                continue
    return None


def write_ordered_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_row(args: argparse.Namespace, dataset: str, noise_rate: float, seed: int, train_ratio: float | None) -> dict:
    split_seed = int(args.split_seed)
    pretrain_seed = int(seed)
    noise_seed = int(seed)
    classifier_seed = int(seed)
    artifact = embedding_path(dataset, split_seed, pretrain_seed, args.split_mode, train_ratio, args.val_ratio)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {artifact}")
    payload = torch.load(artifact, map_location="cpu")
    h = payload["H"].float()
    h_regions = F.normalize(h, p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index.detach().cpu().long()

    split_token = split_name(args.split_mode, split_seed, train_ratio, args.val_ratio)
    noisy_labels, _, _ = load_or_create_noisy_labels(
        clean_labels,
        masks["train_mask"],
        num_classes,
        dataset=dataset,
        noise_type=args.noise_type,
        noise_rate=noise_rate,
        split_seed=split_seed,
        noise_seed=noise_seed,
        split_name=split_token,
    )
    grouping = build_graph_aware_balls(
        h_regions,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=args.gagb_lambda_h,
        lambda_t=args.gagb_lambda_t,
        max_ball_size=args.gagb_max_ball_size,
        seed=pretrain_seed,
    )
    teacher_probs, _ = train_noisy_linear_teacher(
        h_regions,
        noisy_labels,
        masks["train_mask"],
        num_classes,
        seed=noise_seed,
        epochs=args.teacher_epochs,
        learning_rate=args.teacher_lr,
        weight_decay=args.teacher_weight_decay,
        temperature=args.teacher_temperature,
        device=get_device(args.gpu_id),
    )
    current_path = result_rows_path(
        dataset,
        split_seed,
        pretrain_seed,
        noise_seed,
        classifier_seed,
        args.split_mode,
        train_ratio,
        args.val_ratio,
        args.noise_type,
        noise_rate,
    )
    current_ncr = read_current_reported_ncr(current_path, args.method, args.alpha)
    return compute_gated_node_diagnostics(
        clean_labels,
        noisy_labels,
        masks["train_mask"],
        grouping,
        teacher_probs,
        num_classes=num_classes,
        alpha_max=args.alpha,
        dataset=dataset,
        noise_rate=noise_rate,
        seed=seed,
        method=args.method,
        qb_pred_beta=args.qb_pred_beta,
        min_train_per_group=args.min_train_per_ball,
        eps=args.eps,
        conflict_noisy_support_max=args.conflict_noisy_support_max,
        conflict_target_conf_min=args.conflict_target_conf_min,
        current_reported_ncr=current_ncr,
    )


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _summary_value(row: dict, field: str):
    if field in row:
        return row.get(field)
    return row.get(f"{field}_mean")


def classify_failure(row: dict) -> str:
    lift = float(row.get("gated_noisy_lift_over_base_noise_rate_mean", math.nan))
    possible = float(row.get("possible_correction_rate_before_alpha_mean", math.nan))
    lost = float(row.get("lost_by_alpha_rate_mean", math.nan))
    recomputed = float(row.get("recomputed_target_argmax_ncr_mean", math.nan))
    current = float(row.get("current_reported_ncr_mean", math.nan))
    if not math.isnan(recomputed) and not math.isnan(current) and abs(recomputed - current) > 1e-6:
        return "D. NCR implementation mismatch should be inspected."
    if math.isnan(lift) or lift <= 1.1:
        return "A. Gate does not clearly enrich noisy nodes."
    if math.isnan(possible) or possible <= 0.05:
        return "B. Gate selects some noisy nodes, but q_mix rarely points to the clean label."
    if lost >= 0.2:
        return "C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy."
    return "Mixed. Gate/q_mix has some correction direction, but not enough to move aggregate NCR."


def write_report(path: Path, rows: list[dict], summary: list[dict], command: str) -> None:
    lines = [
        "# GAGB v2 Gated Nodes Diagnostic Report - 2026-06-01",
        "",
        "## Purpose",
        "",
        "This diagnostic checks why `gagb_qb_conflict_gate` keeps `noise_correction_rate` at 0 without changing GAGB construction, GRACE embeddings, target training, or hyperparameters.",
        "",
        "Clean labels are used only for diagnostics in this report.",
        "",
        "## Per-setting Summary",
        "",
        "| Dataset | Noise | Gated Ratio | Gated Noisy Ratio | Lift | Possible Before Alpha | Recomputed NCR | Lost By Alpha | Teacher Gated Noisy Clean Acc | Ball Clean Purity | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row['dataset']} | {_fmt(row['noise_rate'])} | "
            f"{_fmt(row.get('gated_train_ratio_mean'))} | "
            f"{_fmt(row.get('gated_noisy_ratio_mean'))} | "
            f"{_fmt(row.get('gated_noisy_lift_over_base_noise_rate_mean'))} | "
            f"{_fmt(row.get('possible_correction_rate_before_alpha_mean'))} | "
            f"{_fmt(row.get('recomputed_target_argmax_ncr_mean'))} | "
            f"{_fmt(row.get('lost_by_alpha_rate_mean'))} | "
            f"{_fmt(row.get('teacher_gated_noisy_clean_acc_mean'))} | "
            f"{_fmt(_summary_value(row, 'gated_ball_clean_purity_mean'))} | "
            f"{classify_failure(row)} |"
        )
    lines.extend([
        "",
        "## Required Questions",
        "",
    ])
    for row in summary:
        prefix = f"{row['dataset']} noise={_fmt(row['noise_rate'])}"
        lines.extend([
            f"### {prefix}",
            "",
            f"1. Gate selects noisy nodes? `gated_noisy_ratio={_fmt(row.get('gated_noisy_ratio_mean'))}`, base train noise rate `{_fmt(row.get('train_noise_rate_mean'))}`, lift `{_fmt(row.get('gated_noisy_lift_over_base_noise_rate_mean'))}`.",
            f"2. q_mix correction direction? `gated_noisy_qmix_argmax_eq_clean_rate={_fmt(row.get('gated_noisy_qmix_argmax_eq_clean_rate_mean'))}`, `gated_noisy_qmix_clean_minus_noisy_mean={_fmt(_summary_value(row, 'gated_noisy_qmix_clean_minus_noisy_mean'))}`, possible before alpha `{_fmt(row.get('possible_correction_rate_before_alpha_mean'))}`.",
            f"3. Alpha too weak? possible before alpha `{_fmt(row.get('possible_correction_rate_before_alpha_mean'))}` vs recomputed NCR `{_fmt(row.get('recomputed_target_argmax_ncr_mean'))}`, lost by alpha `{_fmt(row.get('lost_by_alpha_rate_mean'))}`.",
            f"4. Teacher independent correction? `teacher_gated_noisy_clean_acc={_fmt(row.get('teacher_gated_noisy_clean_acc_mean'))}`, `teacher_gated_argmax_eq_clean_rate={_fmt(row.get('teacher_gated_argmax_eq_clean_rate_mean'))}`, `teacher_gated_argmax_eq_noisy_rate={_fmt(row.get('teacher_gated_argmax_eq_noisy_rate_mean'))}`.",
            f"5. Gated ball quality? `clean_purity={_fmt(_summary_value(row, 'gated_ball_clean_purity_mean'))}`, `noisy_entropy={_fmt(_summary_value(row, 'gated_ball_noisy_entropy_mean'))}`, `train_count={_fmt(_summary_value(row, 'gated_ball_train_count_mean'))}`.",
            f"6. Main NCR=0 reason: {classify_failure(row)}",
            "",
        ])
    lines.extend([
        "## Reproduction Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Output Files",
        "",
        "- `gated_nodes_per_seed.csv`",
        "- `gated_nodes_summary.csv`",
        "- `gated_nodes_diagnostic_report.md`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rows = []
    for dataset in args.datasets:
        for train_ratio in args.train_ratios:
            for noise_rate in args.noise_rates:
                for seed in args.seeds:
                    rows.append(diagnostic_row(args, dataset, float(noise_rate), int(seed), float(train_ratio)))
    summary = summarize_rows(rows)
    output_dir = args.output_dir
    write_ordered_csv(output_dir / "gated_nodes_per_seed.csv", rows, GATED_DIAGNOSTIC_FIELDS)
    save_csv(output_dir / "gated_nodes_summary.csv", summary)
    command = (
        "py -3 scripts\\diagnose_gagb_v2_gated_nodes.py ^\n"
        "--allow-external-python ^\n"
        f"--python {args.python or 'C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe'} ^\n"
        f"--datasets {' '.join(args.datasets)} ^\n"
        f"--split-mode {args.split_mode} ^\n"
        f"--train-ratios {' '.join(str(x) for x in args.train_ratios)} ^\n"
        f"--val-ratio {args.val_ratio} ^\n"
        f"--noise-rates {' '.join(str(x) for x in args.noise_rates)} ^\n"
        f"--seeds {' '.join(str(x) for x in args.seeds)} ^\n"
        f"--mode {args.mode} ^\n"
        f"--alpha {args.alpha} ^\n"
        f"--method {args.method} ^\n"
        f"--output-dir {args.output_dir}"
    )
    write_report(output_dir / "gated_nodes_diagnostic_report.md", rows, summary, command)
    print(f"(S) | Wrote {len(rows)} per-seed rows to {output_dir / 'gated_nodes_per_seed.csv'}")
    print(f"(S) | Wrote {len(summary)} summary rows to {output_dir / 'gated_nodes_summary.csv'}")
    print(f"(S) | Wrote report to {output_dir / 'gated_nodes_diagnostic_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    parser.add_argument("--split-mode", choices=["ratio", "planetoid"], default="ratio")
    parser.add_argument("--train-ratios", nargs="+", type=float, default=[0.2])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rates", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--mode", choices=["frozen"], default="frozen")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--method", choices=["gagb_qb_conflict_gate"], default="gagb_qb_conflict_gate")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "gagb_v2_gated_diagnostic_20260601")
    parser.add_argument("--gb-min-ball-size", type=int, default=5)
    parser.add_argument("--gagb-max-hops", type=int, default=2)
    parser.add_argument("--gagb-tau-add", type=float, default=0.55)
    parser.add_argument("--gagb-lambda-h", type=float, default=0.7)
    parser.add_argument("--gagb-lambda-t", type=float, default=0.3)
    parser.add_argument("--gagb-max-ball-size", type=int, default=128)
    parser.add_argument("--min-train-per-ball", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--qb-pred-beta", type=float, default=0.5)
    parser.add_argument("--conflict-noisy-support-max", type=float, default=0.35)
    parser.add_argument("--conflict-target-conf-min", type=float, default=0.55)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--allow-external-python", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
