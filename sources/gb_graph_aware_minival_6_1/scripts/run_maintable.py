"""Formal main-table runner for local-consistency reliability selection.

Phase 1 only:
  datasets = Cora, CiteSeer
  noise    = pair 0.4, pair 0.6, symmetric 0.4
  seeds    = 0..4

Primary method is sel_topk. sel_dcgb is an ablation, not the main method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.utils import save_csv, save_json
from scripts.run_deathmatch_common import evaluate_gates, run_seed, write_report
from noisy_label_gb_grace.utils import get_device


METHODS = ["ce", "coteaching", "sel_topk", "sel_dcgb", "sel_noball", "sel_shuffled", "sel_random", "sel_clean_oracle"]
PAIR_T_CRIT_DF4 = 2.78


def _mean(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def paired_stats(a: list[float], b: list[float]) -> dict:
    deltas = [float(x) - float(y) for x, y in zip(a, b)]
    mean_delta = _mean(deltas)
    std_delta = _std(deltas)
    if len(deltas) <= 1 or std_delta == 0.0:
        t_value = math.inf if mean_delta > 0 else (-math.inf if mean_delta < 0 else 0.0)
    else:
        t_value = mean_delta / (std_delta / math.sqrt(len(deltas)))
    return {
        "n": len(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "mean_delta": mean_delta,
        "mean_delta_pp": mean_delta * 100.0,
        "std_delta": std_delta,
        "t_value": t_value,
        "sig_label": "SIG" if abs(t_value) > PAIR_T_CRIT_DF4 else "ns",
    }


def _values(per_seed: dict[int, dict[str, float]], method: str, seeds: list[int]) -> list[float]:
    return [per_seed[s][method] for s in seeds if s in per_seed and method in per_seed[s]]


def evaluate_main_gate(per_seed: dict[int, dict[str, float]], seeds: list[int]) -> dict:
    topk_coteach = paired_stats(_values(per_seed, "sel_topk", seeds), _values(per_seed, "coteaching", seeds))
    topk_noball = paired_stats(_values(per_seed, "sel_topk", seeds), _values(per_seed, "sel_noball", seeds))
    topk_shuffled = paired_stats(_values(per_seed, "sel_topk", seeds), _values(per_seed, "sel_shuffled", seeds))
    topk_ce = paired_stats(_values(per_seed, "sel_topk", seeds), _values(per_seed, "ce", seeds))
    dcgb_topk = paired_stats(_values(per_seed, "sel_dcgb", seeds), _values(per_seed, "sel_topk", seeds))
    return {
        "gate_1_topk_beats_coteaching": {
            **topk_coteach,
            "passes": topk_coteach["wins"] >= 4 and topk_coteach["mean_delta"] >= 0.005,
        },
        "gate_2_topk_beats_controls": {
            "passes": topk_noball["wins"] >= 4 and topk_shuffled["wins"] >= 4,
            "sel_topk_minus_sel_noball": topk_noball,
            "sel_topk_minus_sel_shuffled": topk_shuffled,
        },
        "diagnostic_topk_minus_ce": topk_ce,
        "dcgb_minus_topk": dcgb_topk,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    ap.add_argument("--noise-specs", nargs="+", default=["pair:0.4", "pair:0.6", "symmetric:0.4"])
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
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/maintable_20260608")
    return ap.parse_args()


def _parse_noise(spec: str) -> tuple[str, float]:
    kind, rate = spec.split(":", 1)
    if kind not in {"pair", "symmetric"}:
        raise ValueError(f"Unsupported noise spec: {spec}")
    return kind, float(rate)


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


def _write_setting_report(out_dir: Path, dataset: str, noise_type: str, noise_rate: float, rows: list[dict], main_gate: dict, common_gate: dict):
    summary = _method_summary(rows)
    lines = [
        f"# Main-table setting: {dataset} {noise_type} {noise_rate}",
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
            "## Paired Deltas",
            "",
            "| contrast | delta pp | t | sig | wins |",
            "|---|---:|---:|---|---:|",
        ]
    )
    contrasts = {
        "sel_topk - coteaching": main_gate["gate_1_topk_beats_coteaching"],
        "sel_topk - ce": main_gate["diagnostic_topk_minus_ce"],
        "sel_topk - sel_noball": main_gate["gate_2_topk_beats_controls"]["sel_topk_minus_sel_noball"],
        "sel_topk - sel_shuffled": main_gate["gate_2_topk_beats_controls"]["sel_topk_minus_sel_shuffled"],
        "sel_dcgb - sel_topk": main_gate["dcgb_minus_topk"],
    }
    for name, stats in contrasts.items():
        lines.append(f"| {name} | {stats['mean_delta_pp']:.2f} | {stats['t_value']:.2f} | {stats['sig_label']} | {stats['wins']}/{stats['n']} |")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Gate-1 topk > co-teaching: **{'PASS' if main_gate['gate_1_topk_beats_coteaching']['passes'] else 'FAIL'}**",
            f"- Gate-2 topk > no-ball and shuffled: **{'PASS' if main_gate['gate_2_topk_beats_controls']['passes'] else 'FAIL'}**",
            "",
            "## Raw Gate JSON",
            "",
            "```json",
            json.dumps({"main_gate": main_gate, "deathmatch_gate": common_gate}, indent=2, ensure_ascii=False),
            "```",
        ]
    )
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
            common_gate = evaluate_gates(rows, list(args.seeds), noise_type)
            write_report(setting_dir, run_args, rows, common_gate)
            main_gate = evaluate_main_gate(_per_seed(rows), list(args.seeds))
            save_json(setting_dir / "main_gate.json", main_gate)
            _write_setting_report(setting_dir, dataset, noise_type, noise_rate, rows, main_gate, common_gate)
            save_csv(setting_dir / "main_rows.csv", rows)
            summary = {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                **_method_summary(rows),
                "gate_1_pass": main_gate["gate_1_topk_beats_coteaching"]["passes"],
                "gate_2_pass": main_gate["gate_2_topk_beats_controls"]["passes"],
                "topk_minus_coteaching_pp": main_gate["gate_1_topk_beats_coteaching"]["mean_delta_pp"],
                "topk_minus_coteaching_t": main_gate["gate_1_topk_beats_coteaching"]["t_value"],
                "topk_minus_coteaching_sig": main_gate["gate_1_topk_beats_coteaching"]["sig_label"],
                "topk_minus_ce_pp": main_gate["diagnostic_topk_minus_ce"]["mean_delta_pp"],
                "topk_minus_ce_t": main_gate["diagnostic_topk_minus_ce"]["t_value"],
                "topk_minus_ce_sig": main_gate["diagnostic_topk_minus_ce"]["sig_label"],
                "dcgb_minus_topk_pp": main_gate["dcgb_minus_topk"]["mean_delta_pp"],
                "dcgb_minus_topk_t": main_gate["dcgb_minus_topk"]["t_value"],
                "dcgb_minus_topk_sig": main_gate["dcgb_minus_topk"]["sig_label"],
            }
            setting_rows.append(summary)
            all_rows.extend(rows)

    save_csv(args.out_dir / "maintable_rows.csv", all_rows)
    save_csv(args.out_dir / "maintable_summary.csv", setting_rows)
    save_json(args.out_dir / "maintable_config.json", vars(args))

    lines = [
        "# Formal Main Table Phase 1",
        "",
        "| dataset | noise | ce | coteaching | sel_topk | sel_dcgb | topk-coteach pp/t | topk-CE pp/t | dcgb-topk pp/t | Gate1 | Gate2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in setting_rows:
        noise = f"{row['noise_type']}{row['noise_rate']}"
        lines.append(
            f"| {row['dataset']} | {noise} | {row.get('ce_mean', math.nan):.4f} | {row.get('coteaching_mean', math.nan):.4f} | "
            f"{row.get('sel_topk_mean', math.nan):.4f} | {row.get('sel_dcgb_mean', math.nan):.4f} | "
            f"{row['topk_minus_coteaching_pp']:.2f}/{row['topk_minus_coteaching_t']:.2f} {row['topk_minus_coteaching_sig']} | "
            f"{row['topk_minus_ce_pp']:.2f}/{row['topk_minus_ce_t']:.2f} {row['topk_minus_ce_sig']} | "
            f"{row['dcgb_minus_topk_pp']:.2f}/{row['dcgb_minus_topk_t']:.2f} {row['dcgb_minus_topk_sig']} | "
            f"{'PASS' if row['gate_1_pass'] else 'FAIL'} | {'PASS' if row['gate_2_pass'] else 'FAIL'} |"
        )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {args.out_dir / 'maintable_summary.csv'}")


if __name__ == "__main__":
    main()
