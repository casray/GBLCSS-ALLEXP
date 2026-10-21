from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_mlp8_p2_advisor_smoke import (
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_OUTPUT_DIR as MLP8_P2_OUTPUT_DIR,
    extract_mlp_hidden,
    load_dataset_with_noise,
    train_mlp_with_soft_targets,
)
from utils.frozen_grace_targets import build_soft_targets_from_features, load_frozen_embeddings
from utils.tools import load_conf, setup_seed


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "mlp8_p2_ballparam_rescue_20260703"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _format_param(max_balls: int, min_ball_size: int, split_improvement: float) -> str:
    return f"{int(max_balls)}/{int(min_ball_size)}/{float(split_improvement):g}"


def _mean_train_per_ball(diag: dict) -> float:
    eligible_balls = int(diag.get("eligible_balls", 0) or 0)
    eligible_train_nodes = int(diag.get("eligible_train_nodes", 0) or 0)
    return float(eligible_train_nodes / eligible_balls) if eligible_balls > 0 else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scan_embedding(
    *,
    phase: str,
    dataset_name: str,
    variant: str,
    embedding: torch.Tensor,
    data,
    noisy_label: torch.Tensor,
    seed: int,
    max_balls_values: list[int],
    min_ball_size_values: list[int],
    split_improvement_values: list[float],
) -> list[dict]:
    conf = load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset_name.lower())
    rows: list[dict] = []
    for max_balls in max_balls_values:
        for min_ball_size in min_ball_size_values:
            for split_improvement in split_improvement_values:
                result = build_soft_targets_from_features(
                    ball_features=embedding,
                    noisy_label=noisy_label,
                    clean_label=data.labels,
                    train_mask=data.train_masks,
                    n_classes=data.n_classes,
                    max_balls=int(max_balls),
                    min_ball_size=int(min_ball_size),
                    split_improvement=float(split_improvement),
                    min_train_per_ball=conf.gb["min_train_per_ball"],
                    alpha_baseline=conf.gb["alpha_baseline"],
                    target_mode=conf.gb["target_mode"],
                    target_beta=conf.gb["target_beta"],
                    random_state=seed,
                    random_control=False,
                )
                diag = result.diagnostics
                row = {
                    "phase": phase,
                    "dataset": dataset_name.lower(),
                    "variant": variant,
                    "seed": int(seed),
                    "max_balls": int(max_balls),
                    "min_ball_size": int(min_ball_size),
                    "split_improvement": float(split_improvement),
                    "param": _format_param(max_balls, min_ball_size, split_improvement),
                    "num_balls": int(diag.get("num_balls", 0)),
                    "mean_ball_size": float(diag.get("mean_ball_size", 0.0)),
                    "max_ball_size": int(diag.get("max_ball_size", 0)),
                    "mean_radius": float(diag.get("mean_radius", 0.0)),
                    "eligible_balls": int(diag.get("eligible_balls", 0) or 0),
                    "eligible_train_nodes": int(diag.get("eligible_train_nodes", 0) or 0),
                    "mean_train_per_ball": _mean_train_per_ball(diag),
                    "alpha_positive_train_count": int(diag.get("alpha_positive_train_count", 0) or 0),
                    "mean_alpha_train": float(diag.get("mean_alpha_train", 0.0) or 0.0),
                    "noisy_train_purity_weighted": float(diag.get("noisy_train_purity_weighted", 0.0)),
                    "true_train_purity_weighted": float(diag.get("true_train_purity_weighted", 0.0)),
                    "true_all_purity_weighted": float(diag.get("true_all_purity_weighted", 0.0)),
                }
                rows.append(row)
    return rows


def _default_row(rows: list[dict], phase: str, dataset: str, variant: str) -> dict:
    for row in rows:
        if (
            row["phase"] == phase
            and row["dataset"] == dataset.lower()
            and row["variant"] == variant
            and int(row["max_balls"]) == 64
            and int(row["min_ball_size"]) == 32
            and abs(float(row["split_improvement"]) - 0.05) < 1e-12
        ):
            return row
    raise KeyError(f"missing default row for {phase}/{dataset}/{variant}")


def _best_candidate_row(rows: list[dict], phase: str, dataset: str, variant: str, anchor: dict) -> dict:
    subset = [
        row
        for row in rows
        if row["phase"] == phase and row["dataset"] == dataset.lower() and row["variant"] == variant
    ]
    if not subset:
        raise KeyError(f"missing rows for {phase}/{dataset}/{variant}")

    anchor_balls = float(anchor["num_balls"])
    anchor_purity = float(anchor["true_all_purity_weighted"])

    def score(row: dict) -> tuple[int, float, int, float, float]:
        count_ok = float(row["num_balls"]) >= 0.75 * anchor_balls
        purity_gap = float(row["true_all_purity_weighted"]) - anchor_purity
        rescued = count_ok and purity_gap >= -0.05
        return (
            1 if rescued else 0,
            purity_gap,
            int(row["num_balls"]),
            float(row["true_train_purity_weighted"]),
            -float(row["split_improvement"]),
        )

    return max(subset, key=score)


def _decision_for_candidate(anchor: dict, candidate: dict) -> dict:
    anchor_balls = float(anchor["num_balls"])
    anchor_purity = float(anchor["true_all_purity_weighted"])
    count_ok = float(candidate["num_balls"]) >= 0.75 * anchor_balls
    purity_ok = float(candidate["true_all_purity_weighted"]) >= anchor_purity - 0.05
    return {
        "phase": candidate["phase"],
        "dataset": candidate["dataset"],
        "candidate_variant": candidate["variant"],
        "anchor_variant": anchor["variant"],
        "anchor_param": anchor["param"],
        "candidate_param": candidate["param"],
        "anchor_num_balls": anchor["num_balls"],
        "candidate_num_balls": candidate["num_balls"],
        "anchor_true_all_purity_weighted": anchor["true_all_purity_weighted"],
        "candidate_true_all_purity_weighted": candidate["true_all_purity_weighted"],
        "candidate_minus_anchor_true_all_purity": (
            float(candidate["true_all_purity_weighted"]) - float(anchor["true_all_purity_weighted"])
        ),
        "count_ok": bool(count_ok),
        "purity_ok": bool(purity_ok),
        "gate": "pass" if count_ok and purity_ok else "fail",
    }


def _write_report(path: Path, raw_rows: list[dict], best_rows: list[dict], decisions: list[dict]) -> None:
    lines = [
        "# MLP-8 P2 Ball-Parameter Rescue Diagnostic",
        "",
        "## Protocol",
        "",
        "- Seed: 0",
        "- Noise: uniform 0.4, train-only",
        "- Split: percent 0.2/0.1/0.7",
        "- Scope: geometry diagnostic only; no downstream retraining gate is claimed here.",
        "- Gate: candidate rescued only if ball count recovers to at least 75% of frozen-GRACE default and true-all purity is within 0.05 of frozen-GRACE default.",
        "",
        "## Best Candidate Per P2 Failure Mode",
        "",
        "| Phase | Dataset | Variant | Best param | Num balls | True all purity | Mean train/ball | Alpha+ train | Gate |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    decision_by_key = {
        (row["phase"], row["dataset"], row["candidate_variant"]): row for row in decisions
    }
    for row in best_rows:
        decision = decision_by_key.get((row["phase"], row["dataset"], row["variant"]))
        gate = decision["gate"] if decision else "anchor"
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['variant']} | {row['param']} | "
            f"{row['num_balls']} | {float(row['true_all_purity_weighted']):.4f} | "
            f"{float(row['mean_train_per_ball']):.2f} | {row['alpha_positive_train_count']} | {gate} |"
        )

    lines.extend([
        "",
        "## Gate Decisions",
        "",
        "| Phase | Dataset | Candidate | Param | Balls vs anchor | Purity delta | Count ok | Purity ok | Gate |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ])
    for row in decisions:
        lines.append(
            f"| {row['phase']} | {row['dataset']} | {row['candidate_variant']} | {row['candidate_param']} | "
            f"{row['candidate_num_balls']}/{row['anchor_num_balls']} | "
            f"{float(row['candidate_minus_anchor_true_all_purity']):+.4f} | "
            f"{row['count_ok']} | {row['purity_ok']} | {row['gate']} |"
        )

    pass_count = sum(1 for row in decisions if row["gate"] == "pass")
    lines.extend([
        "",
        "## Decision",
        "",
    ])
    if pass_count:
        lines.append(
            f"- {pass_count}/{len(decisions)} candidate cases pass the geometry rescue gate. These cases may be considered for a separate downstream seed0 check."
        )
    else:
        lines.append(
            "- No P2 candidate passes the geometry rescue gate. Loosening ball construction can be reported as a diagnostic, but it does not justify reopening P2 downstream training."
        )
    lines.extend([
        "- If ball count recovers without true purity recovery, interpret the original P2 failure as representation-level contamination, not merely split-threshold collapse.",
        "",
        "## Files",
        "",
        "- `p2_ballparam_raw.csv`: full parameter grid.",
        "- `p2_ballparam_best.csv`: anchors plus best candidate rows.",
        "- `p2_ballparam_decisions.csv`: gate decisions.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_p2b_z1(args, dataset: str, output_dir: Path) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data, noisy_label, _modified_mask = load_dataset_with_noise(dataset, int(args.seed), args)
    anchor_path = embedding_path_for(args.embedding_root, dataset, "grace")
    anchor_h = load_frozen_embeddings(anchor_path, expected_nodes=int(data.n_nodes))
    f0_target = build_soft_targets_from_features(
        ball_features=anchor_h,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        min_train_per_ball=load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset.lower()).gb["min_train_per_ball"],
        alpha_baseline=load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset.lower()).gb["alpha_baseline"],
        target_mode=load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset.lower()).gb["target_mode"],
        target_beta=load_conf(None, "frozengrace_gb_mlp_softdist_beta05", dataset.lower()).gb["target_beta"],
        random_state=int(args.seed),
        random_control=False,
    ).soft_target
    f0_model, _f0_result = train_mlp_with_soft_targets(data, f0_target, noisy_label, int(args.seed), args.device)
    z1 = extract_mlp_hidden(f0_model, data.feats)
    z1_path = output_dir / "embeddings" / dataset.lower() / "p2b_z1_seed0.pt"
    z1_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z1, z1_path)
    return data, noisy_label, anchor_h, z1, data.labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether looser ball construction rescues MLP-8 P2 representations.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding_root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--mlp8_p2_output_dir", default=str(MLP8_P2_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--max_balls_values", default="64")
    parser.add_argument("--min_ball_size_values", default="16,32,64")
    parser.add_argument("--split_improvement_values", default="0.05,0.01,0.005,0.001,0.0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_balls_values = _parse_csv_ints(args.max_balls_values)
    min_ball_size_values = _parse_csv_ints(args.min_ball_size_values)
    split_improvement_values = _parse_csv_floats(args.split_improvement_values)

    raw_rows: list[dict] = []

    # P2a: Cora frozen GRACE vs GRACE+noisy-CE representation saved by MLP-8.
    p2a_dataset = "cora"
    p2a_data, p2a_noisy_label, _ = load_dataset_with_noise(p2a_dataset, int(args.seed), args)
    p2a_anchor_path = embedding_path_for(args.embedding_root, p2a_dataset, "grace")
    p2a_anchor_h = load_frozen_embeddings(p2a_anchor_path, expected_nodes=int(p2a_data.n_nodes))
    p2a_candidate_path = Path(args.mlp8_p2_output_dir) / "embeddings" / "Cora" / "grace_ce_l0p1" / "H.pt"
    if not p2a_candidate_path.exists():
        raise FileNotFoundError(f"missing P2a candidate embedding: {p2a_candidate_path}")
    p2a_candidate_h = torch.load(p2a_candidate_path, map_location="cpu")
    raw_rows.extend(_scan_embedding(
        phase="p2a",
        dataset_name=p2a_dataset,
        variant="anchor_grace",
        embedding=p2a_anchor_h,
        data=p2a_data,
        noisy_label=p2a_noisy_label,
        seed=int(args.seed),
        max_balls_values=max_balls_values,
        min_ball_size_values=min_ball_size_values,
        split_improvement_values=split_improvement_values,
    ))
    raw_rows.extend(_scan_embedding(
        phase="p2a",
        dataset_name=p2a_dataset,
        variant="grace_ce_l0p1",
        embedding=p2a_candidate_h,
        data=p2a_data,
        noisy_label=p2a_noisy_label,
        seed=int(args.seed),
        max_balls_values=max_balls_values,
        min_ball_size_values=min_ball_size_values,
        split_improvement_values=split_improvement_values,
    ))

    # P2b: reconstruct Z1 exactly from the original F0 training recipe.
    for dataset in ["cora", "citeseer"]:
        data, noisy_label, anchor_h, z1, _clean = _build_p2b_z1(args, dataset, output_dir)
        raw_rows.extend(_scan_embedding(
            phase="p2b",
            dataset_name=dataset,
            variant="F0_graceH",
            embedding=anchor_h,
            data=data,
            noisy_label=noisy_label,
            seed=int(args.seed),
            max_balls_values=max_balls_values,
            min_ball_size_values=min_ball_size_values,
            split_improvement_values=split_improvement_values,
        ))
        raw_rows.extend(_scan_embedding(
            phase="p2b",
            dataset_name=dataset,
            variant="F1_loop_Z1",
            embedding=z1,
            data=data,
            noisy_label=noisy_label,
            seed=int(args.seed),
            max_balls_values=max_balls_values,
            min_ball_size_values=min_ball_size_values,
            split_improvement_values=split_improvement_values,
        ))

    best_rows: list[dict] = []
    decisions: list[dict] = []
    comparisons = [
        ("p2a", "cora", "anchor_grace", "grace_ce_l0p1"),
        ("p2b", "cora", "F0_graceH", "F1_loop_Z1"),
        ("p2b", "citeseer", "F0_graceH", "F1_loop_Z1"),
    ]
    for phase, dataset, anchor_variant, candidate_variant in comparisons:
        anchor = _default_row(raw_rows, phase, dataset, anchor_variant)
        candidate = _best_candidate_row(raw_rows, phase, dataset, candidate_variant, anchor)
        best_rows.extend([anchor, candidate])
        decisions.append(_decision_for_candidate(anchor, candidate))

    _write_csv(output_dir / "p2_ballparam_raw.csv", raw_rows)
    _write_csv(output_dir / "p2_ballparam_best.csv", best_rows)
    _write_csv(output_dir / "p2_ballparam_decisions.csv", decisions)
    _write_report(output_dir / "p2_ballparam_report.md", raw_rows, best_rows, decisions)
    (output_dir / "p2_ballparam_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"WROTE {output_dir / 'p2_ballparam_report.md'}")
    for row in decisions:
        print(
            f"{row['phase']} {row['dataset']} {row['candidate_variant']} "
            f"gate={row['gate']} param={row['candidate_param']} "
            f"balls={row['candidate_num_balls']}/{row['anchor_num_balls']} "
            f"purity_delta={float(row['candidate_minus_anchor_true_all_purity']):+.4f}"
        )


if __name__ == "__main__":
    main()
