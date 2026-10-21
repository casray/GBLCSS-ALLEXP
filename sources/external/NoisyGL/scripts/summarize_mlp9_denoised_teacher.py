from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp9_denoised_teacher_20260702"
MLP8_RUN_DIR = REPO_ROOT / "runs" / "mlp8_replacement_expansion_20260702"

T1 = "mlp_distill_gcn"
T2 = "mlp_distill_gcn_xh"
T3 = "mlp_distill_softdist_xh"
T4 = "mlp_distill_knn_xh"
T5 = "mlp_distill_softdist_xh_shuffled"
DATASETS = ("cora", "citeseer", "pubmed")
NOISE_RATES = (0.4, 0.6)
SEEDS = tuple(range(5))

LABELS = {
    T1: "T1 GLNN teacher=GCN(X)+CE",
    T2: "T2 teacher=GCN[X||H]+CE",
    T3: "T3 teacher=SoftDist-GCN[X||H]",
    T4: "T4 teacher=kNN-GCN[X||H]",
    T5: "T5 teacher=shuffled-SoftDist-GCN[X||H]",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _float(value, default=""):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _load_t1_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for noise_rate in NOISE_RATES:
        token = str(noise_rate).replace(".", "p")
        path = run_dir / f"p1_glnn_uniform{token}_seeds_0.csv"
        for row in read_csv(path):
            if row.get("status", "ok").lower() not in {"", "ok"}:
                continue
            rows.append({
                "dataset": row["dataset"].lower(),
                "method": T1,
                "method_label": LABELS[T1],
                "seed": int(row["seed"]),
                "noise_type": row["noise_type"].lower(),
                "noise_rate": float(row["noise_rate"]),
                "student_test": float(row["test"]),
                "teacher_test": _float(row.get("teacher_best_test")),
                "teacher_best_val": _float(row.get("teacher_best_val")),
                "teacher_target": "ce",
                "teacher_use_xh": 0,
                "teacher_input_dim": "",
                "student_input_dim": "",
                "matched_k": "",
                "matched_num_balls": "",
                "target_builder": "",
                "source": str(path),
            })
    return rows


def _load_new_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("mlp9_uniform*.csv")):
        for row in read_csv(path):
            method = row.get("method") or row.get("base_method")
            if method not in {T1, T2, T3, T4, T5}:
                continue
            if row.get("status", "ok").lower() not in {"", "ok"}:
                continue
            rows.append({
                "dataset": row["dataset"].lower(),
                "method": method,
                "method_label": LABELS[method],
                "seed": int(row["seed"]),
                "noise_type": row["noise_type"].lower(),
                "noise_rate": float(row["noise_rate"]),
                "student_test": float(row["test"]),
                "teacher_test": _float(row.get("teacher_best_test")),
                "teacher_best_val": _float(row.get("teacher_best_val")),
                "teacher_target": row.get("teacher_target", ""),
                "teacher_use_xh": _float(row.get("teacher_use_xh")),
                "teacher_input_dim": _float(row.get("teacher_input_dim")),
                "student_input_dim": _float(row.get("student_input_dim")),
                "matched_k": _float(row.get("matched_k")),
                "matched_num_balls": _float(row.get("matched_num_balls")),
                "target_builder": row.get("target_builder") or row.get("soft_target_builder") or "",
                "source": str(path),
            })
    return rows


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for row in rows:
        key = (row["dataset"], float(row["noise_rate"]), row["method"], int(row["seed"]))
        if key in unique:
            old = float(unique[key]["student_test"])
            new = float(row["student_test"])
            if abs(old - new) > 1e-12:
                raise ValueError(f"conflicting duplicate row for {key}: {old} vs {new}")
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda item: (item["noise_rate"], item["dataset"], item["method"], item["seed"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    aux = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], row["method_label"])
        groups[key].append(float(row["student_test"]))
        aux[key].append(row)
    out = []
    for (dataset, noise_type, noise_rate, method, label), values in sorted(groups.items()):
        teacher_values = [row["teacher_test"] for row in aux[(dataset, noise_type, noise_rate, method, label)] if row["teacher_test"] != ""]
        out.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "method_label": label,
            "n": len(values),
            "student_mean": statistics.mean(values),
            "student_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "teacher_mean": "" if not teacher_values else statistics.mean(teacher_values),
            "teacher_std": "" if len(teacher_values) <= 1 else statistics.stdev(teacher_values),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(aux[(dataset, noise_type, noise_rate, method, label)], key=lambda item: int(item["seed"]))),
        })
    return out


COMPARISONS = [
    ("G1_win_glnn", T3, T1, 0.01, "ge", "T3-student - T1-student >= +1pp"),
    ("G2_denoising_attribution", T3, T2, 0.0, "gt", "T3-student - T2-student > 0"),
    ("G3_ball_attribution", T3, T4, 0.01, "ge", "T3-student - T4-student >= +1pp"),
    ("G4_structure_sanity", T3, T5, 0.0, "gt", "T3-student - T5-student > 0"),
    ("D1_h_input_increment", T2, T1, 0.0, "gt", "T2-student - T1-student > 0"),
    ("D2_knn_denoising_increment", T4, T2, 0.0, "gt", "T4-student - T2-student > 0"),
]


def _passes(delta: float, threshold: float, mode: str) -> bool:
    if mode == "ge":
        return delta >= threshold
    if mode == "gt":
        return delta > threshold
    raise ValueError(f"unknown threshold mode: {mode}")


def paired_delta_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["student_test"])
        for row in rows
    }
    output = []
    for noise_rate in NOISE_RATES:
        for comparison, left_method, right_method, threshold, mode, rule in COMPARISONS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    left = values.get((dataset, noise_rate, left_method, seed))
                    right = values.get((dataset, noise_rate, right_method, seed))
                    if left is None or right is None:
                        continue
                    delta = left - right
                    output.append({
                        "noise_rate": noise_rate,
                        "dataset": dataset,
                        "comparison": comparison,
                        "left_method": left_method,
                        "right_method": right_method,
                        "seed": seed,
                        "left_student_test": left,
                        "right_student_test": right,
                        "delta": delta,
                        "threshold": threshold,
                        "threshold_mode": mode,
                        "seed_pass": int(_passes(delta, threshold, mode)),
                        "rule": rule,
                    })
    return output


def gate_rows(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    gate_names = {item[0] for item in COMPARISONS[:4]}
    grouped = defaultdict(list)
    for row in paired_rows:
        if row["comparison"] not in gate_names:
            continue
        grouped[(float(row["noise_rate"]), row["comparison"], row["dataset"])].append(row)

    for noise_rate in NOISE_RATES:
        for gate, _, _, threshold, mode, rule in COMPARISONS[:4]:
            dataset_summaries = []
            all_deltas = []
            dataset_pass_count = 0
            for dataset in DATASETS:
                seed_rows = sorted(grouped.get((noise_rate, gate, dataset), []), key=lambda item: int(item["seed"]))
                deltas = [float(row["delta"]) for row in seed_rows]
                all_deltas.extend(deltas)
                pass_count = sum(1 for delta in deltas if _passes(delta, threshold, mode))
                dataset_pass = int(len(deltas) == len(SEEDS) and pass_count >= 4)
                dataset_pass_count += dataset_pass
                mean_delta = "" if not deltas else statistics.mean(deltas)
                std_delta = "" if len(deltas) <= 1 else statistics.stdev(deltas)
                dataset_summaries.append(
                    f"{dataset}:{_fmt_delta(mean_delta)} {pass_count}/{len(deltas)}"
                )
                output.append({
                    "scope": "dataset",
                    "noise_rate": noise_rate,
                    "gate": gate,
                    "dataset": dataset,
                    "rule": rule,
                    "n": len(deltas),
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "seed_pass_count": pass_count,
                    "dataset_pass": dataset_pass,
                    "dataset_pass_count": "",
                    "decision": "pass" if dataset_pass else "not-pass",
                    "details": ",".join(str(row["seed"]) for row in seed_rows),
                })
            decision = "pass" if dataset_pass_count >= 2 else "not-pass"
            output.append({
                "scope": "overall",
                "noise_rate": noise_rate,
                "gate": gate,
                "dataset": "all",
                "rule": rule,
                "n": len(all_deltas),
                "mean_delta": "" if not all_deltas else statistics.mean(all_deltas),
                "std_delta": "" if len(all_deltas) <= 1 else statistics.stdev(all_deltas),
                "seed_pass_count": "",
                "dataset_pass": "",
                "dataset_pass_count": dataset_pass_count,
                "decision": decision,
                "details": "; ".join(dataset_summaries),
            })
    return output


def decomposition_rows(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in paired_rows:
        if row["comparison"] not in {"D1_h_input_increment", "D2_knn_denoising_increment"}:
            continue
        grouped[(float(row["noise_rate"]), row["comparison"], row["dataset"])].append(float(row["delta"]))
    out = []
    for (noise_rate, comparison, dataset), deltas in sorted(grouped.items()):
        out.append({
            "noise_rate": noise_rate,
            "comparison": comparison,
            "dataset": dataset,
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "positive_count": sum(1 for delta in deltas if delta > 0),
        })
    return out


def knn_matched_k_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row["method"] != T4:
            continue
        out.append({
            "dataset": row["dataset"],
            "noise_rate": float(row["noise_rate"]),
            "seed": int(row["seed"]),
            "matched_k": row.get("matched_k", ""),
            "matched_num_balls": row.get("matched_num_balls", ""),
            "student_test": row.get("student_test", ""),
            "teacher_test": row.get("teacher_test", ""),
        })
    return sorted(out, key=lambda item: (item["noise_rate"], item["dataset"], item["seed"]))


def _fmt_acc(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):.2f}"


def _fmt_delta(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):+.2f}pp"


def final_branch(gates: list[dict[str, object]]) -> str:
    u04 = {
        row["gate"]: row
        for row in gates
        if row.get("scope") == "overall" and float(row["noise_rate"]) == 0.4
    }
    g1 = u04.get("G1_win_glnn", {}).get("decision")
    g2 = u04.get("G2_denoising_attribution", {}).get("decision")
    g3 = u04.get("G3_ball_attribution", {}).get("decision")
    if g1 != "pass":
        return "close_probe_g1_failed"
    if g2 != "pass":
        return "close_probe_g2_failed"
    if g3 == "pass":
        return "expand_t3_to_5seed"
    return "denoised_teacher_possible_but_ball_replaced_by_knn"


def write_report(path: Path, rows, agg_rows, paired_rows, gates, decomp_rows, knn_rows) -> None:
    branch = final_branch(gates)
    seed0_rows = [row for row in rows if int(row["seed"]) == 0]
    lines = [
        "# MLP-9 Denoised Teacher Distillation Report",
        "",
        "## Material Passport",
        "- Artifact type: 5-seed experiment result with seed0 reuse",
        "- Protocol: train-only uniform label noise, percent split 0.2/0.1/0.7",
        "- Student: X-only 2-layer weak MLP for every arm",
        "- Teacher: GCN variants; T2-T5 use [X||H] only inside teacher",
        "- Clean labels: evaluation only; teacher and target construction use noisy train labels",
        "",
        "## Seed0 Snapshot",
        "| Dataset | Noise | Arm | Teacher test | Student test | teacher target | matched k |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for row in sorted(seed0_rows, key=lambda item: (item["noise_rate"], item["dataset"], item["method"])):
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']} | {row['method_label']} | "
            f"{_fmt_acc(row['teacher_test'])} | {_fmt_acc(row['student_test'])} | "
            f"{row.get('teacher_target', '')} | {row.get('matched_k', '')} |"
        )
    lines.extend([
        "",
        "## 5-seed Aggregate",
        "| Dataset | Noise | Arm | n | Student mean +/- std | Teacher mean +/- std | seeds |",
        "|---|---:|---|---:|---:|---:|---|",
    ])
    for row in agg_rows:
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']} | {row['method_label']} | "
            f"{row['n']} | {_fmt_acc(row['student_mean'])}+/-{_fmt_acc(row['student_std'])} | "
            f"{_fmt_acc(row['teacher_mean'])}+/-{_fmt_acc(row['teacher_std'])} | {row.get('seeds', '')} |"
        )
    lines.extend([
        "",
        "## 5-seed Pre-registered Gates",
        "| Noise | Scope | Dataset | Gate | Mean delta | Std | Seed pass | Dataset pass count | Decision | Details |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in gates:
        pass_text = "" if row["seed_pass_count"] == "" else f"{row['seed_pass_count']}/{row['n']}"
        lines.append(
            f"| {row['noise_rate']} | {row['scope']} | {row['dataset']} | {row['gate']} | "
            f"{_fmt_delta(row['mean_delta'])} | {_fmt_delta(row['std_delta'])} | {pass_text} | "
            f"{row['dataset_pass_count']} | {row['decision']} | {row['details']} |"
        )
    lines.extend([
        "",
        "## Decomposition Deltas",
        "| Noise | Dataset | Comparison | Mean delta | Std | Positive |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for row in decomp_rows:
        lines.append(
            f"| {row['noise_rate']} | {row['dataset']} | {row['comparison']} | "
            f"{_fmt_delta(row['mean_delta'])} | {_fmt_delta(row['std_delta'])} | "
            f"{row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## kNN Matched-k Audit",
        "| Noise | Dataset | Seed | matched k | matched balls | Student test | Teacher test |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in knn_rows:
        lines.append(
            f"| {row['noise_rate']} | {row['dataset']} | {row['seed']} | "
            f"{row.get('matched_k', '')} | {row.get('matched_num_balls', '')} | "
            f"{_fmt_acc(row.get('student_test', ''))} | {_fmt_acc(row.get('teacher_test', ''))} |"
        )
    lines.extend([
        "",
        "## Branch Decision",
        f"Decision: **{branch}**",
        "",
        "## Research Record",
        "- Hypothesis: a denoised GCN[X||H] teacher can transmit teacher-side label denoising gains to an X-only MLP student.",
        "- Granular-ball role: train-label soft target for the teacher only; the student never consumes H or ball targets directly.",
        "- Success standard: on uniform 0.4, each dataset passes a gate when at least 4/5 paired seeds satisfy the rule; a gate passes overall when at least 2/3 datasets pass.",
        "- Failure standard: if G1 or G2 fails, the denoised-teacher probe is closed; if G3 fails, kNN replacement remains the teacher-layer explanation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--mlp8_run_dir", default=str(MLP8_RUN_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = deduplicate(_load_t1_rows(Path(args.mlp8_run_dir)) + _load_new_rows(run_dir))
    agg_rows = aggregate(rows)
    paired_rows = paired_delta_rows(rows)
    gates = gate_rows(paired_rows)
    decomp_rows = decomposition_rows(paired_rows)
    knn_rows = knn_matched_k_rows(rows)
    write_csv(run_dir / "mlp9_combined_raw.csv", rows)
    write_csv(run_dir / "mlp9_aggregate.csv", agg_rows)
    write_csv(run_dir / "mlp9_paired_deltas.csv", paired_rows)
    write_csv(run_dir / "mlp9_gates.csv", gates)
    write_csv(run_dir / "mlp9_decomposition_deltas.csv", decomp_rows)
    write_csv(run_dir / "mlp9_knn_matched_k.csv", knn_rows)
    write_report(run_dir / "mlp9_report.md", rows, agg_rows, paired_rows, gates, decomp_rows, knn_rows)
    print(f"MLP9_ROWS={len(rows)}")
    print(f"MLP9_BRANCH={final_branch(gates)}")
    print(f"WROTE {run_dir / 'mlp9_report.md'}")


if __name__ == "__main__":
    main()
