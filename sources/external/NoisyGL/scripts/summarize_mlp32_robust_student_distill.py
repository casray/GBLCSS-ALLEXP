import argparse
import csv
import statistics
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull", "amazonpho", "pubmed", "coauthorph")
DIRECT = "frozengrace_gb_mlp_softdist_beta05_small_loss"
PLAIN = "mlp_distill_gcn_xh"
ROBUST = "mlp_distill_gcn_xh_lcss_small_loss"
SHUFFLED = "mlp_distill_gcn_xh_lcss_small_loss_shuffled"


def _validate(row):
    if row.get("status") != "ok":
        raise ValueError(f"non-ok row: {row.get('dataset')} {row.get('method')}")
    if row.get("noise_type") != "uniform" or abs(float(row["noise_rate"]) - 0.4) > 1e-12:
        raise ValueError("MLP-32 requires uniform 0.4")
    if row.get("noise_scope") != "train" or int(row["seed"]) != 0:
        raise ValueError("MLP-32 requires train-only seed0")
    if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
        raise ValueError("validation/test labels were modified")
    if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
        raise ValueError("modified labels are not restricted to train")


def build_summary(direct_rows, distill_rows):
    keyed = {}
    for row in direct_rows + distill_rows:
        if row.get("dataset") not in DATASETS or row.get("method") not in {DIRECT, PLAIN, ROBUST, SHUFFLED}:
            continue
        _validate(row)
        key = (row["dataset"], row["method"])
        if key in keyed:
            raise ValueError(f"duplicate row: {key}")
        keyed[key] = row

    deltas = []
    for dataset in DATASETS:
        rows = {}
        for method in (DIRECT, PLAIN, ROBUST, SHUFFLED):
            key = (dataset, method)
            if key not in keyed:
                raise ValueError(f"missing row: {key}")
            rows[method] = keyed[key]
        hashes = {(row["split_hash"], row["noise_hash"]) for row in rows.values()}
        if len(hashes) != 1:
            raise ValueError(f"protocol hash mismatch: {dataset}")
        direct = float(rows[DIRECT]["test"])
        plain = float(rows[PLAIN]["test"])
        robust = float(rows[ROBUST]["test"])
        shuffled = float(rows[SHUFFLED]["test"])
        teacher = float(rows[ROBUST]["teacher_best_test"])
        deltas.append(
            {
                "dataset": dataset,
                "direct": direct,
                "plain_distill": plain,
                "robust_student": robust,
                "shuffled_student": shuffled,
                "teacher_test": teacher,
                "robust_minus_plain": robust - plain,
                "robust_minus_direct": robust - direct,
                "real_minus_shuffled": robust - shuffled,
                "teacher_retention": robust / teacher if teacher > 0 else 0.0,
            }
        )

    above_plain = sum(row["robust_minus_plain"] > 0 for row in deltas)
    mean_plain_gain = statistics.mean(row["robust_minus_plain"] for row in deltas)
    above_direct = sum(row["robust_minus_direct"] > 0 for row in deltas)
    structure = sum(row["real_minus_shuffled"] >= 0.01 for row in deltas)
    retention = sum(row["teacher_retention"] >= 0.95 for row in deltas)
    regressions = sum(row["robust_minus_plain"] < -0.01 for row in deltas)
    reasons = []
    if above_plain < 4:
        reasons.append("fewer_than_4_of_6_above_plain_distillation")
    if mean_plain_gain < 0.01:
        reasons.append("mean_plain_distillation_gain_below_1pp")
    if above_direct < 4:
        reasons.append("fewer_than_4_of_6_above_direct_method")
    if structure < 4:
        reasons.append("fewer_than_4_of_6_structure_gains_1pp")
    if retention < 5:
        reasons.append("fewer_than_5_of_6_teacher_retention_95pct")
    if regressions:
        reasons.append("plain_distillation_regression_over_1pp")
    gate = {
        "pass": int(not reasons),
        "above_plain_datasets": above_plain,
        "mean_robust_minus_plain": mean_plain_gain,
        "above_direct_datasets": above_direct,
        "structure_1pp_datasets": structure,
        "teacher_retention_95pct_datasets": retention,
        "regressions_over_1pp": regressions,
        "reasons": ";".join(reasons),
    }
    return deltas, gate


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(path, deltas, gate):
    lines = [
        "# MLP-32 Robust-Student Distillation Seed0 Report",
        "",
        "| Dataset | Direct | Plain distill | Robust student | Shuffled | Teacher | vs plain | vs direct | real-shuffled | retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {100*row['direct']:.2f} | {100*row['plain_distill']:.2f} | "
            f"{100*row['robust_student']:.2f} | {100*row['shuffled_student']:.2f} | "
            f"{100*row['teacher_test']:.2f} | {100*row['robust_minus_plain']:+.2f}pp | "
            f"{100*row['robust_minus_direct']:+.2f}pp | {100*row['real_minus_shuffled']:+.2f}pp | "
            f"{100*row['teacher_retention']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Pass: {gate['pass']}",
            f"- Above plain distillation: {gate['above_plain_datasets']}/6",
            f"- Mean robust minus plain: {100*gate['mean_robust_minus_plain']:+.2f}pp",
            f"- Above direct method: {gate['above_direct_datasets']}/6",
            f"- Real-shuffled >=1pp: {gate['structure_1pp_datasets']}/6",
            f"- Teacher retention >=95%: {gate['teacher_retention_95pct_datasets']}/6",
            f"- Regressions over 1pp: {gate['regressions_over_1pp']}",
            f"- Reasons: {gate['reasons'] or 'pass'}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True)
    parser.add_argument("--distill", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    direct_rows = _read(args.direct)
    distill_rows = []
    for path in args.distill:
        distill_rows.extend(_read(path))
    deltas, gate = build_summary(direct_rows, distill_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "mlp32_deltas.csv", deltas)
    _write(output / "mlp32_gate.csv", [gate])
    _report(output / "mlp32_report.md", deltas, gate)


if __name__ == "__main__":
    main()
