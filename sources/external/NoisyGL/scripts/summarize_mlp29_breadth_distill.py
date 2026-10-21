import argparse
import csv
import statistics
from pathlib import Path


DATASETS = ("corafull", "coauthorcs", "amazonpho", "pubmed", "amazoncom", "coauthorph")

LCSS = "frozengrace_gb_mlp_softdist_beta05"
SMALL_ONLY = "mlp_small_loss"
SMALL_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SMALL_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"
DIRECT_METHODS = (LCSS, SMALL_ONLY, SMALL_COMBO, SMALL_SHUFFLED)

DISTILL_BARE = "mlp_distill_gcn_xh"
DISTILL_SOFT = "mlp_distill_softdist_beta05_xh"
DISTILL_KNN = "mlp_distill_knn_beta05_xh"
DISTILL_ROBUST = "mlp_distill_softdist_beta05_small_loss_xh"
DISTILL_ROBUST_SHUFFLED = "mlp_distill_softdist_beta05_small_loss_xh_shuffled"
DISTILL_METHODS = (
    DISTILL_BARE,
    DISTILL_SOFT,
    DISTILL_KNN,
    DISTILL_ROBUST,
    DISTILL_ROBUST_SHUFFLED,
)


def _validate(rows, methods, phase):
    selected = [row for row in rows if row.get("dataset") in DATASETS and row.get("method") in methods]
    by_key = {}
    for row in selected:
        if row.get("status") not in {None, "", "ok"}:
            raise ValueError(f"non-ok {phase} row: {row}")
        if row.get("noise_type") and row["noise_type"] != "uniform":
            raise ValueError("MLP-29 requires uniform noise")
        if row.get("noise_rate") and abs(float(row["noise_rate"]) - 0.6) > 1e-12:
            raise ValueError("MLP-29 requires noise_rate=0.6")
        if row.get("noise_scope") and row["noise_scope"] != "train":
            raise ValueError("MLP-29 requires train-only noise")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError("validation/test labels were modified")
        by_key[(row["dataset"], row["method"], int(row["seed"]))] = row
    for dataset in DATASETS:
        hashes = set()
        for method in methods:
            key = (dataset, method, 0)
            if key not in by_key:
                raise ValueError(f"missing {phase} row: {key}")
            row = by_key[key]
            if row.get("split_hash"):
                hashes.add((row["split_hash"], row["noise_hash"]))
        if len(hashes) > 1:
            raise ValueError(f"{phase} protocol hash mismatch: {dataset}")
    return by_key


def build_summary(direct_rows, distill_rows):
    direct = _validate(direct_rows, DIRECT_METHODS, "direct")
    distill = _validate(distill_rows, DISTILL_METHODS, "distill")
    direct_deltas = []
    distill_deltas = []

    for dataset in DATASETS:
        value = lambda table, method: float(table[(dataset, method, 0)]["test"])
        lcss = value(direct, LCSS)
        small = value(direct, SMALL_ONLY)
        combo = value(direct, SMALL_COMBO)
        shuffled = value(direct, SMALL_SHUFFLED)
        direct_deltas.append(
            {
                "dataset": dataset,
                "lcss": lcss,
                "small_only": small,
                "combo": combo,
                "shuffled_combo": shuffled,
                "synergy_delta": combo - max(lcss, small),
                "structure_delta": combo - shuffled,
            }
        )

        bare = value(distill, DISTILL_BARE)
        soft = value(distill, DISTILL_SOFT)
        knn = value(distill, DISTILL_KNN)
        robust = value(distill, DISTILL_ROBUST)
        robust_shuffled = value(distill, DISTILL_ROBUST_SHUFFLED)
        distill_deltas.append(
            {
                "dataset": dataset,
                "bare_teacher": bare,
                "soft_teacher": soft,
                "knn_teacher": knn,
                "robust_teacher": robust,
                "robust_teacher_shuffled": robust_shuffled,
                "robust_minus_soft": robust - soft,
                "robust_minus_bare": robust - bare,
                "robust_minus_knn": robust - knn,
                "structure_delta": robust - robust_shuffled,
            }
        )

    direct_synergy = [row["synergy_delta"] for row in direct_deltas]
    direct_structure = [row["structure_delta"] for row in direct_deltas]
    direct_reasons = []
    if sum(value > 0 for value in direct_synergy) < 4:
        direct_reasons.append("fewer_than_4_of_6_synergy_positive")
    if statistics.mean(direct_synergy) < 0.01:
        direct_reasons.append("mean_synergy_below_1pp")
    if min(direct_synergy) < -0.01:
        direct_reasons.append("regression_over_1pp")
    if sum(value >= 0.01 for value in direct_structure) < 4:
        direct_reasons.append("fewer_than_4_of_6_structure_gain_1pp")

    teacher_gain = [row["robust_minus_soft"] for row in distill_deltas]
    teacher_structure = [row["structure_delta"] for row in distill_deltas]
    distill_reasons = []
    if sum(value > 0 for value in teacher_gain) < 4:
        distill_reasons.append("fewer_than_4_of_6_above_plain_teacher")
    if statistics.mean(teacher_gain) < 0.005:
        distill_reasons.append("mean_teacher_gain_below_0p5pp")
    if min(teacher_gain) < -0.01:
        distill_reasons.append("teacher_regression_over_1pp")
    if sum(value >= 0.01 for value in teacher_structure) < 4:
        distill_reasons.append("fewer_than_4_of_6_teacher_structure_gain_1pp")

    gates = [
        {
            "phase": "direct",
            "pass": int(not direct_reasons),
            "positive_datasets": sum(value > 0 for value in direct_synergy),
            "mean_primary_delta": statistics.mean(direct_synergy),
            "minimum_primary_delta": min(direct_synergy),
            "structure_gain_1pp_datasets": sum(value >= 0.01 for value in direct_structure),
            "reasons": ";".join(direct_reasons),
        },
        {
            "phase": "distill",
            "pass": int(not distill_reasons),
            "positive_datasets": sum(value > 0 for value in teacher_gain),
            "mean_primary_delta": statistics.mean(teacher_gain),
            "minimum_primary_delta": min(teacher_gain),
            "structure_gain_1pp_datasets": sum(value >= 0.01 for value in teacher_structure),
            "reasons": ";".join(distill_reasons),
        },
    ]
    return direct_deltas, distill_deltas, gates


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(path, direct, distill, gates):
    lines = [
        "# MLP-29 Breadth and Robust-Teacher Distillation Seed0",
        "",
        "## Direct X-only MLP",
        "",
        "| Dataset | LCSS | Small-loss | Combination | Shuffled | Synergy | Real-shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in direct:
        lines.append(
            f"| {row['dataset']} | {100*row['lcss']:.2f} | {100*row['small_only']:.2f} | "
            f"{100*row['combo']:.2f} | {100*row['shuffled_combo']:.2f} | "
            f"{100*row['synergy_delta']:+.2f}pp | {100*row['structure_delta']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## X-only student after teacher distillation",
            "",
            "| Dataset | Bare | Beta0.5 SoftDist | Beta0.5 kNN | Robust SoftDist | Robust shuffled | Robust-plain | Robust-kNN | Real-shuffled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in distill:
        lines.append(
            f"| {row['dataset']} | {100*row['bare_teacher']:.2f} | {100*row['soft_teacher']:.2f} | "
            f"{100*row['knn_teacher']:.2f} | {100*row['robust_teacher']:.2f} | "
            f"{100*row['robust_teacher_shuffled']:.2f} | {100*row['robust_minus_soft']:+.2f}pp | "
            f"{100*row['robust_minus_knn']:+.2f}pp | {100*row['structure_delta']:+.2f}pp |"
        )
    lines.extend(["", "## Gates", ""])
    for row in gates:
        lines.append(
            f"- {row['phase']}: pass={row['pass']}, positive={row['positive_datasets']}/6, "
            f"mean={100*row['mean_primary_delta']:+.2f}pp, min={100*row['minimum_primary_delta']:+.2f}pp, "
            f"structure>=1pp={row['structure_gain_1pp_datasets']}/6, reasons={row['reasons'] or 'pass'}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_input", required=True)
    parser.add_argument("--distill_input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    direct, distill, gates = build_summary(_read(args.direct_input), _read(args.distill_input))
    output = Path(args.output_dir)
    _write(output / "mlp29_direct_deltas.csv", direct)
    _write(output / "mlp29_distill_deltas.csv", distill)
    _write(output / "mlp29_gate_decisions.csv", gates)
    _report(output / "mlp29_report.md", direct, distill, gates)


if __name__ == "__main__":
    main()
