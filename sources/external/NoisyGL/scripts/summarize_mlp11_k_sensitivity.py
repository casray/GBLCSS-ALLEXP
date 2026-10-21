from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp11_k_sensitivity_20260702"
MLP5_RUN_DIR = REPO_ROOT / "runs" / "mlp_maintable_20260701"
MLP8_RUN_DIR = REPO_ROOT / "runs" / "mlp8_replacement_expansion_20260702"
MLP10_RUN_DIR = REPO_ROOT / "runs" / "mlp10_baseline_refresh_20260702"

KNN = "knn_mlp_softdist_beta05"
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
DATASETS = ("cora", "citeseer", "pubmed", "coauthorcs")
K_VALUES = (3, 5, 10, 20, 50, 100)
SEEDS = tuple(range(5))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _float(value, default=""):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def normalize(row: dict[str, str], source: Path, source_group: str) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if not method:
        return None
    test = row.get("test") or row.get("test_acc") or ""
    if test == "":
        return None
    return {
        "source_group": source_group,
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "seed": int(row["seed"]),
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "status": row.get("status", "ok").strip().lower(),
        "valid": _float(row.get("valid")),
        "test": float(test),
        "matched_k": _float(row.get("matched_k")),
        "matched_num_balls": _float(row.get("matched_num_balls")),
        "matched_mean_train_per_ball": _float(row.get("matched_mean_train_per_ball")),
        "target_builder": row.get("target_builder") or row.get("soft_target_builder") or "",
        "source": str(source),
    }


def load_new_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("stageA_knn_k*_uniform0p4_seeds_*.csv")):
        for raw in read_csv(path):
            row = normalize(raw, path, "mlp11")
            if row is None or row["method"] != KNN:
                continue
            rows.append(row)
    return rows


def load_reference_rows(mlp5_dir: Path, mlp10_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in read_csv(mlp5_dir / "main_uniform0p4_seeds_0_1_2_3_4.csv"):
        row = normalize(raw, mlp5_dir / "main_uniform0p4_seeds_0_1_2_3_4.csv", "mlp5")
        if row and row["dataset"] in {"cora", "citeseer", "pubmed"} and row["method"] in {OURS, SHUFFLED}:
            rows.append(row)
    for raw in read_csv(mlp10_dir / "mlp10_combined_raw.csv"):
        row = normalize(raw, mlp10_dir / "mlp10_combined_raw.csv", "mlp10")
        if (
            row
            and row["dataset"] == "coauthorcs"
            and row["method"] in {OURS, SHUFFLED}
            and row["noise_type"] == "uniform"
            and math.isclose(float(row["noise_rate"]), 0.4, abs_tol=1e-12)
        ):
            rows.append(row)
    return rows


def load_p0_rows(mlp8_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = mlp8_dir / "p0_replacement_uniform0p4_seeds_0_1_2_3_4.csv"
    for raw in read_csv(path):
        row = normalize(raw, path, "mlp8_p0")
        if row and row["method"] == KNN:
            rows.append(row)
    return rows


def aggregate_knn(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] not in {"", "ok"}:
            continue
        grouped[(row["dataset"], int(row["matched_k"]))].append(row)
    out = []
    for (dataset, k), items in sorted(grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), item[0][1])):
        tests = [float(row["test"]) for row in items]
        valids = [float(row["valid"]) for row in items if row["valid"] != ""]
        out.append({
            "dataset": dataset,
            "k": k,
            "n": len(items),
            "test_mean": statistics.mean(tests),
            "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "valid_mean": statistics.mean(valids) if valids else "",
            "valid_std": statistics.stdev(valids) if len(valids) > 1 else 0.0,
            "seeds": " ".join(str(row["seed"]) for row in sorted(items, key=lambda x: int(x["seed"]))),
        })
    return out


def aggregate_reference(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] in {"", "ok"}:
            grouped[(row["dataset"], row["method"])].append(row)
    out = []
    for (dataset, method), items in sorted(grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), item[0][1])):
        tests = [float(row["test"]) for row in items]
        out.append({
            "dataset": dataset,
            "method": method,
            "n": len(items),
            "test_mean": statistics.mean(tests),
            "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "seeds": " ".join(str(row["seed"]) for row in sorted(items, key=lambda x: int(x["seed"]))),
        })
    return out


def best_k_rows(curve_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        by_dataset[row["dataset"]].append(row)
    out = []
    for dataset in DATASETS:
        candidates = by_dataset.get(dataset, [])
        if not candidates:
            continue
        best = sorted(candidates, key=lambda row: (-float(row["test_mean"]), int(row["k"])))[0]
        out.append({
            "dataset": dataset,
            "best_k": int(best["k"]),
            "best_test_mean": float(best["test_mean"]),
            "best_test_std": float(best["test_std"]),
            "best_valid_mean": best["valid_mean"],
        })
    return out


def global_k_rows(curve_rows: list[dict[str, object]], best_rows: list[dict[str, object]]) -> tuple[int | None, list[dict[str, object]]]:
    by_k: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        by_k[int(row["k"])].append(row)
    pooled = []
    for k, items in by_k.items():
        if len({row["dataset"] for row in items}) == len(DATASETS):
            pooled.append((statistics.mean(float(row["test_mean"]) for row in items), k))
    if not pooled:
        return None, []
    _, global_k = sorted(pooled, key=lambda item: (-item[0], item[1]))[0]
    curve_by_dataset_k = {(row["dataset"], int(row["k"])): row for row in curve_rows}
    best_by_dataset = {row["dataset"]: row for row in best_rows}
    out = []
    for dataset in DATASETS:
        global_row = curve_by_dataset_k.get((dataset, global_k))
        best = best_by_dataset.get(dataset)
        if global_row is None or best is None:
            continue
        loss = float(best["best_test_mean"]) - float(global_row["test_mean"])
        out.append({
            "dataset": dataset,
            "global_k": global_k,
            "global_test_mean": float(global_row["test_mean"]),
            "best_k": int(best["best_k"]),
            "best_test_mean": float(best["best_test_mean"]),
            "loss_vs_best": loss,
            "loss_ge_2pp": int(loss >= 0.02),
        })
    return global_k, out


def r3_val_selection_rows(curve_rows: list[dict[str, object]], best_rows: list[dict[str, object]], ref_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in curve_rows:
        by_dataset[row["dataset"]].append(row)
    best_by_dataset = {row["dataset"]: row for row in best_rows}
    ball_by_dataset = {
        row["dataset"]: row for row in ref_rows if row["method"] == OURS
    }
    out = []
    for dataset in DATASETS:
        candidates = [row for row in by_dataset.get(dataset, []) if row["valid_mean"] != ""]
        if not candidates:
            continue
        selected = sorted(candidates, key=lambda row: (-float(row["valid_mean"]), int(row["k"])))[0]
        best = best_by_dataset.get(dataset)
        ball = ball_by_dataset.get(dataset)
        out.append({
            "dataset": dataset,
            "val_selected_k": int(selected["k"]),
            "val_selected_valid_mean": float(selected["valid_mean"]),
            "val_selected_test_mean": float(selected["test_mean"]),
            "test_best_k": "" if best is None else int(best["best_k"]),
            "test_best_mean": "" if best is None else float(best["best_test_mean"]),
            "loss_vs_test_best": "" if best is None else float(best["best_test_mean"]) - float(selected["test_mean"]),
            "ball_mean": "" if ball is None else float(ball["test_mean"]),
            "delta_vs_ball": "" if ball is None else float(selected["test_mean"]) - float(ball["test_mean"]),
        })
    return out


def c123_rows(best_rows, global_rows, ref_rows) -> list[dict[str, object]]:
    best_ks = [int(row["best_k"]) for row in best_rows]
    c1 = bool(best_ks) and max(best_ks) / max(1, min(best_ks)) >= 4
    c2 = any(int(row["loss_ge_2pp"]) == 1 for row in global_rows)
    ball_by_dataset = {row["dataset"]: row for row in ref_rows if row["method"] == OURS}
    c3_details = []
    for best in best_rows:
        ball = ball_by_dataset.get(best["dataset"])
        if ball is None:
            continue
        gap = float(best["best_test_mean"]) - float(ball["test_mean"])
        within_1pp = gap <= 0.01
        within_1std = gap <= float(best["best_test_std"])
        c3_details.append({
            "dataset": best["dataset"],
            "ball_mean": float(ball["test_mean"]),
            "best_k": int(best["best_k"]),
            "best_mean": float(best["best_test_mean"]),
            "best_std": float(best["best_test_std"]),
            "gap_best_minus_ball": gap,
            "within_1pp_or_1std": int(within_1pp or within_1std),
        })
    c3_pass_count = sum(int(row["within_1pp_or_1std"]) for row in c3_details)
    return [
        {
            "condition": "C1_k_variation",
            "passes": int(c1),
            "detail": "" if not best_ks else f"best_k min={min(best_ks)} max={max(best_ks)} ratio={max(best_ks) / max(1, min(best_ks)):.2f}",
        },
        {
            "condition": "C2_k_sensitive",
            "passes": int(c2),
            "detail": f"datasets_loss_ge_2pp={sum(int(row['loss_ge_2pp']) for row in global_rows)}",
        },
        {
            "condition": "C3_ball_tracks_best_k",
            "passes": int(c3_pass_count >= 3),
            "detail": f"pass_count={c3_pass_count}/4",
        },
        *[
            {"condition": f"C3_detail_{row['dataset']}", "passes": row["within_1pp_or_1std"], "detail": f"gap={row['gap_best_minus_ball']:.6f}, best_std={row['best_std']:.6f}"}
            for row in c3_details
        ],
    ]


def regression_rows(new_rows, p0_rows) -> list[dict[str, object]]:
    new = {
        (row["dataset"], int(row["seed"]), int(row["matched_k"])): row
        for row in new_rows
        if row["dataset"] in {"cora", "citeseer"} and int(row["matched_k"]) == 10
    }
    out = []
    for p0 in p0_rows:
        if p0["dataset"] not in {"cora", "citeseer"}:
            continue
        p0_k = int(p0["matched_k"])
        seed = int(p0["seed"])
        row = new.get((p0["dataset"], seed, 10))
        if p0_k != 10:
            out.append({
                "dataset": p0["dataset"],
                "seed": seed,
                "p0_matched_k": p0_k,
                "new_k": 10,
                "p0_test": p0["test"],
                "new_test": "" if row is None else row["test"],
                "delta": "",
                "status": f"skipped_p0_matched_k_{p0_k}",
            })
            continue
        delta = "" if row is None else float(row["test"]) - float(p0["test"])
        out.append({
            "dataset": p0["dataset"],
            "seed": seed,
            "p0_matched_k": p0_k,
            "new_k": 10,
            "p0_test": p0["test"],
            "new_test": "" if row is None else row["test"],
            "delta": delta,
            "status": "pass" if delta != "" and abs(float(delta)) < 1e-12 else "fail",
        })
    return out


def completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        grouped[(row["dataset"], int(row["matched_k"]))].add(int(row["seed"]))
    out = []
    for dataset in DATASETS:
        for k in K_VALUES:
            seeds = grouped.get((dataset, k), set())
            out.append({
                "dataset": dataset,
                "k": k,
                "expected": len(SEEDS),
                "observed": len(seeds),
                "missing": " ".join(str(seed) for seed in SEEDS if seed not in seeds),
                "seeds": " ".join(str(seed) for seed in sorted(seeds)),
            })
    return out


def _fmt(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):.2f}"


def _fmt_delta(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):+.2f}pp"


def write_report(path: Path, curve, refs, best, global_k, global_loss, r3, gates, regression, complete) -> None:
    lines = [
        "# MLP-11 k-Sensitivity Report",
        "",
        "## Material Passport",
        "- Protocol: train-only uniform 0.4, percent split 0.2/0.1/0.7.",
        "- kNN target: `knn_mlp_softdist_beta05`, multiclass alpha, beta=0.5.",
        "- Ball references are reused; no ball reruns.",
        "",
        "## C1/C2/C3",
        "| Condition | Pass | Detail |",
        "|---|---:|---|",
    ]
    for row in gates:
        lines.append(f"| {row['condition']} | {row['passes']} | {row['detail']} |")
    lines.extend([
        "",
        f"Global k*: `{'' if global_k is None else global_k}`",
        "",
        "## k Curve",
        "| Dataset | k | n | Test mean±std | Valid mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in curve:
        lines.append(
            f"| {row['dataset']} | {row['k']} | {row['n']} | "
            f"{_fmt(row['test_mean'])}±{_fmt(row['test_std'])} | {_fmt(row['valid_mean'])} |"
        )
    lines.extend([
        "",
        "## Ball/Shuffled References",
        "| Dataset | Method | n | Test mean±std |",
        "|---|---|---:|---:|",
    ])
    for row in refs:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {_fmt(row['test_mean'])}±{_fmt(row['test_std'])} |")
    lines.extend([
        "",
        "## Best-k",
        "| Dataset | Best k | Best test | Best std |",
        "|---|---:|---:|---:|",
    ])
    for row in best:
        lines.append(f"| {row['dataset']} | {row['best_k']} | {_fmt(row['best_test_mean'])} | {_fmt(row['best_test_std'])} |")
    lines.extend([
        "",
        "## Global-k Loss",
        "| Dataset | Global k | Global test | Best k | Best test | Loss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in global_loss:
        lines.append(
            f"| {row['dataset']} | {row['global_k']} | {_fmt(row['global_test_mean'])} | "
            f"{row['best_k']} | {_fmt(row['best_test_mean'])} | {_fmt_delta(row['loss_vs_best'])} |"
        )
    lines.extend([
        "",
        "## R3 Val-selected k",
        "| Dataset | Val k | Val-selected test | Test best k | Test best | Δ vs best | Ball | Δ vs ball |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in r3:
        lines.append(
            f"| {row['dataset']} | {row['val_selected_k']} | {_fmt(row['val_selected_test_mean'])} | "
            f"{row['test_best_k']} | {_fmt(row['test_best_mean'])} | {_fmt_delta(row['loss_vs_test_best'])} | "
            f"{_fmt(row['ball_mean'])} | {_fmt_delta(row['delta_vs_ball'])} |"
        )
    lines.extend([
        "",
        "## Regression Check",
        "| Dataset | Seed | P0 matched-k | New k | P0 test | New test | Delta | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in regression:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['p0_matched_k']} | {row['new_k']} | "
            f"{_fmt(row['p0_test'])} | {_fmt(row['new_test'])} | {_fmt_delta(row['delta'])} | {row['status']} |"
        )
    missing = [row for row in complete if int(row["observed"]) != int(row["expected"])]
    lines.extend([
        "",
        "## Completeness",
        f"- Expected groups: {len(DATASETS) * len(K_VALUES)}",
        f"- Missing groups: {len(missing)}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--mlp5_dir", default=str(MLP5_RUN_DIR))
    parser.add_argument("--mlp8_dir", default=str(MLP8_RUN_DIR))
    parser.add_argument("--mlp10_dir", default=str(MLP10_RUN_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    new_rows = load_new_rows(run_dir)
    ref_rows = load_reference_rows(Path(args.mlp5_dir), Path(args.mlp10_dir))
    p0_rows = load_p0_rows(Path(args.mlp8_dir))
    curve = aggregate_knn(new_rows)
    refs = aggregate_reference(ref_rows)
    best = best_k_rows(curve)
    global_k, global_loss = global_k_rows(curve, best)
    r3 = r3_val_selection_rows(curve, best, refs)
    gates = c123_rows(best, global_loss, refs)
    regression = regression_rows(new_rows, p0_rows)
    complete = completeness(new_rows)

    write_csv(run_dir / "mlp11_combined_raw.csv", new_rows + ref_rows)
    write_csv(run_dir / "mlp11_knn_curve.csv", curve)
    write_csv(run_dir / "mlp11_reference_rows.csv", refs)
    write_csv(run_dir / "mlp11_best_k.csv", best)
    write_csv(run_dir / "mlp11_global_k_loss.csv", global_loss)
    write_csv(run_dir / "mlp11_r3_val_selected.csv", r3)
    write_csv(run_dir / "mlp11_c123_gates.csv", gates)
    write_csv(run_dir / "mlp11_regression_check.csv", regression)
    write_csv(run_dir / "mlp11_completeness.csv", complete)
    write_report(run_dir / "mlp11_report.md", curve, refs, best, global_k, global_loss, r3, gates, regression, complete)

    print(f"MLP11_NEW_ROWS={len(new_rows)}")
    print(f"MLP11_REFERENCE_ROWS={len(ref_rows)}")
    print(f"MLP11_GLOBAL_K={'' if global_k is None else global_k}")
    for row in gates:
        if row["condition"].startswith("C"):
            print(f"{row['condition']}={row['passes']} {row['detail']}")
    print(f"WROTE {run_dir / 'mlp11_report.md'}")


if __name__ == "__main__":
    main()
