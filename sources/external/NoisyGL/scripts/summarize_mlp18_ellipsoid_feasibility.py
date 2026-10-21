from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp18_ellipsoid_feasibility_20260706"
DEFAULT_METHODS = {
    "candidate": "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05",
    "candidate_shuffled": "frozengrace_gb_mlp_softdist_ellipsoid_pca_shrinkage_beta05_shuffled",
    "current": "frozengrace_gb_mlp_softdist_beta05",
    "current_shuffled": "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "knn": "knn_mlp_softdist_beta05",
}


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return float(default)
    return float(value)


def _as_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return int(default)
    return int(float(value))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


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


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if str(row.get("status", "ok")).lower() not in {"", "ok"}:
            continue
        groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    out = []
    for (dataset, method), group in sorted(groups.items()):
        values = [_as_float(row, "test") for row in group if row.get("test", "") not in ("", None)]
        if not values:
            continue
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "n": int(len(values)),
                "test_mean": float(np.mean(values)),
                "test_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return out


def paired_delta_rows(rows: list[dict], candidate_method: str, reference_method: str, comparison: str) -> list[dict]:
    by_key = {
        (str(row["dataset"]), _as_int(row, "seed"), str(row["method"])): row
        for row in rows
        if str(row.get("status", "ok")).lower() in {"", "ok"}
    }
    keys = sorted(
        (dataset, seed)
        for dataset, seed, method in by_key
        if method == candidate_method and (dataset, seed, reference_method) in by_key
    )
    out = []
    for dataset, seed in keys:
        cand = by_key[(dataset, seed, candidate_method)]
        ref = by_key[(dataset, seed, reference_method)]
        out.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "comparison": comparison,
                "candidate_method": candidate_method,
                "reference_method": reference_method,
                "candidate_test": _as_float(cand, "test"),
                "reference_test": _as_float(ref, "test"),
                "test_delta": _as_float(cand, "test") - _as_float(ref, "test"),
                "candidate_config_id": cand.get("config_id", ""),
                "reference_config_id": ref.get("config_id", ""),
                "split_hash": cand.get("split_hash", ""),
                "noise_hash": cand.get("noise_hash", ""),
            }
        )
    return out


def gate_rows(deltas: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in deltas:
        groups.setdefault((str(row["dataset"]), str(row["comparison"])), []).append(row)
    out = []
    for (dataset, comparison), group in sorted(groups.items()):
        values = [_as_float(row, "test_delta") for row in group]
        out.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "n": int(len(values)),
                "delta_mean": float(np.mean(values)) if values else 0.0,
                "delta_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "positive_seed_count": int(sum(value > 0 for value in values)),
                "plus_1pp_seed_count": int(sum(value >= 0.01 for value in values)),
                "minus_0p5pp_or_worse_count": int(sum(value <= -0.005 for value in values)),
            }
        )
    return out


def write_report(path: Path, aggregate: list[dict], deltas: list[dict], gates: list[dict]) -> None:
    lines = [
        "# MLP-18 ellipsoid downstream summary",
        "",
        "This report summarizes downstream runs only; offline gates remain in `mlp18_report.md`.",
        "",
        "## Aggregate",
        "",
    ]
    for row in aggregate:
        lines.append(
            f"- {row['dataset']} / {row['method']}: n={row['n']}, "
            f"test={_as_float(row, 'test_mean'):.4f}±{_as_float(row, 'test_std'):.4f}"
        )
    lines.extend(["", "## Paired gates", ""])
    for row in gates:
        lines.append(
            f"- {row['dataset']} / {row['comparison']}: delta={_as_float(row, 'delta_mean'):+.4f}, "
            f"positive={_as_int(row, 'positive_seed_count')}/{_as_int(row, 'n')}, "
            f">=1pp={_as_int(row, 'plus_1pp_seed_count')}/{_as_int(row, 'n')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize MLP-18 downstream CSV files.")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--inputs", default="downstream_seed0_raw.csv,downstream_5seed_raw.csv")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    rows = []
    for item in str(args.inputs).split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            path = run_dir / path
        rows.extend(read_csv(path))
    aggregate = aggregate_rows(rows)
    deltas = []
    deltas.extend(paired_delta_rows(rows, DEFAULT_METHODS["candidate"], DEFAULT_METHODS["current"], "candidate_vs_current"))
    deltas.extend(paired_delta_rows(rows, DEFAULT_METHODS["candidate"], DEFAULT_METHODS["candidate_shuffled"], "candidate_vs_shuffled"))
    deltas.extend(paired_delta_rows(rows, DEFAULT_METHODS["candidate"], DEFAULT_METHODS["knn"], "candidate_vs_knn"))
    gates = gate_rows(deltas)
    write_csv(run_dir / "downstream_aggregate.csv", aggregate)
    write_csv(run_dir / "downstream_paired_deltas.csv", deltas)
    write_csv(run_dir / "downstream_gate.csv", gates)
    write_report(run_dir / "downstream_report.md", aggregate, deltas, gates)
    print(f"MLP18_SUMMARY run_dir={run_dir}")


if __name__ == "__main__":
    main()
