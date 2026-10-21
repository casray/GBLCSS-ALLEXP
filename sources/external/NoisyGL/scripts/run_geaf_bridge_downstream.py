from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import run_one, write_csv  # noqa: E402


DEFAULT_METHODS = (
    "gcn_xh_ce,"
    "frozengrace_gb_gcn_softdist_xh,"
    "frozengrace_gb_gcn_softdist_xh_shuffled,"
    "frozengrace_geaf_gcn_xh,"
    "frozengrace_geaf_gcn_xh_shuffled"
)
GEAF_METHODS = {"frozengrace_geaf_gcn_xh", "frozengrace_geaf_gcn_xh_shuffled"}


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_passed_gate_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if _truthy(row.get("passed"))]


def _error_row(dataset: str, method: str, seed: int, confidence: float, args, exc: Exception) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "geaf_majority_confidence": float(confidence),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "train": "",
        "valid": "",
        "test": "",
    }


def write_report(path: Path, rows: list[dict], passed_gate_rows: list[dict]) -> None:
    lines = [
        "# GEAF bridge downstream seed0 report",
        "",
        f"Gate settings consumed: {len(passed_gate_rows)}",
        f"Downstream rows: {len(rows)}",
        "",
    ]
    if not passed_gate_rows:
        lines.extend(
            [
                "No GEAF offline setting passed the gate.",
                "",
                "Decision: 当前 H 空间粒椭球过滤无法提供可用训练标签修正，按规则未启动 downstream。",
            ]
        )
    else:
        lines.extend(
            [
                "## Results",
                "",
                "| dataset | confidence | method | status | test | error |",
                "|---|---:|---|---|---:|---|",
            ]
        )
        for row in rows:
            test = row.get("test", "")
            try:
                test_text = f"{float(test):.4f}"
            except (TypeError, ValueError):
                test_text = ""
            lines.append(
                "| {dataset} | {confidence:.2f} | {method} | {status} | {test} | {error} |".format(
                    dataset=row.get("dataset", ""),
                    confidence=_as_float(row, "geaf_majority_confidence"),
                    method=row.get("method", ""),
                    status=row.get("status", ""),
                    test=test_text,
                    error=str(row.get("error", "")),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Run downstream seed0 only for GEAF bridge gate-passed settings.")
    parser.add_argument("--input_gate", required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "geaf_bridge_downstream_seed0_20260629"))
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = Path(args.input_gate)
    if not gate_path.is_absolute():
        gate_path = (REPO_ROOT / gate_path).resolve()
    passed_gate_rows = read_passed_gate_rows(gate_path)
    methods = _parse_str_list(args.methods)
    rows = []
    for gate_row in passed_gate_rows:
        dataset = str(gate_row["dataset"]).lower()
        confidence = _as_float(gate_row, "majority_confidence")
        for method in methods:
            run_args = SimpleNamespace(
                device=args.device,
                split_type=args.split_type,
                train_percent=args.train_percent,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
                noise_type=args.noise_type,
                noise_rate=args.noise_rate,
                noise_scope=args.noise_scope,
                debug=bool(args.debug),
                config_overrides={"gb.min_majority_confidence": confidence} if method in GEAF_METHODS else {},
            )
            print(f"GEAF_DOWNSTREAM dataset={dataset} method={method} seed={args.seed} confidence={confidence}", flush=True)
            try:
                row = run_one(run_args, dataset, method, int(args.seed))
                row["geaf_majority_confidence"] = float(confidence)
                row["gate_hard_net_mean"] = gate_row.get("hard_net_mean", "")
                row["gate_hard_net_delta_mean"] = gate_row.get("hard_net_delta_mean", "")
            except Exception as exc:
                row = _error_row(dataset, method, int(args.seed), confidence, args, exc)
            rows.append(row)
            write_csv(output_dir / "geaf_bridge_downstream_seed0_summary.csv", rows)
    write_report(output_dir / "geaf_bridge_downstream_seed0_report.md", rows, passed_gate_rows)
    print(f"GEAF_DOWNSTREAM rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
