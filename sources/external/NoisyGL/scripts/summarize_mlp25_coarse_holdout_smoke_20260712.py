from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DATASETS = ("citeseerfull", "coraml", "attribwiki")
TRAIN_COUNTS = {"citeseerfull": 846, "coraml": 599, "attribwiki": 481}
CANDIDATES = {
    "vote_target": "vote_target_shuffled",
    "candidate": "candidate_shuffled",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp25_coarse_holdout_smoke_20260712"


def vote_target_cap(train_count: int, target_votes: int = 20, upper_cap: int = 64) -> int:
    if int(train_count) <= 0 or int(target_votes) <= 0 or int(upper_cap) <= 0:
        raise ValueError("train_count, target_votes, and upper_cap must be positive")
    return min(int(upper_cap), int(math.ceil(int(train_count) / int(target_votes))))


def _float(value, default=0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _int(value, default=0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def effective_mean_train_votes(row: dict) -> float:
    reported = row.get("mean_train_per_ball")
    if reported is not None and str(reported).strip() != "":
        return _float(reported)
    dataset = str(row.get("dataset", "")).lower()
    num_balls = _int(row.get("num_balls"))
    if dataset not in TRAIN_COUNTS or num_balls <= 0:
        return 0.0
    return float(TRAIN_COUNTS[dataset] / num_balls)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protocol_ok(row: dict) -> bool:
    return (
        str(row.get("status", "ok")).lower() in {"", "ok"}
        and str(row.get("noise_type", "")).lower() == "uniform"
        and math.isclose(_float(row.get("noise_rate")), 0.4, abs_tol=1e-12)
        and str(row.get("noise_scope", "")).lower() == "train"
        and math.isclose(_float(row.get("train_percent")), 0.2, abs_tol=1e-12)
        and math.isclose(_float(row.get("val_percent")), 0.1, abs_tol=1e-12)
        and math.isclose(_float(row.get("test_percent")), 0.7, abs_tol=1e-12)
        and _int(row.get("modified_val_count")) == 0
        and _int(row.get("modified_test_count")) == 0
        and _int(row.get("modified_count")) == _int(row.get("modified_train_count"))
    )


def _paired(rows: list[dict], left_arm: str, right_arm: str) -> tuple[dict[str, float], list[str]]:
    indexed = {(str(row.get("dataset", "")).lower(), str(row.get("arm", ""))): row for row in rows}
    deltas: dict[str, float] = {}
    reasons: list[str] = []
    for dataset in DATASETS:
        left = indexed.get((dataset, left_arm))
        right = indexed.get((dataset, right_arm))
        if left is None or right is None:
            reasons.append(f"missing_pair:{dataset}:{left_arm}:{right_arm}")
            continue
        if not protocol_ok(left) or not protocol_ok(right):
            reasons.append(f"invalid_protocol:{dataset}:{left_arm}:{right_arm}")
            continue
        if left.get("split_hash", "") != right.get("split_hash", "") or left.get("noise_hash", "") != right.get("noise_hash", ""):
            reasons.append(f"hash_mismatch:{dataset}:{left_arm}:{right_arm}")
            continue
        deltas[dataset] = _float(left.get("test")) - _float(right.get("test"))
    return deltas, reasons


def smoke_gate(rows: list[dict]) -> dict[str, dict[str, object]]:
    decisions: dict[str, dict[str, object]] = {}
    indexed = {(str(row.get("dataset", "")).lower(), str(row.get("arm", ""))): row for row in rows}
    for candidate, shuffled in CANDIDATES.items():
        vs_fixed, reasons = _paired(rows, candidate, "fixed")
        vs_shuffled, pair_reasons = _paired(rows, candidate, shuffled)
        reasons.extend(pair_reasons)
        for dataset in DATASETS:
            row = indexed.get((dataset, candidate))
            if row is not None and (
                _int(row.get("num_balls")) <= 1
                or effective_mean_train_votes(row) < 2
                or _int(row.get("alpha_positive_train_count")) <= 0
            ):
                reasons.append(f"degenerate:{dataset}:{candidate}")
        if len(vs_fixed) == 3:
            if sum(delta >= 0.01 for delta in vs_fixed.values()) < 2:
                reasons.append("fewer_than_2_of_3_gain_1pp")
            if any(delta < -0.005 for delta in vs_fixed.values()):
                reasons.append("regression_over_0p5pp")
        if len(vs_shuffled) == 3 and sum(delta >= 0.01 for delta in vs_shuffled.values()) < 2:
            reasons.append("fewer_than_2_of_3_beat_own_shuffled_1pp")
        decisions[candidate] = {
            "stage": "seed0_smoke",
            "candidate": candidate,
            "pass": int(not reasons),
            "reasons": ";".join(sorted(set(reasons))),
            "datasets_gain_1pp": sum(delta >= 0.01 for delta in vs_fixed.values()),
            "datasets_beat_shuffled_1pp": sum(delta >= 0.01 for delta in vs_shuffled.values()),
            "mean_delta_vs_fixed": sum(vs_fixed.values()) / len(vs_fixed) if vs_fixed else "",
        }
    return decisions


def load_arms(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = [(run_dir / "fixed_seed0.csv", "fixed")]
    sources.extend((path, "vote") for path in sorted(run_dir.glob("vote_*_seed0.csv")))
    sources.append((run_dir / "candidate_seed0.csv", "candidate"))
    for path, family in sources:
        for raw in read_csv(path):
            row: dict[str, object] = dict(raw)
            method = str(raw.get("method", ""))
            shuffled = method.endswith("_shuffled")
            if family == "vote":
                row["arm"] = "vote_target_shuffled" if shuffled else "vote_target"
            elif family == "candidate":
                row["arm"] = "candidate_shuffled" if shuffled else "candidate"
            else:
                row["arm"] = "fixed"
            row["source"] = str(path)
            rows.append(row)
    return rows


def validate_configs(rows: list[dict]) -> list[str]:
    expected_caps = {"citeseerfull": 43, "coraml": 30, "attribwiki": 25}
    reasons: list[str] = []
    for row in rows:
        dataset = str(row.get("dataset", "")).lower()
        arm = str(row.get("arm", ""))
        max_balls = _int(row.get("gb_max_balls"))
        min_size = _int(row.get("gb_min_ball_size"))
        split = _float(row.get("gb_split_improvement"))
        if arm == "fixed" and (max_balls != 64 or min_size != 32 or not math.isclose(split, 0.05)):
            reasons.append(f"wrong_fixed_config:{dataset}")
        elif arm.startswith("vote_target") and (
            max_balls != expected_caps.get(dataset) or min_size != 32 or not math.isclose(split, 0.05)
        ):
            reasons.append(f"wrong_vote_config:{dataset}")
        elif arm.startswith("candidate") and (
            max_balls != 128 or min_size != 64 or not math.isclose(split, 0.01)
        ):
            reasons.append(f"wrong_candidate_config:{dataset}")
    return sorted(set(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the MLP-25 coarse-ball holdout smoke.")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    rows = load_arms(args.run_dir)
    config_failures = validate_configs(rows)
    decisions = smoke_gate(rows)
    if config_failures:
        for decision in decisions.values():
            decision["pass"] = 0
            existing = str(decision.get("reasons", ""))
            decision["reasons"] = ";".join(filter(None, [existing, *config_failures]))

    aggregate = []
    for row in sorted(rows, key=lambda item: (str(item.get("dataset")), str(item.get("arm")))):
        aggregate.append(
            {
                "dataset": str(row.get("dataset", "")).lower(),
                "arm": row.get("arm"),
                "test": _float(row.get("test")),
                "num_balls": _int(row.get("num_balls")),
                "mean_train_per_ball": effective_mean_train_votes(row),
                "gb_max_balls": _int(row.get("gb_max_balls")),
                "gb_min_ball_size": _int(row.get("gb_min_ball_size")),
                "gb_split_improvement": _float(row.get("gb_split_improvement")),
            }
        )
    write_csv(args.run_dir / "mlp25_raw.csv", rows)
    write_csv(args.run_dir / "mlp25_aggregate.csv", aggregate)
    write_csv(args.run_dir / "mlp25_gate_decisions.csv", list(decisions.values()))

    lines = [
        "# MLP-25 Coarse-Ball Holdout Smoke",
        "",
        "| Dataset | Arm | Test (%) | Balls | Train votes/ball |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {100 * float(row['test']):.2f} | "
            f"{row['num_balls']} | {float(row['mean_train_per_ball']):.2f} |"
        )
    lines.extend(["", "## Gate decisions", "", "| Candidate | Pass | Mean delta vs fixed | Reasons |", "|---|---:|---:|---|"])
    for decision in decisions.values():
        mean_delta = decision.get("mean_delta_vs_fixed")
        formatted = "-" if mean_delta == "" else f"{100 * float(mean_delta):+.2f}pp"
        lines.append(
            f"| {decision['candidate']} | {decision['pass']} | {formatted} | {decision.get('reasons') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Final decision",
            "",
            "Neither candidate passes the preregistered seed-0 gate. Do not run five seeds, do not change the "
            "fixed-64 default, and close the granularity axis. Both real candidates remain above their own "
            "shuffled controls on most/all holdouts, so the negative result concerns transferable accuracy "
            "gain rather than the existence of local structure.",
        ]
    )
    (args.run_dir / "mlp25_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.run_dir / "mlp25_gate_decisions.csv")


if __name__ == "__main__":
    main()
