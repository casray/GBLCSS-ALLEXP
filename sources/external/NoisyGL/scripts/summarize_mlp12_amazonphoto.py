from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp12_amazonphoto_20260702"

DATASET = "amazonpho"
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
METHODS = (
    OURS,
    SHUFFLED,
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_fine",
    "mlp_dividemix_lite",
    "mlp_ct",
    "mlp_logitclip",
    "mlp_anl",
)
REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "Small-loss",
    "mlp_coteaching": "Co-teaching",
    "mlp_jocor": "JoCoR",
    "mlp_elr": "ELR",
    "mlp_sce": "SCE",
    "mlp_fine": "FINE",
    "mlp_dividemix_lite": "DivideMix-lite",
    "mlp_ct": "CT",
    "mlp_logitclip": "LogitClip",
    "mlp_anl": "ANL",
    OURS: "Ours beta=0.5",
    SHUFFLED: "Shuffled",
}


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


def normalize(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if not method:
        return None
    test = row.get("test") or row.get("test_acc") or ""
    if test == "":
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": int(row["seed"]),
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": row.get("noise_scope", "train"),
        "split_hash": row.get("split_hash", ""),
        "noise_hash": row.get("noise_hash", ""),
        "status": row.get("status", "ok").strip().lower(),
        "error": row.get("error", ""),
        "valid": _float(row.get("valid")),
        "test": float(test),
        "actual_train_noise_rate": _float(row.get("actual_train_noise_rate")),
        "modified_train_count": _float(row.get("modified_train_count")),
        "modified_val_count": _float(row.get("modified_val_count")),
        "modified_test_count": _float(row.get("modified_test_count")),
        "logit_clip_tau": _float(row.get("logit_clip_tau")),
        "num_balls": _float(row.get("num_balls")),
        "mean_train_per_ball": _float(row.get("mean_train_per_ball")),
        "noisy_train_purity_weighted": _float(row.get("noisy_train_purity_weighted")),
        "true_train_purity_weighted": _float(row.get("true_train_purity_weighted")),
        "source": str(source),
    }


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp12_"):
            continue
        for raw in read_csv(path):
            row = normalize(raw, path)
            if row and row["dataset"] == DATASET:
                rows.append(row)
    return rows


def load_d0(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for raw in read_csv(run_dir / "d0" / "orthogonality_summary.csv"):
        if (raw.get("dataset") or "").strip().lower() != DATASET:
            continue
        rows.append({
            "dataset": DATASET,
            "num_balls_mean": _float(raw.get("num_balls_mean")),
            "true_all_purity_weighted_mean": _float(raw.get("true_all_purity_weighted_mean")),
            "true_train_purity_weighted_mean": _float(raw.get("true_train_purity_weighted_mean")),
            "noisy_train_purity_weighted_mean": _float(raw.get("noisy_train_purity_weighted_mean")),
            "ball_auc_mean": _float(raw.get("ball_auc_mean")),
            "ball_auc_in_gcn_clean50_mean": _float(raw.get("ball_auc_in_gcn_clean50_mean")),
        })
    return rows


def selected_logitclip_tau(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "mlp_logitclip" or row["seed"] != 0 or row["logit_clip_tau"] == "" or row["valid"] == "":
            continue
        grouped[float(row["logit_clip_tau"])].append(float(row["valid"]))
    out = []
    if grouped:
        mean_valid, tau, n = sorted(
            ((statistics.mean(values), tau, len(values)) for tau, values in grouped.items()),
            key=lambda item: (-item[0], item[1]),
        )[0]
        out.append({
            "dataset": DATASET,
            "selected_tau": tau,
            "mean_seed0_noisy_valid": mean_valid,
            "n_settings": n,
        })
    return out


def filter_selected_logitclip(rows: list[dict[str, object]], selected: list[dict[str, object]]) -> list[dict[str, object]]:
    if not selected:
        return [row for row in rows if row["method"] != "mlp_logitclip"]
    tau = float(selected[0]["selected_tau"])
    out = []
    for row in rows:
        if row["method"] != "mlp_logitclip":
            out.append(row)
        elif row["logit_clip_tau"] != "" and math.isclose(float(row["logit_clip_tau"]), tau, abs_tol=1e-12):
            out.append(row)
    return out


def estimate_train_count(row: dict[str, object]) -> int | str:
    modified = row.get("modified_train_count")
    rate = row.get("actual_train_noise_rate")
    if modified == "" or rate == "" or float(rate) <= 0.0:
        return ""
    return int(round(float(modified) / float(rate)))


def split_audit_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row["seed"] != 0:
            continue
        train_count = estimate_train_count(row)
        out.append({
            "noise_type": row["noise_type"],
            "noise_rate": row["noise_rate"],
            "method": row["method"],
            "seed": row["seed"],
            "estimated_train_count": train_count,
            "train_count_ok": int(train_count != "" and abs(int(train_count) - 1530) <= 1),
            "modified_val_count": row["modified_val_count"],
            "modified_test_count": row["modified_test_count"],
            "noise_train_only": int(str(row["modified_val_count"]) in {"", "0", "0.0"} and str(row["modified_test_count"]) in {"", "0", "0.0"}),
            "source": row["source"],
        })
    return out


def d0_gate_rows(d0_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not d0_rows:
        return [{
            "dataset": DATASET,
            "num_balls_mean": "",
            "true_all_purity_weighted_mean": "",
            "ball_auc_in_gcn_clean50_mean": "",
            "passes": 0,
            "reason": "missing_d0",
        }]
    row = d0_rows[0]
    passes = row["num_balls_mean"] != "" and row["true_all_purity_weighted_mean"] != "" and float(row["num_balls_mean"]) > 1 and float(row["true_all_purity_weighted_mean"]) >= 0.65
    return [{
        **row,
        "passes": int(passes),
        "reason": "pass" if passes else "num_balls_or_purity_below_gate",
    }]


def smoke_gate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["noise_type"], float(row["noise_rate"]), row["method"]): row
        for row in rows
        if row["seed"] == 0 and row["method"] in {OURS, SHUFFLED, "mlp_ce"}
    }
    out = []
    for noise_type, noise_rate in REGIMES:
        ours = values.get((noise_type, noise_rate, OURS))
        ce = values.get((noise_type, noise_rate, "mlp_ce"))
        shuffled = values.get((noise_type, noise_rate, SHUFFLED))
        status_ok = all(item is not None and item["status"] in {"", "ok"} for item in [ours, ce, shuffled])
        train_counts = [estimate_train_count(item) for item in [ours, ce, shuffled] if item is not None]
        train_ok = bool(train_counts) and all(count != "" and abs(int(count) - 1530) <= 1 for count in train_counts)
        noise_train_only = all(
            item is not None
            and str(item["modified_val_count"]) in {"", "0", "0.0"}
            and str(item["modified_test_count"]) in {"", "0", "0.0"}
            for item in [ours, ce, shuffled]
        )
        direction_ok = bool(ours and ce and shuffled and float(ours["test"]) > float(ce["test"]) and float(ours["test"]) > float(shuffled["test"]))
        out.append({
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "ours_seed0": "" if ours is None else float(ours["test"]),
            "ce_seed0": "" if ce is None else float(ce["test"]),
            "shuffled_seed0": "" if shuffled is None else float(shuffled["test"]),
            "status_ok": int(status_ok),
            "train_count_ok": int(train_ok),
            "noise_train_only": int(noise_train_only),
            "passes_direction": int(direction_ok),
            "passes": int(status_ok and train_ok and noise_train_only and direction_ok),
        })
    return out


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] in METHODS and row["status"] in {"", "ok"}:
            grouped[(row["noise_type"], float(row["noise_rate"]), row["method"])].append(row)
    out = []
    for (noise_type, noise_rate, method), items in sorted(grouped.items()):
        tests = [float(row["test"]) for row in items]
        out.append({
            "dataset": DATASET,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n": len(items),
            "mean": statistics.mean(tests),
            "std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "seeds": " ".join(str(row["seed"]) for row in sorted(items, key=lambda x: int(x["seed"]))),
        })
    return out


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test"])
        for row in rows
        if row["method"] in METHODS and row["status"] in {"", "ok"}
    }
    out = []
    for noise_type, noise_rate in REGIMES:
        for baseline in METHODS:
            if baseline == OURS:
                continue
            deltas = []
            seeds = []
            for seed in SEEDS:
                ours = values.get((noise_type, noise_rate, OURS, seed))
                base = values.get((noise_type, noise_rate, baseline, seed))
                if ours is None or base is None:
                    continue
                deltas.append(ours - base)
                seeds.append(seed)
            if not deltas:
                continue
            out.append({
                "dataset": DATASET,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "baseline": baseline,
                "n": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_count": sum(delta > 0 for delta in deltas),
                "seeds": " ".join(str(seed) for seed in seeds),
                "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
            })
    return out


def completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], set[int]] = defaultdict(set)
    for row in rows:
        if row["method"] in METHODS:
            grouped[(row["noise_type"], float(row["noise_rate"]), row["method"])].add(int(row["seed"]))
    out = []
    for noise_type, noise_rate in REGIMES:
        for method in METHODS:
            seeds = grouped.get((noise_type, noise_rate, method), set())
            out.append({
                "dataset": DATASET,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "method": method,
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


def write_report(path: Path, d0_gate, smoke_gate, selected_tau, agg_rows, delta_rows, complete_rows) -> None:
    d0_pass = bool(d0_gate) and int(d0_gate[0]["passes"]) == 1
    smoke_pass_count = sum(int(row["passes_direction"]) for row in smoke_gate)
    smoke_all_status = all(int(row["status_ok"]) == 1 and int(row["train_count_ok"]) == 1 and int(row["noise_train_only"]) == 1 for row in smoke_gate) if smoke_gate else False
    smoke_pass = smoke_all_status and smoke_pass_count >= 3
    missing = [row for row in complete_rows if int(row["observed"]) != int(row["expected"])]
    lines = [
        "# MLP-12 Amazon-Photo Report",
        "",
        "## Material Passport",
        "- Dataset: Amazon-Photo (`amazonpho`), train-only label noise, percent split 0.2/0.1/0.7.",
        "- Backbone: X-only 2-layer MLP unless method explicitly uses frozen GRACE targets.",
        "- Clean labels: evaluation and diagnostics only.",
        "",
        "## Gates",
        f"- D0 geometry gate: {'pass' if d0_pass else 'not-pass'}",
        f"- A1 smoke gate: {'pass' if smoke_pass else 'not-pass'} ({smoke_pass_count}/5 direction settings)",
        f"- Selected LogitClip tau: `{'' if not selected_tau else selected_tau[0]['selected_tau']}`",
        "",
        "### D0 Details",
        "| num balls | true all purity | true train purity | noisy train purity | ball AUC | ball AUC in GCN-clean50 | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d0_gate:
        lines.append(
            f"| {row.get('num_balls_mean', '')} | {row.get('true_all_purity_weighted_mean', '')} | "
            f"{row.get('true_train_purity_weighted_mean', '')} | {row.get('noisy_train_purity_weighted_mean', '')} | "
            f"{row.get('ball_auc_mean', '')} | {row.get('ball_auc_in_gcn_clean50_mean', '')} | {row.get('passes', '')} |"
        )
    lines.extend([
        "",
        "### A1 Smoke Details",
        "| Noise | Ours | CE | Shuffled | direction | train/scope ok |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in smoke_gate:
        train_scope_ok = int(row["status_ok"]) == 1 and int(row["train_count_ok"]) == 1 and int(row["noise_train_only"]) == 1
        lines.append(
            f"| {row['noise_type']}{row['noise_rate']} | {_fmt(row['ours_seed0'])} | {_fmt(row['ce_seed0'])} | "
            f"{_fmt(row['shuffled_seed0'])} | {row['passes_direction']} | {int(train_scope_ok)} |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "| Noise | Method | n | mean±std |",
        "|---|---|---:|---:|",
    ])
    for row in agg_rows:
        lines.append(
            f"| {row['noise_type']}{row['noise_rate']} | {row['method_label']} | {row['n']} | "
            f"{_fmt(row['mean'])}±{_fmt(row['std'])} |"
        )
    lines.extend([
        "",
        "## Ours Paired Deltas",
        "| Noise | Baseline | n | Δ | pos |",
        "|---|---|---:|---:|---:|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['noise_type']}{row['noise_rate']} | {row['baseline']} | {row['n']} | "
            f"{_fmt_delta(row['mean_delta'])} | {row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## Completeness",
        f"- Expected method-setting groups: {len(REGIMES) * len(METHODS)}",
        f"- Missing groups: {len(missing)}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    raw_rows = load_rows(run_dir)
    selected_tau = selected_logitclip_tau(raw_rows)
    rows = filter_selected_logitclip(raw_rows, selected_tau)
    d0 = load_d0(run_dir)
    d0_gate = d0_gate_rows(d0)
    smoke_gate = smoke_gate_rows(rows)
    split_audit = split_audit_rows(rows)
    agg = aggregate(rows)
    deltas = paired_deltas(rows)
    complete = completeness(rows)

    write_csv(run_dir / "mlp12_selected_logitclip_tau.csv", selected_tau)
    write_csv(run_dir / "mlp12_combined_raw.csv", rows)
    write_csv(run_dir / "mlp12_d0_gate.csv", d0_gate)
    write_csv(run_dir / "mlp12_a1_smoke_gate.csv", smoke_gate)
    write_csv(run_dir / "mlp12_split_audit.csv", split_audit)
    write_csv(run_dir / "mlp12_aggregate.csv", agg)
    write_csv(run_dir / "mlp12_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp12_completeness.csv", complete)
    write_report(run_dir / "mlp12_report.md", d0_gate, smoke_gate, selected_tau, agg, deltas, complete)

    d0_pass = int(d0_gate[0]["passes"]) == 1 if d0_gate else False
    smoke_pass_count = sum(int(row["passes_direction"]) for row in smoke_gate)
    smoke_all_status = all(int(row["status_ok"]) == 1 and int(row["train_count_ok"]) == 1 and int(row["noise_train_only"]) == 1 for row in smoke_gate) if smoke_gate else False
    smoke_pass = smoke_all_status and smoke_pass_count >= 3
    print(f"MLP12_D0={'pass' if d0_pass else 'not-pass'}")
    print(f"MLP12_A1={'pass' if smoke_pass else 'not-pass'} direction={smoke_pass_count}/5")
    print(f"MLP12_SELECTED_LOGITCLIP_TAU={'' if not selected_tau else selected_tau[0]['selected_tau']}")
    print(f"MLP12_ROWS={len(rows)}")
    print(f"WROTE {run_dir / 'mlp12_report.md'}")


if __name__ == "__main__":
    main()
