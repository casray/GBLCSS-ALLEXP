from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp10_baseline_refresh_20260702"
MLP5_RUN_DIR = REPO_ROOT / "runs" / "mlp_maintable_20260701"
MLP7_RUN_DIR = REPO_ROOT / "runs" / "mlp7_baselines_20260701"
MLP2_RUN_DIR = REPO_ROOT / "runs" / "mlp_dualrobust_20260701"

OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
NEW_METHODS = ("mlp_logitclip", "mlp_anl")
COAUTHOR_NEW_METHODS = (
    OURS,
    SHUFFLED,
    "mlp_fine",
    "mlp_dividemix_lite",
    "mlp_ct",
    "mlp_logitclip",
    "mlp_anl",
)
OLD_MAIN_METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_fine",
    "mlp_dividemix_lite",
    OURS,
    SHUFFLED,
)
COAUTHOR_REUSE_METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
)
CT = "mlp_ct"
DATASETS_OLD = ("cora", "citeseer", "pubmed")
DATASETS_ALL = ("cora", "citeseer", "pubmed", "coauthorcs")
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
    fieldnames = []
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


def regime_name(noise_type: str, noise_rate: float) -> str:
    return f"{noise_type}{str(noise_rate).replace('.', 'p')}"


def normalize(row: dict[str, str], source: Path, source_group: str) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if not method:
        return None
    status = row.get("status", "ok").strip().lower()
    test = row.get("test") or row.get("test_acc") or ""
    if not test:
        return None
    return {
        "source_group": source_group,
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": int(row["seed"]),
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": row.get("noise_scope", "train"),
        "split_hash": row.get("split_hash", ""),
        "noise_hash": row.get("noise_hash", ""),
        "config_id": row.get("config_id", ""),
        "status": status,
        "error": row.get("error", ""),
        "valid": _float(row.get("valid")),
        "test": float(test),
        "actual_train_noise_rate": _float(row.get("actual_train_noise_rate")),
        "strategy": row.get("strategy", ""),
        "logit_clip_tau": _float(row.get("logit_clip_tau")),
        "anl_active_weight": _float(row.get("anl_active_weight")),
        "anl_passive_weight": _float(row.get("anl_passive_weight")),
        "source": str(source),
    }


def load_run_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp10_"):
            continue
        for row in read_csv(path):
            normalized = normalize(row, path, "mlp10")
            if normalized is not None:
                rows.append(normalized)
    return rows


def _load_rows_from_files(paths: list[Path], source_group: str, allowed: set[str]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        for row in read_csv(path):
            normalized = normalize(row, path, source_group)
            if normalized is None or normalized["method"] not in allowed:
                continue
            if normalized["status"] not in {"", "ok"}:
                continue
            rows.append(normalized)
    return rows


def load_reuse_rows(mlp5_dir: Path, mlp7_dir: Path, mlp2_dir: Path) -> list[dict[str, object]]:
    rows = []
    rows.extend(_load_rows_from_files(
        sorted(mlp5_dir.glob("main_*_seeds_0_1_2_3_4.csv")),
        "mlp5",
        set(OLD_MAIN_METHODS),
    ))
    rows.extend(_load_rows_from_files(
        sorted(mlp7_dir.glob("main_ct_*_seeds_0_1_2_3_4.csv")),
        "mlp7",
        {CT},
    ))
    rows.extend(_load_rows_from_files(
        sorted(mlp2_dir.glob("label_*_coauthorcs_seeds_0_1_2_3_4.csv")),
        "mlp2",
        set(COAUTHOR_REUSE_METHODS),
    ))
    return rows


def selected_logitclip_tau(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] != "mlp_logitclip" or row["status"] not in {"", "ok"}:
            continue
        if row["seed"] != 0 or row["logit_clip_tau"] == "" or row["valid"] == "":
            continue
        grouped[(row["dataset"], float(row["logit_clip_tau"]))].append(float(row["valid"]))
    out = []
    for dataset in sorted({key[0] for key in grouped}):
        candidates = []
        for (candidate_dataset, tau), values in grouped.items():
            if candidate_dataset != dataset:
                continue
            candidates.append((statistics.mean(values), tau, len(values)))
        if not candidates:
            continue
        mean_valid, tau, n = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        out.append({
            "dataset": dataset,
            "selected_tau": tau,
            "mean_seed0_noisy_valid": mean_valid,
            "n_settings": n,
        })
    return out


def _selected_tau_map(rows: list[dict[str, object]]) -> dict[str, float]:
    return {row["dataset"]: float(row["selected_tau"]) for row in selected_logitclip_tau(rows)}


def filter_logitclip_to_selected(rows: list[dict[str, object]], selected: dict[str, float]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if row["method"] != "mlp_logitclip":
            out.append(row)
            continue
        tau = row["logit_clip_tau"]
        if tau == "":
            continue
        dataset_tau = selected.get(row["dataset"])
        if dataset_tau is not None and math.isclose(float(tau), dataset_tau, abs_tol=1e-12):
            out.append(row)
    return out


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for row in rows:
        key = (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
            "" if row["method"] != "mlp_logitclip" else row["logit_clip_tau"],
        )
        if key in unique:
            previous = float(unique[key]["test"])
            current = float(row["test"])
            if not math.isclose(previous, current, abs_tol=1e-12):
                raise ValueError(f"conflicting duplicate for {key}: {previous} vs {current}")
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (
        row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])
    ))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] not in {"", "ok"}:
            continue
        key = (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], row["method_label"])
        grouped[key].append(float(row["test"]))
    out = []
    for (dataset, noise_type, noise_rate, method, label), values in sorted(grouped.items()):
        out.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "method_label": label,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return out


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test"])
        for row in rows
        if row["status"] in {"", "ok"}
    }
    out = []
    for dataset in DATASETS_ALL:
        for noise_type, noise_rate in REGIMES:
            for baseline in sorted({key[3] for key in values if key[0] == dataset and key[1] == noise_type and key[2] == noise_rate}):
                if baseline == OURS:
                    continue
                deltas = []
                seeds = []
                for seed in SEEDS:
                    left = values.get((dataset, noise_type, noise_rate, OURS, seed))
                    right = values.get((dataset, noise_type, noise_rate, baseline, seed))
                    if left is None or right is None:
                        continue
                    deltas.append(left - right)
                    seeds.append(seed)
                if not deltas:
                    continue
                out.append({
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "comparison": f"ours_vs_{baseline}",
                    "baseline": baseline,
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                    "seeds": " ".join(str(seed) for seed in seeds),
                })
    return out


def ga1_rows(new_rows: list[dict[str, object]], reuse_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ce = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), int(row["seed"])): float(row["test"])
        for row in reuse_rows
        if row["method"] == "mlp_ce"
    }
    selected = _selected_tau_map(new_rows)
    rows = []
    for row in new_rows:
        if row["dataset"] not in DATASETS_OLD or row["seed"] != 0:
            continue
        if row["method"] == "mlp_logitclip":
            if not math.isclose(float(row["logit_clip_tau"]), selected.get(row["dataset"], -1.0), abs_tol=1e-12):
                continue
        elif row["method"] != "mlp_anl":
            continue
        ce_acc = ce.get((row["dataset"], row["noise_type"], float(row["noise_rate"]), 0))
        threshold_ok = "" if ce_acc is None else int(float(row["test"]) >= ce_acc - 0.02)
        rows.append({
            "dataset": row["dataset"],
            "noise_type": row["noise_type"],
            "noise_rate": row["noise_rate"],
            "method": row["method"],
            "status": row["status"],
            "test": row["test"],
            "ce_seed0": "" if ce_acc is None else ce_acc,
            "passes_ce_minus_2pp": threshold_ok,
        })
    return rows


def ga2_rows(new_rows: list[dict[str, object]], reuse_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test"])
        for row in new_rows + reuse_rows
        if row["dataset"] == "coauthorcs" and row["status"] in {"", "ok"}
    }
    out = []
    for noise_type, noise_rate in REGIMES:
        ours = values.get((noise_type, noise_rate, OURS, 0))
        ce = values.get((noise_type, noise_rate, "mlp_ce", 0))
        shuffled = values.get((noise_type, noise_rate, SHUFFLED, 0))
        out.append({
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "ours_seed0": "" if ours is None else ours,
            "ce_seed0": "" if ce is None else ce,
            "shuffled_seed0": "" if shuffled is None else shuffled,
            "passes": int(ours is not None and ce is not None and shuffled is not None and ours > ce and ours > shuffled),
        })
    return out


def ga3_rows(new_rows: list[dict[str, object]], reuse_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    reference = {
        (row["noise_type"], float(row["noise_rate"]), int(row["seed"])): row
        for row in reuse_rows
        if row["dataset"] == "coauthorcs" and row["method"] == "mlp_ce"
    }
    out = []
    for row in new_rows:
        if row["dataset"] != "coauthorcs" or int(row["seed"]) != 0:
            continue
        ref = reference.get((row["noise_type"], float(row["noise_rate"]), 0))
        out.append({
            "noise_type": row["noise_type"],
            "noise_rate": row["noise_rate"],
            "method": row["method"],
            "split_match": int(ref is not None and row["split_hash"] == ref["split_hash"]),
            "noise_match": int(ref is not None and row["noise_hash"] == ref["noise_hash"]),
            "reference_source": "" if ref is None else ref["source"],
            "new_source": row["source"],
        })
    return out


def a4_rows(new_rows: list[dict[str, object]], reuse_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ce = {
        int(row["seed"]): float(row["test"])
        for row in reuse_rows
        if row["dataset"] == "citeseer"
        and row["method"] == "mlp_ce"
        and row["noise_type"] == "uniform"
        and math.isclose(float(row["noise_rate"]), 0.4, abs_tol=1e-12)
        and int(row["seed"]) in {1, 2, 3, 4}
    }
    anl = {
        int(row["seed"]): row
        for row in new_rows
        if row["dataset"] == "citeseer"
        and row["method"] == "mlp_anl"
        and row["noise_type"] == "uniform"
        and math.isclose(float(row["noise_rate"]), 0.4, abs_tol=1e-12)
        and int(row["seed"]) in {1, 2, 3, 4}
    }
    out = []
    for seed in (1, 2, 3, 4):
        row = anl.get(seed)
        ce_acc = ce.get(seed)
        delta = "" if row is None or ce_acc is None else float(row["test"]) - ce_acc
        out.append({
            "dataset": "citeseer",
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "method": "mlp_anl",
            "seed": seed,
            "test": "" if row is None else row["test"],
            "ce_same_seed": "" if ce_acc is None else ce_acc,
            "delta_vs_ce": delta,
            "passes_ge_minus_2pp": int(delta != "" and delta >= -0.02),
            "status": "" if row is None else row["status"],
        })
    return out


def completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(set)
    for row in rows:
        grouped[(row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"])].add(int(row["seed"]))
    out = []
    for key, seeds in sorted(grouped.items()):
        out.append({
            "dataset": key[0],
            "noise_type": key[1],
            "noise_rate": key[2],
            "method": key[3],
            "observed": len(seeds),
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


def write_report(path: Path, selected, ga1, ga2, ga3, a4, aggregate_rows, delta_rows, complete_rows) -> None:
    ga1_pass = bool(ga1) and all(row["status"] in {"", "ok"} and row["passes_ce_minus_2pp"] == 1 for row in ga1)
    ga2_pass_count = sum(int(row["passes"]) for row in ga2)
    ga2_pass = ga2_pass_count >= 3
    ga3_pass = bool(ga3) and all(row["split_match"] == 1 and row["noise_match"] == 1 for row in ga3)
    branch = "expand_5seed" if ga1_pass and ga2_pass and ga3_pass else "stop_after_smoke"
    a4_pass_count = sum(int(row["passes_ge_minus_2pp"]) for row in a4)
    a4_pass = len(a4) == 4 and a4_pass_count >= 3
    revised_branch = "include_anl_after_a4" if a4_pass else "drop_anl_after_a4"
    lines = [
        "# MLP-10 Baseline Refresh Report",
        "",
        "## Material Passport",
        "- Artifact type: baseline refresh and Coauthor-CS expansion",
        "- Protocol: train-only label noise, percent split 0.2/0.1/0.7",
        "- Backbone: X-only 2-layer MLP unless method name explicitly says otherwise",
        "- Clean labels: evaluation only",
        "",
        "## Selected LogitClip Tau",
        "| Dataset | selected tau | mean seed0 noisy-valid | n settings |",
        "|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(f"| {row['dataset']} | {row['selected_tau']} | {_fmt(row['mean_seed0_noisy_valid'])} | {row['n_settings']} |")
    lines.extend([
        "",
        "## Smoke Gates",
        f"- GA1 implementation health: {'pass' if ga1_pass else 'not-pass'}",
        f"- GA2 Coauthor-CS feasibility: {'pass' if ga2_pass else 'not-pass'} ({ga2_pass_count}/5 settings)",
        f"- GA3 protocol reuse audit: {'pass' if ga3_pass else 'not-pass'}",
        f"- Branch: **{branch}**",
        "",
        "## Revised 7.3 Gate",
        f"- A4 ANL targeted audit: {'pass' if a4_pass else 'not-pass'} ({a4_pass_count}/4 seeds >= CE-2pp)",
        f"- Revised ANL decision: **{revised_branch}**",
        "",
        "### A4 Details",
        "| Seed | ANL test | CE same seed | ANL-CE | pass >= -2pp |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in a4:
        lines.append(
            f"| {row['seed']} | {_fmt(row['test'])} | {_fmt(row['ce_same_seed'])} | "
            f"{_fmt_delta(row['delta_vs_ce'])} | {row['passes_ge_minus_2pp']} |"
        )
    lines.extend([
        "",
        "### GA1 Details",
        "| Dataset | Noise | Method | test | CE seed0 | pass CE-2pp | status |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in ga1:
        lines.append(
            f"| {row['dataset']} | {row['noise_type']}{row['noise_rate']} | {row['method']} | "
            f"{_fmt(row['test'])} | {_fmt(row['ce_seed0'])} | {row['passes_ce_minus_2pp']} | {row['status']} |"
        )
    lines.extend([
        "",
        "### GA2 Details",
        "| Noise | Ours seed0 | CE seed0 | Shuffled seed0 | pass |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in ga2:
        lines.append(
            f"| {row['noise_type']}{row['noise_rate']} | {_fmt(row['ours_seed0'])} | "
            f"{_fmt(row['ce_seed0'])} | {_fmt(row['shuffled_seed0'])} | {row['passes']} |"
        )
    lines.extend([
        "",
        "### GA3 Details",
        "| Noise | Method | split match | noise match |",
        "|---|---|---:|---:|",
    ])
    for row in ga3:
        lines.append(
            f"| {row['noise_type']}{row['noise_rate']} | {row['method']} | "
            f"{row['split_match']} | {row['noise_match']} |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "| Dataset | Noise | Method | n | mean +/- std |",
        "|---|---|---|---:|---:|",
    ])
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {row['noise_type']}{row['noise_rate']} | {row['method_label']} | "
            f"{row['n']} | {_fmt(row['mean'])}+/-{_fmt(row['std'])} |"
        )
    lines.extend([
        "",
        "## Ours Paired Deltas",
        "| Dataset | Noise | Baseline | n | mean delta | std | positive |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['dataset']} | {row['noise_type']}{row['noise_rate']} | {row['baseline']} | "
            f"{row['n']} | {_fmt_delta(row['mean_delta'])} | {_fmt_delta(row['std_delta'])} | "
            f"{row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## Completeness",
        f"- observed method-setting groups: {len(complete_rows)}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MLP10_BRANCH={branch}")
    print(f"MLP10_REVISED_BRANCH={revised_branch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--mlp5_dir", default=str(MLP5_RUN_DIR))
    parser.add_argument("--mlp7_dir", default=str(MLP7_RUN_DIR))
    parser.add_argument("--mlp2_dir", default=str(MLP2_RUN_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    new_rows = load_run_rows(run_dir)
    reuse_rows = load_reuse_rows(Path(args.mlp5_dir), Path(args.mlp7_dir), Path(args.mlp2_dir))
    selected = selected_logitclip_tau(new_rows)
    selected_map = _selected_tau_map(new_rows)
    combined = deduplicate(filter_logitclip_to_selected(new_rows + reuse_rows, selected_map))
    agg = aggregate(combined)
    deltas = paired_deltas(combined)
    ga1 = ga1_rows(new_rows, reuse_rows)
    ga2 = ga2_rows(new_rows, reuse_rows)
    ga3 = ga3_rows(new_rows, reuse_rows)
    a4 = a4_rows(new_rows, reuse_rows)
    complete = completeness(combined)

    write_csv(run_dir / "mlp10_selected_logitclip_tau.csv", selected)
    write_csv(run_dir / "mlp10_combined_raw.csv", combined)
    write_csv(run_dir / "mlp10_aggregate.csv", agg)
    write_csv(run_dir / "mlp10_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp10_ga1.csv", ga1)
    write_csv(run_dir / "mlp10_ga2.csv", ga2)
    write_csv(run_dir / "mlp10_ga3_protocol_audit.csv", ga3)
    write_csv(run_dir / "mlp10_a4_anl_audit.csv", a4)
    write_csv(run_dir / "mlp10_completeness.csv", complete)
    write_report(run_dir / "mlp10_report.md", selected, ga1, ga2, ga3, a4, agg, deltas, complete)
    print(f"MLP10_NEW_ROWS={len(new_rows)}")
    print(f"MLP10_COMBINED_ROWS={len(combined)}")
    print(f"WROTE {run_dir / 'mlp10_report.md'}")


if __name__ == "__main__":
    main()
