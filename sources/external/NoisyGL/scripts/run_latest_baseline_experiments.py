from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import sys
import time
from collections import defaultdict
from pathlib import Path
from queue import Empty
from statistics import mean, stdev
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import error_row_for, run_one, write_csv
from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "coauthorcs")
LATEST_METHODS = ("tfr", "procon", "gd2")
MAIN_METHODS = (
    "gcn",
    "gcn_xh_ce",
    "nrgnn",
    "nrgnn_xh",
    "frozengrace_gb_gcn_softdist_xh",
    "frozengrace_gb_gcn_softdist_xh_shuffled",
    "tfr",
    "procon",
    "gd2",
)
APPENDIX_METHODS = ("unionnet", "crgnn", "cgnn", "tss", "r2lp")
SEEDS = (0, 1, 2, 3, 4)


def _row_key(row: dict) -> tuple[str, str, int]:
    return str(row["dataset"]), str(row["method"]), int(row["seed"])


def pending_combinations(
        combinations: list[tuple[str, str, int]], rows: list[dict]
) -> list[tuple[str, str, int]]:
    completed = {_row_key(row) for row in rows}
    return [combination for combination in combinations if combination not in completed]


def upsert_row(rows: list[dict], replacement: dict) -> list[dict]:
    replacement_key = _row_key(replacement)
    updated = list(rows)
    for index, row in enumerate(updated):
        if _row_key(row) == replacement_key:
            updated[index] = replacement
            return updated
    updated.append(replacement)
    return updated


def make_timeout_row(template: dict, method: str, timeout_seconds: float) -> dict:
    row = dict(template)
    row.update({
        "method": method,
        "config_id": "",
        "wall_time_seconds": float(timeout_seconds),
        "parameter_count": "",
        "status": "timeout",
        "error": f"TimeoutError: exceeded {timeout_seconds:g} seconds",
        "train": "",
        "valid": "",
        "test": "",
    })
    return row


def _queue_target(queue, target, target_args) -> None:
    try:
        queue.put(("ok", target(*target_args)))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_with_timeout(target, target_args: tuple, timeout_seconds: float):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_queue_target,
        args=(queue, target, target_args),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
        return False, None
    try:
        status, payload = queue.get(timeout=5)
    except Empty as exc:
        raise RuntimeError(
            f"child exited with code {process.exitcode} without a result"
        ) from exc
    finally:
        queue.close()
    if status == "error":
        raise RuntimeError(payload)
    return True, payload


def replace_with_retry(
        source,
        target,
        attempts: int = 10,
        delay_seconds: float = 0.1,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_seconds)


def build_grid(method: str) -> list[dict]:
    if method == "tfr":
        return [
            {
                "training.lr": lr,
                "model.pseudo_ratio": pseudo_ratio,
                "model.beta": beta,
            }
            for lr, pseudo_ratio, beta in itertools.product(
                (0.01, 0.05, 0.1), (0.2, 0.4), (0.1, 0.5))
        ]
    if method == "procon":
        return [
            {
                "model.peer_k": peer_k,
                "model.alpha": alpha,
                "model.lambda_supervised": loss_weight,
            }
            for peer_k, alpha, loss_weight in itertools.product(
                (5, 10, 20), (0.25, 0.5), (0.01, 0.1))
        ]
    if method == "gd2":
        return [
            {
                "training.lr": lr,
                "model.tau": tau,
                "model.gamma": gamma,
            }
            for lr, tau, gamma in itertools.product(
                (0.001, 0.005, 0.01), (0.3, 0.5), (0.9, 0.95))
        ]
    raise KeyError(f"no tuning grid for method={method}")


def select_best_config(rows: list[dict]) -> dict:
    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        raise ValueError("no successful tuning run")
    return max(
        successful,
        key=lambda row: (float(row["valid"]), -int(row["config_index"])),
    )


def audit_protocol_hashes(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        if row.get("noise_scope", "train") != "train":
            raise ValueError(f"non-train noise scope found: {row}")
        if int(row.get("modified_val_count", 0)) != 0:
            raise ValueError(f"validation labels were corrupted: {row}")
        if int(row.get("modified_test_count", 0)) != 0:
            raise ValueError(f"test labels were corrupted: {row}")
        grouped[(row["dataset"], int(row["seed"]))].append(row)
    for key, group in grouped.items():
        split_hashes = {row["split_hash"] for row in group}
        noise_hashes = {row["noise_hash"] for row in group}
        if len(split_hashes) != 1 or len(noise_hashes) != 1:
            raise ValueError(
                f"protocol fingerprint mismatch for dataset/seed={key}: "
                f"split={split_hashes}, noise={noise_hashes}")


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), group in sorted(grouped.items()):
        successful = {
            int(row["seed"]): float(row["test"])
            for row in group
            if row.get("status") == "ok"
        }
        complete = set(successful) == set(SEEDS)
        values = [successful[seed] for seed in SEEDS] if complete else []
        summary.append({
            "dataset": dataset,
            "method": method,
            "n_success": len(successful),
            "status": "complete" if complete else "incomplete",
            "mean": mean(values) if complete else "",
            "std": stdev(values) if complete else "",
        })
    return summary


def paired_delta_rows(rows: list[dict]) -> list[dict]:
    by_key = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test"])
        for row in rows
        if row.get("status") == "ok"
    }
    comparisons = [
        ("frozengrace_gb_gcn_softdist_xh", method)
        for method in (
            "gcn_xh_ce",
            "frozengrace_gb_gcn_softdist_xh_shuffled",
            "nrgnn_xh",
            "tfr",
            "procon",
            "gd2",
        )
    ]
    output = []
    for dataset in DATASETS:
        for left, right in comparisons:
            deltas = []
            for seed in SEEDS:
                left_key = (dataset, left, seed)
                right_key = (dataset, right, seed)
                if left_key in by_key and right_key in by_key:
                    deltas.append(by_key[left_key] - by_key[right_key])
            output.append({
                "dataset": dataset,
                "comparison": f"{left}_minus_{right}",
                "n": len(deltas),
                "mean_delta": mean(deltas) if len(deltas) == 5 else "",
                "std_delta": stdev(deltas) if len(deltas) == 5 else "",
                "positive_seed_count": sum(delta > 0 for delta in deltas),
                "status": "complete" if len(deltas) == 5 else "incomplete",
            })
    return output


def _args(device: str, overrides: dict):
    return SimpleNamespace(
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        noise_type="uniform",
        noise_rate=0.4,
        noise_scope="train",
        config_overrides=overrides,
        debug=False,
    )


def _run_and_capture(dataset, method, seed, device, overrides):
    try:
        return run_one(_args(device, overrides), dataset, method, seed)
    except Exception as exc:
        return error_row_for(
            dataset, method, seed, "uniform", 0.4, exc, noise_scope="train")


def run_tuning(output_dir: Path, device: str) -> dict:
    rows = []
    selected = {}
    output_path = output_dir / "tuning_seed0.csv"
    for dataset in DATASETS:
        selected[dataset] = {}
        for method in LATEST_METHODS:
            method_rows = []
            for config_index, overrides in enumerate(build_grid(method)):
                row = _run_and_capture(dataset, method, 0, device, overrides)
                row["config_index"] = config_index
                row["config_overrides"] = json.dumps(overrides, sort_keys=True)
                rows.append(row)
                method_rows.append(row)
                write_csv(output_path, rows)
            best = select_best_config(method_rows)
            selected[dataset][method] = {
                "config_index": int(best["config_index"]),
                "config_id": best["config_id"],
                "valid": float(best["valid"]),
                "overrides": json.loads(best["config_overrides"]),
            }
    audit_protocol_hashes(rows)
    (output_dir / "selected_configs.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8")
    write_config_snapshots(output_dir, selected)
    return selected


def write_config_snapshots(output_dir: Path, selected: dict) -> None:
    snapshot_dir = output_dir / "config_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for dataset, methods in selected.items():
        for method, selection in methods.items():
            conf = load_conf(None, method, dataset)
            snapshot = {
                "dataset": dataset,
                "method": method,
                "base": {
                    "model": dict(conf.model),
                    "training": dict(conf.training),
                    "dataset": dict(conf.dataset),
                },
                "selection": selection,
            }
            (snapshot_dir / f"{method}_{dataset}.json").write_text(
                json.dumps(snapshot, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def load_selected(output_dir: Path) -> dict:
    path = output_dir / "selected_configs.json"
    if not path.exists():
        raise FileNotFoundError(f"run tuning first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_formal(output_dir: Path, device: str, selected: dict) -> list[dict]:
    rows = []
    output_path = output_dir / "formal_main_raw.csv"
    for dataset in DATASETS:
        for method in MAIN_METHODS:
            overrides = selected.get(dataset, {}).get(method, {}).get("overrides", {})
            for seed in SEEDS:
                row = _run_and_capture(dataset, method, seed, device, overrides)
                rows.append(row)
                write_csv(output_path, rows)
    audit_protocol_hashes(rows)
    return rows


def run_appendix(
        output_dir: Path,
        device: str,
        timeout_seconds: float = 1800.0,
        combinations: list[tuple[str, str, int]] | None = None,
        timed_runner=run_with_timeout,
) -> list[dict]:
    output_path = output_dir / "appendix_raw.csv"
    rows = read_csv(output_path)
    requested = combinations or [
        (dataset, method, seed)
        for dataset in DATASETS
        for method in APPENDIX_METHODS
        for seed in SEEDS
    ]
    formal_rows = read_csv(output_dir / "formal_main_raw.csv")
    protocol_templates = {
        (row["dataset"], int(row["seed"])): row
        for row in formal_rows
        if row.get("status") == "ok"
    }
    for dataset, method, seed in pending_combinations(requested, rows):
        template = protocol_templates.get((dataset, seed))
        if template is None:
            raise ValueError(
                f"missing formal protocol template for dataset/seed={(dataset, seed)}")
        try:
            completed, row = timed_runner(
                _run_and_capture,
                (dataset, method, seed, device, {}),
                timeout_seconds,
            )
            if not completed:
                row = make_timeout_row(template, method, timeout_seconds)
        except Exception as exc:
            row = error_row_for(
                dataset, method, seed, "uniform", 0.4, exc, noise_scope="train")
        for key in (
                "split_hash", "noise_hash", "modified_count", "actual_noise_rate",
                "modified_train_count", "actual_train_noise_rate",
                "modified_val_count", "modified_test_count"):
            if row.get(key, "") == "":
                row[key] = template.get(key, "")
        rows = upsert_row(rows, row)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        write_csv(temporary_path, rows)
        replace_with_retry(temporary_path, output_path)
    audit_protocol_hashes(rows)
    return rows


def run_smoke(output_dir: Path, device: str) -> list[dict]:
    rows = []
    for dataset, epochs in (("cora", 2), ("coauthorcs", 1)):
        for method in LATEST_METHODS:
            row = _run_and_capture(
                dataset,
                method,
                0,
                device,
                {"training.n_epochs": epochs},
            )
            rows.append(row)
            write_csv(output_dir / "smoke_raw.csv", rows)
    audit_protocol_hashes(rows)
    return rows


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_report(output_dir: Path, formal_rows: list[dict], appendix_rows: list[dict]) -> None:
    combined = formal_rows + appendix_rows
    audit_protocol_hashes(combined)
    write_csv(output_dir / "summary.csv", summarize_rows(combined))
    write_csv(output_dir / "paired_deltas.csv", paired_delta_rows(formal_rows))
    complete_count = sum(
        row["status"] == "complete" for row in summarize_rows(combined))
    report = [
        "# Latest Baselines Train-Only Protocol Audit",
        "",
        "- Split: 20% / 10% / 70%",
        "- Noise: uniform 0.4, train labels only",
        "- Validation: clean labels, used for model/config selection",
        "- Test: clean labels, never used for selection",
        "- ProCon and GD2: paper reimplementations, not official code",
        f"- Complete five-seed method/dataset cells: {complete_count}",
        "",
        "Historical global-noise results are legacy evidence and must not be mixed with this table.",
    ]
    (output_dir / "protocol_audit_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Tune and run the 2025 graph noisy-label baselines under train-only noise.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "tune", "formal", "appendix", "summarize", "all"],
        default="smoke",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--appendix_timeout_seconds", type=float, default=1800.0)
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "runs" / "latest_baselines_trainonly_20260622"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        run_smoke(output_dir, args.device)
        return
    selected = None
    if args.mode in {"tune", "all"}:
        selected = run_tuning(output_dir, args.device)
    if args.mode in {"formal", "all"}:
        selected = selected or load_selected(output_dir)
        run_formal(output_dir, args.device, selected)
    if args.mode in {"appendix", "all"}:
        run_appendix(
            output_dir,
            args.device,
            timeout_seconds=args.appendix_timeout_seconds,
        )
    if args.mode in {"summarize", "all"}:
        write_report(
            output_dir,
            read_csv(output_dir / "formal_main_raw.csv"),
            read_csv(output_dir / "appendix_raw.csv"),
        )


if __name__ == "__main__":
    main()
