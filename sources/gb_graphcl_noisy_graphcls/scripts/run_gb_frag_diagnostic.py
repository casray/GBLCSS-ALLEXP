from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gb_graphcl.data import corrupt_symmetric_labels, load_dataset, load_split_and_noise, save_split_and_noise, stratified_split
from gb_graphcl.granular_ball import build_granular_balls
from gb_graphcl.reliability import compute_reliability, gb_consensus_reliability
from gb_graphcl.selection import train_with_reweight


GB_RELAXED = {
    "purity_threshold": 0.6,
    "min_samples": 10,
    "max_size": 200,
}
W_MIN = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["NCI1", "PROTEINS", "MUTAG"])
    parser.add_argument("--noise-type", default="symmetric", choices=["symmetric"])
    parser.add_argument("--noise-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def _rate_tag(rate: float) -> str:
    return str(rate).replace(".", "p")


def _artifact_path(dataset: str, seed: int, noise_type: str, noise_rate: float) -> Path:
    return ROOT / "artifacts" / dataset / f"seed{seed}_noise{noise_type}_{_rate_tag(noise_rate)}" / "split.pt"


def _gb_stats(sizes, purities):
    return {
        "num_balls": int(len(sizes)),
        "mean_size": float(sum(sizes) / max(len(sizes), 1)),
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "mean_purity": float(sum(purities) / max(len(purities), 1)),
    }


def _write_results_csv(path: Path, results: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "test_acc", "selection_precision", "noise_exclusion_recall"])
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow(
                {
                    "method": method,
                    "test_acc": metrics["test_acc_at_best_val"],
                    "selection_precision": metrics["selection_precision"],
                    "noise_exclusion_recall": metrics["noise_exclusion_recall"],
                }
            )


def _write_trace_csv(path: Path, trace: list[dict]):
    if not trace:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in trace for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)


def _delta(results: dict, a: str, b: str) -> float:
    return results[a]["test_acc_at_best_val"] - results[b]["test_acc_at_best_val"]


def _check_line(name: str, delta: float, passed: bool, threshold_text: str) -> str:
    label = "PASS" if passed else "FAIL"
    return f"- {name}: {delta:.4f} ({label}; {threshold_text})"


def _verdict(results: dict) -> tuple[str, list[str]]:
    checks = [
        (
            "rw_gb_relaxed - rw_noball",
            _delta(results, "rw_gb_relaxed", "rw_noball"),
            _delta(results, "rw_gb_relaxed", "rw_noball") > 0.0,
            "PASS if > 0",
        ),
        (
            "rw_gb_relaxed - rw_shuffled_relaxed",
            _delta(results, "rw_gb_relaxed", "rw_shuffled_relaxed"),
            _delta(results, "rw_gb_relaxed", "rw_shuffled_relaxed") > 0.005,
            "PASS if > 0.005",
        ),
        (
            "rw_gb_relaxed - rw_random_relaxed",
            _delta(results, "rw_gb_relaxed", "rw_random_relaxed"),
            _delta(results, "rw_gb_relaxed", "rw_random_relaxed") > 0.0,
            "PASS if > 0",
        ),
        (
            "rw_gb_relaxed - rw_kmeans",
            _delta(results, "rw_gb_relaxed", "rw_kmeans"),
            _delta(results, "rw_gb_relaxed", "rw_kmeans") >= -0.005,
            "PASS if >= -0.005",
        ),
    ]
    failed = [name for name, _, passed, _ in checks if not passed]
    if failed:
        verdict = "FAIL: " + "; ".join(failed)
    else:
        verdict = "PASS: all 4 pre-registered checks passed"
    lines = [_check_line(name, delta, passed, threshold_text) for name, delta, passed, threshold_text in checks]
    return verdict, lines


def _format_report(results: dict, gb_stats: dict, dataset: str, noise_type: str, noise_rate: float, seed: int):
    verdict, check_lines = _verdict(results)
    lines = [
        f"# GB Fragmentation Diagnostic: {dataset} {noise_type} {noise_rate} seed{seed}",
        "",
        "## GB Statistics",
        "### gb_orig (purity=0.7, min=5, max=100)",
        f"- num_balls: {gb_stats['gb_orig']['num_balls']}",
        f"- mean_size: {gb_stats['gb_orig']['mean_size']}",
        f"- min_size: {gb_stats['gb_orig']['min_size']}",
        f"- max_size: {gb_stats['gb_orig']['max_size']}",
        f"- mean_purity: {gb_stats['gb_orig']['mean_purity']}",
        "",
        "### gb_relaxed (purity=0.6, min=10, max=200)",
        f"- num_balls: {gb_stats['gb_relaxed']['num_balls']}",
        f"- mean_size: {gb_stats['gb_relaxed']['mean_size']}",
        f"- min_size: {gb_stats['gb_relaxed']['min_size']}",
        f"- max_size: {gb_stats['gb_relaxed']['max_size']}",
        f"- mean_purity: {gb_stats['gb_relaxed']['mean_purity']}",
        "",
        "## Ranked Test Accuracy",
    ]
    for method, metrics in sorted(results.items(), key=lambda item: item[1]["test_acc_at_best_val"], reverse=True):
        lines.append(
            f"- {method}: test_acc={metrics['test_acc_at_best_val']:.4f}, "
            f"precision={metrics['selection_precision']:.4f}, "
            f"noise_exclusion={metrics['noise_exclusion_recall']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Pre-registered Deltas",
            *check_lines,
            "",
            "## Fragmentation Repair Delta",
            f"- rw_gb_relaxed - rw_gb_orig: {_delta(results, 'rw_gb_relaxed', 'rw_gb_orig'):.4f} (positive = repair helped)",
            "",
            "## Verdict",
            verdict,
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_path = ROOT / "checkpoints" / "embeddings" / args.dataset / f"pretrain{args.seed}" / "H.pt"
    payload = torch.load(emb_path, map_location="cpu", weights_only=False)
    embeddings = payload["embeddings"].float()
    clean_labels = payload["labels"].long()
    num_classes = int(payload["num_classes"])

    split_path = _artifact_path(args.dataset, args.seed, args.noise_type, args.noise_rate)
    if split_path.exists():
        artifact = load_split_and_noise(split_path)
        train_idx = artifact["train_idx"]
        val_idx = artifact["val_idx"]
        test_idx = artifact["test_idx"]
        noisy_labels = artifact["noisy_labels"].long()
    else:
        dataset = load_dataset(args.dataset, root=str(ROOT / "data"))
        train_idx, val_idx, test_idx = stratified_split(dataset, seed=args.seed)
        noisy_labels = corrupt_symmetric_labels(clean_labels, train_idx, args.noise_rate, seed=args.seed)
        save_split_and_noise(split_path, train_idx, val_idx, test_idx, noisy_labels, clean_labels)

    gb_orig_cfg = cfg["granular_ball"]
    group_orig, _, sizes_orig, purities_orig = build_granular_balls(
        embeddings,
        noisy_labels,
        train_idx,
        purity_threshold=gb_orig_cfg["purity_threshold"],
        min_samples=gb_orig_cfg["min_samples"],
        max_size=gb_orig_cfg["max_size"],
    )
    group_relaxed, _, sizes_relaxed, purities_relaxed = build_granular_balls(
        embeddings,
        noisy_labels,
        train_idx,
        purity_threshold=GB_RELAXED["purity_threshold"],
        min_samples=GB_RELAXED["min_samples"],
        max_size=GB_RELAXED["max_size"],
    )
    stats = {
        "gb_orig": _gb_stats(sizes_orig, purities_orig),
        "gb_relaxed": _gb_stats(sizes_relaxed, purities_relaxed),
    }

    n_clusters = max(1, stats["gb_relaxed"]["num_balls"])
    reliabilities = {
        "rw_noball": compute_reliability("none", embeddings, noisy_labels, train_idx),
        "rw_kmeans": compute_reliability("kmeans", embeddings, noisy_labels, train_idx, n_clusters=n_clusters),
        "rw_gb_orig": gb_consensus_reliability(group_orig, noisy_labels, train_idx),
        "rw_gb_relaxed": compute_reliability("gb", embeddings, noisy_labels, train_idx, group_ids=group_relaxed),
        "rw_shuffled_relaxed": compute_reliability("shuffled", embeddings, noisy_labels, train_idx, group_ids=group_relaxed, seed=args.seed),
        "rw_random_relaxed": compute_reliability("random", embeddings, noisy_labels, train_idx, group_sizes=sizes_relaxed, seed=args.seed),
    }

    cls_cfg = cfg["classifier"]
    sel_cfg = cfg["selection"]
    common = {
        "embeddings": embeddings,
        "noisy_labels": noisy_labels,
        "clean_labels": clean_labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "num_classes": num_classes,
        "device": device,
        "epochs": cls_cfg["epochs"],
        "lr": cls_cfg["lr"],
        "wd": cls_cfg["weight_decay"],
        "patience": cls_cfg["patience"],
        "seed": args.seed,
    }

    results = {}
    for method, reliability in reliabilities.items():
        results[method] = train_with_reweight(
            **common,
            reliability=reliability,
            warmup=sel_cfg["warmup"],
            ramp=sel_cfg["ramp"],
            w_min=W_MIN,
            trace=True,
        )

    report_dir = ROOT / "reports" / f"gb_frag_diag_{args.dataset}_{args.noise_type}_{_rate_tag(args.noise_rate)}_seed{args.seed}"
    _write_results_csv(report_dir / "results.csv", results)
    (report_dir / "gb_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for method, metrics in results.items():
        _write_trace_csv(report_dir / "traces" / f"trace_{method}.csv", metrics.get("trace", []))
    report = _format_report(results, stats, args.dataset, args.noise_type, args.noise_rate, args.seed)
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved={report_dir}")


if __name__ == "__main__":
    main()
