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
from gb_graphcl.evaluate import format_smoke_report
from gb_graphcl.granular_ball import build_granular_balls
from gb_graphcl.reliability import compute_reliability
from gb_graphcl.selection import train_ce_baseline, train_clean_oracle, train_sce_baseline, train_with_selection


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

    gb_cfg = cfg["granular_ball"]
    group_ids, _, sizes, purities = build_granular_balls(
        embeddings,
        noisy_labels,
        train_idx,
        purity_threshold=gb_cfg["purity_threshold"],
        min_samples=gb_cfg["min_samples"],
        max_size=gb_cfg["max_size"],
    )
    stats = _gb_stats(sizes, purities)
    k = max(1, int(round(stats["mean_size"])))
    n_clusters = max(1, stats["num_balls"])

    reliabilities = {
        "sel_topk": compute_reliability("topk", embeddings, noisy_labels, train_idx, k=k),
        "sel_kmeans": compute_reliability("kmeans", embeddings, noisy_labels, train_idx, n_clusters=n_clusters),
        "sel_gb": compute_reliability("gb", embeddings, noisy_labels, train_idx, group_ids=group_ids),
        "sel_shuffled": compute_reliability("shuffled", embeddings, noisy_labels, train_idx, group_ids=group_ids, seed=args.seed),
        "sel_random": compute_reliability("random", embeddings, noisy_labels, train_idx, group_sizes=sizes, seed=args.seed),
        "sel_noball": compute_reliability("none", embeddings, noisy_labels, train_idx),
    }

    cls_cfg = cfg["classifier"]
    sel_cfg = cfg["selection"]
    sce_cfg = cfg["sce"]
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
    rho = 1.0 - args.noise_rate
    results = {
        "ce": train_ce_baseline(**common),
        "sce": train_sce_baseline(**common, alpha=sce_cfg["alpha"], beta=sce_cfg["beta"]),
    }
    for method in ["sel_topk", "sel_kmeans", "sel_gb", "sel_shuffled", "sel_random", "sel_noball"]:
        results[method] = train_with_selection(
            **common,
            reliability=reliabilities[method],
            warmup=sel_cfg["warmup"],
            ramp=sel_cfg["ramp"],
            rho=rho,
        )
    results["clean_oracle"] = train_clean_oracle(**common)

    report_dir = ROOT / "reports" / f"smoke_{args.dataset}_{args.noise_type}_{_rate_tag(args.noise_rate)}_seed{args.seed}"
    _write_results_csv(report_dir / "results.csv", results)
    (report_dir / "gb_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report = format_smoke_report(results, stats, args.dataset, f"{args.noise_type}_{args.noise_rate}", args.seed)
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved={report_dir}")


if __name__ == "__main__":
    main()
