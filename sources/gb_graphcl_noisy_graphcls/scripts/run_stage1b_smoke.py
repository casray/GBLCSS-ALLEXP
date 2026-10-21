from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gb_graphcl.data import corrupt_symmetric_labels, load_dataset, load_split_and_noise, save_split_and_noise, stratified_split
from gb_graphcl.granular_ball import build_granular_balls
from gb_graphcl.reliability import compute_reliability, compute_soft_targets
from gb_graphcl.selection import (
    _train_loop,
    train_ce_baseline,
    train_with_reweight,
    train_with_smoothing,
)


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


def _train_sce_with_trace(common: dict, alpha: float, beta: float, trace: bool):
    def sce_loss(logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pred = torch.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = F.one_hot(labels, num_classes=common["num_classes"]).float().clamp(min=1e-4, max=1.0)
        rce = -(pred * torch.log(one_hot)).sum(dim=1)
        return alpha * ce + beta * rce

    return _train_loop(
        common["embeddings"],
        common["noisy_labels"],
        common["clean_labels"],
        common["train_idx"],
        common["val_idx"],
        common["test_idx"],
        common["num_classes"],
        common["device"],
        common["seed"],
        common["epochs"],
        common["lr"],
        common["wd"],
        common["patience"],
        sce_loss,
        trace=trace,
    )


def _train_clean_oracle_with_trace(common: dict, trace: bool):
    train_idx_t = torch.tensor(common["train_idx"], dtype=torch.long)
    clean_mask = common["noisy_labels"][train_idx_t] == common["clean_labels"][train_idx_t]
    clean_train_idx = train_idx_t[clean_mask].tolist()
    oracle_common = dict(common)
    oracle_common["noisy_labels"] = common["clean_labels"]
    oracle_common["clean_labels"] = common["clean_labels"]
    oracle_common["train_idx"] = clean_train_idx
    return train_ce_baseline(**oracle_common, trace=trace)


def _delta(results: dict, a: str, b: str) -> float:
    return results[a]["test_acc_at_best_val"] - results[b]["test_acc_at_best_val"]


def _best_method(results: dict, methods: list[str]):
    return max(methods, key=lambda method: results[method]["test_acc_at_best_val"])


def _format_report(results: dict, gb_stats: dict, dataset: str, noise: str, seed: int):
    route_a = ["rw_topk", "rw_kmeans", "rw_gb", "rw_shuffled", "rw_random", "rw_noball"]
    route_b = ["sm_topk", "sm_kmeans", "sm_gb", "sm_shuffled", "sm_random", "sm_noball"]
    route_a_best = _best_method(results, route_a)
    route_b_best = _best_method(results, route_b)
    lines = [
        f"# Stage1b Smoke Report: {dataset} {noise} seed{seed}",
        "",
        "## Ranked Test Accuracy",
    ]
    for method, metrics in sorted(results.items(), key=lambda item: item[1]["test_acc_at_best_val"], reverse=True):
        lines.append(
            f"- {method}: test={metrics['test_acc_at_best_val']:.4f}, "
            f"precision={metrics['selection_precision']:.4f}, "
            f"noise_exclusion={metrics['noise_exclusion_recall']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Baselines",
            f"- headroom clean_oracle - ce: {_delta(results, 'clean_oracle', 'ce'):.4f}",
            "",
            "## Route A: Reweight",
            f"- best: {route_a_best} ({results[route_a_best]['test_acc_at_best_val']:.4f})",
            f"- route_a_best - ce: {_delta(results, route_a_best, 'ce'):.4f}",
            f"- rw_gb - rw_topk: {_delta(results, 'rw_gb', 'rw_topk'):.4f}",
            f"- rw_gb - rw_kmeans: {_delta(results, 'rw_gb', 'rw_kmeans'):.4f}",
            f"- rw_gb - rw_shuffled: {_delta(results, 'rw_gb', 'rw_shuffled'):.4f}",
            f"- rw_gb - rw_random: {_delta(results, 'rw_gb', 'rw_random'):.4f}",
            "",
            "## Route B: Label Smoothing",
            f"- best: {route_b_best} ({results[route_b_best]['test_acc_at_best_val']:.4f})",
            f"- route_b_best - ce: {_delta(results, route_b_best, 'ce'):.4f}",
            f"- sm_gb - sm_topk: {_delta(results, 'sm_gb', 'sm_topk'):.4f}",
            f"- sm_gb - sm_kmeans: {_delta(results, 'sm_gb', 'sm_kmeans'):.4f}",
            f"- sm_gb - sm_shuffled: {_delta(results, 'sm_gb', 'sm_shuffled'):.4f}",
            f"- sm_gb - sm_random: {_delta(results, 'sm_gb', 'sm_random'):.4f}",
            "",
            "## GB Stats",
            f"- num_balls: {gb_stats['num_balls']}",
            f"- mean_size: {gb_stats['mean_size']}",
            f"- min_size: {gb_stats['min_size']}",
            f"- max_size: {gb_stats['max_size']}",
            f"- mean_purity: {gb_stats['mean_purity']}",
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

    source_kwargs = {
        "topk": {"k": k},
        "kmeans": {"n_clusters": n_clusters},
        "gb": {"group_ids": group_ids},
        "shuffled": {"group_ids": group_ids, "seed": args.seed},
        "random": {"group_sizes": sizes, "seed": args.seed},
        "none": {},
    }
    reliabilities = {
        name: compute_reliability(name, embeddings, noisy_labels, train_idx, **kwargs)
        for name, kwargs in source_kwargs.items()
    }
    soft_targets = {
        name: compute_soft_targets(name, embeddings, noisy_labels, train_idx, num_classes, **kwargs)
        for name, kwargs in source_kwargs.items()
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

    results = {
        "ce": train_ce_baseline(**common, trace=True),
        "sce": _train_sce_with_trace(common, alpha=sce_cfg["alpha"], beta=sce_cfg["beta"], trace=True),
        "clean_oracle": _train_clean_oracle_with_trace(common, trace=True),
    }
    # Stage1b fixed diagnostic hyperparameter; do not change configs/default.yaml.
    w_min = 0.1
    source_to_method = {
        "topk": "topk",
        "kmeans": "kmeans",
        "gb": "gb",
        "shuffled": "shuffled",
        "random": "random",
        "none": "noball",
    }
    for source, suffix in source_to_method.items():
        results[f"rw_{suffix}"] = train_with_reweight(
            **common,
            reliability=reliabilities[source],
            warmup=sel_cfg["warmup"],
            ramp=sel_cfg["ramp"],
            w_min=w_min,
            trace=True,
        )
    for source, suffix in source_to_method.items():
        results[f"sm_{suffix}"] = train_with_smoothing(
            **common,
            soft_targets=soft_targets[source],
            warmup=sel_cfg["warmup"],
            ramp=sel_cfg["ramp"],
            trace=True,
        )

    report_dir = ROOT / "reports" / f"stage1b_smoke_{args.dataset}_{args.noise_type}_{_rate_tag(args.noise_rate)}_seed{args.seed}"
    _write_results_csv(report_dir / "results.csv", results)
    (report_dir / "gb_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for method, metrics in results.items():
        _write_trace_csv(report_dir / "traces" / f"trace_{method}.csv", metrics.get("trace", []))
    report = _format_report(results, stats, args.dataset, f"{args.noise_type}_{args.noise_rate}", args.seed)
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved={report_dir}")


if __name__ == "__main__":
    main()
