"""Selection diagnostic — answer: did selection actually engage, and did it help or hurt?

Runs ce / sel_topk / sel_gb / sel_noball with trace=True on a single seed.
Saves per-epoch CSV traces and a summary report.  No gate claims, no parameter changes.
"""
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

from gb_graphcl.data import (
    corrupt_symmetric_labels,
    load_dataset,
    load_split_and_noise,
    save_split_and_noise,
    stratified_split,
)
from gb_graphcl.granular_ball import build_granular_balls
from gb_graphcl.reliability import compute_reliability
from gb_graphcl.selection import (
    train_ce_baseline,
    train_with_selection,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="NCI1")
    p.add_argument("--noise-type", default="symmetric")
    p.add_argument("--noise-rate", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    return p.parse_args()


def _rate_tag(r: float) -> str:
    return str(r).replace(".", "p")


def _artifact_path(dataset, seed, noise_type, noise_rate):
    return ROOT / "artifacts" / dataset / f"seed{seed}_noise{noise_type}_{_rate_tag(noise_rate)}" / "split.pt"


def _save_trace_csv(path: Path, trace: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trace:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)


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
        embeddings, noisy_labels, train_idx,
        purity_threshold=gb_cfg["purity_threshold"],
        min_samples=gb_cfg["min_samples"],
        max_size=gb_cfg["max_size"],
    )
    k = max(1, int(round(sum(sizes) / max(len(sizes), 1))))
    n_clusters = max(1, len(sizes))

    rel_topk = compute_reliability("topk", embeddings, noisy_labels, train_idx, k=k)
    rel_gb = compute_reliability("gb", embeddings, noisy_labels, train_idx, group_ids=group_ids)
    rel_none = compute_reliability("none", embeddings, noisy_labels, train_idx)

    cls_cfg = cfg["classifier"]
    sel_cfg = cfg["selection"]
    rho = 1.0 - args.noise_rate
    common = dict(
        embeddings=embeddings,
        noisy_labels=noisy_labels,
        clean_labels=clean_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=num_classes,
        device=device,
        epochs=cls_cfg["epochs"],
        lr=cls_cfg["lr"],
        wd=cls_cfg["weight_decay"],
        patience=cls_cfg["patience"],
        seed=args.seed,
        trace=True,
    )

    methods = {}
    print("Running ce ...")
    methods["ce"] = train_ce_baseline(**common)
    for name, rel in [("sel_topk", rel_topk), ("sel_gb", rel_gb), ("sel_noball", rel_none)]:
        print(f"Running {name} ...")
        methods[name] = train_with_selection(
            **common,
            reliability=rel,
            warmup=sel_cfg["warmup"],
            ramp=sel_cfg["ramp"],
            rho=rho,
        )

    out_dir = ROOT / "reports" / f"diag_{args.dataset}_{args.noise_type}_{_rate_tag(args.noise_rate)}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    summary_lines.append(f"# Selection Diagnostic: {args.dataset} {args.noise_type} {args.noise_rate} seed{args.seed}")
    summary_lines.append("")

    for name, res in methods.items():
        trace = res.get("trace", [])
        _save_trace_csv(out_dir / f"trace_{name}.csv", trace)
        stopped = res.get("stopped_epoch", "?")
        test_acc = res["test_acc_at_best_val"]
        summary_lines.append(f"## {name}")
        summary_lines.append(f"- test_acc_at_best_val: {test_acc:.4f}")
        summary_lines.append(f"- stopped_epoch: {stopped}")
        if trace:
            warmup_end = min(sel_cfg["warmup"], len(trace) - 1)
            ramp_end = min(sel_cfg["warmup"] + sel_cfg["ramp"], len(trace) - 1)
            summary_lines.append(f"- val_acc at epoch 0: {trace[0]['val_acc']:.4f}")
            summary_lines.append(f"- val_acc at warmup end (epoch {warmup_end}): {trace[warmup_end]['val_acc']:.4f}")
            if ramp_end < len(trace):
                summary_lines.append(f"- val_acc at ramp end (epoch {ramp_end}): {trace[ramp_end]['val_acc']:.4f}")
            summary_lines.append(f"- val_acc at stop (epoch {stopped}): {trace[-1]['val_acc']:.4f}")
            summary_lines.append(f"- keep_ratio at stop: {trace[-1]['keep_ratio']:.4f}")
            summary_lines.append(f"- selected_clean_rate at stop: {trace[-1]['selected_clean_rate']:.4f}")
            summary_lines.append(f"- selected_count at stop: {trace[-1]['selected_count']}")

            best_val_epoch = max(range(len(trace)), key=lambda i: trace[i]["val_acc"])
            summary_lines.append(f"- best val_acc: {trace[best_val_epoch]['val_acc']:.4f} at epoch {best_val_epoch}")
            early_stop_before_ramp = stopped < sel_cfg["warmup"] + sel_cfg["ramp"]
            summary_lines.append(f"- early_stop_before_ramp_end: {early_stop_before_ramp}")
        summary_lines.append("")

    summary_lines.append("## Key diagnostic questions")
    summary_lines.append("")

    ce_stopped = methods["ce"].get("stopped_epoch", 999)
    sel_topk_stopped = methods["sel_topk"].get("stopped_epoch", 999)
    sel_gb_stopped = methods["sel_gb"].get("stopped_epoch", 999)

    summary_lines.append(f"1. CE stopped at epoch {ce_stopped}, sel_topk at {sel_topk_stopped}, sel_gb at {sel_gb_stopped}")

    ramp_end_epoch = sel_cfg["warmup"] + sel_cfg["ramp"]
    for name in ["sel_topk", "sel_gb", "sel_noball"]:
        stopped = methods[name].get("stopped_epoch", 999)
        if stopped < ramp_end_epoch:
            summary_lines.append(f"   WARNING: {name} early-stopped at epoch {stopped} < ramp end {ramp_end_epoch} — selection never fully engaged!")

    ce_test = methods["ce"]["test_acc_at_best_val"]
    for name in ["sel_topk", "sel_gb", "sel_noball"]:
        delta = methods[name]["test_acc_at_best_val"] - ce_test
        summary_lines.append(f"2. {name} - ce = {delta:+.4f} ({'selection helps' if delta > 0 else 'selection hurts'})")

    report = "\n".join(summary_lines)
    (out_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nTraces saved to {out_dir}")


if __name__ == "__main__":
    main()
