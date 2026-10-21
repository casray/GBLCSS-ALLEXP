"""Two-cell deathmatch runner shared by CiteSeer-symmetric and Cora-pair checks.

All methods run inside the minival protocol on the same frozen GRACE embeddings. No NoisyGL baseline
numbers are imported. Clean labels are used only for validation/test evaluation and diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from noisy_label_gb_grace.balls import build_dcgb_balls, shuffle_grouping
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation import (
    ball_consensus_score,
    knn_train_consensus_score,
    train_coteaching,
    train_frozen_classifier,
    train_frozen_classifier_with_selection,
)
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing.targets import build_topk_group_size
from noisy_label_gb_grace.utils import get_device, mask_to_index, save_csv, save_json
from scripts.extract_embeddings import embedding_path


def parse_args(default_dataset: str, default_noise_type: str, default_noise_rate: float, default_out_dir: Path):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=default_dataset)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="ratio")
    ap.add_argument("--train-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--noise-type", default=default_noise_type, choices=["symmetric", "pair"])
    ap.add_argument("--noise-rate", type=float, default=default_noise_rate)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    ap.add_argument("--min-train-per-ball", type=int, default=3)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--ramp-epochs", type=int, default=20)
    ap.add_argument("--classifier-epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=default_out_dir)
    return ap.parse_args()


def _mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else math.nan


def _rel_auroc(rel, covered, clean, noisy, train_mask):
    train_idx = mask_to_index(train_mask.detach().cpu().bool())
    cov = covered.detach().cpu().bool()[train_idx]
    if int(cov.sum()) == 0:
        return math.nan
    is_noisy = (clean.detach().cpu()[train_idx] != noisy.detach().cpu()[train_idx])[cov].numpy().astype(int)
    if is_noisy.sum() == 0 or is_noisy.sum() == len(is_noisy):
        return math.nan
    suspicion = (1.0 - rel.detach().cpu().float()[train_idx])[cov].numpy()
    return float(roc_auc_score(is_noisy, suspicion))


def _clean_purity(grouping, clean, train_mask, num_classes):
    clean = clean.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    total = 0
    majority = 0
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        local = idx[train_mask[idx]]
        if local.numel() == 0:
            continue
        total += int(local.numel())
        majority += int(torch.bincount(clean[local], minlength=num_classes).max().item())
    return float(majority / max(1, total))


def _row_from_result(seed: int, method: str, result: dict, extra: dict | None = None):
    extra = extra or {}
    return {
        "seed": int(seed),
        "method": method,
        "test_acc_at_best_val": float(result["test_acc_at_best_val"]),
        "best_val_acc": float(result["best_val_acc"]),
        "best_epoch": int(result["best_epoch"]),
        "selected_count": result.get("selected_count", math.nan),
        "selected_coverage": result.get("selected_coverage", math.nan),
        "selection_precision": result.get("selection_precision", math.nan),
        "noise_excluded_recall": result.get("noise_excluded_recall", math.nan),
        "train_acc_at_best_state": result.get("train_acc_at_best_state", math.nan),
        "val_acc_at_best_state": result.get("val_acc_at_best_state", math.nan),
        "test_acc_at_best_state": result.get("test_acc_at_best_state", math.nan),
        **extra,
    }


def run_seed(args, seed: int, device: torch.device):
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    emb_path = embedding_path(args.dataset, args.split_seed, seed, args.split_mode, args.train_ratio, args.val_ratio)
    payload = torch.load(emb_path, map_location="cpu")
    h = payload["H"].float()
    h_norm = F.normalize(h, p=2, dim=1)
    clean = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            args.dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index

    noisy, noisy_path, metadata = load_or_create_noisy_labels(
        clean,
        masks["train_mask"],
        num_classes,
        dataset=args.dataset,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        split_seed=args.split_seed,
        noise_seed=seed,
        split_name=split_token,
    )

    dcgb = build_dcgb_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size,
        seed=seed,
        mode="emb_only",
    )
    shuffled = shuffle_grouping(dcgb, seed=seed, embeddings=h_norm)
    random_g = shuffle_grouping(dcgb, seed=seed + 1, embeddings=h_norm)
    topk = build_topk_group_size(
        dcgb["groups"],
        train_mask=masks["train_mask"],
        min_train_per_group=args.min_train_per_ball,
        match_train_nodes=True,
    )

    keep_ratio_final = 1.0 - float(args.noise_rate)
    common = dict(
        embeddings=h,
        clean_labels=clean,
        noisy_labels=noisy,
        num_classes=num_classes,
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        seed=seed,
        keep_ratio_final=keep_ratio_final,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        epochs=args.classifier_epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )

    rows = []
    noisy_targets = F.one_hot(noisy, num_classes=num_classes).float()
    ce = train_frozen_classifier(
        embeddings=h,
        clean_labels=clean,
        soft_targets=noisy_targets,
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        seed=seed,
        epochs=args.classifier_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
    rows.append(_row_from_result(seed, "ce", ce, {"noise_path": str(noisy_path)}))

    coteaching = train_coteaching(
        **common,
        lr=args.learning_rate,
    )
    rows.append(_row_from_result(seed, "coteaching", coteaching))

    def _selection(method: str, reliability=None, covered=None, grouping=None, use_reliability=True, clean_oracle=False):
        res = train_frozen_classifier_with_selection(
            **common,
            learning_rate=args.learning_rate,
            reliability=reliability,
            reliability_covered=covered,
            use_reliability=use_reliability,
            clean_oracle=clean_oracle,
        )
        extra = {"topk": int(topk), "keep_ratio_final": keep_ratio_final}
        if reliability is not None and covered is not None:
            extra["rel_noise_auroc"] = _rel_auroc(reliability, covered, clean, noisy, masks["train_mask"])
        if grouping is not None:
            extra["clean_purity"] = _clean_purity(grouping, clean, masks["train_mask"], num_classes)
            extra["ball_count"] = int(len(grouping["groups"]))
            extra["avg_ball_size"] = float(sum(len(g) for g in grouping["groups"]) / max(1, len(grouping["groups"])))
        rows.append(_row_from_result(seed, method, res, extra))
        print(
            f"s{seed} | {method:18s} test={res['test_acc_at_best_val']:.4f} "
            f"sel_prec={res.get('selection_precision', math.nan):.4f} "
            f"noise_excl={res.get('noise_excluded_recall', math.nan):.4f}"
        )

    rel_dcgb, cov_dcgb = ball_consensus_score(dcgb, noisy, masks["train_mask"], num_classes)
    rel_shuf, cov_shuf = ball_consensus_score(shuffled, noisy, masks["train_mask"], num_classes)
    rel_rand, cov_rand = ball_consensus_score(random_g, noisy, masks["train_mask"], num_classes)
    rel_topk, cov_topk = knn_train_consensus_score(h_norm, noisy, masks["train_mask"], int(topk), num_classes)
    _selection("sel_dcgb", rel_dcgb, cov_dcgb, dcgb)
    _selection("sel_topk", rel_topk, cov_topk, None)
    _selection("sel_shuffled", rel_shuf, cov_shuf, shuffled)
    _selection("sel_random", rel_rand, cov_rand, random_g)
    _selection("sel_noball", use_reliability=False)
    _selection("sel_clean_oracle", use_reliability=False, clean_oracle=True)

    for row in rows:
        row.update(
            {
                "dataset": args.dataset,
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "split_name": split_token,
                "noise_num_flipped": metadata.get("num_flipped", math.nan),
            }
        )
    print(f"s{seed} | {'ce':18s} test={ce['test_acc_at_best_val']:.4f}")
    print(f"s{seed} | {'coteaching':18s} test={coteaching['test_acc_at_best_val']:.4f}")
    return rows


def _method_values(per_seed: dict, method: str, seeds: list[int]):
    return [per_seed[s][method] for s in seeds if method in per_seed[s]]


def _paired_delta(per_seed: dict, a: str, b: str, seeds: list[int]):
    return [per_seed[s][a] - per_seed[s][b] for s in seeds if a in per_seed[s] and b in per_seed[s]]


def evaluate_gates(rows: list[dict], seeds: list[int], noise_type: str):
    per_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        per_seed.setdefault(int(row["seed"]), {})[row["method"]] = float(row["test_acc_at_best_val"])

    def method_gate_vs(method: str, baseline: str, min_wins: int, min_delta: float):
        deltas = _paired_delta(per_seed, method, baseline, seeds)
        wins = sum(1 for d in deltas if d > 0)
        mean_delta = _mean(deltas)
        return {"method": method, "baseline": baseline, "wins": wins, "mean_delta": mean_delta, "passes": wins >= min_wins and mean_delta >= min_delta}

    gate_a_candidates = [
        method_gate_vs("sel_topk", "coteaching", 4, 0.005),
        method_gate_vs("sel_dcgb", "coteaching", 4, 0.005),
    ]
    gate_a = any(c["passes"] for c in gate_a_candidates)

    gate_b_candidates = []
    for method in ["sel_topk", "sel_dcgb"]:
        parts = [
            method_gate_vs(method, "sel_noball", 4, -1.0),
            method_gate_vs(method, "sel_shuffled", 4, -1.0),
            method_gate_vs(method, "sel_random", 4, -1.0),
        ]
        gate_b_candidates.append({"method": method, "parts": parts, "passes": all(p["wins"] >= 4 for p in parts)})
    gate_b = any(c["passes"] for c in gate_b_candidates)

    best_structured = {}
    for s in seeds:
        vals = per_seed.get(s, {})
        if "sel_topk" in vals and "sel_dcgb" in vals:
            best_structured[s] = max(vals["sel_topk"], vals["sel_dcgb"])
    ce_delta = [best_structured[s] - per_seed[s]["ce"] for s in seeds if s in best_structured and "ce" in per_seed[s]]
    c_threshold = 0.015 if noise_type == "pair" else 0.01
    gate_c = _mean(ce_delta) >= c_threshold

    return {
        "gate_A_beats_coteaching": {"passes": gate_a, "candidates": gate_a_candidates},
        "gate_B_beats_structure_controls": {"passes": gate_b, "candidates": gate_b_candidates},
        "gate_C_beats_ce": {
            "passes": gate_c,
            "mean_delta": _mean(ce_delta),
            "threshold": c_threshold,
            "best_structured_by_seed": best_structured,
        },
    }


def write_report(out_dir: Path, args, rows: list[dict], gates: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(out_dir / "deathmatch_rows.csv", rows)
    save_json(out_dir / "deathmatch_gates.json", {"config": vars(args), "gates": gates})

    methods = ["ce", "coteaching", "sel_topk", "sel_dcgb", "sel_noball", "sel_shuffled", "sel_random", "sel_clean_oracle"]
    grouped = {m: [] for m in methods}
    for row in rows:
        if row["method"] in grouped:
            grouped[row["method"]].append(float(row["test_acc_at_best_val"]))
    lines = [
        f"# Deathmatch report: {args.dataset} {args.noise_type} {args.noise_rate}",
        "",
        f"Setting: `{args.dataset}` / `{args.split_mode}_train{args.train_ratio}_val{args.val_ratio}` / `{args.noise_type}` `{args.noise_rate}` / seeds `{args.seeds}` / frozen GRACE minival.",
        "",
        "## Mean Test Accuracy",
        "",
        "| method | mean test |",
        "|---|---:|",
    ]
    for method in methods:
        lines.append(f"| {method} | {_mean(grouped[method]):.4f} |")
    lines.extend(
        [
            "",
            "## Pre-registered Gates",
            "",
            f"- Gate A beats co-teaching: **{'PASS' if gates['gate_A_beats_coteaching']['passes'] else 'FAIL'}**",
            f"- Gate B beats no-ball/shuffled/random controls: **{'PASS' if gates['gate_B_beats_structure_controls']['passes'] else 'FAIL'}**",
            f"- Gate C beats CE with required margin: **{'PASS' if gates['gate_C_beats_ce']['passes'] else 'FAIL'}** "
            f"(mean delta={gates['gate_C_beats_ce']['mean_delta'] * 100:.2f}pp, threshold={gates['gate_C_beats_ce']['threshold'] * 100:.2f}pp)",
            "",
            "## Gate Details",
            "",
            "```json",
            json.dumps(gates, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(default_dataset: str, default_noise_type: str, default_noise_rate: float, default_out_dir: Path):
    args = parse_args(default_dataset, default_noise_type, default_noise_rate, default_out_dir)
    device = get_device(args.gpu_id)
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(args, seed, device))
    gates = evaluate_gates(rows, list(args.seeds), args.noise_type)
    write_report(args.out_dir, args, rows, gates)
    print(f"saved {args.out_dir / 'deathmatch_rows.csv'}")
    print(json.dumps(gates, indent=2, ensure_ascii=False))

