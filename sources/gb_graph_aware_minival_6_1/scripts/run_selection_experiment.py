"""S2-redesign-v1 — ball-guided sample-selection experiment (one narrow shot).

Commit-style mechanism: small-loss x ball-reliability selection on a frozen GRACE head (hard noisy
labels). The keep-ratio schedule is FIXED a priori and identical across the method and every control;
only the reliability (`rel`) source differs, so controls are size-matched by construction.

Cell (must match the Phase 1 smoke / headroom exactly):
    Cora, ratio_train0.2_val0.1, split0, pretrain0, symmetric 0.4, seed 0, frozen GRACE.
This is a RATIO-SPLIT minival result. NOT planetoid, NOT NoisyGL. Single-cell smoke, not a formal run.

Methods:
    sel_dcgb_emb       -- rel from reciprocal-kNN reliable balls (THE METHOD)
    sel_noball         -- loss-only selection (no ball signal)            [control]
    sel_gb             -- rel from ordinary granular balls                [control]
    sel_shuffled_matched -- rel from shuffle_grouping(dcgb_emb)           [control]
    sel_random_matched -- rel from shuffle_grouping(dcgb_emb, seed+1)     [control, 2nd random draw]
    sel_topk_matched   -- rel from k-NN train neighbors (no-ball matched) [control]
    sel_clean_oracle   -- perfect selection (noisy==clean)               [DIAGNOSTIC upper bound only]

Gate (ALL must hold, else STOP the S2 method line and pivot to S3):
    1. sel_dcgb_emb test > gb soft-target baseline (0.7962 from the smoke)
    2. sel_dcgb_emb test > sel_noball
    3. sel_dcgb_emb test > sel_random_matched AND > sel_shuffled_matched
    4. selection_precision(sel_dcgb_emb) > sel_noball AND > sel_random_matched
    5. fixed keep-ratio (honored by construction)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.balls import build_dcgb_balls, build_granular_balls, shuffle_grouping
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation import (
    ball_consensus_score,
    knn_train_consensus_score,
    train_frozen_classifier_with_selection,
)
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing.targets import build_topk_group_size
from noisy_label_gb_grace.utils import get_device, save_csv, save_json
from scripts.extract_embeddings import embedding_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="ratio")
    ap.add_argument("--train-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--pretrain-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--classifier-seed", type=int, default=0)
    ap.add_argument("--noise-type", default="symmetric")
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    # fixed selection schedule (NOT swept)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--ramp-epochs", type=int, default=20)
    ap.add_argument("--classifier-epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-train-per-ball", type=int, default=3)
    ap.add_argument(
        "--smoke-csv",
        type=Path,
        default=ROOT
        / "results/gate_runs/Cora/symmetric_0p4/ratio_train0p2_val0p1_split0_pretrain0_noise0_clf0/result_rows.csv",
    )
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/s2a_plan_20260607")
    return ap.parse_args()


def _smoke_baseline(smoke_csv: Path, method: str, alpha: str) -> float | None:
    if not smoke_csv.exists():
        return None
    for r in csv.DictReader(smoke_csv.open(encoding="utf-8")):
        if r.get("method") == method and f"{float(r['alpha_max']):g}" == alpha:
            return float(r["test_acc_at_best_val"])
    return None


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu_id)
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)

    emb = embedding_path(
        args.dataset, args.split_seed, args.pretrain_seed,
        args.split_mode, args.train_ratio, args.val_ratio,
    )
    payload = torch.load(emb, map_location="cpu")
    h = payload["H"].float()
    h_regions = F.normalize(h, p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            args.dataset, device=None, split_mode=args.split_mode,
            split_seed=args.split_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index

    noisy_labels, _, _ = load_or_create_noisy_labels(
        clean_labels, masks["train_mask"], num_classes,
        dataset=args.dataset, noise_type=args.noise_type, noise_rate=args.noise_rate,
        split_seed=args.split_seed, noise_seed=args.noise_seed, split_name=split_token,
    )

    # ---- groupings (identical params to run_noisy_classifier) ----
    gb_grouping = build_granular_balls(
        h_regions, min_ball_size=args.gb_min_ball_size,
        min_radius_gain=args.gb_min_radius_gain, seed=args.pretrain_seed,
    )
    dcgb_emb_grouping = build_dcgb_balls(
        h_regions, edge_index, max_hops=args.gagb_max_hops, min_ball_size=args.gb_min_ball_size,
        max_ball_size=args.gagb_max_ball_size, seed=args.pretrain_seed, mode="emb_only",
    )
    shuffled_grouping = shuffle_grouping(dcgb_emb_grouping, seed=args.noise_seed, embeddings=h_regions)
    random_grouping = shuffle_grouping(dcgb_emb_grouping, seed=args.noise_seed + 1, embeddings=h_regions)
    topk = build_topk_group_size(
        dcgb_emb_grouping["groups"], train_mask=masks["train_mask"],
        min_train_per_group=args.min_train_per_ball, match_train_nodes=True,
    )

    # ---- per-node reliability per source (no clean labels) ----
    def consensus(grouping):
        return ball_consensus_score(grouping, noisy_labels, masks["train_mask"], num_classes)

    rel_dcgb, cov_dcgb = consensus(dcgb_emb_grouping)
    rel_gb, cov_gb = consensus(gb_grouping)
    rel_shuf, cov_shuf = consensus(shuffled_grouping)
    rel_rand, cov_rand = consensus(random_grouping)
    rel_topk, cov_topk = knn_train_consensus_score(
        h_regions, noisy_labels, masks["train_mask"], int(topk), num_classes
    )

    keep_ratio_final = 1.0 - float(args.noise_rate)
    common = dict(
        clean_labels=clean_labels, noisy_labels=noisy_labels, num_classes=num_classes,
        train_mask=masks["train_mask"], val_mask=masks["val_mask"], test_mask=masks["test_mask"],
        seed=args.classifier_seed, keep_ratio_final=keep_ratio_final,
        warmup_epochs=args.warmup_epochs, ramp_epochs=args.ramp_epochs,
        epochs=args.classifier_epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, patience=args.patience, device=device,
    )

    specs = [
        ("sel_dcgb_emb", dict(reliability=rel_dcgb, reliability_covered=cov_dcgb, use_reliability=True)),
        ("sel_noball", dict(use_reliability=False)),
        ("sel_gb", dict(reliability=rel_gb, reliability_covered=cov_gb, use_reliability=True)),
        ("sel_shuffled_matched", dict(reliability=rel_shuf, reliability_covered=cov_shuf, use_reliability=True)),
        ("sel_random_matched", dict(reliability=rel_rand, reliability_covered=cov_rand, use_reliability=True)),
        ("sel_topk_matched", dict(reliability=rel_topk, reliability_covered=cov_topk, use_reliability=True)),
        ("sel_clean_oracle", dict(use_reliability=False, clean_oracle=True)),
    ]

    rows = []
    for name, kw in specs:
        res = train_frozen_classifier_with_selection(embeddings=h, **common, **kw)
        rows.append({
            "method": name,
            "test_acc_at_best_val": res["test_acc_at_best_val"],
            "best_val_acc": res["best_val_acc"],
            "best_epoch": res["best_epoch"],
            "selected_coverage": res["selected_coverage"],
            "selected_count": res["selected_count"],
            "selection_precision": res["selection_precision"],
            "noise_excluded_recall": res["noise_excluded_recall"],
            "keep_ratio_final": keep_ratio_final,
            "topk": int(topk),
            "dataset": args.dataset, "split_name": split_token,
            "noise_type": args.noise_type, "noise_rate": args.noise_rate,
        })
        print(f"(R) | {name:22s} test={res['test_acc_at_best_val']:.4f} "
              f"val={res['best_val_acc']:.4f} ep={res['best_epoch']:>3} "
              f"cov={res['selected_coverage']:.3f} sel_prec={res['selection_precision']:.4f} "
              f"noise_excl={res['noise_excluded_recall']:.4f}")

    by = {r["method"]: r for r in rows}
    ce = _smoke_baseline(args.smoke_csv, "ce", "0")
    gb_soft = _smoke_baseline(args.smoke_csv, "gb", "0.3")

    dcgb_t = by["sel_dcgb_emb"]["test_acc_at_best_val"]
    dcgb_p = by["sel_dcgb_emb"]["selection_precision"]
    g1 = (gb_soft is not None) and (dcgb_t > gb_soft)
    g2 = dcgb_t > by["sel_noball"]["test_acc_at_best_val"]
    g3 = (dcgb_t > by["sel_random_matched"]["test_acc_at_best_val"]
          and dcgb_t > by["sel_shuffled_matched"]["test_acc_at_best_val"])
    g4 = (dcgb_p > by["sel_noball"]["selection_precision"]
          and dcgb_p > by["sel_random_matched"]["selection_precision"])
    g5 = True
    passed = all([g1, g2, g3, g4, g5])

    print("\n=== context (from Phase 1 smoke) ===")
    print(f"  ce={ce}  gb_soft_target={gb_soft}  clean_oracle_CE=0.8221 (headroom 3.43pp)")
    print("=== GATE (all must hold) ===")
    print(f"  1. dcgb {dcgb_t:.4f} > gb_soft {gb_soft}: {g1}")
    print(f"  2. dcgb {dcgb_t:.4f} > noball {by['sel_noball']['test_acc_at_best_val']:.4f}: {g2}")
    print(f"  3. dcgb > random {by['sel_random_matched']['test_acc_at_best_val']:.4f} "
          f"& shuffled {by['sel_shuffled_matched']['test_acc_at_best_val']:.4f}: {g3}")
    print(f"  4. sel_prec dcgb {dcgb_p:.4f} > noball {by['sel_noball']['selection_precision']:.4f} "
          f"& random {by['sel_random_matched']['selection_precision']:.4f}: {g4}")
    print(f"  5. fixed keep-ratio: {g5}")
    print(f">>> VERDICT: {'PASS — S2 method line survives' if passed else 'STOP — pivot to S3'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir / "s2a_redesign_v1_selection.csv", rows)
    save_json(args.out_dir / "s2a_redesign_v1_selection.json", {
        "setting": (f"{args.dataset} ratio_train{args.train_ratio}_val{args.val_ratio} "
                    f"{args.noise_type}{args.noise_rate} seed{args.classifier_seed} frozen-GRACE "
                    f"pretrain{args.pretrain_seed} (NOT planetoid, NOT NoisyGL)"),
        "schedule": {"warmup_epochs": args.warmup_epochs, "ramp_epochs": args.ramp_epochs,
                     "keep_ratio_final": keep_ratio_final, "topk": int(topk)},
        "context": {"ce": ce, "gb_soft_target": gb_soft, "clean_oracle_ce": 0.8221, "headroom_pp": 3.43},
        "rows": rows,
        "gate": {"g1_gt_gb": g1, "g2_gt_noball": g2, "g3_gt_random_shuffled": g3,
                 "g4_precision": g4, "g5_fixed_ratio": g5, "passed": passed},
    })
    print(f"saved {args.out_dir / 's2a_redesign_v1_selection.csv'}")


if __name__ == "__main__":
    main()
