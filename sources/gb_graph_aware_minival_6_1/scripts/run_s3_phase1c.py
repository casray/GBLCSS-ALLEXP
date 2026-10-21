"""S3 Phase 1c — density-guided ball-level selection diagnostic (frozen GRACE, multi-seed).

The LAST frozen-space check of whether ball STRUCTURE has any value a kNN consensus lacks.
Q2 showed dcgb (mutual-kNN ball) ties topk (kNN consensus). Here we add a GBmooer-style
*ball-level density* reliability signal (density_score(B) x compactness(i,B) x noisy_consensus(B))
that a per-node kNN consensus cannot express, with shuffled/random matched controls.

NOT a method claim, NOT a schedule change, NOT a representation retrain. Clean labels eval/diagnostic
only. Cell: Cora / symmetric 0.4 / ratio_train0.2 / seeds 0-4 / frozen GRACE.

Methods: ce, gb_soft (from smoke), sel_topk_matched, sel_dcgb_emb, sel_density_gb,
         sel_shuffled_density_matched, sel_random_density_matched, sel_clean_oracle (diagnostic).

Pre-registered verdict:
  sel_density_gb > sel_topk_matched in >=4/5 seeds AND mean test delta >= 0.5pp
  AND selection_precision(density) >= topk AND noise_excl_recall(density) >= topk
  -> ball-level density structure has independent value on frozen GRACE.
  else -> ball structure has no independent increment on frozen GRACE; keeping the granular-ball
          mainline requires a real ball-aware contrastive representation / encoder coupling.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    ball_density_reliability,
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
    ap.add_argument("--noise-type", default="symmetric")
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--ramp-epochs", type=int, default=20)
    ap.add_argument("--classifier-epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--min-train-per-ball", type=int, default=3)
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/s3_phase1c_density_20260607")
    return ap.parse_args()


def _smoke_baseline(dataset, split_token, s, noise_rate, method, alpha):
    rate = f"{noise_rate:g}".replace(".", "p")
    csv_path = (ROOT / "results/gate_runs" / dataset / f"symmetric_{rate}"
                / f"{split_token}_pretrain{s}_noise{s}_clf{s}" / "result_rows.csv")
    if not csv_path.exists():
        return None
    for r in csv.DictReader(csv_path.open(encoding="utf-8")):
        if r.get("method") == method and f"{float(r['alpha_max']):g}" == alpha:
            return float(r["test_acc_at_best_val"])
    return None


def run_seed(args, s, device):
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    emb = embedding_path(args.dataset, args.split_seed, s, args.split_mode, args.train_ratio, args.val_ratio)
    payload = torch.load(emb, map_location="cpu")
    h = payload["H"].float()
    h_regions = F.normalize(h, p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(args.dataset, device=None, split_mode=args.split_mode,
                               split_seed=args.split_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        edge_index = data.edge_index
    noisy_labels, _, _ = load_or_create_noisy_labels(
        clean_labels, masks["train_mask"], num_classes, dataset=args.dataset,
        noise_type=args.noise_type, noise_rate=args.noise_rate,
        split_seed=args.split_seed, noise_seed=s, split_name=split_token,
    )

    dcgb_emb = build_dcgb_balls(h_regions, edge_index, max_hops=args.gagb_max_hops,
                                min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size,
                                seed=s, mode="emb_only")
    shuffled = shuffle_grouping(dcgb_emb, seed=s, embeddings=h_regions)
    random_g = shuffle_grouping(dcgb_emb, seed=s + 1, embeddings=h_regions)
    topk = build_topk_group_size(dcgb_emb["groups"], train_mask=masks["train_mask"],
                                 min_train_per_group=args.min_train_per_ball, match_train_nodes=True)

    rel_dcgb, cov_dcgb = ball_consensus_score(dcgb_emb, noisy_labels, masks["train_mask"], num_classes)
    rel_topk, cov_topk = knn_train_consensus_score(h_regions, noisy_labels, masks["train_mask"], int(topk), num_classes)
    rel_dens, cov_dens = ball_density_reliability(dcgb_emb, h_regions, noisy_labels, masks["train_mask"], num_classes)
    rel_dens_sh, cov_dens_sh = ball_density_reliability(shuffled, h_regions, noisy_labels, masks["train_mask"], num_classes)
    rel_dens_rd, cov_dens_rd = ball_density_reliability(random_g, h_regions, noisy_labels, masks["train_mask"], num_classes)

    keep_ratio_final = 1.0 - float(args.noise_rate)
    common = dict(embeddings=h, clean_labels=clean_labels, noisy_labels=noisy_labels, num_classes=num_classes,
                  train_mask=masks["train_mask"], val_mask=masks["val_mask"], test_mask=masks["test_mask"],
                  seed=s, keep_ratio_final=keep_ratio_final, warmup_epochs=args.warmup_epochs,
                  ramp_epochs=args.ramp_epochs, epochs=args.classifier_epochs, learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay, patience=args.patience, device=device)

    specs = [
        ("sel_dcgb_emb", dict(reliability=rel_dcgb, reliability_covered=cov_dcgb, use_reliability=True)),
        ("sel_topk_matched", dict(reliability=rel_topk, reliability_covered=cov_topk, use_reliability=True)),
        ("sel_density_gb", dict(reliability=rel_dens, reliability_covered=cov_dens, use_reliability=True)),
        ("sel_shuffled_density_matched", dict(reliability=rel_dens_sh, reliability_covered=cov_dens_sh, use_reliability=True)),
        ("sel_random_density_matched", dict(reliability=rel_dens_rd, reliability_covered=cov_dens_rd, use_reliability=True)),
        ("sel_clean_oracle", dict(use_reliability=False, clean_oracle=True)),
    ]
    rows = []
    for name, kw in specs:
        res = train_frozen_classifier_with_selection(**common, **kw)
        rows.append({"seed": s, "method": name,
                     "test_acc_at_best_val": res["test_acc_at_best_val"], "best_val_acc": res["best_val_acc"],
                     "best_epoch": res["best_epoch"], "selected_coverage": res["selected_coverage"],
                     "selection_precision": res["selection_precision"], "noise_excluded_recall": res["noise_excluded_recall"],
                     "loss_mean_selected": res["loss_mean_selected"], "loss_mean_excluded": res["loss_mean_excluded"],
                     "keep_ratio_final": keep_ratio_final, "topk": int(topk)})
        print(f"(R) s{s} | {name:30s} test={res['test_acc_at_best_val']:.4f} "
              f"sel_prec={res['selection_precision']:.4f} noise_excl={res['noise_excluded_recall']:.4f}")
    ce = _smoke_baseline(args.dataset, split_token, s, args.noise_rate, "ce", "0")
    gb_soft = _smoke_baseline(args.dataset, split_token, s, args.noise_rate, "gb", "0.3")
    for r in rows:
        r["ce_smoke"] = ce; r["gb_soft_smoke"] = gb_soft
    return rows, ce, gb_soft


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    all_rows = []
    per_seed = {}
    for s in args.seeds:
        rows, ce, gb_soft = run_seed(args, s, device)
        all_rows.extend(rows)
        d = {r["method"]: r for r in rows}
        per_seed[s] = {"rows": d, "ce": ce, "gb_soft": gb_soft}

    seeds = args.seeds
    def col(method, c):
        return [per_seed[s]["rows"][method][c] for s in seeds if method in per_seed[s]["rows"]]
    def delta(a, b, c):
        return [per_seed[s]["rows"][a][c] - per_seed[s]["rows"][b][c] for s in seeds
                if a in per_seed[s]["rows"] and b in per_seed[s]["rows"]]

    d_test = delta("sel_density_gb", "sel_topk_matched", "test_acc_at_best_val")
    test_wins = sum(1 for x in d_test if x > 0)
    prec_ok = _mean(col("sel_density_gb", "selection_precision")) >= _mean(col("sel_topk_matched", "selection_precision"))
    recall_ok = _mean(col("sel_density_gb", "noise_excluded_recall")) >= _mean(col("sel_topk_matched", "noise_excluded_recall"))
    verdict = (test_wins >= 4) and (_mean(d_test) >= 0.005) and prec_ok and recall_ok

    # density vs its own shuffled/random controls (sanity: structure must beat destroyed structure)
    d_sh = delta("sel_density_gb", "sel_shuffled_density_matched", "test_acc_at_best_val")
    d_rd = delta("sel_density_gb", "sel_random_density_matched", "test_acc_at_best_val")

    order = ["ce", "gb_soft", "sel_clean_oracle", "sel_density_gb", "sel_dcgb_emb",
             "sel_topk_matched", "sel_shuffled_density_matched", "sel_random_density_matched"]
    def val(s, m):
        if m in ("ce", "gb_soft"):
            return per_seed[s][m]
        return per_seed[s]["rows"].get(m, {}).get("test_acc_at_best_val")
    print("\n=== per-seed test acc ===")
    print("seed  " + "  ".join(m[:12].rjust(12) for m in order))
    for s in seeds:
        print(f"  {s}   " + "  ".join((f"{val(s,m):.4f}" if val(s, m) is not None else "  nan ").rjust(12) for m in order))
    print("mean  " + "  ".join(f"{_mean([val(s,m) for s in seeds]):.4f}".rjust(12) for m in order))

    print("\n=== PRE-REGISTERED VERDICT (density vs topk) ===")
    print(f"  density>topk test wins {test_wins}/5, mean delta {_mean(d_test)*100:.2f}pp (need >=4/5 & >=0.50pp)")
    print(f"  sel_precision: density {_mean(col('sel_density_gb','selection_precision')):.4f} "
          f">= topk {_mean(col('sel_topk_matched','selection_precision')):.4f} -> {prec_ok}")
    print(f"  noise_excl_recall: density {_mean(col('sel_density_gb','noise_excluded_recall')):.4f} "
          f">= topk {_mean(col('sel_topk_matched','noise_excluded_recall')):.4f} -> {recall_ok}")
    print(f"  density vs shuffled mean delta {_mean(d_sh)*100:+.2f}pp ; vs random {_mean(d_rd)*100:+.2f}pp")
    print(f">>> VERDICT: {'ball density structure HAS independent value on frozen GRACE' if verdict else 'NO independent increment -> needs ball-aware representation / encoder coupling'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir / "s3_phase1c_rows.csv", all_rows)
    save_json(args.out_dir / "s3_phase1c_summary.json", {
        "setting": (f"{args.dataset} ratio_train{args.train_ratio}_val{args.val_ratio} "
                    f"{args.noise_type}{args.noise_rate} seeds{seeds} frozen-GRACE (NOT planetoid, NOT NoisyGL)"),
        "verdict": {
            "density_gt_topk_test_wins": test_wins, "density_gt_topk_mean_delta_pp": _mean(d_test) * 100,
            "precision_ok": prec_ok, "recall_ok": recall_ok, "passed": verdict,
            "density_vs_shuffled_pp": _mean(d_sh) * 100, "density_vs_random_pp": _mean(d_rd) * 100,
        },
        "per_seed_test": {s: {m: val(s, m) for m in order} for s in seeds},
    })
    print(f"\nsaved {args.out_dir / 's3_phase1c_rows.csv'}")


if __name__ == "__main__":
    main()
