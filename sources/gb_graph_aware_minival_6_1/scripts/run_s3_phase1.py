"""S3 Phase 1 — phenomenon robustness (commit-style ball-guided selection, multi-seed).

Question (NOT a method search): does ball *quality* (purity / noise-AUROC) predict downstream
*conversion*? Seed-0 showed an inversion: dcgb (reliable, high purity) selects WORSE than ordinary gb.
This script checks whether that inversion holds across seeds, recording ball-quality AND conversion
diagnostics per signal per seed. It does NOT explain (Phase 2) and does NOT claim sel_gb is a method.

Cell: Cora, ratio_train0.2_val0.1, split0, symmetric 0.4, seeds 0-4 (each seed s uses
pretrain{s}/noise{s}/clf{s}). NOT planetoid, NOT NoisyGL. Pre-registered gates A/B/C (see below).

Pre-registered phenomena (mechanical verdicts):
  A  sel_gb > sel_dcgb_emb         in >=4/5 seeds AND mean delta >= 1.0 pp
  B  sel_gb > gb_soft              in >=4/5 seeds AND mean delta >= 0.5 pp
  C  sel_dcgb_emb > {shuffled, random, noball}  in >=4/5 seeds (each)

Signals (rel source); clean labels eval/diagnostic only:
  ce(from smoke), gb_soft(from smoke), sel_gb, sel_dcgb_emb, sel_topk_matched,
  sel_noball, sel_shuffled_matched, sel_random_matched, sel_clean_oracle(diagnostic).
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

import csv

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from noisy_label_gb_grace.balls import build_dcgb_balls, build_granular_balls, shuffle_grouping
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation import (
    ball_consensus_score,
    knn_train_consensus_score,
    train_frozen_classifier_with_selection,
)
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing.targets import build_topk_group_size
from noisy_label_gb_grace.utils import get_device, mask_to_index, save_csv, save_json
from scripts.extract_embeddings import embedding_path

# methods that USE a ball/knn reliability signal (for quality-vs-conversion diagnostics)
BALL_SIGNALS = {"sel_gb", "sel_dcgb_emb", "sel_topk_matched", "sel_shuffled_matched", "sel_random_matched"}


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
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/s3_phase1_20260607")
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


def _clean_purity(grouping, clean, noisy, train_mask, num_classes):
    """group-majority clean purity, same definition as run_noisy_classifier._group_label_diagnostics."""
    clean = clean.cpu().long(); noisy = noisy.cpu().long(); tm = train_mask.cpu().bool()
    tot = 0; maj = 0
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        local = idx[tm[idx]]
        if local.numel() == 0:
            continue
        tot += int(local.numel())
        maj += int(torch.bincount(clean[local], minlength=num_classes).max().item())
    return float(maj / max(1, tot))


def _rel_auroc(rel, covered, clean, noisy, train_mask):
    """noise-detection AUROC using the reliability signal: suspicion = 1 - rel on covered train nodes."""
    tm = train_mask.cpu().bool()
    tr = mask_to_index(tm)
    cov = covered.cpu().bool()[tr]
    is_noisy = (clean.cpu()[tr] != noisy.cpu()[tr])[cov].numpy().astype(int)
    susp = (1.0 - rel.cpu().float()[tr])[cov].numpy()
    if is_noisy.sum() == 0 or is_noisy.sum() == len(is_noisy):
        return math.nan
    return float(roc_auc_score(is_noisy, susp))


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

    gb_grouping = build_granular_balls(h_regions, min_ball_size=args.gb_min_ball_size,
                                       min_radius_gain=args.gb_min_radius_gain, seed=s)
    dcgb_emb = build_dcgb_balls(h_regions, edge_index, max_hops=args.gagb_max_hops,
                                min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size,
                                seed=s, mode="emb_only")
    shuffled = shuffle_grouping(dcgb_emb, seed=s, embeddings=h_regions)
    random_g = shuffle_grouping(dcgb_emb, seed=s + 1, embeddings=h_regions)
    topk = build_topk_group_size(dcgb_emb["groups"], train_mask=masks["train_mask"],
                                 min_train_per_group=args.min_train_per_ball, match_train_nodes=True)

    def consensus(g):
        return ball_consensus_score(g, noisy_labels, masks["train_mask"], num_classes)

    rels = {
        "sel_gb": (consensus(gb_grouping), gb_grouping),
        "sel_dcgb_emb": (consensus(dcgb_emb), dcgb_emb),
        "sel_shuffled_matched": (consensus(shuffled), shuffled),
        "sel_random_matched": (consensus(random_g), random_g),
        "sel_topk_matched": (knn_train_consensus_score(h_regions, noisy_labels, masks["train_mask"], int(topk), num_classes), None),
    }

    keep_ratio_final = 1.0 - float(args.noise_rate)
    common = dict(embeddings=h, clean_labels=clean_labels, noisy_labels=noisy_labels, num_classes=num_classes,
                  train_mask=masks["train_mask"], val_mask=masks["val_mask"], test_mask=masks["test_mask"],
                  seed=s, keep_ratio_final=keep_ratio_final, warmup_epochs=args.warmup_epochs,
                  ramp_epochs=args.ramp_epochs, epochs=args.classifier_epochs, learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay, patience=args.patience, device=device)

    specs = [
        ("sel_dcgb_emb", dict(use_reliability=True)),
        ("sel_gb", dict(use_reliability=True)),
        ("sel_topk_matched", dict(use_reliability=True)),
        ("sel_shuffled_matched", dict(use_reliability=True)),
        ("sel_random_matched", dict(use_reliability=True)),
        ("sel_noball", dict(use_reliability=False)),
        ("sel_clean_oracle", dict(use_reliability=False, clean_oracle=True)),
    ]

    rows = []
    for name, kw in specs:
        extra = {}
        if name in rels:
            (rel, cov), grouping = rels[name]
            kw = {**kw, "reliability": rel, "reliability_covered": cov}
            if grouping is not None:
                extra["clean_purity"] = _clean_purity(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes)
                extra["ball_count"] = int(len(grouping["groups"]))
                extra["avg_ball_size"] = float(sum(len(g) for g in grouping["groups"]) / max(1, len(grouping["groups"])))
            extra["rel_noise_auroc"] = _rel_auroc(rel, cov, clean_labels, noisy_labels, masks["train_mask"])
        res = train_frozen_classifier_with_selection(**common, **kw)
        row = {"seed": s, "method": name,
               "test_acc_at_best_val": res["test_acc_at_best_val"],
               "best_val_acc": res["best_val_acc"], "best_epoch": res["best_epoch"],
               "selected_coverage": res["selected_coverage"], "selected_count": res["selected_count"],
               "selection_precision": res["selection_precision"],
               "noise_excluded_recall": res["noise_excluded_recall"],
               "loss_mean_selected": res["loss_mean_selected"], "loss_mean_excluded": res["loss_mean_excluded"],
               "clean_selected_rate_by_class": json.dumps(res["clean_selected_rate_by_class"]),
               "noisy_excluded_rate_by_class": json.dumps(res["noisy_excluded_rate_by_class"]),
               "selected_clean_class_distribution": json.dumps(res["selected_clean_class_distribution"]),
               "selected_noisy_class_distribution": json.dumps(res["selected_noisy_class_distribution"]),
               "keep_ratio_final": keep_ratio_final, "topk": int(topk),
               **{k: extra.get(k, math.nan) for k in ["clean_purity", "rel_noise_auroc", "ball_count", "avg_ball_size"]}}
        rows.append(row)
        print(f"(R) s{s} | {name:22s} test={res['test_acc_at_best_val']:.4f} "
              f"sel_prec={res['selection_precision']:.4f} purity={extra.get('clean_purity', float('nan')):.4f} "
              f"rel_auroc={extra.get('rel_noise_auroc', float('nan')):.4f}")

    ce = _smoke_baseline(args.dataset, split_token, s, args.noise_rate, "ce", "0")
    gb_soft = _smoke_baseline(args.dataset, split_token, s, args.noise_rate, "gb", "0.3")
    for r in rows:
        r["ce_smoke"] = ce
        r["gb_soft_smoke"] = gb_soft
    return rows, ce, gb_soft


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def main():
    args = parse_args()
    device = get_device(args.gpu_id)
    all_rows = []
    per_seed = {}  # seed -> {method: test}
    gb_soft_by_seed = {}
    for s in args.seeds:
        rows, ce, gb_soft = run_seed(args, s, device)
        all_rows.extend(rows)
        per_seed[s] = {r["method"]: r["test_acc_at_best_val"] for r in rows}
        per_seed[s]["gb_soft"] = gb_soft
        per_seed[s]["ce"] = ce
        gb_soft_by_seed[s] = gb_soft

    # ---- pre-registered phenomena (mechanical) ----
    seeds = args.seeds
    def deltas(a, b):
        return [per_seed[s][a] - per_seed[s][b] for s in seeds
                if per_seed[s].get(a) is not None and per_seed[s].get(b) is not None]

    dA = deltas("sel_gb", "sel_dcgb_emb")
    A_wins = sum(1 for d in dA if d > 0)
    A = (A_wins >= 4) and (_mean(dA) >= 0.01)

    dB = deltas("sel_gb", "gb_soft")
    B_wins = sum(1 for d in dB if d > 0)
    B = (B_wins >= 4) and (_mean(dB) >= 0.005)

    C_parts = {}
    C = True
    for ctrl in ["sel_shuffled_matched", "sel_random_matched", "sel_noball"]:
        dc = deltas("sel_dcgb_emb", ctrl)
        wins = sum(1 for d in dc if d > 0)
        C_parts[ctrl] = {"wins": wins, "mean_delta": _mean(dc)}
        C = C and (wins >= 4)

    print("\n=== per-seed test acc ===")
    methods_order = ["ce", "gb_soft", "sel_clean_oracle", "sel_dcgb_emb", "sel_gb",
                     "sel_topk_matched", "sel_noball", "sel_shuffled_matched", "sel_random_matched"]
    hdr = "seed  " + "  ".join(m[:10].rjust(10) for m in methods_order)
    print(hdr)
    for s in seeds:
        print(f"  {s}   " + "  ".join((f"{per_seed[s].get(m):.4f}" if per_seed[s].get(m) is not None else "   nan  ").rjust(10) for m in methods_order))
    print("mean  " + "  ".join(f"{_mean([per_seed[s].get(m) for s in seeds]):.4f}".rjust(10) for m in methods_order))

    print("\n=== PRE-REGISTERED PHENOMENA ===")
    print(f"  A  sel_gb > sel_dcgb_emb : wins {A_wins}/5, mean delta {_mean(dA)*100:.2f}pp  -> {'HOLDS' if A else 'FAILS'} (need >=4/5 & >=1.00pp)")
    print(f"  B  sel_gb > gb_soft      : wins {B_wins}/5, mean delta {_mean(dB)*100:.2f}pp  -> {'HOLDS' if B else 'FAILS'} (need >=4/5 & >=0.50pp)")
    for ctrl, v in C_parts.items():
        print(f"  C  sel_dcgb_emb > {ctrl:20s}: wins {v['wins']}/5, mean delta {v['mean_delta']*100:.2f}pp")
    print(f"  C overall -> {'HOLDS' if C else 'FAILS'} (need >=4/5 each)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir / "s3_phase1_rows.csv", all_rows)
    save_json(args.out_dir / "s3_phase1_summary.json", {
        "setting": (f"{args.dataset} ratio_train{args.train_ratio}_val{args.val_ratio} "
                    f"{args.noise_type}{args.noise_rate} seeds{seeds} frozen-GRACE (NOT planetoid, NOT NoisyGL)"),
        "per_seed_test": per_seed,
        "phenomena": {
            "A_sel_gb_gt_dcgb": {"wins": A_wins, "mean_delta_pp": _mean(dA) * 100, "holds": A},
            "B_sel_gb_gt_gb_soft": {"wins": B_wins, "mean_delta_pp": _mean(dB) * 100, "holds": B},
            "C_dcgb_gt_controls": {**{k: v for k, v in C_parts.items()}, "holds": C},
        },
    })
    print(f"\nsaved {args.out_dir / 's3_phase1_rows.csv'}")


if __name__ == "__main__":
    main()
