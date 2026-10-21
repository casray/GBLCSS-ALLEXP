"""S3 Phase 2 — intermediate ball-quality diagnostic (the early-stop gate).

Before running the full selection gate, check whether ball coupling actually improved dcgb ball quality
(clean_purity / noise-AUROC) vs the frozen GRACE and the same-source sup-baseline. If quality did NOT
improve, selection almost certainly won't either -> early-stop, attribute as "coupling did not improve
ball quality" (clean failure attribution). Clean labels are used ONLY to MEASURE purity/AUROC (eval).

Compares dcgb (emb_only) ball quality on three same-split representations:
  frozen   : original frozen GRACE H (checkpoints/embeddings/...)
  baseline : finetuned with L_sup + L_GRACE, lambda_gb=0 (same-source control)
  coupled  : finetuned with + lambda_gb * L_ball
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

from noisy_label_gb_grace.balls import build_dcgb_balls
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.evaluation import ball_consensus_score
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.utils import mask_to_index
from scripts.extract_embeddings import embedding_path
from scripts.finetune_ball_coupled_encoder import coupled_embedding_path


def _clean_purity(grouping, clean, noisy, train_mask, num_classes):
    clean = clean.cpu().long(); tm = train_mask.cpu().bool()
    tot = maj = 0
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        local = idx[tm[idx]]
        if local.numel() == 0:
            continue
        tot += int(local.numel())
        maj += int(torch.bincount(clean[local], minlength=num_classes).max().item())
    return float(maj / max(1, tot))


def _rel_auroc(rel, covered, clean, noisy, train_mask):
    tm = train_mask.cpu().bool(); tr = mask_to_index(tm)
    cov = covered.cpu().bool()[tr]
    is_noisy = (clean.cpu()[tr] != noisy.cpu()[tr])[cov].numpy().astype(int)
    susp = (1.0 - rel.cpu().float()[tr])[cov].numpy()
    if is_noisy.sum() == 0 or is_noisy.sum() == len(is_noisy):
        return math.nan
    return float(roc_auc_score(is_noisy, susp))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="ratio")
    ap.add_argument("--train-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--pretrain-seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--noise-type", default="symmetric")
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports/s3_phase2_coupling_20260607")
    return ap.parse_args()


def main():
    args = parse_args()
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    _, data = load_dataset(args.dataset, device=None, split_mode=args.split_mode,
                           split_seed=args.split_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    edge_index = data.edge_index

    sources = {
        "frozen": embedding_path(args.dataset, args.split_seed, args.pretrain_seed,
                                 args.split_mode, args.train_ratio, args.val_ratio),
        "baseline": coupled_embedding_path(args.dataset, split_token, args.pretrain_seed, "baseline"),
        "coupled": coupled_embedding_path(args.dataset, split_token, args.pretrain_seed, "coupled"),
    }
    rows = []
    base = None
    for name, path in sources.items():
        payload = torch.load(path, map_location="cpu")
        h = payload["H"].float()
        hn = F.normalize(h, p=2, dim=1)
        clean_labels = payload["labels"].long()
        masks = payload["masks"]
        num_classes = int(payload["num_classes"])
        noisy_labels, _, _ = load_or_create_noisy_labels(
            clean_labels, masks["train_mask"], num_classes, dataset=args.dataset,
            noise_type=args.noise_type, noise_rate=args.noise_rate,
            split_seed=args.split_seed, noise_seed=args.noise_seed, split_name=split_token)
        grouping = build_dcgb_balls(hn, edge_index, max_hops=args.gagb_max_hops,
                                    min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size,
                                    seed=args.pretrain_seed, mode="emb_only")
        rel, cov = ball_consensus_score(grouping, noisy_labels, masks["train_mask"], num_classes)
        purity = _clean_purity(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes)
        auroc = _rel_auroc(rel, cov, clean_labels, noisy_labels, masks["train_mask"])
        ball_count = len(grouping["groups"])
        rows.append({"source": name, "clean_purity": purity, "noise_auroc": auroc, "ball_count": ball_count})
        print(f"  {name:10s} dcgb: clean_purity={purity:.4f}  noise_auroc={auroc:.4f}  balls={ball_count}")

    by = {r["source"]: r for r in rows}
    dpur = by["coupled"]["clean_purity"] - by["frozen"]["clean_purity"]
    dauroc = by["coupled"]["noise_auroc"] - by["frozen"]["noise_auroc"]
    dpur_b = by["coupled"]["clean_purity"] - by["baseline"]["clean_purity"]
    improved = (dpur > 0.002) or (dauroc > 0.005)  # any meaningful improvement vs frozen
    print(f"\n  coupled - frozen   : d_purity={dpur*100:+.2f}pp  d_auroc={dauroc*100:+.2f}pp")
    print(f"  coupled - baseline : d_purity={dpur_b*100:+.2f}pp")
    print(f">>> ball quality improved by coupling? {improved}  "
          f"-> {'RUN full selection gate' if improved else 'EARLY-STOP: coupling did not improve ball quality'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ball_quality_diag.json").write_text(json.dumps({
        "setting": f"{args.dataset} {split_token} {args.noise_type}{args.noise_rate} seed{args.pretrain_seed} (NOT planetoid/NoisyGL)",
        "rows": rows,
        "coupled_minus_frozen": {"d_purity_pp": dpur * 100, "d_auroc_pp": dauroc * 100},
        "coupled_minus_baseline": {"d_purity_pp": dpur_b * 100},
        "improved": improved,
    }, indent=2), encoding="utf-8")
    print(f"saved {args.out_dir / 'ball_quality_diag.json'}")


if __name__ == "__main__":
    main()
