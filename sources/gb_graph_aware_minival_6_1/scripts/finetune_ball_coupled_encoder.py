"""S3 Phase 2 — minimal ball-coupled encoder finetuning (give the granular ball the right knob).

Init a GRACE encoder from a pretrain checkpoint, finetune the FULL encoder at a small lr with
    L = L_sup_noisy + lambda_cl * L_GRACE + lambda_gb * (L_intra + L_inter)
where the ball losses (training/ball_coupling.py) shape the representation around periodically-rebuilt
high-quality dcgb balls. Then dump the finetuned H in the exact extract_embeddings payload format so the
downstream selection experiment can reuse it unchanged.

Two outputs from one init (the ONLY difference is lambda_gb), for a same-source "coupled vs not" control:
  - coupled   : lambda_gb > 0   -> representation "ball_coupled_encoder_H"
  - baseline  : lambda_gb = 0   -> representation "sup_grace_baseline_H" (same finetune, no ball loss)

Clean labels never enter the encoder loss / ball construction / quality gate (eval/diagnostic only).
Cell: Cora / symmetric 0.4 / ratio_train0.2_val0.1 / seed 0 / GRACE init. NOT planetoid, NOT NoisyGL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj

from noisy_label_gb_grace.balls import build_dcgb_balls
from noisy_label_gb_grace.data import extract_masks, load_dataset, split_name
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.training import ball_coupling_losses, label_free_quality_mask
from noisy_label_gb_grace.upstream_grace import _load_upstream_module, build_grace_model, extract_encoder_h
from noisy_label_gb_grace.utils import get_device, mask_to_index, set_seed
from scripts.run_vanilla_grace import checkpoint_path


def coupled_embedding_path(dataset, split_token, pretrain_seed, tag):
    return ROOT / "checkpoints" / "embeddings_coupled" / dataset / f"{split_token}_pretrain{pretrain_seed}_{tag}" / "H.pt"


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
    ap.add_argument("--coupling-epochs", type=int, default=150)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--lambda-cl", type=float, default=1.0)
    ap.add_argument("--lambda-gb", type=float, default=1.0)
    ap.add_argument("--lambda-sup", type=float, default=1.0)
    ap.add_argument("--ball-refresh", type=int, default=10)
    ap.add_argument("--inter-margin", type=float, default=0.5)
    ap.add_argument("--gb-min-ball-size", type=int, default=5)
    ap.add_argument("--gagb-max-hops", type=int, default=2)
    ap.add_argument("--gagb-max-ball-size", type=int, default=128)
    ap.add_argument("--min-train-per-ball", type=int, default=3)
    ap.add_argument("--gpu-id", type=int, default=None)
    return ap.parse_args()


def finetune(args, lambda_gb, tag):
    """One finetuning run; returns the dumped H.pt path."""
    device = get_device(args.gpu_id)
    set_seed(args.pretrain_seed)
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    ckpt = torch.load(checkpoint_path(args.dataset, args.split_seed, args.pretrain_seed,
                                      args.split_mode, args.train_ratio, args.val_ratio), map_location=device)
    config = ckpt["config"]
    dataset, data = load_dataset(args.dataset, device=device, split_mode=args.split_mode,
                                 split_seed=args.split_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    model = build_grace_model(dataset, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model_module = _load_upstream_module("model")

    masks = extract_masks(data)
    train_mask = masks["train_mask"].to(device)
    num_classes = int(dataset.num_classes)
    clean_labels = data.y.detach().cpu().long()
    noisy_labels, _, _ = load_or_create_noisy_labels(
        clean_labels, masks["train_mask"].cpu(), num_classes, dataset=args.dataset,
        noise_type=args.noise_type, noise_rate=args.noise_rate,
        split_seed=args.split_seed, noise_seed=args.noise_seed, split_name=split_token)
    noisy_dev = noisy_labels.to(device)

    head = nn.Linear(config["num_hidden"], num_classes).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                           lr=args.finetune_lr, weight_decay=config.get("weight_decay", 1e-5))
    train_idx = mask_to_index(train_mask)

    group_ids = None
    quality_mask = None
    for epoch in range(1, args.coupling_epochs + 1):
        model.train()
        opt.zero_grad()
        z = model(data.x, data.edge_index)                 # differentiable encoder output
        hn = F.normalize(z, p=2, dim=1)

        # L_sup on noisy train labels (representation must stay classification-usable)
        logits = head(z)
        L_sup = F.cross_entropy(logits[train_idx], noisy_dev[train_idx])

        # L_GRACE two-view contrastive (keep representation healthy, no drift)
        ei1 = dropout_adj(data.edge_index, p=config["drop_edge_rate_1"])[0]
        ei2 = dropout_adj(data.edge_index, p=config["drop_edge_rate_2"])[0]
        x1 = model_module.drop_feature(data.x, config["drop_feature_rate_1"])
        x2 = model_module.drop_feature(data.x, config["drop_feature_rate_2"])
        L_grace = model.loss(model(x1, ei1), model(x2, ei2), batch_size=0)

        # L_ball (only if coupled): periodically rebuild membership from DETACHED H
        L_intra = z.sum() * 0.0
        L_inter = z.sum() * 0.0
        if lambda_gb > 0:
            if group_ids is None or (epoch - 1) % args.ball_refresh == 0:
                grouping = build_dcgb_balls(hn.detach().cpu(), data.edge_index.detach().cpu(),
                                            max_hops=args.gagb_max_hops, min_ball_size=args.gb_min_ball_size,
                                            max_ball_size=args.gagb_max_ball_size, seed=args.pretrain_seed, mode="emb_only")
                group_ids = grouping["group_ids"]
                quality_mask = label_free_quality_mask(group_ids, hn.detach().cpu(), noisy_labels,
                                                        masks["train_mask"].cpu(), num_classes,
                                                        min_train=args.min_train_per_ball)
            L_intra, L_inter = ball_coupling_losses(hn, group_ids, quality_mask, noisy_labels,
                                                    masks["train_mask"], num_classes, margin=args.inter_margin)

        loss = args.lambda_sup * L_sup + args.lambda_cl * L_grace + lambda_gb * (L_intra + L_inter)
        loss.backward()
        opt.step()
        if epoch % max(1, args.coupling_epochs // 5) == 0 or epoch == 1:
            print(f"[{tag}] ep{epoch:>3} loss={float(loss):.4f} sup={float(L_sup):.4f} "
                  f"grace={float(L_grace):.4f} intra={float(L_intra):.4f} inter={float(L_inter):.4f}")

    h_final = extract_encoder_h(model, data.x, data.edge_index, normalize=True).detach().cpu()
    payload = {
        "H": h_final,
        "edge_index": data.edge_index.detach().cpu().long(),
        "labels": data.y.detach().cpu().long(),
        "num_classes": num_classes,
        "masks": {k: v.detach().cpu() for k, v in masks.items()},
        "dataset": args.dataset, "split_mode": args.split_mode, "split_name": split_token,
        "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
        "split_seed": args.split_seed, "pretrain_seed": args.pretrain_seed,
        "representation": "ball_coupled_encoder_H" if lambda_gb > 0 else "sup_grace_baseline_H",
        "lambda_gb": float(lambda_gb), "coupling_epochs": int(args.coupling_epochs),
    }
    out = coupled_embedding_path(args.dataset, split_token, args.pretrain_seed, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[{tag}] saved {out}")
    return out


def main():
    args = parse_args()
    # same-source control: identical init/finetune, only lambda_gb differs
    p_base = finetune(args, lambda_gb=0.0, tag="baseline")
    p_coup = finetune(args, lambda_gb=args.lambda_gb, tag="coupled")
    print(f"\nbaseline H: {p_base}\ncoupled  H: {p_coup}")


if __name__ == "__main__":
    main()
