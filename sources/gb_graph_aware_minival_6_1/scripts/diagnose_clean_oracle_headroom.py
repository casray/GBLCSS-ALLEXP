"""S2a headroom diagnostic — clean-label ORACLE upper bound.

================================ DIAGNOSTIC ONLY ================================
Clean labels are used here SOLELY to measure the oracle upper bound, i.e. how
much test accuracy ANY denoiser could possibly recover in this frozen-GRACE
setting. This script is NOT a method, is NOT part of method selection, and
clean-label training MUST NEVER be wired into the real pipeline / predictors.
It exists to answer one question and then inform a go/stop decision.
================================================================================

Question:
    headroom = clean-oracle-CE test acc  -  noisy-CE test acc

Decision rule (user/GPT, 2026-06-07):
    headroom < ~1pp                  -> no denoising headroom in this setting;
                                        stop iterating S2a here.
    headroom large, but dcgb still   -> headroom exists but ball->downstream
       ties no-ball/shuffled            conversion fails; go to S3 / S2-redesign,
                                        NOT alpha/gate tuning.

Setting (must match the Phase 1 smoke EXACTLY):
    Cora, ratio_train0.2_val0.1, symmetric 0.4, seed 0, frozen GRACE, pretrain0.
This is a RATIO-SPLIT minival result. NOT planetoid, NOT NoisyGL protocol.
pair / high-noise headroom is a separate NoisyGL lane and must not be merged
into this table.
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

from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.utils import get_device, one_hot
from scripts.extract_embeddings import embedding_path

OTHER_ROWS = {  # method -> representative alpha (from the Phase 1 smoke CSV)
    "ce": "0",
    "gb": "0.3",
    "dcgb_emb_only": "0.3",
    "dcgb_qsafe_boundary_alpha09_q50": "0.9",
    "topk_dcgb_train_matched": "0.3",
    "dcgb_shuffled_alpha_matched": "0.3",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", default="ratio")
    ap.add_argument("--train-ratio", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--pretrain-seed", type=int, default=0)
    ap.add_argument("--classifier-seed", type=int, default=0)
    ap.add_argument("--classifier-epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--noise-type", default="symmetric")
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument(
        "--smoke-csv",
        type=Path,
        default=ROOT
        / "results/gate_runs/Cora/symmetric_0p4/ratio_train0p2_val0p1_split0_pretrain0_noise0_clf0/result_rows.csv",
    )
    ap.add_argument("--gpu-id", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/s2a_plan_20260607/s2a_headroom_oracle.json",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu_id)
    emb = embedding_path(
        args.dataset, args.split_seed, args.pretrain_seed,
        args.split_mode, args.train_ratio, args.val_ratio,
    )
    payload = torch.load(emb, map_location="cpu")
    h = payload["H"].float()
    clean = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])

    # ---- ORACLE: train the SAME frozen head on CLEAN one-hot targets ----
    # (diagnostic upper bound; clean labels never enter any method path)
    soft = one_hot(clean, num_classes).float()
    res = train_frozen_classifier(
        h, clean, soft,
        masks["train_mask"], masks["val_mask"], masks["test_mask"],
        seed=args.classifier_seed, epochs=args.classifier_epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, device=device, sample_weights=None,
    )
    oracle = res["test_acc_at_best_val"]

    # ---- pull the other rows from the existing Phase 1 smoke CSV ----
    table: dict[str, dict] = {}
    if args.smoke_csv.exists():
        for r in csv.DictReader(args.smoke_csv.open(encoding="utf-8")):
            m = r.get("method", "")
            if m in OTHER_ROWS and f"{float(r['alpha_max']):g}" == OTHER_ROWS[m]:
                def _f(key):
                    v = r.get(key)
                    return float(v) if v not in (None, "", "nan") else float("nan")
                table[m] = {"alpha": OTHER_ROWS[m], "test": _f("test_acc_at_best_val"),
                            "ncr": _f("noise_correction_rate"), "cdr": _f("clean_damage_rate")}

    noisy_ce = table.get("ce", {}).get("test")
    headroom_pp = (oracle - noisy_ce) * 100 if noisy_ce is not None else None

    print("=== S2a headroom (clean-oracle upper bound) — DIAGNOSTIC ONLY ===")
    print(f"setting: {args.dataset} ratio_train{args.train_ratio}_val{args.val_ratio} "
          f"{args.noise_type}{args.noise_rate} seed{args.classifier_seed} frozen-GRACE pretrain{args.pretrain_seed}")
    print(f"clean-oracle CE test = {oracle:.4f}  (best_val {res['best_val_acc']:.4f}, epoch {res['best_epoch']})")
    if noisy_ce is not None:
        print(f"noisy CE test        = {noisy_ce:.4f}")
        print(f">>> HEADROOM (oracle - noisy CE) = {headroom_pp:.2f} pp")
    for m in ["ce", "gb", "dcgb_emb_only", "dcgb_qsafe_boundary_alpha09_q50",
              "topk_dcgb_train_matched", "dcgb_shuffled_alpha_matched"]:
        if m in table:
            t = table[m]
            print(f"  {m:36s} a={t['alpha']:>3}  test={t['test']:.4f}  "
                  f"NCR={t['ncr']:.4f}  CDR={t['cdr']:.4f}")

    out = {
        "setting": (f"{args.dataset} ratio_train{args.train_ratio}_val{args.val_ratio} "
                    f"{args.noise_type}{args.noise_rate} seed{args.classifier_seed} "
                    f"frozen-GRACE pretrain{args.pretrain_seed} (NOT planetoid, NOT NoisyGL)"),
        "clean_oracle_ce_test": oracle,
        "clean_oracle_best_val": res["best_val_acc"],
        "noisy_ce_test": noisy_ce,
        "headroom_pp": headroom_pp,
        "rows": table,
        "note": "clean labels used ONLY for oracle upper-bound diagnosis; never in method selection or training.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
