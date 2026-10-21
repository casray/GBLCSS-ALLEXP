from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from noisy_label_gb_grace.data import extract_masks, load_dataset, split_name
from noisy_label_gb_grace.upstream_grace import build_grace_model, extract_encoder_h
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device
from scripts.run_vanilla_grace import checkpoint_path


def embedding_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    split_mode: str = "planetoid",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "embeddings" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "H.pt"


def run(args: argparse.Namespace) -> Path:
    device = get_device(args.gpu_id)
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    checkpoint = torch.load(
        args.checkpoint or checkpoint_path(
            args.dataset,
            args.split_seed,
            args.pretrain_seed,
            args.split_mode,
            args.train_ratio,
            args.val_ratio,
        ),
        map_location=device,
    )
    dataset, data = load_dataset(
        args.dataset,
        device=device,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    model = build_grace_model(dataset, checkpoint["config"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    h = extract_encoder_h(model, data.x, data.edge_index, normalize=True).detach().cpu()
    out_path = args.output or embedding_path(args.dataset, args.split_seed, args.pretrain_seed, args.split_mode, args.train_ratio, args.val_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "H": h,
        "edge_index": data.edge_index.detach().cpu().long(),
        "labels": data.y.detach().cpu().long(),
        "num_classes": int(dataset.num_classes),
        "masks": extract_masks(data),
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "split_name": split_token,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "split_seed": int(args.split_seed),
        "pretrain_seed": int(args.pretrain_seed),
        "checkpoint_path": str(args.checkpoint or checkpoint_path(
            args.dataset,
            args.split_seed,
            args.pretrain_seed,
            args.split_mode,
            args.train_ratio,
            args.val_ratio,
        )),
        "representation": "normalized_encoder_H",
    }
    torch.save(payload, out_path)
    print(f"(S) | Saved normalized encoder H to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed", "DBLP"], default="Cora")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["planetoid", "ratio"], default="planetoid")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--pretrain-seed", type=int, default=39788)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
