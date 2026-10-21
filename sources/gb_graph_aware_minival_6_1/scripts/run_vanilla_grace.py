from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from noisy_label_gb_grace.data import extract_masks, load_dataset, split_name
from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.upstream_grace import build_grace_model, extract_encoder_h, load_grace_config, train_grace_epoch
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, one_hot, save_json, set_seed


def checkpoint_path(
    dataset: str,
    split_seed: int,
    pretrain_seed: int,
    split_mode: str = "planetoid",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "grace" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "checkpoint.pt"


def run(args: argparse.Namespace) -> Path:
    device = get_device(args.gpu_id)
    config = load_grace_config(args.dataset)
    config["seed"] = int(args.pretrain_seed)
    if args.epochs is not None:
        config["num_epochs"] = int(args.epochs)
    set_seed(args.pretrain_seed)
    split_token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    dataset, data = load_dataset(
        args.dataset,
        device=device,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    model = build_grace_model(dataset, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    losses = []
    for epoch in range(1, int(config["num_epochs"]) + 1):
        loss = train_grace_epoch(model, optimizer, data.x, data.edge_index, config)
        losses.append({"epoch": epoch, "loss": loss})
        if args.log_every and epoch % args.log_every == 0:
            print(f"(T) | dataset={args.dataset}, epoch={epoch:04d}, loss={loss:.4f}")

    with torch.no_grad():
        h = extract_encoder_h(model, data.x, data.edge_index, normalize=True).detach().cpu()
    masks = extract_masks(data)
    clean_targets = one_hot(data.y.detach().cpu(), dataset.num_classes)
    eval_result = train_frozen_classifier(
        h,
        data.y.detach().cpu(),
        clean_targets,
        masks["train_mask"],
        masks["val_mask"],
        masks["test_mask"],
        seed=args.classifier_seed,
        epochs=args.classifier_epochs,
        device=torch.device("cpu") if args.eval_cpu else device,
    )

    path = checkpoint_path(args.dataset, args.split_seed, args.pretrain_seed, args.split_mode, args.train_ratio, args.val_ratio)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": args.dataset,
            "split_mode": args.split_mode,
            "split_name": split_token,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "split_seed": int(args.split_seed),
            "pretrain_seed": int(args.pretrain_seed),
            "config": config,
            "model_state_dict": model.state_dict(),
            "losses": losses,
            "vanilla_eval": eval_result,
            "upstream_commit": args.upstream_commit,
        },
        path,
    )
    save_json(path.parent / "vanilla_eval.json", eval_result)
    save_json(path.parent / "losses.json", losses)
    print(f"(S) | Saved GRACE checkpoint to {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed", "DBLP"], default="Cora")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=["planetoid", "ratio"], default="planetoid")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--pretrain-seed", type=int, default=39788)
    parser.add_argument("--classifier-seed", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-cpu", action="store_true")
    parser.add_argument("--upstream-commit", default="b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
