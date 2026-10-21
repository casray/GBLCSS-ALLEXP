from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import load_public_dataset, set_seed
from gblcss.grace import train_grace


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRACE and export the frozen representation used by GB-LCSS.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Optional smoke-test override.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    key = args.dataset.strip().lower()
    grace_config = yaml.safe_load(
        (ROOT / "configs" / "representations" / "grace.yaml").read_text(encoding="utf-8")
    )
    if key not in grace_config["datasets"]:
        raise ValueError(f"no reported GRACE configuration for dataset: {key}")
    config = {**grace_config["common"], **grace_config["datasets"][key]}
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)

    dataset_config = yaml.safe_load(
        (ROOT / "configs" / "datasets" / f"{key}.yaml").read_text(encoding="utf-8")
    )
    set_seed(args.seed)
    graph = load_public_dataset(dataset_config["name"], args.data_root, args.seed)
    output = train_grace(graph.x, graph.edge_index, config, device=args.device)

    destination = Path(args.output) if args.output else ROOT / "representations_generated" / key / "grace" / "H.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output.representation, destination)
    with destination.with_name("losses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows({"epoch": index + 1, "loss": loss} for index, loss in enumerate(output.losses))
    print(f"representation={destination} shape={tuple(output.representation.shape)}")


if __name__ == "__main__":
    main()
