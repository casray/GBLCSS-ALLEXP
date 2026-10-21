from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import load_public_dataset
from gblcss.representation import load_frozen_representation


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public graph and validate its frozen GRACE representation.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    args = parser.parse_args()
    config_path = ROOT / "configs" / "datasets" / f"{args.dataset.lower()}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    graph = load_public_dataset(config["name"], args.data_root, seed=0)
    representation = load_frozen_representation(ROOT / config["representation"], graph.num_nodes)
    print(f"{config['name']}: {graph.num_nodes} nodes, H={tuple(representation.shape)}")


if __name__ == "__main__":
    main()
