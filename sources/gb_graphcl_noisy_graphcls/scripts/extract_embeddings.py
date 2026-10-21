from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gb_graphcl.data import dataset_labels, load_dataset
from gb_graphcl.encoder import GINEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["NCI1", "PROTEINS", "MUTAG"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset, root=str(ROOT / "data"))
    ckpt_path = ROOT / "checkpoints" / args.dataset / f"pretrain{args.seed}" / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = GINEncoder(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    loader = DataLoader([dataset[i] for i in range(len(dataset))], batch_size=256, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = encoder(batch.x, batch.edge_index, batch.batch)
            chunks.append(h.cpu())
    embeddings = F.normalize(torch.cat(chunks, dim=0), dim=1)
    labels = dataset_labels(dataset)
    num_classes = int(torch.unique(labels).numel())

    out_dir = ROOT / "checkpoints" / "embeddings" / args.dataset / f"pretrain{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "H.pt"
    torch.save({"embeddings": embeddings, "labels": labels, "num_classes": num_classes}, out_path)
    print(f"saved={out_path} shape={tuple(embeddings.shape)} num_classes={num_classes}")


if __name__ == "__main__":
    main()

