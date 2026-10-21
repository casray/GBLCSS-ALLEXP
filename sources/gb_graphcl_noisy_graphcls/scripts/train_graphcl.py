from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gb_graphcl.contrastive import augment_edge_perturbation, augment_node_dropping, train_graphcl_epoch
from gb_graphcl.data import load_dataset, stratified_split
from gb_graphcl.encoder import GINEncoder, ProjectionHead


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["NCI1", "PROTEINS", "MUTAG"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset, root=str(ROOT / "data"))
    train_idx, val_idx, _ = stratified_split(dataset, seed=args.seed)
    pretrain_idx = train_idx + val_idx
    pretrain_data = [dataset[i] for i in pretrain_idx]
    input_dim = int(pretrain_data[0].x.size(-1))

    enc_cfg = cfg["encoder"]
    con_cfg = cfg["contrastive"]
    encoder = GINEncoder(input_dim=input_dim, hidden_dim=enc_cfg["hidden_dim"], num_layers=enc_cfg["num_layers"]).to(device)
    proj_head = ProjectionHead(hidden_dim=enc_cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=con_cfg["lr"],
    )
    loader = DataLoader(pretrain_data, batch_size=con_cfg["batch_size"], shuffle=True)
    aug1 = lambda data: augment_node_dropping(data, drop_ratio=con_cfg["drop_ratio"])
    aug2 = lambda data: augment_edge_perturbation(data, perturb_ratio=con_cfg["perturb_ratio"])

    final_loss = 0.0
    for epoch in range(con_cfg["epochs"]):
        final_loss = train_graphcl_epoch(
            encoder,
            proj_head,
            loader,
            optimizer,
            device,
            aug1,
            aug2,
            temperature=con_cfg["temperature"],
        )
        print(f"epoch={epoch + 1:03d} loss={final_loss:.6f}")

    out_dir = ROOT / "checkpoints" / args.dataset / f"pretrain{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": enc_cfg["hidden_dim"],
            "num_layers": enc_cfg["num_layers"],
            "dataset": args.dataset,
            "seed": args.seed,
        },
        out_dir / "encoder.pt",
    )
    print(f"saved={out_dir / 'encoder.pt'} final_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()

