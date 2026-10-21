"""Train independent reconstruction encoders for the relation diagnostic."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "mgbcc_main"))
from model.autoencoder import AutoEncoder, Normalize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(ROOT / "data" / "Caltech101-20.npz")
    result: dict[str, np.ndarray] = {}
    log = {"epochs": args.epochs, "seed": args.seed, "lr": args.lr, "device": device,
           "batch_size": 128, "latent_dim": 128, "views": [3, 4], "loss": {}}
    for view_id in [3, 4]:
        x = np.asarray(data[f"view{view_id}"], dtype=np.float32)
        low, high = x.min(axis=0), x.max(axis=0)
        x = (x - low) / (high - low + 1e-12)
        tensor = torch.from_numpy(x)
        loader = DataLoader(TensorDataset(tensor), batch_size=128, shuffle=True)
        model = AutoEncoder(x.shape[1], 128, middle_dims=[2000, 500, 500])
        model.encoder.middle_layers.append(Normalize())
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        curve = []
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                _, reconstruction = model(batch)
                loss = nn.functional.mse_loss(reconstruction, batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total += loss.item() * batch.shape[0]
            curve.append(total / len(tensor))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"view {view_id} epoch {epoch+1}: MSE={curve[-1]:.6f}", flush=True)
        model.eval()
        with torch.no_grad():
            result[f"view{view_id}"] = model.encoder(tensor.to(device)).cpu().numpy()
        log["loss"][str(view_id)] = curve
    np.savez_compressed(ROOT / "data" / "Caltech101-20_reconstruction_z.npz", **result)
    (ROOT / "pretrain_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
