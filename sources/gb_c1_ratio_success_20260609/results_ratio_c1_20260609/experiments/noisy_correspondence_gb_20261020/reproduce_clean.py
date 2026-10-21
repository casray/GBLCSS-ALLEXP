"""Single-configuration clean Caltech check using MGBCC's released modules."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "mgbcc_main"))
from granular.base import MVGBList  # noqa: E402
from granular.granular_loss import MultiviewGCLoss  # noqa: E402
from model.autoencoder import MultiviewAutoEncoder, Normalize  # noqa: E402
from utils.score import cluster_metric  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = np.load(ROOT / "data" / "Caltech101-20.npz")
    order = np.random.permutation(len(raw["labels"]))  # author loader shuffles once
    x = []
    for view in [3, 4]:
        arr = torch.from_numpy(np.asarray(raw[f"view{view}"][order], dtype=np.float32)).to(device)
        arr = (arr - arr.min(dim=0, keepdim=True).values) / (
            arr.max(dim=0, keepdim=True).values - arr.min(dim=0, keepdim=True).values + 1e-12)
        x.append(arr)
    y = np.asarray(raw["labels"][order], dtype=np.int32)
    rows = torch.arange(len(y))
    loader = DataLoader(TensorDataset(rows), batch_size=128, shuffle=True, num_workers=0)
    model = MultiviewAutoEncoder([1984, 512], 128, [[2000, 500, 500], [2000, 500, 500]])
    for view in range(2):
        model[view].encoder.middle_layers.append(Normalize())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    criterion = MultiviewGCLoss()
    log = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for (ids,) in loader:
            batch = [v[ids] for v in x]
            hidden, reconstruction = model(batch)
            # Source labels are passed into GBList but not used by KMeans or the loss.
            no_labels = torch.zeros(len(ids), dtype=torch.int32, device=device)
            balls = MVGBList(hidden, no_labels, args.p)
            loss = criterion(balls) + 10 * sum(nn.functional.mse_loss(a, b) for a, b in zip(batch, reconstruction))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        scheduler.step()
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                hidden, _ = model(x)
                embedding = torch.stack(hidden).mean(0).cpu().numpy()
            predicted = KMeans(n_clusters=20, n_init="auto", random_state=args.seed).fit_predict(embedding)
            acc, nmi, purity = cluster_metric(y, predicted)
            item = {"epoch": epoch + 1, "acc": acc * 100, "nmi": nmi * 100,
                    "purity": purity * 100, "loss": epoch_loss / len(loader),
                    "elapsed_s": time.perf_counter() - start}
            log.append(item)
            print(json.dumps(item), flush=True)
    out = ROOT / "clean_single_config_caltech.csv"
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    (ROOT / "clean_single_config_config.json").write_text(
        json.dumps({**vars(args), "device": device, "batch_size": 128, "latent_dim": 128,
                    "views": [3, 4], "loss_weights": [1, 10],
                    "selected_epoch_policy": "fixed final epoch; peak ACC is diagnostic only"}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
