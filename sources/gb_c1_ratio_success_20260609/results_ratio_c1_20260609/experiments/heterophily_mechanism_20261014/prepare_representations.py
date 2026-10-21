"""Label-free GRACE and audited GraphACL portable extension; no checkpoint selection."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import to_undirected, dropout_edge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "GB-LCSS_code/src"))
from gblcss.grace import GRACE, Encoder, _drop_features, _row_normalize_features

ROMAN_SHA256 = "a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roman_inputs(path):
    """Read only attributes/edges, retaining official node order; bidirected as upstream."""
    if sha256(path) != ROMAN_SHA256:
        raise ValueError("Roman-empire source hash differs from audited official artifact")
    with np.load(path) as data:
        features = torch.from_numpy(data["node_features"].copy()).float()
        edges = torch.from_numpy(data["edges"].T.copy()).long()
    return features, to_undirected(edges, num_nodes=features.shape[0])


def chunked_grace_loss(first, second, temperature=.7, chunk_size=512):
    """Exact all-node objective; checkpoint each anchor block to bound activation memory."""
    if chunk_size < 1 or temperature <= 0:
        raise ValueError("chunk_size and temperature must be positive")
    first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
    total = first.new_zeros(())
    n = first.shape[0]
    for anchors, other in [(first, second), (second, first)]:
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            def block(a, b, start=start, end=end):
                within = torch.exp(a[start:end] @ a.T / temperature)
                between = torch.exp(a[start:end] @ b.T / temperature)
                row = torch.arange(end - start, device=a.device)
                col = torch.arange(start, end, device=a.device)
                denominator = within.sum(1) + between.sum(1) - within[row, col]
                return -torch.log(between[row, col] / denominator.clamp_min(1e-12)).sum()
            total = total + checkpoint(block, anchors, other, use_reentrant=False)
    return total / (2 * n)


class SymmetricGraphConv(nn.Module):
    """DGL GraphConv(norm='both') algebra, with no implicit self loops."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edges):
        src, dst = edges
        degree_out = torch.bincount(src, minlength=len(x)).to(x.dtype).clamp_min(1).rsqrt()
        degree_in = torch.bincount(dst, minlength=len(x)).to(x.dtype).clamp_min(1).rsqrt()
        # Transform first for bounded memory. Matrix associativity matches DGL's algebra.
        z = (x * degree_out[:, None]) @ self.weight
        output = torch.zeros((len(x), z.shape[1]), dtype=x.dtype, device=x.device)
        output.index_add_(0, dst, z[src])
        return output * degree_in[:, None] + self.bias


class GraphACL(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_mlp, ema):
        super().__init__()
        self.online = nn.ModuleList([SymmetricGraphConv(in_dim, hidden_dim), SymmetricGraphConv(hidden_dim, hidden_dim)])
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        layers = []
        for i in range(num_mlp):
            if i: layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.projector = nn.Sequential(*layers)
        self.ema = ema

    def encode(self, x, edges):
        return self.online[1](self.online[0](x, edges), edges)

    def forward(self, x, edges):
        online = self.encode(x, edges)
        with torch.no_grad():
            target = self.target[1](self.target[0](x, edges), edges)
        return F.normalize(self.projector(online), dim=1), F.normalize(online, dim=1), F.normalize(target, dim=1)

    @torch.no_grad()
    def update_target(self):
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(self.ema).add_(online, alpha=1 - self.ema)


def graphacl_loss(projected, online, target, positive, negative, temperature, lambda_loss=1.):
    pos = torch.exp(torch.einsum('nd,nkd->nk', projected, target[positive]) / temperature)
    neg = torch.exp(torch.einsum('nd,nkd->nk', online, online[negative]) / temperature).sum(1, keepdim=True)
    return (lambda_loss * torch.log(pos + neg) - torch.log(pos)).mean()


def sample_graphacl_indices(neighbors, sample_size):
    """Preserve official Python/NumPy sampling, including low-degree list padding."""
    positive, negative = [], []
    for adjacent in neighbors:
        if not adjacent:
            raise ValueError("Official GraphACL does not define isolated-node sampling")
        if len(adjacent) < sample_size:
            adjacent += np.random.choice(adjacent, sample_size - len(adjacent)).tolist()
            pos = adjacent
        else:
            pos = random.sample(adjacent, sample_size)
        positive.append(list(pos))
        negative.append(random.sample(range(len(neighbors)), sample_size))
    return torch.tensor(positive), torch.tensor(negative)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    x, edges = load_roman_inputs(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "H.pt").exists():
        raise FileExistsError("Refusing to overwrite completed representation")
    config = vars(args).copy()
    config.update(data_sha256=sha256(args.data), script_sha256=sha256(__file__),
                  node_count=len(x), edge_count=edges.shape[1],
                  label_usage="none; fixed last epoch, no evaluation", torch_version=torch.__version__)
    if args.action == "train-grace":
        config.update(hidden_dim=256, projection_dim=256, layers=2, learning_rate=.001,
                      weight_decay=1e-5, edge_drop_rates=[.1, .4], feature_mask_rates=[.1, 0],
                      temperature=.7, activation="relu", feature_preprocessing="per-row min shift and sum normalization, original GB-LCSS GRACE")
        model = GRACE(Encoder(x.shape[1], 256, nn.ReLU(), 2), 256, 256, .7).to(args.device)
        x = _row_normalize_features(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    else:
        config.update(hidden_dim=512, layers=2, learning_rate=.0001, weight_decay=.0003,
                      temperature=.5, ema=.95, sample_size=5, num_mlp=1,
                      feature_preprocessing="official raw features; Roman extension, not official tuned config",
                      source_commit="c3aa4b2ba8b901c630140c7ab689a8cb4a693457",
                      embedding_epoch="post-final-update (upstream returns pre-final-update)")
        model = GraphACL(x.shape[1], 512, 1, .95).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=.0003)
        neighbors = [[] for _ in range(len(x))]
        for src, dst in edges.T.tolist(): neighbors[src].append(dst)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    x, edges = x.to(args.device), edges.to(args.device)
    start = time.perf_counter()
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    with (output / "training.jsonl").open("w", encoding="utf-8") as logfile:
        for epoch in range(args.epochs):
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.action == "train-grace":
                edge1, _ = dropout_edge(edges, p=.1, training=True)
                edge2, _ = dropout_edge(edges, p=.4, training=True)
                first = model.projection(model(_drop_features(x, .1), edge1))
                second = model.projection(model(_drop_features(x, 0.), edge2))
                loss = chunked_grace_loss(first, second, .7, args.chunk_size)
            else:
                positive, negative = sample_graphacl_indices(neighbors, 5)
                projected, online, target = model(x, edges)
                loss = graphacl_loss(projected, online, target, positive.to(args.device), negative.to(args.device), .5)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite SSL loss")
            loss.backward()
            optimizer.step()
            if args.action == "train-graphacl": model.update_target()
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            record = dict(epoch=epoch + 1, loss=float(loss.detach()), seconds=time.perf_counter() - tick)
            logfile.write(json.dumps(record) + "\n")
            logfile.flush()
            if epoch < 3 or (epoch + 1) % 10 == 0: print(json.dumps(record), flush=True)
    model.eval()
    with torch.no_grad():
        h = model(x, edges) if args.action == "train-grace" else model.encode(x, edges)
        h = F.normalize(h, dim=1).cpu().float()
    torch.save(h, output / "H.pt")
    torch.save(model.state_dict(), output / "model.pt")
    result = dict(status="completed", epochs=args.epochs, shape=list(h.shape),
                  elapsed_seconds=time.perf_counter() - start, representation_sha256=sha256(output / "H.pt"),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                  exploratory_single_ssl_seed=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['train-grace', 'train-graphacl'], required=True)
    parser.add_argument('--data', default=str(HERE / 'data/roman_empire.npz'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1: parser.error('--epochs must be positive')
    run(args)


if __name__ == '__main__':
    main()
