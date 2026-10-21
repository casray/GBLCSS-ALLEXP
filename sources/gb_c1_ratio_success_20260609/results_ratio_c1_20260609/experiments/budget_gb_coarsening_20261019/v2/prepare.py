"""CPU-only, label-free preparation, amortized separately from per-arm training."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import psutil
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from granularity import build_tree
from runtime import Stage, atomic_json, sha256_file

DATA_FILES = {
    "cora": "external/NoisyGL/data/Cora/processed/data.pt",
    "citeseer": "external/NoisyGL/data/CiteSeer/processed/data.pt",
    "pubmed": "external/NoisyGL/data/PubMed/processed/data.pt",
    "amazonpho": "external/NoisyGL/data/Photo/processed/data.pt",
    "coauthorcs": "external/NoisyGL/data/CS/processed/data.pt",
    "corafull": "gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt",
}


def rownorm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prepare(config_path, dataset, root):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    folder = Path(root) / "cache" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    signature = sha256_file(config_path)
    preparation_sources = {name: sha256_file(Path(__file__).parent / name)
                           for name in ("prepare.py", "granularity.py", "runtime.py")}
    if (folder / "metadata.json").exists():
        old = json.loads((folder / "metadata.json").read_text())
        if old["config_sha256"] != signature:
            raise RuntimeError("Prepared data belongs to a different configuration")
        if old.get("preparation_sources") != preparation_sources:
            raise RuntimeError("Prepared data source fingerprint mismatch")
        for name, expected in old["cache_files"].items():
            file = folder / name
            if (not file.is_file() or file.stat().st_size != expected["bytes"]
                    or sha256_file(file) != expected["sha256"]):
                raise RuntimeError(f"Prepared cache integrity failed: {name}")
        print(f"Preparation reused: {dataset}", flush=True)
        return
    torch.set_num_threads(config["cpu_threads"])
    times = {}
    source = Path(config["data_root"]) / DATA_FILES[dataset]
    with Stage(times, "load_data_s"):
        from torch_geometric.data import Data
        raw = torch.load(source, map_location="cpu", weights_only=False)
        data = Data.from_dict(raw[0]) if isinstance(raw[0], dict) else raw[0]
        x = data.x.numpy().astype(np.float32, copy=False)
        y = data.y.numpy().astype(np.int64, copy=False)
        edges = data.edge_index.numpy()
        n, d = x.shape
        classes = int(y.max()) + 1
        np.save(folder / "x.npy", x)
        np.save(folder / "y.npy", y)
    with Stage(times, "adjacency_s"):
        p = sp.coo_matrix((np.ones(edges.shape[1], np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
        p = ((p + p.T) > 0).astype(np.float32)
        p.setdiag(0)
        p.eliminate_zeros()
        p += sp.eye(n, dtype=np.float32, format="csr")
        inv = sp.diags(1 / np.sqrt(np.asarray(p.sum(1)).ravel()))
        p = (inv @ p @ inv).tocsr().astype(np.float32)
        sp.save_npz(folder / "p.npz", p, compressed=False)
    with threadpool_limits(limits=config["cpu_threads"]):
        with Stage(times, "cached_features_s"):
            features = np.lib.format.open_memmap(folder / "features.npy", mode="w+", dtype="float32", shape=(n, 3 * d))
            features[:, :d] = x
            features[:, d:2*d] = p @ x
            features[:, 2*d:] = p @ np.asarray(features[:, d:2*d])
            features.flush()
            del features
        with Stage(times, "coordinates_s"):
            xs = sp.csr_matrix(rownorm(x))
            svd = TruncatedSVD(n_components=min(config["svd_dim"], d - 1), random_state=config["tree_seed"])
            u = svd.fit_transform(xs).astype(np.float32)
            pu = p @ u
            ppu = p @ pu
            channels = [rownorm(a) for a in (u, pu, ppu)]
            root_sses = [float(np.square(a - a.mean(0)).sum()) for a in channels]
            z = np.concatenate([a / np.sqrt(max(s, 1e-12)) for a, s in zip(channels, root_sses)], axis=1)
        with Stage(times, "tree_s"):
            ratios = config["ratios"] + (config["corafull_sensitivity"] if dataset == "corafull" else [])
            max_budget = max(round(n * r) for r in ratios)
            max_leaves = min(n // config["min_leaf"], config["tree_leaf_factor"] * max_budget)
            tree = build_tree(z, max_leaves=max_leaves, seed=config["tree_seed"], min_leaf=config["min_leaf"])
            with open(folder / "tree.pkl", "wb") as f:
                pickle.dump(tree, f, protocol=5)
    with Stage(times, "provenance_s"):
        source_hash = sha256_file(source)
        cache_files = {name: {"bytes": (folder / name).stat().st_size,
                              "sha256": sha256_file(folder / name)}
                       for name in ("x.npy", "y.npy", "p.npz", "features.npy", "tree.pkl")}
    leaves = int((tree.children[:, 0] < 0).sum())
    if leaves < max_budget:
        raise RuntimeError(f"Candidate tree has {leaves} leaves but needs {max_budget}")
    atomic_json(folder / "metadata.json", {
        "dataset": dataset, "N": n, "D": d, "C": classes, "E": int(p.nnz),
        "config_sha256": signature, "source": str(source), "source_sha256": source_hash,
        "times": times, "candidate_leaves": leaves, "candidate_nodes": len(tree.members),
        "preparation_sources": preparation_sources, "cache_files": cache_files,
        "preparation_host_peak_gib": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss) / 2**30,
        "tree_seed": config["tree_seed"], "root_channel_sses": root_sses,
        "scope": "Unlabeled geometry; class count is dataset metadata. All labels remain CPU-only until per-job split.",
    })
    print(json.dumps({"dataset": dataset, "N": n, "candidate_leaves": leaves, "times": times}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=DATA_FILES)
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    prepare(args.config, args.dataset, args.run_root)
