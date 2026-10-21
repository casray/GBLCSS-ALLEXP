from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub/src"))
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed

DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]
DATA_FILES = {
    "cora": PROJECT / "external/NoisyGL/data/Cora/processed/data.pt",
    "citeseer": PROJECT / "external/NoisyGL/data/CiteSeer/processed/data.pt",
    "corafull": PROJECT / "gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt",
    "coauthorcs": PROJECT / "external/NoisyGL/data/CS/processed/data.pt",
    "amazonpho": PROJECT / "external/NoisyGL/data/Photo/processed/data.pt",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def raw_data(name):
    stored = torch.load(DATA_FILES[name], map_location="cpu", weights_only=False)
    return Data.from_dict(stored[0]) if isinstance(stored[0], dict) else stored[0]


def load_data(name, device="cuda"):
    raw = raw_data(name)
    n = raw.x.shape[0]
    indices = percent_split_indices(n, 0)
    masks = []
    for index in indices:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    graph = GraphData(name, raw.x.float(), raw.edge_index.long(), raw.y.long(),
                      *masks, train_index=torch.tensor(indices[0]),
                      val_index=torch.tensor(indices[1]), test_index=torch.tensor(indices[2]))
    inject_train_label_noise(graph, "uniform", .4, 0)
    audit = dict(dataset=name, seed=0, noise_type="uniform", noise_rate=.4,
                 noise_scope="train", split_hash=fingerprint(*indices),
                 noise_hash=fingerprint(graph.y, graph.noisy_y),
                 num_nodes=n, num_classes=graph.num_classes,
                 modified_train_count=int((graph.noisy_y[masks[0]] != graph.y[masks[0]]).sum()),
                 modified_val_count=int((graph.noisy_y[masks[1]] != graph.y[masks[1]]).sum()),
                 modified_test_count=int((graph.noisy_y[masks[2]] != graph.y[masks[2]]).sum()),
                 features="raw main-experiment attributes", data_file=str(DATA_FILES[name]))
    reference = json.loads((WORKSPACE / f"experiments/main_protocol_replay_20260922/matrix/{name}/default/seed0.json").read_text(encoding="utf-8-sig"))
    for key in ["split_hash", "noise_hash"]:
        if audit[key] != reference[key]:
            raise ValueError(f"{name}: historical {key} mismatch")
    assert audit["modified_val_count"] == audit["modified_test_count"] == 0
    # Only training labels and a separate validation vector enter the adapters.
    labels = torch.full_like(graph.y, -1)
    labels[masks[0]] = graph.noisy_y[masks[0]]
    data = Data(x=graph.x, edge_index=graph.edge_index, y=labels,
                train_index=graph.train_index, val_index=graph.val_index,
                test_index=graph.test_index, train_mask=masks[0], val_mask=masks[1],
                test_mask=masks[2], val_y=graph.y[graph.val_index],
                num_classes=graph.num_classes, name=name)
    return data.to(device), audit


def final_metrics(name, probabilities):
    raw = raw_data(name)
    indices = percent_split_indices(raw.x.shape[0], 0)
    clean = raw.y.long()
    if probabilities.shape != (len(clean), int(clean.max()) + 1):
        raise ValueError("invalid probability shape")
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(len(clean)), atol=1e-5)
    pred = probabilities.argmax(1)
    return {key: float((pred[idx] == clean[idx]).double().mean())
            for key, idx in zip(["train_clean", "valid", "test"], indices)}
