from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_standard_downstream


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
METHOD = "frozengrace_gb_mlp_softdist_beta05"
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"
GCN_HIDDEN_PATHS = {
    "cora": REPO_ROOT
    / "runs"
    / "mlp15_gcn_hidden_balls_20260703"
    / "embeddings"
    / "Cora"
    / "gcn_hidden_train0p20"
    / "H.pt",
    "citeseer": REPO_ROOT
    / "runs"
    / "mlp15_gcn_hidden_balls_20260703"
    / "embeddings"
    / "CiteSeer"
    / "gcn_hidden_train0p20"
    / "H.pt",
}


def formal_label_free_cells() -> list[tuple[str, str, int]]:
    return [
        (dataset, source, seed)
        for dataset in DATASETS
        for source in ("raw_x", "adj2_x")
        for seed in range(5)
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_tensors(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    propagation_steps: int = 2,
) -> dict[str, torch.Tensor]:
    if int(propagation_steps) <= 0:
        raise ValueError("propagation_steps must be positive")
    features = features.detach().cpu().float()
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    adjacency = adjacency.detach().cpu().float()
    if not getattr(adjacency, "is_sparse", False):
        adjacency = adjacency.to_sparse()
    propagated = features
    for _ in range(int(propagation_steps)):
        propagated = torch.sparse.mm(adjacency, propagated)
    return {"raw_x": features, f"adj{propagation_steps}_x": propagated}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def prepare_artifacts(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    artifact_root = output_dir / "representation_artifacts"
    manifest_path = output_dir / "representation_artifact_manifest.csv"
    manifest: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASETS:
        run_standard_downstream.setup_seed(0)
        data = run_standard_downstream._load_dataset(
            dataset,
            device="cpu",
            split_type="percent",
            train_percent=0.2,
            val_percent=0.1,
            test_percent=0.7,
        )
        dataset_dir = artifact_root / dataset
        raw_path = dataset_dir / "raw_x.pt"
        adj_path = dataset_dir / "adj2_x.pt"
        if not raw_path.exists() or not adj_path.exists():
            sources = build_source_tensors(data.feats, data.adj, propagation_steps=2)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            torch.save(sources["raw_x"], raw_path)
            torch.save(sources["adj2_x"], adj_path)
            del sources
        for source, path in (("raw_x", raw_path), ("adj2_x", adj_path)):
            row = {
                "dataset": dataset,
                "source": source,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "num_nodes": int(data.labels.shape[0]),
                "feature_dim": int(data.dim_feats),
                "label_free": 1,
                "propagation_steps": 0 if source == "raw_x" else 2,
            }
            manifest[(dataset, source)] = row
    for dataset, path in GCN_HIDDEN_PATHS.items():
        if not path.exists():
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid GCN-hidden artifact: {path}")
        manifest[(dataset, "gcn_hidden_noisy_seed0")] = {
            "dataset": dataset,
            "source": "gcn_hidden_noisy_seed0",
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "num_nodes": int(tensor.shape[0]),
            "feature_dim": int(tensor.shape[1]),
            "label_free": 0,
            "propagation_steps": "",
        }
    _write_csv(manifest_path, list(manifest.values()))
    return manifest


def _run_args(
    device: str,
    embedding_path: str,
    *,
    leave_one_out: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        noise_type="uniform",
        noise_rate=0.4,
        noise_scope="train",
        debug=False,
        config_overrides={
            "gb.embedding_path": embedding_path,
            "gb.tie_mode": "distribution_only",
            "gb.leave_one_out": bool(leave_one_out),
        },
    )


def run_sources(
    *,
    output_dir: Path,
    device: str,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...],
    sources: tuple[str, ...],
    include_gcn_hidden: bool,
    leave_one_out: bool = False,
) -> Path:
    manifest = prepare_artifacts(output_dir)
    output_path = output_dir / "representation_source_raw.csv"
    rows = _read_csv(output_path)
    completed = {
        (str(row["dataset"]), str(row["representation_source"]), int(row["seed"]))
        for row in rows
    }
    cells = [
        (dataset, source, seed)
        for dataset in datasets
        for source in sources
        for seed in seeds
    ]
    if include_gcn_hidden and 0 in seeds:
        cells.extend(
            (dataset, "gcn_hidden_noisy_seed0", 0)
            for dataset in sorted(GCN_HIDDEN_PATHS)
            if (dataset, "gcn_hidden_noisy_seed0") in manifest
        )
    for dataset, source, seed in cells:
        key = (dataset, source, seed)
        if key in completed:
            continue
        artifact = manifest[(dataset, source)]
        method = LOO_METHOD if leave_one_out else METHOD
        row = run_standard_downstream.run_one(
            _run_args(
                device,
                str(artifact["path"]),
                leave_one_out=leave_one_out,
            ),
            dataset,
            method,
            seed,
        )
        if row.get("status") != "ok":
            raise RuntimeError(f"representation-source run failed: {key}: {row}")
        if row.get("tie_mode") != "distribution_only":
            raise RuntimeError(f"representation-source row is not tie-safe: {key}")
        if bool(leave_one_out) and int(row.get("leave_one_out", 0)) != 1:
            raise RuntimeError(f"representation-source row is not leave-one-out: {key}")
        row.update(
            {
                "representation_source": source,
                "representation_path": artifact["path"],
                "representation_sha256": artifact["sha256"],
                "representation_label_free": artifact["label_free"],
                "representation_propagation_steps": artifact["propagation_steps"],
            }
        )
        rows.append(row)
        completed.add(key)
        _write_csv(output_path, rows)
        print(
            f"MLP43_SOURCE dataset={dataset} source={source} seed={seed} "
            f"balls={row.get('num_balls', '')} test={float(row['test']):.6f}"
        )
    return output_path


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not seeds or any(seed not in range(5) for seed in seeds):
        raise ValueError("seeds must be a non-empty subset of 0,1,2,3,4")
    return seeds


def _parse_subset(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected = tuple(token.strip().lower() for token in value.split(",") if token.strip())
    if not selected or any(item not in allowed for item in selected):
        raise ValueError(f"{label} must be a non-empty subset of {','.join(allowed)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--sources", default="raw_x,adj2_x")
    parser.add_argument("--include-gcn-hidden", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_sources(
        output_dir=args.output_dir,
        device=args.device,
        seeds=_parse_seeds(args.seeds),
        datasets=_parse_subset(args.datasets, DATASETS, "datasets"),
        sources=_parse_subset(args.sources, ("raw_x", "adj2_x"), "sources"),
        include_gcn_hidden=bool(args.include_gcn_hidden),
        leave_one_out=bool(args.leave_one_out),
    )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
