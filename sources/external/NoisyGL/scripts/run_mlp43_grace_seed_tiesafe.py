from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_standard_downstream
from scripts.run_mlp37_grace_seed_variance import load_pretrain_manifest
from utils.frozen_grace_targets import build_frozen_grace_soft_targets
from utils.labelnoise import label_process
from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "corafull")
H_SEEDS = (0, 1, 2)
DOWNSTREAM_SEEDS = (0, 1, 2)
METHOD = "frozengrace_gb_mlp_softdist_beta05"
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"


def formal_run_cells() -> list[tuple[str, int, int]]:
    return [
        (dataset, h_seed, downstream_seed)
        for dataset in DATASETS
        for h_seed in H_SEEDS
        for downstream_seed in DOWNSTREAM_SEEDS
    ]


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("H", "embedding", "embeddings", "x"):
            if key in loaded and torch.is_tensor(loaded[key]):
                loaded = loaded[key]
                break
    if not torch.is_tensor(loaded):
        raise ValueError(f"embedding artifact does not contain a tensor: {path}")
    return loaded.detach().cpu()


def validate_embedding(path: str | Path, expected_nodes: int) -> dict[str, object]:
    path = Path(path).resolve()
    tensor = _load_tensor(path)
    if tensor.ndim != 2:
        raise ValueError(f"embedding must be a matrix: {path}")
    if int(tensor.shape[0]) != int(expected_nodes):
        raise ValueError(
            f"embedding node count mismatch: expected {expected_nodes}, "
            f"found {tensor.shape[0]} at {path}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"embedding contains non-finite values: {path}")
    norms = torch.linalg.vector_norm(tensor.float(), dim=1)
    return {
        "h_num_nodes": int(tensor.shape[0]),
        "h_dim": int(tensor.shape[1]),
        "h_row_norm_mean": float(norms.mean().item()),
        "h_row_norm_std": float(norms.std(unbiased=True).item()),
    }


def _run_args(
    device: str,
    embedding_path: str,
    *,
    leave_one_out: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=str(device),
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        noise_type="uniform",
        noise_rate=0.4,
        noise_scope="train",
        debug=False,
        config_overrides={
            "gb.embedding_path": str(embedding_path),
            "gb.tie_mode": "distribution_only",
            "gb.leave_one_out": bool(leave_one_out),
        },
    )


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: ""}).to_dict("records")


def _hash_numpy(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _geometry_row(
    *,
    dataset: str,
    h_seed: int,
    downstream_seed: int,
    embedding_path: Path,
    assignment_root: Path,
    leave_one_out: bool = False,
) -> dict[str, object]:
    run_standard_downstream.setup_seed(int(downstream_seed))
    data = run_standard_downstream._load_dataset(
        dataset,
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        random_seed=int(downstream_seed),
        eligible_mask=data.train_masks,
        debug=False,
    )
    method = LOO_METHOD if leave_one_out else METHOD
    conf = load_conf(None, method, dataset)
    gb = conf.gb
    result = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=int(gb["max_balls"]),
        min_ball_size=int(gb["min_ball_size"]),
        split_improvement=float(gb["split_improvement"]),
        min_train_per_ball=int(gb["min_train_per_ball"]),
        alpha_baseline=str(gb["alpha_baseline"]),
        target_mode=str(gb["target_mode"]),
        target_beta=float(gb["target_beta"]),
        random_state=int(downstream_seed),
        random_control=False,
        tie_mode="distribution_only",
        leave_one_out=bool(leave_one_out),
    )
    assignment = np.asarray(result.assignment, dtype=np.int64)
    assignment_path = (
        assignment_root / dataset / f"hseed{h_seed}_dseed{downstream_seed}.npy"
    )
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(assignment_path, assignment)
    modified = np.asarray(modified_mask, dtype=np.int64)
    return {
        "dataset": dataset,
        "h_seed": int(h_seed),
        "seed": int(downstream_seed),
        "embedding_path": str(embedding_path.resolve()),
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": _hash_numpy(assignment),
        "split_hash": run_standard_downstream._split_fingerprint(
            data.train_masks, data.val_masks, data.test_masks
        ),
        "noise_hash": run_standard_downstream._noise_fingerprint(
            data.labels, noisy_label
        ),
        **run_standard_downstream._noise_stats(
            data.labels,
            noisy_label,
            modified,
            data.train_masks,
            data.val_masks,
            data.test_masks,
        ),
        **result.diagnostics,
    }


def run_grid(
    *,
    manifest_path: Path,
    output_dir: Path,
    device: str,
    resume: bool,
    leave_one_out: bool = False,
) -> tuple[Path, Path]:
    manifest = load_pretrain_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "grace_seed_tiesafe_raw.csv"
    geometry_path = output_dir / "grace_seed_tiesafe_geometry.csv"

    node_counts: dict[str, int] = {}
    embedding_diagnostics: dict[tuple[str, int], dict[str, object]] = {}
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
        node_counts[dataset] = int(data.labels.shape[0])
        for h_seed in H_SEEDS:
            embedding_diagnostics[(dataset, h_seed)] = validate_embedding(
                manifest[(dataset, h_seed)]["embedding_path"],
                expected_nodes=node_counts[dataset],
            )

    rows = _load_existing(raw_path) if resume else []
    completed = {
        (str(row["dataset"]), int(row["h_seed"]), int(row["seed"])) for row in rows
    }
    for dataset, h_seed, downstream_seed in formal_run_cells():
        key = (dataset, h_seed, downstream_seed)
        if key in completed:
            continue
        manifest_row = manifest[(dataset, h_seed)]
        method = LOO_METHOD if leave_one_out else METHOD
        row = run_standard_downstream.run_one(
            _run_args(
                device,
                manifest_row["embedding_path"],
                leave_one_out=leave_one_out,
            ),
            dataset,
            method,
            downstream_seed,
        )
        if row.get("status") != "ok":
            raise RuntimeError(f"MLP-43 GRACE-seed run failed for {key}: {row}")
        if row.get("tie_mode") != "distribution_only":
            raise RuntimeError(f"MLP-43 GRACE-seed run is not tie-safe: {key}")
        if bool(leave_one_out) and int(row.get("leave_one_out", 0)) != 1:
            raise RuntimeError(f"MLP-43 GRACE-seed run is not leave-one-out: {key}")
        row.update(
            {
                "h_seed": int(h_seed),
                "h_path": str(manifest_row["embedding_path"]),
                "h_sha256": str(manifest_row["embedding_sha256"]),
                **embedding_diagnostics[(dataset, h_seed)],
            }
        )
        rows.append(row)
        completed.add(key)
        run_standard_downstream.write_csv(raw_path, rows)
        print(
            f"MLP43_HSEED dataset={dataset} hseed={h_seed} "
            f"dseed={downstream_seed} test={float(row['test']):.6f}"
        )

    geometry_rows = _load_existing(geometry_path) if resume else []
    geometry_completed = {
        (str(row["dataset"]), int(row["h_seed"]), int(row["seed"]))
        for row in geometry_rows
    }
    for dataset, h_seed, downstream_seed in formal_run_cells():
        key = (dataset, h_seed, downstream_seed)
        if key in geometry_completed:
            continue
        row = _geometry_row(
            dataset=dataset,
            h_seed=h_seed,
            downstream_seed=downstream_seed,
            embedding_path=Path(manifest[(dataset, h_seed)]["embedding_path"]),
            assignment_root=output_dir / "grace_seed_assignments",
            leave_one_out=leave_one_out,
        )
        geometry_rows.append(row)
        geometry_completed.add(key)
        run_standard_downstream.write_csv(geometry_path, geometry_rows)
    return raw_path, geometry_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--leave-one-out", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path, geometry_path = run_grid(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        resume=not args.no_resume,
        leave_one_out=bool(args.leave_one_out),
    )
    print(f"WROTE {raw_path}")
    print(f"WROTE {geometry_path}")


if __name__ == "__main__":
    main()
