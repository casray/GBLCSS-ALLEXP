from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts import run_standard_downstream
from scripts.pretrain_mlp37_grace_seeds import sha256_file
from utils.frozen_grace_targets import build_frozen_grace_soft_targets
from utils.labelnoise import label_process
from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "corafull")
H_SEEDS = (0, 1, 2)
DOWNSTREAM_SEEDS = (0, 1, 2)
LCSS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
REFERENCES = ("mlp_ce", "mlp_jocor")
H_METHODS = (LCSS, SHUFFLED)


def formal_run_cells() -> list[tuple[str, str, int, int | None]]:
    cells: list[tuple[str, str, int, int | None]] = []
    for dataset in DATASETS:
        for seed in DOWNSTREAM_SEEDS:
            for method in REFERENCES:
                cells.append((dataset, method, seed, None))
        for h_seed in H_SEEDS:
            for seed in DOWNSTREAM_SEEDS:
                for method in H_METHODS:
                    cells.append((dataset, method, seed, h_seed))
    return cells


def load_pretrain_manifest(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"dataset", "h_seed", "embedding_path", "embedding_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"pretrain manifest missing columns: {sorted(missing)}")
    manifest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        dataset = str(row["dataset"]).strip().lower()
        h_seed = int(row["h_seed"])
        key = (dataset, h_seed)
        if key in manifest:
            raise ValueError(f"duplicate pretrain manifest cell: {key}")
        embedding_path = Path(str(row["embedding_path"])).resolve()
        if not embedding_path.exists():
            raise ValueError(f"missing H artifact for {key}: {embedding_path}")
        actual_sha = sha256_file(embedding_path)
        expected_sha = str(row["embedding_sha256"]).strip().lower()
        if actual_sha.lower() != expected_sha:
            raise ValueError(f"H SHA256 mismatch for {key}")
        row["embedding_path"] = str(embedding_path)
        manifest[key] = row
    expected = {(dataset, h_seed) for dataset in DATASETS for h_seed in H_SEEDS}
    if set(manifest) != expected:
        raise ValueError(
            f"incomplete pretrain manifest: missing={sorted(expected - set(manifest))}, "
            f"extra={sorted(set(manifest) - expected)}"
        )
    return manifest


def decorate_h_row(row: dict[str, Any], *, h_seed: int, manifest_row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "h_seed": int(h_seed),
        "h_path": str(manifest_row["embedding_path"]),
        "h_sha256": str(manifest_row["embedding_sha256"]),
        "h_config_source_dataset": str(manifest_row.get("config_source_dataset", "")),
        "h_pretrain_wall_time_seconds": manifest_row.get("wall_time_seconds", ""),
        "h_peak_memory_allocated_bytes": manifest_row.get("peak_memory_allocated_bytes", ""),
        "h_peak_memory_reserved_bytes": manifest_row.get("peak_memory_reserved_bytes", ""),
    }


def _reference_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "h_seed": "", "h_path": "", "h_sha256": "", "h_config_source_dataset": ""}


def _run_args(device: str, embedding_path: str | None = None) -> SimpleNamespace:
    overrides = {"gb.embedding_path": str(embedding_path)} if embedding_path else {}
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
        config_overrides=overrides,
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, int | None]:
    raw_h_seed = row.get("h_seed", "")
    h_seed = None if raw_h_seed == "" or pd.isna(raw_h_seed) else int(raw_h_seed)
    return str(row["dataset"]), str(row["method"]), int(row["seed"]), h_seed


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: ""}).to_dict("records")


def _hash_numpy(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_geometry_row(
    *,
    dataset: str,
    downstream_seed: int,
    h_seed: int,
    embedding_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
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
    conf = load_conf(None, LCSS, dataset)
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
    )
    assignment = np.asarray(result.assignment, dtype=np.int64)
    assignment_dir = Path(output_root) / "assignments" / dataset
    assignment_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = assignment_dir / f"hseed{h_seed}_dseed{downstream_seed}.npy"
    np.save(assignment_path, assignment)
    target_np = result.soft_target.detach().cpu().numpy().astype(np.float32, copy=False)
    modified = np.asarray(modified_mask, dtype=np.int64)
    return {
        "dataset": dataset,
        "h_seed": int(h_seed),
        "seed": int(downstream_seed),
        "embedding_path": str(Path(embedding_path).resolve()),
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": _hash_numpy(assignment),
        "soft_target_sha256": _hash_numpy(target_np),
        "split_hash": run_standard_downstream._split_fingerprint(
            data.train_masks, data.val_masks, data.test_masks
        ),
        "noise_hash": run_standard_downstream._noise_fingerprint(data.labels, noisy_label),
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


def run_formal_grid(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    device: str,
    resume: bool = True,
) -> tuple[Path, Path]:
    manifest = load_pretrain_manifest(manifest_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "mlp37_raw.csv"
    geometry_path = output_root / "mlp37_geometry.csv"
    rows = _load_existing(raw_path) if resume else []
    completed = {_row_key(row) for row in rows}

    for dataset, method, seed, h_seed in formal_run_cells():
        key = (dataset, method, seed, h_seed)
        if key in completed:
            continue
        if h_seed is None:
            row = run_standard_downstream.run_one(_run_args(device), dataset, method, seed)
            row = _reference_row(row)
        else:
            manifest_row = manifest[(dataset, h_seed)]
            row = run_standard_downstream.run_one(
                _run_args(device, manifest_row["embedding_path"]), dataset, method, seed
            )
            row = decorate_h_row(row, h_seed=h_seed, manifest_row=manifest_row)
        if row.get("status") != "ok":
            raise RuntimeError(f"MLP-37 run failed: {key}: {row.get('error', '')}")
        rows.append(row)
        completed.add(key)
        run_standard_downstream.write_csv(raw_path, rows)
        print(
            f"MLP37_DOWNSTREAM dataset={dataset} method={method} "
            f"dseed={seed} hseed={h_seed} test={float(row['test']):.6f}"
        )

    geometry_rows = _load_existing(geometry_path) if resume else []
    geometry_keys = {
        (str(row["dataset"]), int(row["h_seed"]), int(row["seed"])) for row in geometry_rows
    }
    for dataset in DATASETS:
        for h_seed in H_SEEDS:
            manifest_row = manifest[(dataset, h_seed)]
            for seed in DOWNSTREAM_SEEDS:
                key = (dataset, h_seed, seed)
                if key in geometry_keys:
                    continue
                row = build_geometry_row(
                    dataset=dataset,
                    downstream_seed=seed,
                    h_seed=h_seed,
                    embedding_path=manifest_row["embedding_path"],
                    output_root=output_root,
                )
                geometry_rows.append(row)
                geometry_keys.add(key)
                run_standard_downstream.write_csv(geometry_path, geometry_rows)
                print(
                    f"MLP37_GEOMETRY dataset={dataset} hseed={h_seed} dseed={seed} "
                    f"balls={row.get('num_balls', '')}"
                )
    return raw_path, geometry_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal MLP-37 3x3 downstream grid.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path, geometry_path = run_formal_grid(
        manifest_path=args.manifest,
        output_root=args.output_root,
        device=args.device,
        resume=not args.no_resume,
    )
    print(f"WROTE {raw_path}")
    print(f"WROTE {geometry_path}")


if __name__ == "__main__":
    main()
