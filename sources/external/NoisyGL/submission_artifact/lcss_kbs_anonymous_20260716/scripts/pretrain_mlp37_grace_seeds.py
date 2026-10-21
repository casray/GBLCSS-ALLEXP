from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = REPO_ROOT.parents[1] / "gb_graph_aware_minival_6_1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "mlp37_grace_seed_variance_20260716"

if str(DEFAULT_UPSTREAM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DEFAULT_UPSTREAM_ROOT / "src"))

from noisy_label_gb_grace.data import load_dataset  # noqa: E402
from noisy_label_gb_grace.upstream_grace import (  # noqa: E402
    build_grace_model,
    extract_encoder_h,
    load_grace_config,
    train_grace_epoch,
)
from noisy_label_gb_grace.utils import set_seed  # noqa: E402


CONFIG_FALLBACKS = {"corafull": "DBLP"}


def _canonical_dataset_name(dataset: str) -> str:
    aliases = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "corafull": "CoraFull",
        "cora-full": "CoraFull",
    }
    key = str(dataset).strip().lower()
    if key not in aliases:
        raise ValueError(f"unsupported MLP-37 dataset: {dataset}")
    return aliases[key]


def resolve_grace_config(dataset: str) -> tuple[dict[str, Any], str]:
    canonical = _canonical_dataset_name(dataset)
    source = CONFIG_FALLBACKS.get(canonical.lower(), canonical)
    return dict(load_grace_config(source)), source


def seed_output_dir(output_root: str | Path, dataset: str, h_seed: int) -> Path:
    canonical = _canonical_dataset_name(dataset)
    return Path(output_root) / "pretrain" / canonical.lower() / f"hseed{int(h_seed)}"


def build_checkpoint_payload(
    *,
    dataset: str,
    h_seed: int,
    config: dict[str, Any],
    config_source: str,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    losses: list[dict[str, Any]],
    wall_time_seconds: float,
    peak_memory_allocated_bytes: int,
    peak_memory_reserved_bytes: int,
) -> dict[str, Any]:
    return {
        "dataset": _canonical_dataset_name(dataset),
        "h_seed": int(h_seed),
        "config": dict(config),
        "config_source_dataset": str(config_source),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "losses": losses,
        "wall_time_seconds": float(wall_time_seconds),
        "peak_memory_allocated_bytes": int(peak_memory_allocated_bytes),
        "peak_memory_reserved_bytes": int(peak_memory_reserved_bytes),
        "label_free": True,
        "used_labels_or_masks_in_training": False,
        "training_inputs": ["x", "edge_index"],
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_losses(path: Path, losses: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(losses)


def prepare_device_measurement(device: torch.device) -> None:
    if device.type != "cuda":
        return
    # ROCm rejects peak-stat calls before the CUDA-compatible context is selected.
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def pretrain_one(
    *,
    dataset_name: str,
    h_seed: int,
    device_name: str,
    output_root: str | Path,
    data_root: str | Path | None = None,
    epochs_override: int | None = None,
    log_every: int = 20,
) -> dict[str, Any]:
    canonical = _canonical_dataset_name(dataset_name)
    config, config_source = resolve_grace_config(canonical)
    config["seed"] = int(h_seed)
    if epochs_override is not None:
        config["num_epochs"] = int(epochs_override)

    set_seed(int(h_seed))
    device = torch.device(device_name)
    prepare_device_measurement(device)

    started = time.perf_counter()
    dataset, data = load_dataset(
        canonical,
        data_root=Path(data_root) if data_root else None,
        device=device,
    )
    model = build_grace_model(dataset, config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    losses: list[dict[str, Any]] = []
    epochs = int(config["num_epochs"])
    for epoch in range(1, epochs + 1):
        loss = train_grace_epoch(model, optimizer, data.x, data.edge_index, config)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError(
                f"non-finite GRACE loss for {canonical} hseed={h_seed} epoch={epoch}: {loss}"
            )
        losses.append({"epoch": epoch, "loss": float(loss)})
        if int(log_every) > 0 and (epoch == 1 or epoch % int(log_every) == 0 or epoch == epochs):
            print(
                f"MLP37_PRETRAIN dataset={canonical} hseed={h_seed} "
                f"epoch={epoch}/{epochs} loss={loss:.6f}"
            )

    with torch.no_grad():
        h = extract_encoder_h(model, data.x, data.edge_index, normalize=True).detach().cpu().float()
    wall_time = time.perf_counter() - started
    if not torch.isfinite(h).all():
        raise FloatingPointError(f"non-finite H for {canonical} hseed={h_seed}")

    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    out_dir = seed_output_dir(output_root, canonical, h_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    h_path = out_dir / "H.pt"
    checkpoint_path = out_dir / "checkpoint.pt"
    losses_path = out_dir / "losses.csv"
    metadata_path = out_dir / "metadata.json"

    torch.save(h, h_path)
    payload = build_checkpoint_payload(
        dataset=canonical,
        h_seed=h_seed,
        config=config,
        config_source=config_source,
        model_state_dict=_cpu_tree(model.state_dict()),
        optimizer_state_dict=_cpu_tree(optimizer.state_dict()),
        losses=losses,
        wall_time_seconds=wall_time,
        peak_memory_allocated_bytes=peak_allocated,
        peak_memory_reserved_bytes=peak_reserved,
    )
    torch.save(payload, checkpoint_path)
    _write_losses(losses_path, losses)

    metadata = {
        **{key: value for key, value in payload.items() if key not in {"model_state_dict", "optimizer_state_dict", "losses"}},
        "embedding_shape": [int(h.shape[0]), int(h.shape[1])],
        "max_row_norm_error": float((h.norm(p=2, dim=1) - 1.0).abs().max().item()),
        "final_loss": float(losses[-1]["loss"]),
        "embedding_path": str(h_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "embedding_sha256": sha256_file(h_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "embedding_size_bytes": int(h_path.stat().st_size),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the frozen 3x3 GRACE seed grid for MLP-37.")
    parser.add_argument("--datasets", default="Cora,CiteSeer,CoraFull")
    parser.add_argument("--h-seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-only override; omit for formal runs.")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    h_seeds = [int(item.strip()) for item in args.h_seeds.split(",") if item.strip()]
    rows = []
    summary_path = Path(args.output_root) / "grace_pretrain_manifest.csv"
    for dataset in datasets:
        for h_seed in h_seeds:
            row = pretrain_one(
                dataset_name=dataset,
                h_seed=h_seed,
                device_name=args.device,
                output_root=args.output_root,
                data_root=args.data_root,
                epochs_override=args.epochs,
                log_every=args.log_every,
            )
            rows.append(row)
            _write_summary(summary_path, rows)
    print(f"WROTE {summary_path}")


if __name__ == "__main__":
    main()
