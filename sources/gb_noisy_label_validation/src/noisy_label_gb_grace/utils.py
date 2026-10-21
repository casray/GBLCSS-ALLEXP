from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_GRACE = PROJECT_ROOT / "upstream" / "GRACE"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).view(-1)
    return mask.long().view(-1)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels.long(), num_classes).float()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()

