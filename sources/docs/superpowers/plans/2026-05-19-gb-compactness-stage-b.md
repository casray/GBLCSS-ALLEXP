# GB Compactness Stage B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Stage B experiment project that tests whether label-free granular-ball compactness can shape a graph SSL encoder and improve granular-ball quality without representation collapse.

**Architecture:** Create an isolated Python package under `gb_denoise_stage_b_compactness/`. Keep SSL training, ball generation, compactness loss, diagnostics, and reporting in separate modules so Stage B can reuse Stage A decisions later without coupling to unfinished Stage A code.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, NumPy, pandas, PyYAML, scikit-learn, pytest.

---

## Implementation Context

Spec source:

- `docs/superpowers/specs/2026-05-19-gb-compactness-stage-b-design.md`

New project root:

- `gb_denoise_stage_b_compactness/`

Important constraints:

- Do not modify existing experiment-history markdown files.
- Do not modify `gb_denoise_stage_a_negfree_vs_infonce/`.
- Do not use labels in SSL training, granular-ball generation, compactness loss, or reliable-ball gating.
- Labels may be used only for linear evaluation and diagnostic purity.
- If implementation starts from `main`, first create a branch named `codex/stage-b-gb-compactness`.

Recommended preflight commands:

```powershell
git status --short
git switch -c codex/stage-b-gb-compactness
```

If the branch already exists:

```powershell
git switch codex/stage-b-gb-compactness
```

---

## File Structure

Create these files:

```text
gb_denoise_stage_b_compactness/
  README.md
  pyproject.toml
  configs/
    stage_b.yaml
  src/
    gb_stage_b/
      __init__.py
      augment.py
      balls.py
      cli.py
      compactness.py
      config.py
      data.py
      diagnostics.py
      eval.py
      losses.py
      models.py
      report.py
      train.py
      utils.py
  tests/
    test_balls.py
    test_compactness.py
    test_config.py
    test_diagnostics.py
    test_losses.py
    test_report.py
    test_smoke_cli.py
```

Responsibilities:

- `config.py`: typed config loader, defaults, validation.
- `utils.py`: deterministic seeding, filesystem helpers, JSON/YAML helpers.
- `data.py`: Planetoid loader plus tiny synthetic graph for tests/smoke runs.
- `models.py`: GCN encoder, MLP projector, predictor, and SSL model wrapper.
- `augment.py`: edge dropout and feature masking.
- `losses.py`: InfoNCE and negative-free SSL losses.
- `balls.py`: label-free angular granular-ball generation, ball metrics, shuffled and size-matched assignment controls.
- `compactness.py`: stop-gradient center compactness loss.
- `diagnostics.py`: ball-quality, representation-collapse, and optional purity diagnostics.
- `eval.py`: logistic regression linear evaluation.
- `train.py`: Stage B training loop and run artifact writing.
- `report.py`: result discovery, summary CSV, paired deltas.
- `cli.py`: command line entry point.

---

### Task 1: Project Skeleton and Import Smoke Test

**Files:**

- Create: `gb_denoise_stage_b_compactness/pyproject.toml`
- Create: `gb_denoise_stage_b_compactness/README.md`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/__init__.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_config.py`

- [ ] **Step 1: Write the failing import test**

Create `gb_denoise_stage_b_compactness/tests/test_config.py`:

```python
from gb_stage_b import __version__


def test_package_imports():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_config.py -q
```

Expected: FAIL because the package and `pyproject.toml` do not exist yet.

- [ ] **Step 3: Create package metadata**

Create `gb_denoise_stage_b_compactness/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gb-stage-b"
version = "0.1.0"
description = "Stage B experiments for label-free granular-ball compactness encoder shaping."
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "pandas>=2.0",
  "pyyaml>=6.0",
  "scikit-learn>=1.3",
  "torch>=2.0",
  "torch-geometric>=2.4",
  "tqdm>=4.66"
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[project.scripts]
gb-stage-b = "gb_stage_b.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/__init__.py`:

```python
"""Stage B experiments for granular-ball compactness encoder shaping."""

__version__ = "0.1.0"

__all__ = [
    "augment",
    "balls",
    "cli",
    "compactness",
    "config",
    "data",
    "diagnostics",
    "eval",
    "losses",
    "models",
    "report",
    "train",
    "utils",
]
```

Create `gb_denoise_stage_b_compactness/README.md`:

````markdown
# GB Denoise Stage B

This experiment tests whether a label-free granular-ball compactness loss can shape a graph SSL encoder and improve granular-ball quality.

The project follows the Stage B design in `../docs/superpowers/specs/2026-05-19-gb-compactness-stage-b-design.md`.

Labels are used only for linear evaluation and diagnostic purity. Labels are never used for SSL training, granular-ball generation, reliable-ball gating, or compactness loss.

## Quick Smoke Run

```powershell
cd gb_denoise_stage_b_compactness
python -m pip install -e .[dev]
pytest -q
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Synthetic --method B0 --seed 0 --epochs 4 --warmup-epochs 2
python -m gb_stage_b.cli summarize --results results
```
````

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cd gb_denoise_stage_b_compactness
python -m pip install -e .[dev]
python -m pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "chore: scaffold stage b experiment package"
```

---

### Task 2: Configuration and Utility Layer

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/config.py`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/utils.py`
- Create: `gb_denoise_stage_b_compactness/configs/stage_b.yaml`
- Modify: `gb_denoise_stage_b_compactness/tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Replace `gb_denoise_stage_b_compactness/tests/test_config.py` with:

```python
from pathlib import Path

import pytest

from gb_stage_b import __version__
from gb_stage_b.config import StageBConfig, load_config
from gb_stage_b.utils import ensure_dir, set_seed


def test_package_imports():
    assert __version__ == "0.1.0"


def test_load_config_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
dataset: Synthetic
method: B1
ssl_objective: negfree
epochs: 10
warmup_epochs: 4
lambda_gb: 0.005
seed: 0
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg, StageBConfig)
    assert cfg.dataset == "Synthetic"
    assert cfg.method == "B1"
    assert cfg.ball_space == "H"
    assert cfg.assignment_control == "true"


def test_config_rejects_bad_warmup():
    with pytest.raises(ValueError, match="warmup_epochs"):
        StageBConfig(dataset="Synthetic", epochs=3, warmup_epochs=4)


def test_utils_seed_and_dir(tmp_path):
    set_seed(123)
    out = ensure_dir(tmp_path / "nested")
    assert out.exists()
    assert out.is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_config.py -q
```

Expected: FAIL because `config.py` and `utils.py` are missing.

- [ ] **Step 3: Implement config and utilities**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StageBConfig:
    dataset: str = "Cora"
    method: str = "B0"
    ssl_objective: str = "negfree"
    seed: int = 0
    epochs: int = 1000
    warmup_epochs: int = 300
    hidden_dim: int = 256
    proj_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    edge_drop: float = 0.2
    feat_mask: float = 0.2
    lambda_gb: float = 0.005
    ball_space: str = "H"
    ball_refresh_interval: int = 0
    assignment_control: str = "true"
    reliable_ball_min_size: int = 3
    reliable_ball_max_size: int = 256
    reliable_ball_ad_percentile: float = 50.0
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must be between 0 and epochs")
        if self.method not in {"B0", "B1", "B2", "B3", "B4"}:
            raise ValueError("method must be one of B0, B1, B2, B3, B4")
        if self.ssl_objective not in {"negfree", "infonce"}:
            raise ValueError("ssl_objective must be negfree or infonce")
        if self.ball_space not in {"H", "Z"}:
            raise ValueError("ball_space must be H or Z")
        if self.assignment_control not in {"true", "shuffled", "size_matched_random"}:
            raise ValueError("assignment_control must be true, shuffled, or size_matched_random")
        if self.lambda_gb < 0:
            raise ValueError("lambda_gb must be non-negative")
        if not 0 <= self.reliable_ball_ad_percentile <= 100:
            raise ValueError("reliable_ball_ad_percentile must be in [0, 100]")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> StageBConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return StageBConfig(**data)
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/utils.py`:

```python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

Create `gb_denoise_stage_b_compactness/configs/stage_b.yaml`:

```yaml
dataset: Synthetic
method: B0
ssl_objective: negfree
seed: 0
epochs: 1000
warmup_epochs: 300
hidden_dim: 256
proj_dim: 256
lr: 0.001
weight_decay: 0.00001
edge_drop: 0.2
feat_mask: 0.2
lambda_gb: 0.005
ball_space: H
ball_refresh_interval: 0
assignment_control: true
reliable_ball_min_size: 3
reliable_ball_max_size: 256
reliable_ball_ad_percentile: 50.0
results_dir: results
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add stage b config utilities"
```

---

### Task 3: Synthetic and Planetoid Data Loading

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/data.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_diagnostics.py`

- [ ] **Step 1: Write failing data tests**

Create `gb_denoise_stage_b_compactness/tests/test_diagnostics.py`:

```python
import torch

from gb_stage_b.data import load_graph


def test_synthetic_graph_has_masks_and_edges():
    graph = load_graph("Synthetic", seed=0)
    assert graph.x.ndim == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.y.shape[0] == graph.x.shape[0]
    assert graph.train_mask.dtype is torch.bool
    assert graph.val_mask.sum().item() > 0
    assert graph.test_mask.sum().item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_diagnostics.py -q
```

Expected: FAIL because `data.py` is missing.

- [ ] **Step 3: Implement graph loading**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/data.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def _mask_from_indices(num_nodes: int, indices: list[int]) -> torch.Tensor:
    out = torch.zeros(num_nodes, dtype=torch.bool)
    out[indices] = True
    return out


def synthetic_graph(seed: int = 0) -> Data:
    gen = torch.Generator().manual_seed(seed)
    num_per_class = 20
    centers = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5], [-1.0, -1.0, 0.5, 0.5]])
    xs = []
    ys = []
    for cls, center in enumerate(centers):
        xs.append(center + 0.08 * torch.randn(num_per_class, center.numel(), generator=gen))
        ys.extend([cls] * num_per_class)
    x = torch.cat(xs, dim=0).float()
    y = torch.tensor(ys, dtype=torch.long)
    edges = []
    for start in range(0, x.size(0), num_per_class):
        for i in range(start, start + num_per_class):
            edges.append((i, start + ((i - start + 1) % num_per_class)))
            edges.append((start + ((i - start + 1) % num_per_class), i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=y)
    train_idx = list(range(0, 10)) + list(range(20, 30)) + list(range(40, 50))
    val_idx = list(range(10, 15)) + list(range(30, 35)) + list(range(50, 55))
    test_idx = list(range(15, 20)) + list(range(35, 40)) + list(range(55, 60))
    data.train_mask = _mask_from_indices(x.size(0), train_idx)
    data.val_mask = _mask_from_indices(x.size(0), val_idx)
    data.test_mask = _mask_from_indices(x.size(0), test_idx)
    return data


def load_graph(name: str, seed: int = 0, root: str | Path = "data") -> Data:
    if name == "Synthetic":
        return synthetic_graph(seed)
    if name not in {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("dataset must be Synthetic, Cora, CiteSeer, or PubMed")
    dataset = Planetoid(root=str(Path(root) / name), name=name)
    return dataset[0]
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add graph data loaders"
```

---

### Task 4: Models, Augmentations, and SSL Losses

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/models.py`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/augment.py`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/losses.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_losses.py`

- [ ] **Step 1: Write failing model/loss tests**

Create `gb_denoise_stage_b_compactness/tests/test_losses.py`:

```python
import torch

from gb_stage_b.augment import drop_edges, mask_features
from gb_stage_b.data import load_graph
from gb_stage_b.losses import infonce_loss, negative_free_loss
from gb_stage_b.models import StageBModel


def test_model_forward_shapes():
    data = load_graph("Synthetic", seed=0)
    model = StageBModel(in_dim=data.x.size(1), hidden_dim=16, proj_dim=8)
    h, z, p = model(data.x, data.edge_index)
    assert h.shape == (data.x.size(0), 16)
    assert z.shape == (data.x.size(0), 8)
    assert p.shape == (data.x.size(0), 8)


def test_ssl_losses_are_finite():
    z1 = torch.randn(12, 8)
    z2 = torch.randn(12, 8)
    p1 = torch.randn(12, 8)
    p2 = torch.randn(12, 8)
    assert torch.isfinite(infonce_loss(z1, z2))
    assert torch.isfinite(negative_free_loss(p1, z2, p2, z1))


def test_augmentations_preserve_shapes():
    data = load_graph("Synthetic", seed=0)
    edge_index = drop_edges(data.edge_index, drop_prob=0.5, seed=0)
    x = mask_features(data.x, mask_prob=0.5, seed=0)
    assert edge_index.shape[0] == 2
    assert x.shape == data.x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_losses.py -q
```

Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement models, augmentations, and SSL losses**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/models.py`:

```python
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageBModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim)
        self.projector = MLP(hidden_dim, hidden_dim, proj_dim)
        self.predictor = MLP(proj_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, edge_index)
        z = self.projector(h)
        p = self.predictor(z)
        return h, z, p
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/augment.py`:

```python
from __future__ import annotations

import torch


def drop_edges(edge_index: torch.Tensor, drop_prob: float, seed: int | None = None) -> torch.Tensor:
    if drop_prob <= 0:
        return edge_index
    gen = torch.Generator(device=edge_index.device)
    if seed is not None:
        gen.manual_seed(seed)
    keep = torch.rand(edge_index.size(1), generator=gen, device=edge_index.device) >= drop_prob
    if keep.sum().item() == 0:
        keep[0] = True
    return edge_index[:, keep]


def mask_features(x: torch.Tensor, mask_prob: float, seed: int | None = None) -> torch.Tensor:
    if mask_prob <= 0:
        return x
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)
    mask = torch.rand(x.size(1), generator=gen, device=x.device) < mask_prob
    out = x.clone()
    out[:, mask] = 0
    return out
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    z1 = _normalize(z1)
    z2 = _normalize(z2)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def negative_free_loss(p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    p1 = _normalize(p1)
    p2 = _normalize(p2)
    z1 = _normalize(z1.detach())
    z2 = _normalize(z2.detach())
    return 0.5 * (1 - (p1 * z2).sum(dim=-1).mean()) + 0.5 * (1 - (p2 * z1).sum(dim=-1).mean())
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_losses.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add ssl models and losses"
```

---

### Task 5: Granular-Ball Generator and Assignment Controls

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/balls.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_balls.py`

- [ ] **Step 1: Write failing ball tests**

Create `gb_denoise_stage_b_compactness/tests/test_balls.py`:

```python
import torch

from gb_stage_b.balls import (
    BallBatch,
    generate_angular_balls,
    make_ball_batch,
    reliable_ball_mask,
    shuffled_assignments,
    size_matched_random_assignments,
)


def test_generate_angular_balls_covers_all_nodes():
    x = torch.randn(30, 8)
    balls = generate_angular_balls(x, min_size=3, quality_threshold=0.35)
    assert isinstance(balls, BallBatch)
    assert balls.assignment.shape == (30,)
    assert set(balls.assignment.tolist()) == set(range(balls.num_balls))
    assert balls.centers.shape[0] == balls.num_balls


def test_reliable_ball_mask_filters_by_size_and_ad():
    x = torch.randn(20, 4)
    balls = generate_angular_balls(x, min_size=2, quality_threshold=0.4)
    mask = reliable_ball_mask(balls, min_size=2, max_size=20, ad_percentile=100.0)
    assert mask.dtype == torch.bool
    assert mask.shape == (balls.num_balls,)
    assert mask.any()


def test_assignment_controls_preserve_shape_and_sizes():
    assignment = torch.tensor([0, 0, 1, 1, 1, 2])
    shuffled = shuffled_assignments(assignment, seed=0)
    random_assignment = size_matched_random_assignments(assignment, seed=0)
    assert shuffled.shape == assignment.shape
    assert random_assignment.shape == assignment.shape
    assert sorted(torch.bincount(random_assignment).tolist()) == sorted(torch.bincount(assignment).tolist())


def test_make_ball_batch_recomputes_centers_for_control_assignment():
    x = torch.randn(12, 4)
    assignment = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    balls = make_ball_batch(x, assignment)
    assert balls.centers.shape == (4, 4)
    assert balls.sizes.tolist() == [3, 3, 3, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_balls.py -q
```

Expected: FAIL because `balls.py` is missing.

- [ ] **Step 3: Implement ball generation and controls**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/balls.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BallBatch:
    assignment: torch.Tensor
    centers: torch.Tensor
    sizes: torch.Tensor
    average_angular_distance: torch.Tensor
    radius: torch.Tensor

    @property
    def num_balls(self) -> int:
        return int(self.centers.size(0))


def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    sim = (a * b).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(sim)


def make_ball_batch(x: torch.Tensor, assignment: torch.Tensor) -> BallBatch:
    x = F.normalize(x.detach(), dim=-1, eps=1e-8)
    centers = []
    sizes = []
    ads = []
    radii = []
    for ball_id in sorted(assignment.unique().tolist()):
        idx = torch.where(assignment == ball_id)[0]
        members = x[idx]
        center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).squeeze(0)
        dist = angular_distance(members, center.expand_as(members))
        centers.append(center)
        sizes.append(idx.numel())
        ads.append(dist.mean())
        radii.append(dist.max())
    remap = {old: new for new, old in enumerate(sorted(assignment.unique().tolist()))}
    compact_assignment = torch.tensor([remap[int(v)] for v in assignment.tolist()], device=assignment.device)
    return BallBatch(
        assignment=compact_assignment,
        centers=torch.stack(centers, dim=0),
        sizes=torch.tensor(sizes, dtype=torch.long, device=x.device),
        average_angular_distance=torch.stack(ads),
        radius=torch.stack(radii),
    )


def generate_angular_balls(x: torch.Tensor, min_size: int = 3, quality_threshold: float = 0.35) -> BallBatch:
    x = F.normalize(x.detach(), dim=-1, eps=1e-8)
    groups = [torch.arange(x.size(0), device=x.device)]
    final = []
    while groups:
        idx = groups.pop()
        members = x[idx]
        center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).squeeze(0)
        dist = angular_distance(members, center.expand_as(members))
        if idx.numel() < 2 * min_size or float(dist.mean()) <= quality_threshold:
            final.append(idx)
            continue
        farthest = int(torch.argmax(dist).item())
        c1 = members[farthest]
        dist_to_c1 = angular_distance(members, c1.expand_as(members))
        second = int(torch.argmax(dist_to_c1).item())
        c2 = members[second]
        d1 = angular_distance(members, c1.expand_as(members))
        d2 = angular_distance(members, c2.expand_as(members))
        left = idx[d1 <= d2]
        right = idx[d1 > d2]
        if left.numel() < min_size or right.numel() < min_size:
            final.append(idx)
        else:
            groups.extend([left, right])
    assignment = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for ball_id, idx in enumerate(final):
        assignment[idx] = ball_id
    return make_ball_batch(x, assignment)


def reliable_ball_mask(balls: BallBatch, min_size: int, max_size: int, ad_percentile: float) -> torch.Tensor:
    threshold = torch.quantile(balls.average_angular_distance, ad_percentile / 100.0)
    return (balls.sizes >= min_size) & (balls.sizes <= max_size) & (balls.average_angular_distance <= threshold)


def shuffled_assignments(assignment: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=assignment.device).manual_seed(seed)
    perm = torch.randperm(assignment.numel(), generator=gen, device=assignment.device)
    return assignment[perm]


def size_matched_random_assignments(assignment: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=assignment.device).manual_seed(seed)
    sizes = torch.bincount(assignment).tolist()
    perm = torch.randperm(assignment.numel(), generator=gen, device=assignment.device)
    out = torch.empty_like(assignment)
    cursor = 0
    for ball_id, size in enumerate(sizes):
        out[perm[cursor : cursor + size]] = ball_id
        cursor += size
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_balls.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add angular granular-ball generation"
```

---

### Task 6: GB Compactness Loss

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/compactness.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_compactness.py`

- [ ] **Step 1: Write failing compactness tests**

Create `gb_denoise_stage_b_compactness/tests/test_compactness.py`:

```python
import torch

from gb_stage_b.balls import generate_angular_balls, reliable_ball_mask
from gb_stage_b.compactness import compactness_loss


def test_compactness_loss_is_finite_and_backprops_to_z_only():
    z = torch.randn(24, 8, requires_grad=True)
    balls = generate_angular_balls(z.detach(), min_size=3, quality_threshold=0.4)
    reliable = reliable_ball_mask(balls, min_size=3, max_size=100, ad_percentile=100.0)
    loss, stats = compactness_loss(z, balls.assignment, balls.centers, reliable)
    assert torch.isfinite(loss)
    assert stats["covered_nodes"] > 0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_compactness_loss_zero_when_no_reliable_balls():
    z = torch.randn(10, 4, requires_grad=True)
    balls = generate_angular_balls(z.detach(), min_size=2, quality_threshold=0.4)
    reliable = torch.zeros(balls.num_balls, dtype=torch.bool)
    loss, stats = compactness_loss(z, balls.assignment, balls.centers, reliable)
    assert loss.item() == 0.0
    assert stats["covered_nodes"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_compactness.py -q
```

Expected: FAIL because `compactness.py` is missing.

- [ ] **Step 3: Implement compactness loss**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/compactness.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from gb_stage_b.balls import angular_distance


def compactness_loss(
    z: torch.Tensor,
    assignment: torch.Tensor,
    centers: torch.Tensor,
    reliable_balls: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    z = F.normalize(z, dim=-1, eps=1e-8)
    centers = F.normalize(centers.detach(), dim=-1, eps=1e-8)
    node_reliable = reliable_balls[assignment]
    if node_reliable.sum().item() == 0:
        return z.sum() * 0.0, {"covered_nodes": 0.0, "covered_ratio": 0.0, "mean_distance": 0.0}
    selected_z = z[node_reliable]
    selected_centers = centers[assignment[node_reliable]]
    dist = angular_distance(selected_z, selected_centers)
    loss = dist.mean()
    return loss, {
        "covered_nodes": float(node_reliable.sum().item()),
        "covered_ratio": float(node_reliable.float().mean().item()),
        "mean_distance": float(dist.detach().mean().item()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_compactness.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add gb compactness loss"
```

---

### Task 7: Diagnostics and Linear Evaluation

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/diagnostics.py`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/eval.py`
- Modify: `gb_denoise_stage_b_compactness/tests/test_diagnostics.py`

- [ ] **Step 1: Extend diagnostics tests**

Append to `gb_denoise_stage_b_compactness/tests/test_diagnostics.py`:

```python
import torch

from gb_stage_b.balls import generate_angular_balls
from gb_stage_b.diagnostics import ball_diagnostics, representation_diagnostics
from gb_stage_b.eval import linear_eval


def test_diagnostics_return_expected_keys():
    graph = load_graph("Synthetic", seed=0)
    balls = generate_angular_balls(graph.x, min_size=3, quality_threshold=0.4)
    ball_stats = ball_diagnostics(balls, labels=graph.y)
    rep_stats = representation_diagnostics(graph.x)
    assert "num_balls" in ball_stats
    assert "label_purity_diagnostic" in ball_stats
    assert "effective_rank" in rep_stats
    assert rep_stats["feature_std_mean"] >= 0


def test_linear_eval_runs_on_synthetic_graph():
    graph = load_graph("Synthetic", seed=0)
    acc = linear_eval(graph.x, graph.y, graph.train_mask, graph.val_mask, graph.test_mask, seed=0)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_diagnostics.py -q
```

Expected: FAIL because diagnostics and eval modules are missing.

- [ ] **Step 3: Implement diagnostics and evaluation**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/diagnostics.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from gb_stage_b.balls import BallBatch


def _diagnostic_purity(assignment: torch.Tensor, labels: torch.Tensor) -> float:
    purities = []
    for ball_id in assignment.unique().tolist():
        idx = assignment == int(ball_id)
        counts = torch.bincount(labels[idx])
        purities.append(float(counts.max().item() / idx.sum().item()))
    return float(sum(purities) / len(purities)) if purities else 0.0


def ball_diagnostics(balls: BallBatch, labels: torch.Tensor | None = None) -> dict[str, float]:
    out = {
        "num_balls": float(balls.num_balls),
        "mean_ball_size": float(balls.sizes.float().mean().item()),
        "max_ball_size": float(balls.sizes.max().item()),
        "singleton_ratio": float((balls.sizes == 1).float().mean().item()),
        "mean_radius": float(balls.radius.mean().item()),
        "mean_average_angular_distance": float(balls.average_angular_distance.mean().item()),
    }
    if labels is not None:
        out["label_purity_diagnostic"] = _diagnostic_purity(balls.assignment.cpu(), labels.cpu())
    return out


def representation_diagnostics(z: torch.Tensor) -> dict[str, float]:
    z = F.normalize(z.detach(), dim=-1, eps=1e-8)
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    probs = s / s.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    effective_rank = torch.exp(entropy)
    sim = z @ z.t()
    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
    return {
        "effective_rank": float(effective_rank.item()),
        "feature_std_mean": float(z.std(dim=0).mean().item()),
        "mean_pairwise_cosine": float(upper.mean().item()) if upper.numel() else 0.0,
        "embedding_uniformity": float(torch.pdist(z).pow(2).mul(-2).exp().mean().log().item()) if z.size(0) > 1 else 0.0,
    }
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/eval.py`:

```python
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def linear_eval(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> float:
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    train_idx = train_mask.detach().cpu().numpy().astype(bool)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)
    clf = LogisticRegression(max_iter=1000, random_state=seed, multi_class="auto")
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    return float(accuracy_score(y[test_idx], pred))
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add diagnostics and linear evaluation"
```

---

### Task 8: Stage B Training Loop and CLI

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/train.py`
- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/cli.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_smoke_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `gb_denoise_stage_b_compactness/tests/test_smoke_cli.py`:

```python
from pathlib import Path

from gb_stage_b.cli import main


def test_cli_smoke_run(tmp_path):
    config = tmp_path / "stage_b.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: B1
ssl_objective: negfree
seed: 0
epochs: 4
warmup_epochs: 2
hidden_dim: 16
proj_dim: 8
lambda_gb: 0.005
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )
    main(["run", "--config", str(config)])
    run_dirs = list((tmp_path / "results").glob("Synthetic/B1/seed_0/*"))
    assert run_dirs
    assert (run_dirs[0] / "metrics.json").exists()
    assert (run_dirs[0] / "history.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_smoke_cli.py -q
```

Expected: FAIL because `train.py` and `cli.py` are missing.

- [ ] **Step 3: Implement training loop and CLI**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/train.py`:

```python
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from gb_stage_b.augment import drop_edges, mask_features
from gb_stage_b.balls import generate_angular_balls, make_ball_batch, reliable_ball_mask, shuffled_assignments, size_matched_random_assignments
from gb_stage_b.compactness import compactness_loss
from gb_stage_b.config import StageBConfig
from gb_stage_b.data import load_graph
from gb_stage_b.diagnostics import ball_diagnostics, representation_diagnostics
from gb_stage_b.eval import linear_eval
from gb_stage_b.losses import infonce_loss, negative_free_loss
from gb_stage_b.models import StageBModel
from gb_stage_b.utils import ensure_dir, set_seed, write_json


def _ssl_loss(cfg: StageBConfig, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    if cfg.ssl_objective == "infonce":
        return infonce_loss(z1, z2)
    return negative_free_loss(p1, z2, p2, z1)


def _controlled_assignment(cfg: StageBConfig, assignment: torch.Tensor) -> torch.Tensor:
    if cfg.method == "B3" or cfg.assignment_control == "shuffled":
        return shuffled_assignments(assignment, seed=cfg.seed)
    if cfg.method == "B4" or cfg.assignment_control == "size_matched_random":
        return size_matched_random_assignments(assignment, seed=cfg.seed)
    return assignment


def _build_balls(cfg: StageBConfig, h: torch.Tensor):
    balls = generate_angular_balls(h.detach(), min_size=cfg.reliable_ball_min_size, quality_threshold=0.35)
    assignment = _controlled_assignment(cfg, balls.assignment)
    if not torch.equal(assignment, balls.assignment):
        balls = make_ball_batch(h.detach(), assignment)
    reliable = reliable_ball_mask(
        balls,
        min_size=cfg.reliable_ball_min_size,
        max_size=cfg.reliable_ball_max_size,
        ad_percentile=cfg.reliable_ball_ad_percentile,
    )
    return balls, reliable


def run_experiment(cfg: StageBConfig) -> Path:
    set_seed(cfg.seed)
    data = load_graph(cfg.dataset, seed=cfg.seed)
    model = StageBModel(data.x.size(1), cfg.hidden_dim, cfg.proj_dim)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    balls = None
    reliable = None
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        x1 = mask_features(data.x, cfg.feat_mask, seed=cfg.seed + epoch * 2)
        x2 = mask_features(data.x, cfg.feat_mask, seed=cfg.seed + epoch * 2 + 1)
        e1 = drop_edges(data.edge_index, cfg.edge_drop, seed=cfg.seed + epoch * 2)
        e2 = drop_edges(data.edge_index, cfg.edge_drop, seed=cfg.seed + epoch * 2 + 1)
        h1, z1, p1 = model(x1, e1)
        h2, z2, p2 = model(x2, e2)
        loss_ssl = _ssl_loss(cfg, p1, z1, p2, z2)
        loss_gb = loss_ssl.new_tensor(0.0)
        gb_stats = {"covered_nodes": 0.0, "covered_ratio": 0.0, "mean_distance": 0.0}
        should_use_gb = cfg.method in {"B1", "B2", "B3", "B4"} and epoch >= cfg.warmup_epochs
        should_refresh = balls is None or (cfg.method == "B2" and cfg.ball_refresh_interval > 0 and epoch % cfg.ball_refresh_interval == 0)
        if should_use_gb and should_refresh:
            with torch.no_grad():
                h_full, z_full, _ = model(data.x, data.edge_index)
                space = h_full if cfg.ball_space == "H" else z_full
                balls, reliable = _build_balls(cfg, F.normalize(space, dim=-1, eps=1e-8))
        if should_use_gb and balls is not None and reliable is not None:
            current = h1 if cfg.ball_space == "H" else z1
            loss_gb, gb_stats = compactness_loss(current, balls.assignment, balls.centers, reliable)
        loss = loss_ssl + cfg.lambda_gb * loss_gb
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({
            "epoch": epoch,
            "loss_ssl": float(loss_ssl.detach().item()),
            "loss_compact": float(loss_gb.detach().item()),
            "loss_compact_weighted": float((cfg.lambda_gb * loss_gb).detach().item()),
            **gb_stats,
        })
    model.eval()
    with torch.no_grad():
        h, z, _ = model(data.x, data.edge_index)
        final_space = F.normalize(h if cfg.ball_space == "H" else z, dim=-1, eps=1e-8)
        final_balls = generate_angular_balls(final_space, min_size=cfg.reliable_ball_min_size, quality_threshold=0.35)
    metrics = {}
    metrics.update(ball_diagnostics(final_balls, labels=data.y))
    metrics.update(representation_diagnostics(final_space))
    metrics["linear_eval_acc"] = linear_eval(final_space, data.y, data.train_mask, data.val_mask, data.test_mask, seed=cfg.seed)
    metrics["dataset"] = cfg.dataset
    metrics["method"] = cfg.method
    metrics["seed"] = cfg.seed
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(cfg.results_dir) / cfg.dataset / cfg.method / f"seed_{cfg.seed}" / run_id)
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", {"history": history})
    return out_dir
```

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/cli.py`:

```python
from __future__ import annotations

import argparse

from gb_stage_b.config import load_config
from gb_stage_b.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gb-stage-b")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--method")
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--warmup-epochs", type=int, dest="warmup_epochs")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(
            args.config,
            overrides={
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
            },
        )
        out = run_experiment(cfg)
        print(f"wrote run artifacts to {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_smoke_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit training and run-only CLI**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add stage b training loop"
```

---

### Task 9: Reporting and Paired Summary

**Files:**

- Create: `gb_denoise_stage_b_compactness/src/gb_stage_b/report.py`
- Create: `gb_denoise_stage_b_compactness/tests/test_report.py`
- Modify: `gb_denoise_stage_b_compactness/src/gb_stage_b/cli.py`
- Modify: `gb_denoise_stage_b_compactness/tests/test_smoke_cli.py`

- [ ] **Step 1: Write failing report tests**

Create `gb_denoise_stage_b_compactness/tests/test_report.py`:

```python
from pathlib import Path

from gb_stage_b.report import summarize_results
from gb_stage_b.utils import write_json


def test_summarize_results_writes_summary_and_paired_delta(tmp_path):
    b0 = tmp_path / "Cora" / "B0" / "seed_0" / "run"
    b1 = tmp_path / "Cora" / "B1" / "seed_0" / "run"
    write_json(b0 / "metrics.json", {"dataset": "Cora", "method": "B0", "seed": 0, "linear_eval_acc": 0.7, "mean_radius": 0.5})
    write_json(b1 / "metrics.json", {"dataset": "Cora", "method": "B1", "seed": 0, "linear_eval_acc": 0.8, "mean_radius": 0.4})
    out = summarize_results(tmp_path)
    assert out == tmp_path / "stage_b_summary.csv"
    assert out.exists()
    assert (tmp_path / "stage_b_paired_deltas.csv").exists()
```

Append to `gb_denoise_stage_b_compactness/tests/test_smoke_cli.py`:

```python

def test_cli_summarize_existing_results(tmp_path):
    from gb_stage_b.utils import write_json

    run_dir = tmp_path / "results" / "Synthetic" / "B0" / "seed_0" / "run"
    write_json(run_dir / "metrics.json", {"dataset": "Synthetic", "method": "B0", "seed": 0, "linear_eval_acc": 0.5})
    main(["summarize", "--results", str(tmp_path / "results")])
    assert (tmp_path / "results" / "stage_b_summary.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_report.py tests/test_smoke_cli.py -q
```

Expected: FAIL because `report.py` is missing.

- [ ] **Step 3: Implement report summarization**

Create `gb_denoise_stage_b_compactness/src/gb_stage_b/report.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gb_stage_b.utils import read_json


def _metric_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*/seed_*/*/metrics.json"))


def summarize_results(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    rows = []
    for path in _metric_files(results_dir):
        row = read_json(path)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    df = pd.DataFrame(rows)
    summary_path = results_dir / "stage_b_summary.csv"
    if df.empty:
        df.to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(results_dir / "stage_b_paired_deltas.csv", index=False)
        return summary_path
    df.to_csv(summary_path, index=False)
    deltas = []
    keys = ["linear_eval_acc", "mean_radius", "mean_average_angular_distance", "effective_rank"]
    for (dataset, seed), group in df.groupby(["dataset", "seed"]):
        base = group[group["method"] == "B0"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            if row["method"] == "B0":
                continue
            item = {"dataset": dataset, "seed": seed, "method": row["method"]}
            for key in keys:
                if key in row and key in base_row:
                    item[f"delta_{key}"] = row[key] - base_row[key]
            deltas.append(item)
    pd.DataFrame(deltas).to_csv(results_dir / "stage_b_paired_deltas.csv", index=False)
    return summary_path
```

Replace `gb_denoise_stage_b_compactness/src/gb_stage_b/cli.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from gb_stage_b.config import load_config
from gb_stage_b.report import summarize_results
from gb_stage_b.train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gb-stage-b")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--method")
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--warmup-epochs", type=int, dest="warmup_epochs")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--results", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(
            args.config,
            overrides={
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
            },
        )
        out = run_experiment(cfg)
        print(f"wrote run artifacts to {out}")
    elif args.command == "summarize":
        out = summarize_results(Path(args.results))
        print(f"wrote summary to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest tests/test_report.py tests/test_smoke_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "feat: add stage b reporting"
```

---

### Task 10: Full Test Run and Minimal Gate Commands

**Files:**

- Modify: `gb_denoise_stage_b_compactness/README.md`
- No code module changes unless tests reveal a defect.

- [ ] **Step 1: Run full test suite**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run B0 smoke experiment**

```powershell
cd gb_denoise_stage_b_compactness
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Synthetic --method B0 --seed 0 --epochs 4 --warmup-epochs 2
```

Expected: command prints `wrote run artifacts to ...`; a `results/Synthetic/B0/seed_0/<timestamp>/metrics.json` file exists.

- [ ] **Step 3: Run B1 smoke experiment**

```powershell
cd gb_denoise_stage_b_compactness
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Synthetic --method B1 --seed 0 --epochs 4 --warmup-epochs 2
```

Expected: command prints `wrote run artifacts to ...`; a `results/Synthetic/B1/seed_0/<timestamp>/metrics.json` file exists.

- [ ] **Step 4: Summarize smoke results**

```powershell
cd gb_denoise_stage_b_compactness
python -m gb_stage_b.cli summarize --results results
```

Expected: `results/stage_b_summary.csv` and `results/stage_b_paired_deltas.csv` exist.

- [ ] **Step 5: Update README with real commands**

Append this section to `gb_denoise_stage_b_compactness/README.md`:

````markdown
## Minimal Gate

The first real Stage B gate should run paired B0/B1/B3/B4 experiments with the same dataset and seeds:

```powershell
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B0 --seed 0
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B1 --seed 0
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B3 --seed 0
python -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B4 --seed 0
python -m gb_stage_b.cli summarize --results results
```

Pass condition: B1 must improve ball diagnostics over B0 and beat B3/B4 without collapse diagnostics worsening.
````

- [ ] **Step 6: Run full tests again**

```powershell
cd gb_denoise_stage_b_compactness
python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```powershell
git add gb_denoise_stage_b_compactness
git commit -m "docs: document stage b minimal gate"
```

---

## Self-Review Checklist

Before handing off implementation:

- [ ] The plan implements every spec section: config, ball generation, compactness loss, B0/B1/B3/B4 controls, diagnostics, reporting, and smoke commands.
- [ ] No task uses label purity or majority labels in training.
- [ ] The only label use is in `linear_eval` and `label_purity_diagnostic`.
- [ ] Each task has a failing test before implementation.
- [ ] Each task has a concrete command and expected outcome.
- [ ] Each task has a commit command.
- [ ] No existing markdown history files or Stage A project files are modified.
