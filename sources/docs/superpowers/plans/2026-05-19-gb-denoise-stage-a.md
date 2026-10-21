# GB Denoise Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Stage A experiment that compares GRACE / InfoNCE and negative-free graph SSL representations by their downstream evaluation and granular-ball generation quality.

**Architecture:** Create a self-contained experiment folder, `gb_denoise_stage_a_negfree_vs_infonce`, with focused Python modules for config loading, data loading, augmentations, encoders, SSL objectives, granular-ball generation, diagnostics, training, and reporting. Keep labels out of SSL training and ball generation; labels are used only for linear evaluation and post-hoc diagnostics.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, PyYAML, pytest.

---

## File Structure

Create this new folder and files:

```text
gb_denoise_stage_a_negfree_vs_infonce/
  README.md
  pyproject.toml
  configs/
    stage_a.yaml
  src/
    gb_stage_a/
      __init__.py
      augment.py
      balls.py
      cli.py
      config.py
      data.py
      diagnostics.py
      eval.py
      losses.py
      models.py
      report.py
      train.py
      utils.py
  scripts/
    run_smoke.ps1
    run_stage_a.ps1
  tests/
    test_augment.py
    test_balls.py
    test_config.py
    test_diagnostics.py
    test_losses.py
    test_report.py
  results/
    .gitkeep
```

Responsibilities:

- `config.py`: typed experiment config and YAML loading.
- `data.py`: Planetoid data loading without changing split.
- `augment.py`: edge drop and feature mask.
- `models.py`: GCN encoder, projector, predictor, and model wrappers.
- `losses.py`: GRACE InfoNCE and SimSiam-style negative-free losses.
- `balls.py`: label-free angle-space granular-ball generation.
- `diagnostics.py`: embedding and ball metrics.
- `eval.py`: logistic-regression linear evaluation.
- `train.py`: run one method/dataset/seed and save metrics.
- `report.py`: aggregate run metrics into CSV and Markdown summary.
- `cli.py`: command-line entry point.

All paths below are relative to repository root unless stated otherwise.

---

### Task 1: Scaffold Package And Project Metadata

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/README.md`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/pyproject.toml`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/__init__.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/results/.gitkeep`

- [ ] **Step 1: Create package directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path `
  'gb_denoise_stage_a_negfree_vs_infonce\src\gb_stage_a', `
  'gb_denoise_stage_a_negfree_vs_infonce\configs', `
  'gb_denoise_stage_a_negfree_vs_infonce\scripts', `
  'gb_denoise_stage_a_negfree_vs_infonce\tests', `
  'gb_denoise_stage_a_negfree_vs_infonce\results' | Out-Null
New-Item -ItemType File -Force -Path 'gb_denoise_stage_a_negfree_vs_infonce\results\.gitkeep' | Out-Null
```

Expected: directories exist under `gb_denoise_stage_a_negfree_vs_infonce`.

- [ ] **Step 2: Write `pyproject.toml`**

Create `gb_denoise_stage_a_negfree_vs_infonce/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gb-stage-a"
version = "0.1.0"
description = "Stage A diagnostics for negative-free versus InfoNCE representations and granular-ball quality."
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

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: Write package marker**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/__init__.py`:

```python
"""Stage A experiments for granular-ball diagnostics."""

__all__: list[str] = []
```

- [ ] **Step 4: Write README**

Create `gb_denoise_stage_a_negfree_vs_infonce/README.md`:

```markdown
# GB Denoise Stage A

This experiment compares InfoNCE / GRACE and negative-free graph SSL representations by granular-ball quality.

Stage A does not train with granular-ball loss. It trains SSL encoders, extracts embeddings, generates label-free granular balls, and reports post-hoc diagnostics.

Labels are used only for linear evaluation and diagnostic purity. Labels are never used for SSL training or granular-ball generation.

## Quick Smoke Run

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m pip install -e .[dev]
pytest -q
python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method raw_features --seed 0 --epochs 0
python -m gb_stage_a.cli summarize --results results
```
```

- [ ] **Step 5: Run metadata check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m pip install -e .[dev]
pytest -q
```

Expected: install succeeds; pytest reports no tests collected or all tests passing, depending on current progress.

- [ ] **Step 6: Commit scaffold**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce
git commit -m "chore: scaffold GB denoise stage A experiment"
```

---

### Task 2: Add Config Loading

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/configs/stage_a.yaml`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_config.py`:

```python
from pathlib import Path

import pytest

from gb_stage_a.config import load_config


def test_load_default_config_has_expected_methods():
    cfg = load_config(Path("configs/stage_a.yaml"))
    assert cfg.datasets == ["Cora", "CiteSeer", "PubMed"]
    assert cfg.methods == [
        "raw_features",
        "grace_infonce_h",
        "grace_infonce_z",
        "negfree_h",
        "negfree_z",
    ]
    assert cfg.training.epochs > 0
    assert cfg.balls.min_ball_size >= 2


def test_config_rejects_unknown_method(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
datasets: [Cora]
methods: [bad_method]
seeds: [0]
training:
  epochs: 1
  hidden_dim: 16
  out_dim: 8
  proj_dim: 8
  lr: 0.001
  weight_decay: 0.0
  drop_edge_rate_1: 0.2
  drop_edge_rate_2: 0.2
  drop_feature_rate_1: 0.2
  drop_feature_rate_2: 0.2
  eval_epochs: 10
balls:
  min_ball_size: 4
  max_depth: 8
  min_split_gain: 0.001
paths:
  data_root: data
  results_dir: results
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown method"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_config.py -q
```

Expected: FAIL because `gb_stage_a.config` does not exist.

- [ ] **Step 3: Write default YAML config**

Create `gb_denoise_stage_a_negfree_vs_infonce/configs/stage_a.yaml`:

```yaml
datasets:
  - Cora
  - CiteSeer
  - PubMed

methods:
  - raw_features
  - grace_infonce_h
  - grace_infonce_z
  - negfree_h
  - negfree_z

seeds:
  - 0
  - 1
  - 2
  - 3
  - 4

training:
  epochs: 200
  hidden_dim: 256
  out_dim: 256
  proj_dim: 256
  lr: 0.001
  weight_decay: 0.00001
  drop_edge_rate_1: 0.2
  drop_edge_rate_2: 0.4
  drop_feature_rate_1: 0.3
  drop_feature_rate_2: 0.4
  eval_epochs: 1000

balls:
  min_ball_size: 8
  max_depth: 12
  min_split_gain: 0.001

paths:
  data_root: data
  results_dir: results
```

- [ ] **Step 4: Implement config loader**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KNOWN_METHODS = {
    "raw_features",
    "grace_infonce_h",
    "grace_infonce_z",
    "negfree_h",
    "negfree_z",
}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    hidden_dim: int
    out_dim: int
    proj_dim: int
    lr: float
    weight_decay: float
    drop_edge_rate_1: float
    drop_edge_rate_2: float
    drop_feature_rate_1: float
    drop_feature_rate_2: float
    eval_epochs: int


@dataclass(frozen=True)
class BallConfig:
    min_ball_size: int
    max_depth: int
    min_split_gain: float


@dataclass(frozen=True)
class PathConfig:
    data_root: Path
    results_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: list[str]
    methods: list[str]
    seeds: list[int]
    training: TrainingConfig
    balls: BallConfig
    paths: PathConfig


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    return data


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = _require_mapping(raw, "config")

    datasets = list(data["datasets"])
    methods = list(data["methods"])
    seeds = [int(seed) for seed in data["seeds"]]

    unknown = sorted(set(methods) - KNOWN_METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")

    training = TrainingConfig(**_require_mapping(data["training"], "training"))
    balls = BallConfig(**_require_mapping(data["balls"], "balls"))
    paths_raw = _require_mapping(data["paths"], "paths")
    paths = PathConfig(
        data_root=Path(paths_raw["data_root"]),
        results_dir=Path(paths_raw["results_dir"]),
    )

    if training.epochs < 0:
        raise ValueError("training.epochs must be non-negative")
    if training.eval_epochs <= 0:
        raise ValueError("training.eval_epochs must be positive")
    if balls.min_ball_size < 2:
        raise ValueError("balls.min_ball_size must be at least 2")
    if balls.max_depth < 1:
        raise ValueError("balls.max_depth must be positive")

    return ExperimentConfig(
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        training=training,
        balls=balls,
        paths=paths,
    )
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit config**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/configs/stage_a.yaml gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_config.py
git commit -m "feat: add stage A config loader"
```

---

### Task 3: Add Reproducibility Utilities

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/utils.py`

- [ ] **Step 1: Write utility module**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/utils.py`:

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
```

- [ ] **Step 2: Run import check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -c "from gb_stage_a.utils import set_seed, get_device; set_seed(0); print(get_device())"
```

Expected: prints `cpu` or `cuda`.

- [ ] **Step 3: Commit utilities**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/utils.py
git commit -m "feat: add reproducibility utilities"
```

---

### Task 4: Add Data Loading

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/data.py`

- [ ] **Step 1: Write Planetoid data loader**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/data.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_planetoid(name: str, root: Path):
    if name not in {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError(f"Unsupported dataset: {name}")
    dataset = Planetoid(root=str(root), name=name, transform=NormalizeFeatures())
    data = dataset[0]
    if not hasattr(data, "train_mask") or data.train_mask is None:
        raise ValueError(f"{name} has no train_mask")
    return dataset, data


def mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).view(-1)
```

- [ ] **Step 2: Run Cora load smoke test**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -c "from pathlib import Path; from gb_stage_a.data import load_planetoid; ds, data = load_planetoid('Cora', Path('data')); print(ds.num_features, ds.num_classes, data.num_nodes)"
```

Expected: downloads Cora if missing and prints feature count, class count, and node count.

- [ ] **Step 3: Commit data loader**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/data.py
git commit -m "feat: add Planetoid data loader"
```

---

### Task 5: Add Graph Augmentations

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/augment.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_augment.py`

- [ ] **Step 1: Write failing augmentation tests**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_augment.py`:

```python
import torch

from gb_stage_a.augment import drop_edges, mask_features


def test_mask_features_zero_rate_keeps_input():
    x = torch.ones(4, 3)
    out = mask_features(x, 0.0)
    assert torch.equal(out, x)


def test_mask_features_full_rate_zeros_columns():
    x = torch.ones(4, 3)
    out = mask_features(x, 1.0)
    assert torch.count_nonzero(out) == 0


def test_drop_edges_zero_rate_keeps_edges():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    out = drop_edges(edge_index, 0.0)
    assert torch.equal(out, edge_index)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_augment.py -q
```

Expected: FAIL because `gb_stage_a.augment` does not exist.

- [ ] **Step 3: Implement augmentations**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/augment.py`:

```python
from __future__ import annotations

import torch


def drop_edges(edge_index: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError("drop_rate must be in [0, 1]")
    if drop_rate == 0.0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) >= drop_rate
    if keep_mask.sum() == 0:
        keep_mask[torch.randint(0, num_edges, (1,), device=edge_index.device)] = True
    return edge_index[:, keep_mask]


def mask_features(x: torch.Tensor, mask_rate: float) -> torch.Tensor:
    if not 0.0 <= mask_rate <= 1.0:
        raise ValueError("mask_rate must be in [0, 1]")
    if mask_rate == 0.0:
        return x
    out = x.clone()
    feature_mask = torch.rand(x.size(1), device=x.device) < mask_rate
    out[:, feature_mask] = 0.0
    return out


def make_two_views(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    drop_edge_rate_1: float,
    drop_edge_rate_2: float,
    drop_feature_rate_1: float,
    drop_feature_rate_2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = mask_features(x, drop_feature_rate_1)
    x2 = mask_features(x, drop_feature_rate_2)
    edge_index1 = drop_edges(edge_index, drop_edge_rate_1)
    edge_index2 = drop_edges(edge_index, drop_edge_rate_2)
    return x1, edge_index1, x2, edge_index2
```

- [ ] **Step 4: Run augmentation tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_augment.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit augmentations**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/augment.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_augment.py
git commit -m "feat: add graph augmentations"
```

---

### Task 6: Add Models

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/models.py`

- [ ] **Step 1: Implement encoder and wrappers**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/models.py`:

```python
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.prelu(h)
        return self.conv2(h, edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRACEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim)
        self.projector = MLP(out_dim, proj_dim, proj_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x, edge_index)
        z = self.project(h)
        return h, z


class NegativeFreeModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, proj_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim)
        self.projector = MLP(out_dim, proj_dim, proj_dim)
        self.predictor = MLP(proj_dim, proj_dim, proj_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x, edge_index)
        z = self.project(h)
        p = self.predict(z)
        return h, z, p
```

- [ ] **Step 2: Run model smoke check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -c "import torch; from gb_stage_a.models import GRACEModel; m=GRACEModel(8,16,16,16); x=torch.randn(5,8); edge=torch.tensor([[0,1,2,3],[1,2,3,4]]); h,z=m(x,edge); print(h.shape, z.shape)"
```

Expected: prints `torch.Size([5, 16]) torch.Size([5, 16])`.

- [ ] **Step 3: Commit models**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/models.py
git commit -m "feat: add SSL model modules"
```

---

### Task 7: Add SSL Losses

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/losses.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_losses.py`

- [ ] **Step 1: Write failing loss tests**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_losses.py`:

```python
import torch

from gb_stage_a.losses import grace_infonce_loss, simsiam_loss


def test_grace_loss_is_scalar():
    z1 = torch.randn(6, 4)
    z2 = torch.randn(6, 4)
    loss = grace_infonce_loss(z1, z2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_simsiam_loss_is_scalar():
    p1 = torch.randn(6, 4)
    p2 = torch.randn(6, 4)
    z1 = torch.randn(6, 4)
    z2 = torch.randn(6, 4)
    loss = simsiam_loss(p1, p2, z1, z2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_losses.py -q
```

Expected: FAIL because `gb_stage_a.losses` does not exist.

- [ ] **Step 3: Implement losses**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def grace_infonce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    z1 = _normalized(z1)
    z2 = _normalized(z2)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.mm(representations, representations.t()) / tau
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels])
    mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))
    return F.cross_entropy(similarity, labels)


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = _normalized(p)
    z = _normalized(z.detach())
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * negative_cosine_similarity(p1, z2) + 0.5 * negative_cosine_similarity(p2, z1)
```

- [ ] **Step 4: Run loss tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_losses.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit losses**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/losses.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_losses.py
git commit -m "feat: add InfoNCE and negative-free losses"
```

---

### Task 8: Add Granular-Ball Generator

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/balls.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_balls.py`

- [ ] **Step 1: Write failing ball tests**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_balls.py`:

```python
import torch

from gb_stage_a.balls import generate_balls


def test_generate_balls_covers_all_points():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    balls = generate_balls(x, min_ball_size=2, max_depth=4, min_split_gain=0.0)
    covered = sorted(idx for ball in balls for idx in ball.indices)
    assert covered == [0, 1, 2, 3]


def test_generate_balls_uses_no_labels():
    x = torch.randn(8, 4)
    balls = generate_balls(x, min_ball_size=2, max_depth=3, min_split_gain=0.0)
    assert all(hasattr(ball, "center") for ball in balls)
    assert all(ball.size >= 1 for ball in balls)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_balls.py -q
```

Expected: FAIL because `gb_stage_a.balls` does not exist.

- [ ] **Step 3: Implement ball generator**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/balls.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Ball:
    indices: list[int]
    center: torch.Tensor
    radius: float
    average_angular_distance: float
    depth: int

    @property
    def size(self) -> int:
        return len(self.indices)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def _angular_distance(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    cosine = torch.clamp(torch.matmul(x, center), -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(cosine)


def _make_ball(x: torch.Tensor, indices: list[int], depth: int) -> Ball:
    subset = x[indices]
    center = F.normalize(subset.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
    distances = _angular_distance(subset, center)
    return Ball(
        indices=indices,
        center=center.detach().cpu(),
        radius=float(distances.max().detach().cpu()),
        average_angular_distance=float(distances.mean().detach().cpu()),
        depth=depth,
    )


def _split_indices(x: torch.Tensor, indices: list[int]) -> tuple[list[int], list[int]]:
    subset = x[indices]
    sim = torch.mm(subset, subset.t())
    dist = 1.0 - sim
    flat_idx = int(torch.argmax(dist).item())
    n = subset.size(0)
    i = flat_idx // n
    j = flat_idx % n
    pivot_a = subset[i]
    pivot_b = subset[j]
    dist_a = _angular_distance(subset, pivot_a)
    dist_b = _angular_distance(subset, pivot_b)
    left = []
    right = []
    for local_idx, global_idx in enumerate(indices):
        if dist_a[local_idx] <= dist_b[local_idx]:
            left.append(global_idx)
        else:
            right.append(global_idx)
    if not left or not right:
        midpoint = len(indices) // 2
        return indices[:midpoint], indices[midpoint:]
    return left, right


def generate_balls(
    embeddings: torch.Tensor,
    min_ball_size: int,
    max_depth: int,
    min_split_gain: float,
) -> list[Ball]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if min_ball_size < 2:
        raise ValueError("min_ball_size must be at least 2")
    x = _normalize(embeddings.detach().cpu().float())
    final: list[Ball] = []
    queue: list[tuple[list[int], int]] = [(list(range(x.size(0))), 0)]

    while queue:
        indices, depth = queue.pop(0)
        parent = _make_ball(x, indices, depth)
        if len(indices) < 2 * min_ball_size or depth >= max_depth:
            final.append(parent)
            continue

        left, right = _split_indices(x, indices)
        if len(left) < min_ball_size or len(right) < min_ball_size:
            final.append(parent)
            continue

        left_ball = _make_ball(x, left, depth + 1)
        right_ball = _make_ball(x, right, depth + 1)
        weighted_child_ad = (
            left_ball.average_angular_distance * left_ball.size
            + right_ball.average_angular_distance * right_ball.size
        ) / parent.size
        gain = parent.average_angular_distance - weighted_child_ad

        if gain >= min_split_gain:
            queue.append((left, depth + 1))
            queue.append((right, depth + 1))
        else:
            final.append(parent)

    return final
```

- [ ] **Step 4: Run ball tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_balls.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit ball generator**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/balls.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_balls.py
git commit -m "feat: add label-free granular-ball generator"
```

---

### Task 9: Add Diagnostics

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/diagnostics.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics tests**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_diagnostics.py`:

```python
import torch

from gb_stage_a.balls import generate_balls
from gb_stage_a.diagnostics import ball_metrics, embedding_metrics


def test_embedding_metrics_have_expected_keys():
    metrics = embedding_metrics(torch.randn(12, 5))
    assert "embedding_std" in metrics
    assert "effective_rank" in metrics
    assert "embedding_uniformity" in metrics


def test_ball_metrics_have_expected_keys():
    x = torch.randn(12, 5)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    balls = generate_balls(x, min_ball_size=2, max_depth=3, min_split_gain=0.0)
    metrics = ball_metrics(balls, y)
    assert metrics["num_balls"] >= 1
    assert "label_purity_diagnostic" in metrics
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_diagnostics.py -q
```

Expected: FAIL because `gb_stage_a.diagnostics` does not exist.

- [ ] **Step 3: Implement diagnostics**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/diagnostics.py`:

```python
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from gb_stage_a.balls import Ball


def embedding_metrics(embeddings: torch.Tensor) -> dict[str, float]:
    x = embeddings.detach().cpu().float()
    x_norm = F.normalize(x, p=2, dim=1)
    std = float(x.std(dim=0).mean().item())
    sim = torch.mm(x_norm, x_norm.t())
    n = x.size(0)
    off_diag = sim[~torch.eye(n, dtype=torch.bool)]
    uniformity = float(torch.log(torch.exp(2.0 * off_diag).mean()).item())
    singular_values = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
    probs = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    effective_rank = float(torch.exp(entropy).item())
    return {
        "embedding_std": std,
        "embedding_uniformity": uniformity,
        "effective_rank": effective_rank,
    }


def _purity(labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return math.nan
    values, counts = labels.unique(return_counts=True)
    del values
    return float(counts.max().item() / labels.numel())


def ball_metrics(balls: list[Ball], labels: torch.Tensor | None) -> dict[str, float]:
    sizes = np.array([ball.size for ball in balls], dtype=float)
    radii = np.array([ball.radius for ball in balls], dtype=float)
    angular = np.array([ball.average_angular_distance for ball in balls], dtype=float)

    metrics = {
        "num_balls": float(len(balls)),
        "mean_ball_size": float(sizes.mean()) if len(sizes) else math.nan,
        "max_ball_size": float(sizes.max()) if len(sizes) else math.nan,
        "singleton_ratio": float((sizes == 1).mean()) if len(sizes) else math.nan,
        "mean_radius": float(radii.mean()) if len(radii) else math.nan,
        "mean_average_angular_distance": float(angular.mean()) if len(angular) else math.nan,
    }

    if labels is None:
        metrics["label_purity_diagnostic"] = math.nan
    else:
        purities = []
        weights = []
        cpu_labels = labels.detach().cpu()
        for ball in balls:
            purities.append(_purity(cpu_labels[ball.indices]))
            weights.append(ball.size)
        metrics["label_purity_diagnostic"] = float(np.average(purities, weights=weights))

    return metrics


def assignment_vector(balls: list[Ball], num_nodes: int) -> np.ndarray:
    assignment = np.full(num_nodes, -1, dtype=int)
    for ball_id, ball in enumerate(balls):
        for idx in ball.indices:
            assignment[idx] = ball_id
    if (assignment < 0).any():
        raise ValueError("Some nodes were not assigned to a ball")
    return assignment


def pairwise_same_ball_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Assignments must have the same shape")
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    upper = np.triu(np.ones_like(same_a, dtype=bool), k=1)
    set_a = same_a[upper]
    set_b = same_b[upper]
    union = np.logical_or(set_a, set_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(set_a, set_b).sum() / union)
```

- [ ] **Step 4: Run diagnostics tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit diagnostics**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/diagnostics.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_diagnostics.py
git commit -m "feat: add embedding and ball diagnostics"
```

---

### Task 10: Add Linear Evaluation

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/eval.py`

- [ ] **Step 1: Implement linear evaluation**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/eval.py`:

```python
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def linear_evaluation(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    train_idx = train_mask.detach().cpu().numpy().astype(bool)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)

    clf = LogisticRegression(
        max_iter=3000,
        random_state=seed,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    target = y[test_idx]
    return {
        "linear_eval_accuracy": float(accuracy_score(target, pred)),
        "linear_eval_f1_micro": float(f1_score(target, pred, average="micro")),
        "linear_eval_f1_macro": float(f1_score(target, pred, average="macro")),
    }
```

- [ ] **Step 2: Run import check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -c "from gb_stage_a.eval import linear_evaluation; print(linear_evaluation.__name__)"
```

Expected: prints `linear_evaluation`.

- [ ] **Step 3: Commit evaluation module**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/eval.py
git commit -m "feat: add linear evaluation"
```

---

### Task 11: Add Training And One-Run Execution

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/train.py`

- [ ] **Step 1: Implement training module**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/train.py`:

```python
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

from gb_stage_a.augment import make_two_views
from gb_stage_a.balls import generate_balls
from gb_stage_a.config import ExperimentConfig
from gb_stage_a.data import load_planetoid
from gb_stage_a.diagnostics import ball_metrics, embedding_metrics
from gb_stage_a.eval import linear_evaluation
from gb_stage_a.losses import grace_infonce_loss, simsiam_loss
from gb_stage_a.models import GRACEModel, NegativeFreeModel
from gb_stage_a.utils import ensure_dir, get_device, set_seed, write_json


def _train_grace(model: GRACEModel, data, cfg: ExperimentConfig, device: torch.device) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    model.train()
    for _ in trange(cfg.training.epochs, desc="train_grace", leave=False):
        x1, edge1, x2, edge2 = make_two_views(
            data.x,
            data.edge_index,
            cfg.training.drop_edge_rate_1,
            cfg.training.drop_edge_rate_2,
            cfg.training.drop_feature_rate_1,
            cfg.training.drop_feature_rate_2,
        )
        _, z1 = model(x1.to(device), edge1.to(device))
        _, z2 = model(x2.to(device), edge2.to(device))
        loss = grace_infonce_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _train_negfree(model: NegativeFreeModel, data, cfg: ExperimentConfig, device: torch.device) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    model.train()
    for _ in trange(cfg.training.epochs, desc="train_negfree", leave=False):
        x1, edge1, x2, edge2 = make_two_views(
            data.x,
            data.edge_index,
            cfg.training.drop_edge_rate_1,
            cfg.training.drop_edge_rate_2,
            cfg.training.drop_feature_rate_1,
            cfg.training.drop_feature_rate_2,
        )
        _, z1, p1 = model(x1.to(device), edge1.to(device))
        _, z2, p2 = model(x2.to(device), edge2.to(device))
        loss = simsiam_loss(p1, p2, z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _extract_embedding(method: str, model, data, device: torch.device) -> torch.Tensor:
    if method == "raw_features":
        return data.x.detach().cpu()
    model.eval()
    if method == "grace_infonce_h":
        h, _ = model(data.x.to(device), data.edge_index.to(device))
        return h.detach().cpu()
    if method == "grace_infonce_z":
        _, z = model(data.x.to(device), data.edge_index.to(device))
        return z.detach().cpu()
    if method == "negfree_h":
        h, _, _ = model(data.x.to(device), data.edge_index.to(device))
        return h.detach().cpu()
    if method == "negfree_z":
        _, z, _ = model(data.x.to(device), data.edge_index.to(device))
        return z.detach().cpu()
    raise ValueError(f"Unsupported method: {method}")


def run_one(
    cfg: ExperimentConfig,
    dataset_name: str,
    method: str,
    seed: int,
    config_path: Path,
) -> dict[str, float | int | str]:
    set_seed(seed)
    device = get_device()
    dataset, data = load_planetoid(dataset_name, cfg.paths.data_root)
    data = data.to(device)

    model = None
    if method.startswith("grace_infonce"):
        model = GRACEModel(
            dataset.num_features,
            cfg.training.hidden_dim,
            cfg.training.out_dim,
            cfg.training.proj_dim,
        ).to(device)
        _train_grace(model, data, cfg, device)
    elif method.startswith("negfree"):
        model = NegativeFreeModel(
            dataset.num_features,
            cfg.training.hidden_dim,
            cfg.training.out_dim,
            cfg.training.proj_dim,
        ).to(device)
        _train_negfree(model, data, cfg, device)
    elif method != "raw_features":
        raise ValueError(f"Unsupported method: {method}")

    embedding = _extract_embedding(method, model, data, device)
    labels = data.y.detach().cpu()
    balls = generate_balls(
        embedding,
        min_ball_size=cfg.balls.min_ball_size,
        max_depth=cfg.balls.max_depth,
        min_split_gain=cfg.balls.min_split_gain,
    )
    metrics = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "config_path": str(config_path),
        **embedding_metrics(embedding),
        **ball_metrics(balls, labels),
        **linear_evaluation(
            embedding,
            labels,
            data.train_mask.detach().cpu(),
            data.test_mask.detach().cpu(),
            seed,
        ),
    }

    out_dir = ensure_dir(cfg.paths.results_dir / "runs" / dataset_name / method / f"seed_{seed}")
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    write_json(out_dir / "config_snapshot.json", {"config": asdict(cfg)})
    torch.save(embedding, out_dir / "embedding.pt")
    return metrics
```

- [ ] **Step 2: Run raw feature smoke execution**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -c "from pathlib import Path; from gb_stage_a.config import load_config; from gb_stage_a.train import run_one; cfg=load_config(Path('configs/stage_a.yaml')); print(run_one(cfg, 'Cora', 'raw_features', 0, Path('configs/stage_a.yaml')))"
```

Expected: creates `results/runs/Cora/raw_features/seed_0/metrics.csv` and prints a metrics dictionary.

- [ ] **Step 3: Commit training module**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/train.py
git commit -m "feat: add one-run Stage A training pipeline"
```

---

### Task 12: Add CLI And Run Scripts

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/scripts/run_smoke.ps1`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/scripts/run_stage_a.ps1`

- [ ] **Step 1: Implement CLI**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from gb_stage_a.config import load_config
from gb_stage_a.train import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description="GB Stage A experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--method", required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--epochs", type=int)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--results", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config(args.config)
        if args.epochs is not None:
            object.__setattr__(cfg.training, "epochs", args.epochs)
        metrics = run_one(cfg, args.dataset, args.method, args.seed, args.config)
        print(metrics)
    elif args.command == "summarize":
        from gb_stage_a.report import summarize_results

        summarize_results(args.results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add smoke script**

Create `gb_denoise_stage_a_negfree_vs_infonce/scripts/run_smoke.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method raw_features --seed 0 --epochs 0
python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method grace_infonce_h --seed 0 --epochs 1
python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method negfree_h --seed 0 --epochs 1
python -m gb_stage_a.cli summarize --results results
```

- [ ] **Step 3: Add full run script**

Create `gb_denoise_stage_a_negfree_vs_infonce/scripts/run_stage_a.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$datasets = @("Cora", "CiteSeer", "PubMed")
$methods = @("raw_features", "grace_infonce_h", "grace_infonce_z", "negfree_h", "negfree_z")
$seeds = @(0, 1, 2, 3, 4)

foreach ($dataset in $datasets) {
  foreach ($method in $methods) {
    foreach ($seed in $seeds) {
      python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset $dataset --method $method --seed $seed
    }
  }
}

python -m gb_stage_a.cli summarize --results results
```

- [ ] **Step 4: Run CLI import check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m gb_stage_a.cli --help
```

Expected: prints CLI help with `run` and `summarize`.

- [ ] **Step 5: Commit CLI**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py gb_denoise_stage_a_negfree_vs_infonce/scripts/run_smoke.ps1 gb_denoise_stage_a_negfree_vs_infonce/scripts/run_stage_a.ps1
git commit -m "feat: add Stage A CLI and scripts"
```

---

### Task 13: Add Result Aggregation

**Files:**
- Create: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/report.py`
- Create: `gb_denoise_stage_a_negfree_vs_infonce/tests/test_report.py`

- [ ] **Step 1: Write failing report test**

Create `gb_denoise_stage_a_negfree_vs_infonce/tests/test_report.py`:

```python
from pathlib import Path

import pandas as pd

from gb_stage_a.report import summarize_results


def test_summarize_results_writes_outputs(tmp_path):
    run_dir = tmp_path / "runs" / "Cora" / "raw_features" / "seed_0"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "Cora",
                "method": "raw_features",
                "seed": 0,
                "mean_radius": 0.5,
                "linear_eval_accuracy": 0.7,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    summarize_results(tmp_path)
    assert (tmp_path / "stage_a_metrics.csv").exists()
    assert (tmp_path / "stage_a_summary.md").exists()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_report.py -q
```

Expected: FAIL because `gb_stage_a.report` does not exist.

- [ ] **Step 3: Implement report module**

Create `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/report.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _collect_metrics(results_dir: Path) -> pd.DataFrame:
    files = sorted((results_dir / "runs").glob("*/*/seed_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv files found under {results_dir / 'runs'}")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _format_summary(df: pd.DataFrame) -> str:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"dataset", "method", "config_path"} and pd.api.types.is_numeric_dtype(df[col])
    ]
    grouped = df.groupby(["dataset", "method"], as_index=False)[numeric_cols].agg(["mean", "std"])
    lines = [
        "# Stage A Summary",
        "",
        "This report aggregates per-run metrics. Labels are diagnostic only and are not used for SSL training or granular-ball generation.",
        "",
    ]
    flat = grouped.reset_index()
    lines.append(flat.to_markdown(index=False))
    lines.append("")

    if {"dataset", "method", "mean_radius", "linear_eval_accuracy"}.issubset(df.columns):
        lines.extend(
            [
                "## Main Comparison Reminder",
                "",
                "Primary comparison is `negfree_h` vs `grace_infonce_h`. Smaller radius alone is not success unless collapse diagnostics and linear evaluation remain healthy.",
                "",
            ]
        )
    return "\n".join(lines)


def summarize_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    df = _collect_metrics(results_dir)
    df.to_csv(results_dir / "stage_a_metrics.csv", index=False)
    (results_dir / "stage_a_summary.md").write_text(_format_summary(df), encoding="utf-8")
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest tests/test_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit reporting**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/report.py gb_denoise_stage_a_negfree_vs_infonce/tests/test_report.py
git commit -m "feat: add Stage A result aggregation"
```

---

### Task 14: Fix CLI Config Override

**Files:**
- Modify: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py`
- Modify: `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py`

- [ ] **Step 1: Add config helper for epoch override**

Modify `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py` by adding this function at the end:

```python
def with_epochs(cfg: ExperimentConfig, epochs: int) -> ExperimentConfig:
    training = TrainingConfig(
        epochs=epochs,
        hidden_dim=cfg.training.hidden_dim,
        out_dim=cfg.training.out_dim,
        proj_dim=cfg.training.proj_dim,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        drop_edge_rate_1=cfg.training.drop_edge_rate_1,
        drop_edge_rate_2=cfg.training.drop_edge_rate_2,
        drop_feature_rate_1=cfg.training.drop_feature_rate_1,
        drop_feature_rate_2=cfg.training.drop_feature_rate_2,
        eval_epochs=cfg.training.eval_epochs,
    )
    return ExperimentConfig(
        datasets=cfg.datasets,
        methods=cfg.methods,
        seeds=cfg.seeds,
        training=training,
        balls=cfg.balls,
        paths=cfg.paths,
    )
```

- [ ] **Step 2: Update CLI to use helper**

Modify imports in `gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py`:

```python
from gb_stage_a.config import load_config, with_epochs
```

Replace this block:

```python
        if args.epochs is not None:
            object.__setattr__(cfg.training, "epochs", args.epochs)
```

with:

```python
        if args.epochs is not None:
            cfg = with_epochs(cfg, args.epochs)
```

- [ ] **Step 3: Run CLI smoke help**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method raw_features --seed 0 --epochs 0
```

Expected: run completes and writes raw feature metrics.

- [ ] **Step 4: Commit CLI fix**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/config.py gb_denoise_stage_a_negfree_vs_infonce/src/gb_stage_a/cli.py
git commit -m "fix: support CLI epoch override"
```

---

### Task 15: Run Tests And Smoke Pipeline

**Files:**
- Modify only if smoke exposes a bug in files already created.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run smoke script**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1
```

Expected:

- `results/runs/Cora/raw_features/seed_0/metrics.csv`
- `results/runs/Cora/grace_infonce_h/seed_0/metrics.csv`
- `results/runs/Cora/negfree_h/seed_0/metrics.csv`
- `results/stage_a_metrics.csv`
- `results/stage_a_summary.md`

- [ ] **Step 3: Inspect summary**

Run:

```powershell
Get-Content -LiteralPath 'results\stage_a_summary.md' | Select-Object -First 40
```

Expected: Markdown table includes `Cora`, `raw_features`, `grace_infonce_h`, and `negfree_h`.

- [ ] **Step 4: Commit smoke fixes or results policy**

If only code changed:

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce
git commit -m "test: verify Stage A smoke pipeline"
```

If generated result files are large, do not commit generated `results/runs`. Commit only code and `.gitkeep`.

---

### Task 16: Update README With Final Commands

**Files:**
- Modify: `gb_denoise_stage_a_negfree_vs_infonce/README.md`

- [ ] **Step 1: Replace README with final usage**

Replace `gb_denoise_stage_a_negfree_vs_infonce/README.md` with:

```markdown
# GB Denoise Stage A

This experiment tests whether negative-sample-free graph SSL representations are more suitable than GRACE / InfoNCE representations for label-free granular-ball generation.

Stage A does not train with granular-ball loss. It only trains SSL encoders, extracts embeddings, generates granular balls, and reports diagnostics.

Labels are used only for linear evaluation and post-hoc purity diagnostics. Labels are never used for SSL training or granular-ball generation.

## Setup

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m pip install -e .[dev]
```

If PyTorch Geometric installation fails, install the PyTorch/PyG wheels matching the local CUDA or CPU environment, then rerun the editable install.

## Unit Tests

```powershell
pytest -q
```

## Smoke Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1
```

## Full Stage A Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stage_a.ps1
```

## Outputs

- `results/runs/<dataset>/<method>/seed_<seed>/metrics.csv`
- `results/stage_a_metrics.csv`
- `results/stage_a_summary.md`

## Main Comparison

The primary comparison is `negfree_h` versus `grace_infonce_h`.

`grace_infonce_z` and `negfree_z` are diagnostic views of projector-space behavior and should not replace the main encoder-space comparison.

## Claim Boundary

Smaller ball radius is not sufficient evidence of success. A successful result must also avoid representation collapse and preserve downstream linear evaluation.
```

- [ ] **Step 2: Run README command check**

Run:

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Commit README**

```powershell
git add gb_denoise_stage_a_negfree_vs_infonce/README.md
git commit -m "docs: document Stage A experiment workflow"
```

---

## Self-Review Checklist

Spec coverage:

- Stage A only: covered by folder scope, tasks, and README.
- InfoNCE vs negative-free: covered by `GRACEModel`, `NegativeFreeModel`, losses, methods, and run scripts.
- Same data/split/encoder/augmentation/seed: covered by config and shared training pipeline.
- Same ball algorithm: covered by `balls.py`.
- Diagnostics: covered by `diagnostics.py`, `eval.py`, `report.py`.
- No labels in SSL or ball generation: covered by `train.py` flow and README claim boundary.
- Stage B deferred: no GB loss task exists.

Plan hygiene scan:

- No task contains unresolved placeholders.
- Every created file has concrete content.
- Every test has exact command and expected result.

Type consistency:

- `ExperimentConfig`, `TrainingConfig`, `BallConfig`, and `PathConfig` are defined before use.
- Method names match config, CLI, and training code.
- Ball metrics consume `Ball` objects returned by `generate_balls`.
