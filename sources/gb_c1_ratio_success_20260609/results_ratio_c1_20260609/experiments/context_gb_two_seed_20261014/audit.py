"""Independent, read-only audit for the Context-GB experiment outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_ARMS = ["grace", "pingda", "pingda_context", "pingda_hreg"]
DEFAULT_SPACES = [
    "grace_h", "pingda_h", "pingda_mean_r", "context_h", "context_r", "hreg_h"
]
DEFAULT_NOISE = {"clean": 0.0, "uniform": 0.4, "pair": 0.4}
HEADS = ["mlp", "gcn"]
STUDENT_EPOCHS_DEFAULT = 200


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*arrays: Any) -> str:
    """Match the experiment's shape-plus-contiguous-bytes SHA-256 fingerprint."""
    digest = hashlib.sha256()
    for value in arrays:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def split_indices(n_nodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the project's existing ordered 20/10/70 split exactly."""
    rng = np.random.RandomState(int(seed))
    remaining = np.arange(int(n_nodes))
    train = rng.choice(remaining, int(n_nodes * 0.2), replace=False)
    remaining = np.setdiff1d(remaining, train)
    validation = rng.choice(remaining, int(n_nodes * 0.1), replace=False)
    remaining = np.setdiff1d(remaining, validation)
    test = rng.choice(remaining, int(n_nodes * 0.7), replace=False)
    return train, validation, test


def noisy_labels(clean: torch.Tensor, train_index: np.ndarray, noise: str,
                 rate: float, seed: int) -> torch.Tensor:
    """Reproduce training-only uniform/pair noise without touching held-out labels."""
    labels = clean.detach().cpu().long().clone()
    if rate <= 0:
        return labels
    classes = int(labels.max().item()) + 1
    if noise == "uniform":
        transition = np.float64(rate) / np.float64(classes - 1) * np.ones(
            (classes, classes), dtype=np.float64
        )
        np.fill_diagonal(transition, 1.0 - np.float64(rate))
        diagonal = np.arange(classes)
        transition[diagonal, diagonal] = (
            transition[diagonal, diagonal] + 1.0 - transition.sum(axis=0)
        )
    elif noise == "pair":
        transition = (1.0 - rate) * np.eye(classes)
        for class_index in range(classes):
            transition[class_index, class_index - 1] = rate
    else:
        raise ValueError(f"unsupported noise type: {noise}")

    rng = np.random.RandomState(int(seed))
    sampled: list[int] = []
    for label in labels[torch.as_tensor(train_index, dtype=torch.long)]:
        draw = rng.multinomial(1, transition[int(label)], 1)[0]
        sampled.append(int(np.flatnonzero(draw)[0]))
    labels[torch.as_tensor(train_index, dtype=torch.long)] = torch.tensor(
        sampled, dtype=torch.long
    )
    return labels


def load_graph(path: str | Path) -> tuple[torch.Tensor, int]:
    loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    raw = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
    labels = raw["y"] if isinstance(raw, dict) else raw.y
    edge_index = raw["edge_index"] if isinstance(raw, dict) else raw.edge_index
    labels = torch.as_tensor(labels).detach().cpu().long().view(-1)
    edges = torch.as_tensor(edge_index).detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"invalid edge_index shape: {tuple(edges.shape)}")
    return labels, int(edges.shape[1])


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("logits and labels have incompatible shapes")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def expected_student_cells(datasets: list[str], seeds: list[int],
                           spaces: list[str], budgets: list[int],
                           noises: list[str]) -> set[tuple[Any, ...]]:
    cells: set[tuple[Any, ...]] = set()
    for dataset in datasets:
        for seed in seeds:
            for noise in noises:
                for head in HEADS:
                    cells.add((dataset, int(seed), noise, head, "CE", 0))
                    for space in spaces:
                        for budget in budgets:
                            cells.add((dataset, int(seed), noise, head, space, int(budget)))
    return cells


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                     encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _same_number(first: Any, second: Any, *, atol: float = 1e-10) -> bool:
    if first == "":
        first = None
    if second == "":
        second = None
    if first is None or second is None:
        return first is None and second is None
    try:
        a, b = float(first), float(second)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol


class Auditor:
    def __init__(self, output: Path, partial: bool):
        self.output = output
        self.partial = partial
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checked: dict[str, int] = {
            "source_files": 0,
            "protocols": 0,
            "pretrains": 0,
            "students": 0,
            "logits": 0,
            "target_hashes": 0,
            "noise_hashes": 0,
        }
        self.rows: list[dict[str, Any]] = []
        self.split_notes: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}
        self.datasets: list[str] = []
        self.seeds: list[int] = []
        self.arms: list[str] = []
        self.spaces: list[str] = []
        self.budgets: list[int] = []
        self.noise_rates: dict[str, float] = {}
        self.graph_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def _load_config(self) -> bool:
        path = self.output / "config.json"
        if not path.is_file():
            self.error(f"missing config: {path}")
            return False
        try:
            self.config = _read_json(path)
            self.datasets = [str(value) for value in self.config["datasets"]]
            self.seeds = [int(value) for value in self.config["seeds"]]
            self.arms = [str(value) for value in self.config.get("arms", DEFAULT_ARMS)]
            self.spaces = [str(value) for value in self.config.get("spaces", DEFAULT_SPACES)]
            self.budgets = [int(value) for value in self.config.get("budgets", [64, 128, 256])]
            self.noise_rates = {str(key): float(value) for key, value in
                                self.config.get("noise", DEFAULT_NOISE).items()}
        except (KeyError, TypeError, ValueError) as exc:
            self.error(f"invalid config: {exc}")
            return False
        if set(self.noise_rates) != set(DEFAULT_NOISE):
            self.error(f"noise conditions differ from the registered three: {sorted(self.noise_rates)}")
        if self.arms != DEFAULT_ARMS:
            self.error(f"pretraining arms differ from the registered four: {self.arms}")
        if self.spaces != DEFAULT_SPACES:
            self.error(f"student spaces differ from the registered six: {self.spaces}")
        if self.budgets != [64, 128, 256]:
            self.error(f"registered GB budgets differ: {self.budgets}")
        scope = str(self.config.get("scope", "")).lower()
        if "knn" not in scope or "kmeans" not in scope:
            self.error("config.scope does not explicitly record the KNN/KMeans comparator exclusion")
        return True

    def _verify_manifest(self) -> None:
        path = self.output / "source_manifest.json"
        if not path.is_file():
            self.error(f"missing source manifest: {path}")
            return
        try:
            manifest = _read_json(path)
        except Exception as exc:  # noqa: BLE001 - report malformed artifacts, do not hide them
            self.error(f"cannot read source manifest: {exc}")
            return
        if not manifest:
            self.error("source manifest is empty")
            return
        for source, expected in manifest.items():
            source_path = Path(source)
            if not source_path.is_file():
                self.error(f"manifest source is missing: {source}")
                continue
            try:
                actual = file_sha256(source_path)
            except OSError as exc:
                self.error(f"cannot hash manifest source {source}: {exc}")
                continue
            if actual != expected:
                self.error(f"manifest hash mismatch: {source}")
                continue
            self.checked["source_files"] += 1

    def _graph_info(self, dataset: str, seed: int) -> dict[str, Any] | None:
        key = (dataset, int(seed))
        if key in self.graph_cache:
            return self.graph_cache[key]
        spec = self.config.get("data", {}).get(dataset)
        if not isinstance(spec, dict) or not spec.get("path"):
            self.error(f"config has no data path for {dataset}")
            return None
        path = Path(spec["path"])
        if not path.is_file():
            self.error(f"dataset file is missing for {dataset}: {path}")
            return None
        try:
            data_sha = file_sha256(path)
            if data_sha != spec.get("sha256"):
                self.error(f"dataset hash mismatch for {dataset}: {path}")
                return None
            clean, edge_count = load_graph(path)
            train, validation, test = split_indices(len(clean), seed)
            all_indices = np.concatenate([train, validation, test])
            if any(len(np.unique(part)) != len(part) for part in (train, validation, test)):
                self.error(f"duplicate split indices for {dataset}/seed{seed}")
            if len(np.unique(all_indices)) != len(all_indices):
                self.error(f"overlapping split indices for {dataset}/seed{seed}")
            if len(all_indices) and (int(all_indices.min()) < 0 or int(all_indices.max()) >= len(clean)):
                self.error(f"out-of-range split index for {dataset}/seed{seed}")
            covered = np.zeros(len(clean), dtype=bool)
            covered[all_indices] = True
            unassigned = int((~covered).sum())
            self.split_notes.append({
                "dataset": dataset,
                "seed": int(seed),
                "train_nodes": int(len(train)),
                "validation_nodes": int(len(validation)),
                "test_nodes": int(len(test)),
                "unassigned_nodes": unassigned,
            })
            if unassigned:
                self.warnings.append(
                    f"{dataset}/seed{seed}: existing 20/10/70 split leaves {unassigned} nodes unassigned"
                )
            protocol_path = self.output / "protocol" / dataset / f"seed{seed}.json"
            protocol = _read_json(protocol_path) if protocol_path.is_file() else None
            split_hash = fingerprint(train, validation, test)
            if protocol is not None:
                self.checked["protocols"] += 1
                expected_protocol = {
                    "split_hash": split_hash,
                    "nodes": len(clean),
                    "classes": int(clean.max()) + 1,
                    "edges": edge_count,
                }
                for field, expected in expected_protocol.items():
                    if protocol.get(field) != expected:
                        self.error(
                            f"protocol mismatch {dataset}/seed{seed} field {field}: "
                            f"{protocol.get(field)!r} != {expected!r}"
                        )
            elif not self.partial:
                self.error(f"missing protocol file: {protocol_path}")
            info = {
                "path": path,
                "clean": clean,
                "edges": edge_count,
                "train": train,
                "validation": validation,
                "test": test,
                "split_hash": split_hash,
                "noisy": {},
                "noise_hash": {},
            }
            self.graph_cache[key] = info
            return info
        except Exception as exc:  # noqa: BLE001
            self.error(f"cannot inspect {dataset}/seed{seed}: {type(exc).__name__}: {exc}")
            return None

    def _noise_for(self, info: dict[str, Any], dataset: str, seed: int,
                   noise_name: str) -> tuple[torch.Tensor, str] | None:
        if noise_name in info["noisy"]:
            return info["noisy"][noise_name], info["noise_hash"][noise_name]
        if noise_name not in self.noise_rates:
            self.error(f"unknown noise condition: {noise_name}")
            return None
        rate = self.noise_rates[noise_name]
        kind = "uniform" if noise_name == "clean" else noise_name
        try:
            labels = noisy_labels(info["clean"], info["train"], kind, rate, seed)
            noise_hash = fingerprint(labels[torch.as_tensor(info["train"], dtype=torch.long)])
            info["noisy"][noise_name] = labels
            info["noise_hash"][noise_name] = noise_hash
            return labels, noise_hash
        except Exception as exc:  # noqa: BLE001
            self.error(f"cannot recompute noise for {dataset}/seed{seed}/{noise_name}: {exc}")
            return None

    def _verify_pretrains(self) -> set[tuple[str, int, str]]:
        found: set[tuple[str, int, str]] = set()
        base = self.output / "pretrain"
        for path in base.glob("*/seed*/*/complete.json") if base.exists() else []:
            try:
                dataset = path.parents[2].name
                seed_name = path.parents[1].name
                arm = path.parent.name
                seed = int(seed_name.removeprefix("seed"))
                key = (dataset, seed, arm)
                found.add(key)
                if dataset not in self.datasets or seed not in self.seeds or arm not in self.arms:
                    self.error(f"unexpected pretraining cell: {key}")
                    continue
                complete = _read_json(path)
                config_path = path.parent / "config.json"
                loss_path = path.parent / "loss.csv"
                if not config_path.is_file() or not loss_path.is_file():
                    self.error(f"missing pretrain config/loss: {path.parent}")
                else:
                    pretrain_config = _read_json(config_path)
                    for field, expected in (("dataset", dataset), ("seed", seed), ("arm", arm)):
                        if pretrain_config.get(field) != expected:
                            self.error(f"pretrain config mismatch {key} field {field}")
                    expected_epochs = int(self.config.get("pretrain_epochs", pretrain_config["epochs"]))
                    with loss_path.open(encoding="utf-8-sig", newline="") as stream:
                        losses = list(csv.DictReader(stream))
                    if (pretrain_config.get("epochs") != expected_epochs or
                            [int(row["epoch"]) for row in losses] != list(range(1, expected_epochs + 1))):
                        self.error(f"pretrain epoch coverage mismatch: {key}")
                    for row in losses:
                        if any(not math.isfinite(float(value)) for value in row.values()):
                            self.error(f"non-finite pretrain loss/diagnostic: {key}")
                            break
                        if arm in ("pingda_context", "pingda_hreg") and int(row["valid_roots"]) < 2:
                            self.error(f"insufficient context roots: {key}")
                            break
                representation_path = path.parent / "representation.pt"
                checkpoint_path = path.parent / "checkpoint.pt"
                if not representation_path.is_file() or not checkpoint_path.is_file():
                    self.error(f"incomplete pretrain artifact: {path.parent}")
                    continue
                if complete.get("sha256") != file_sha256(representation_path):
                    self.error(f"representation hash mismatch: {representation_path}")
                    continue
                info = self._graph_info(dataset, seed)
                if info is not None:
                    representation = torch.load(representation_path, map_location="cpu", weights_only=True)
                    h = representation.get("h") if isinstance(representation, dict) else None
                    c = representation.get("c") if isinstance(representation, dict) else None
                    if not torch.is_tensor(h) or h.shape[0] != len(info["clean"]):
                        self.error(f"invalid H representation shape: {representation_path}")
                    if not torch.is_tensor(c) or c.shape[0] != len(info["clean"]):
                        self.error(f"invalid context representation shape: {representation_path}")
                    if (torch.is_tensor(h) and not torch.isfinite(h).all()) or (torch.is_tensor(c) and not torch.isfinite(c).all()):
                        self.error(f"non-finite saved representation: {representation_path}")
                    if isinstance(representation, dict) and "structural_valid" in representation:
                        valid = representation["structural_valid"]
                        if not torch.is_tensor(valid) or valid.shape != (len(info["clean"]),):
                            self.error(f"invalid structural_valid shape: {representation_path}")
                self.checked["pretrains"] += 1
            except Exception as exc:  # noqa: BLE001
                self.error(f"cannot audit pretrain {path}: {type(exc).__name__}: {exc}")
        expected = {
            (dataset, seed, arm)
            for dataset in self.datasets for seed in self.seeds for arm in self.arms
        }
        missing = expected - found
        if missing and not self.partial:
            self.error(f"missing {len(missing)} pretrain completion records; first: {sorted(missing)[:5]}")
        return found

    def _target_hash(self, dataset: str, seed: int, noise: str, space: str,
                     budget: int, info: dict[str, Any], noisy: torch.Tensor) -> str | None:
        train_tensor = torch.as_tensor(info["train"], dtype=torch.long)
        if space == "CE":
            target = F.one_hot(noisy, num_classes=int(info["clean"].max()) + 1).float()
        else:
            target_path = (self.output / "targets" / dataset / f"seed{seed}" / noise /
                           space / f"B{budget}" / "targets.pt")
            if not target_path.is_file():
                self.error(f"missing target artifact: {target_path}")
                return None
            try:
                artifact = torch.load(target_path, map_location="cpu", weights_only=True)
                target = artifact.get("targets") if isinstance(artifact, dict) else None
                alpha = artifact.get("alpha") if isinstance(artifact, dict) else None
                assignment = artifact.get("assignment") if isinstance(artifact, dict) else None
                if not torch.is_tensor(target) or target.ndim != 2 or target.shape[0] != len(info["clean"]):
                    self.error(f"invalid target tensor: {target_path}")
                    return None
                if not torch.is_tensor(alpha) or alpha.shape != (len(info["clean"]),):
                    self.error(f"invalid target alpha: {target_path}")
                    return None
                if not torch.is_tensor(assignment) or assignment.shape != (len(info["clean"]),):
                    self.error(f"invalid target assignment: {target_path}")
                    return None
                if (not torch.isfinite(target).all() or not torch.isfinite(alpha).all() or
                        (target < -1e-6).any() or (target > 1.0 + 1e-6).any() or
                        (alpha < -1e-6).any() or (alpha > 1.0 + 1e-6).any()):
                    self.error(f"target artifact contains invalid probabilities/weights: {target_path}")
                    return None
                if not torch.allclose(target.sum(dim=1), torch.ones(len(target)), atol=1e-5, rtol=0):
                    self.error(f"target rows do not sum to one: {target_path}")
                    return None
                if space in ("pingda_mean_r", "context_r"):
                    self._verify_soft_target_fallback(
                        dataset, seed, noise, space, budget, target, alpha, assignment
                    )
            except Exception as exc:  # noqa: BLE001
                self.error(f"cannot load target artifact {target_path}: {exc}")
                return None
        return fingerprint(target[train_tensor])

    def _verify_soft_target_fallback(self, dataset: str, seed: int, noise: str,
                                     space: str, budget: int, target: torch.Tensor,
                                     alpha: torch.Tensor, assignment: torch.Tensor) -> None:
        """Check that invalid context rows keep the matching full-H targets."""
        base_space = "context_h" if space == "context_r" else "pingda_h"
        arm = "pingda_context" if space == "context_r" else "pingda"
        representation_path = self.output / "pretrain" / dataset / f"seed{seed}" / arm / "representation.pt"
        base_path = (self.output / "targets" / dataset / f"seed{seed}" / noise /
                     base_space / f"B{budget}" / "targets.pt")
        if not representation_path.is_file() or not base_path.is_file():
            self.error(f"missing context fallback reference artifacts for {dataset}/seed{seed}/{space}/B{budget}")
            return
        try:
            representation = torch.load(representation_path, map_location="cpu", weights_only=True)
            base_artifact = torch.load(base_path, map_location="cpu", weights_only=True)
            h, c = representation.get("h"), representation.get("c")
            structural_valid = representation.get("structural_valid")
            base_target = base_artifact.get("targets")
            base_alpha = base_artifact.get("alpha")
            base_assignment = base_artifact.get("assignment")
            n = len(target)
            if (not torch.is_tensor(h) or not torch.is_tensor(c) or
                    not torch.is_tensor(structural_valid) or
                    h.shape[0] != n or c.shape[0] != n or structural_valid.shape != (n,)):
                self.error(f"invalid representation needed to check fallback: {representation_path}")
                return
            valid = structural_valid.bool() & (h.norm(dim=1) > 1e-8) & (c.norm(dim=1) > 1e-8)
            invalid = ~valid
            if not bool(invalid.any()):
                return
            if (not torch.is_tensor(base_target) or not torch.is_tensor(base_alpha) or
                    not torch.is_tensor(base_assignment) or base_target.shape != target.shape or
                    base_alpha.shape != alpha.shape or base_assignment.shape != assignment.shape):
                self.error(f"invalid base-H fallback artifact: {base_path}")
                return
            if not torch.equal(target[invalid], base_target[invalid]):
                self.error(f"soft-target fallback differs from {base_space} for invalid nodes: {space}/B{budget}")
            if not torch.equal(alpha[invalid], base_alpha[invalid]):
                self.error(f"alpha fallback differs from {base_space} for invalid nodes: {space}/B{budget}")
            if bool(valid.any()):
                offset = int(assignment[valid].max().item()) + 1
                expected_assignment = base_assignment[invalid] + offset
                if not torch.equal(assignment[invalid], expected_assignment):
                    self.error(f"assignment fallback differs from {base_space} for invalid nodes: {space}/B{budget}")
        except Exception as exc:  # noqa: BLE001
            self.error(f"cannot verify soft-target fallback for {space}/B{budget}: {exc}")

    def _student_files(self) -> dict[tuple[Any, ...], Path]:
        files: dict[tuple[Any, ...], Path] = {}
        base = self.output / "students"
        for path in base.glob("*/seed*/*/*/*/result.json") if base.exists() else []:
            try:
                row = _read_json(path)
                key = (
                    str(row["dataset"]), int(row["seed"]), str(row["noise"]),
                    str(row["head"]), str(row["space"]), int(row["budget"]),
                )
                if key in files:
                    self.error(f"duplicate student result key: {key}")
                files[key] = path
                if "knn" in key[4].lower() or "kmeans" in key[4].lower():
                    self.error(f"prohibited KNN/KMeans comparator row: {key}")
            except Exception as exc:  # noqa: BLE001
                self.error(f"cannot parse student result {path}: {type(exc).__name__}: {exc}")
        return files

    def _audit_student(self, key: tuple[Any, ...], result_path: Path) -> None:
        dataset, seed, noise, head, space, budget = key
        if (dataset not in self.datasets or seed not in self.seeds or
                noise not in self.noise_rates or head not in HEADS or
                space not in ["CE", *self.spaces] or
                (space == "CE" and budget != 0) or
                (space != "CE" and budget not in self.budgets)):
            self.error(f"student result is outside the registered grid: {key}")
            return
        try:
            row = _read_json(result_path)
            folder = result_path.parent
            config_path = folder / "config.json"
            logits_path = folder / "logits.pt"
            if not config_path.is_file() or not logits_path.is_file():
                self.error(f"student lacks config/logits: {folder}")
                return
            config = _read_json(config_path)
            for field, expected in zip(
                ["dataset", "seed", "noise", "head", "space", "budget"], key
            ):
                if config.get(field) != expected:
                    self.error(f"student config mismatch {key} field {field}")
            info = self._graph_info(dataset, seed)
            if info is None:
                return
            noise_info = self._noise_for(info, dataset, seed, noise)
            if noise_info is None:
                return
            noisy, noise_hash = noise_info
            if row.get("split_hash") != info["split_hash"] or config.get("split_hash") != info["split_hash"]:
                self.error(f"split hash mismatch for {key}")
            if row.get("noise_hash") != noise_hash or config.get("noise_hash") != noise_hash:
                self.error(f"noise hash mismatch for {key}")
            self.checked["noise_hashes"] += 1
            target_hash = self._target_hash(dataset, seed, noise, space, budget, info, noisy)
            if target_hash is not None:
                if row.get("target_hash") != target_hash or config.get("target_hash") != target_hash:
                    self.error(f"target hash mismatch for {key}")
                else:
                    self.checked["target_hashes"] += 1
            logits = torch.load(logits_path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(logits) or logits.shape != (len(info["clean"]), int(info["clean"].max()) + 1):
                self.error(f"invalid logits shape for {key}")
                return
            train = torch.as_tensor(info["train"], dtype=torch.long)
            validation = torch.as_tensor(info["validation"], dtype=torch.long)
            test = torch.as_tensor(info["test"], dtype=torch.long)
            train_acc = accuracy(logits[train], noisy[train])
            val_acc = accuracy(logits[validation], info["clean"][validation])
            test_acc = accuracy(logits[test], info["clean"][test])
            for field, actual in (("train_accuracy", train_acc),
                                  ("validation_accuracy", val_acc),
                                  ("test_accuracy", test_acc)):
                # GCN validation accuracy is measured on-device during selection;
                # the saved logits are recomputed on CPU, so allow float32 rounding.
                if not _same_number(row.get(field), actual, atol=1e-6):
                    self.error(f"saved {field} disagrees with logits for {key}")
            epochs = int(config.get("training", {}).get("epochs", STUDENT_EPOCHS_DEFAULT))
            try:
                best_epoch = int(row["best_epoch"])
            except (KeyError, TypeError, ValueError):
                best_epoch = -1
            if not 1 <= best_epoch <= epochs:
                self.error(f"best_epoch outside 1..{epochs} for {key}: {best_epoch}")
            self.rows.append({
                "dataset": dataset, "seed": seed, "noise": noise, "head": head,
                "space": space, "budget": budget,
                "test_accuracy": test_acc,
                "test_accuracy_pct": 100.0 * test_acc,
                "validation_accuracy": val_acc,
                "train_accuracy": train_acc,
                "best_epoch": best_epoch,
                "split_hash": info["split_hash"],
                "noise_hash": noise_hash,
                "target_hash": target_hash,
            })
            self.checked["students"] += 1
            self.checked["logits"] += 1
        except Exception as exc:  # noqa: BLE001
            self.error(f"cannot audit student {key}: {type(exc).__name__}: {exc}")

    def _summaries(self) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[float]] = {}
        for row in self.rows:
            key = (row["dataset"], row["noise"], row["head"], row["space"], row["budget"])
            groups.setdefault(key, []).append(float(row["test_accuracy_pct"]))
        summary: list[dict[str, Any]] = []
        for key, values in sorted(groups.items()):
            sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append({
                "dataset": key[0], "noise": key[1], "head": key[2],
                "space": key[3], "budget": key[4], "seed_count": len(values),
                "mean_accuracy_pct": float(np.mean(values)),
                "sample_std_pp": sample_std,
            })
        return summary

    def _compare_runner_summary(self, audited: list[dict[str, Any]], complete: bool) -> None:
        path = self.output / "summary.csv"
        if not path.is_file():
            if not self.partial:
                self.error("missing runner summary.csv")
            return
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                runner_rows = list(csv.DictReader(stream))
            by_key = {
                (r["dataset"], r["noise"], r["head"], r["space"], int(r["budget"])): r
                for r in runner_rows
            }
            audited_by_key = {
                (r["dataset"], r["noise"], r["head"], r["space"], int(r["budget"])): r
                for r in audited
            }
            if complete and set(by_key) != set(audited_by_key):
                self.error("runner summary coverage differs from independently audited groups")
            for key in set(by_key) & set(audited_by_key):
                runner, expected = by_key[key], audited_by_key[key]
                if int(runner.get("seed_count", -1)) != expected["seed_count"]:
                    if self.partial:
                        self.warnings.append(f"live runner summary seed_count differs from audit snapshot: {key}")
                        continue
                    self.error(f"runner summary seed_count mismatch: {key}")
                if not _same_number(runner.get("mean_accuracy_pct"), expected["mean_accuracy_pct"]):
                    self.error(f"runner summary mean mismatch: {key}")
                if not _same_number(runner.get("sample_std_pp"), expected["sample_std_pp"]):
                    self.error(f"runner summary sample std mismatch: {key}")
        except Exception as exc:  # noqa: BLE001
            self.error(f"cannot compare runner summary: {type(exc).__name__}: {exc}")

    def run(self) -> dict[str, Any]:
        if not self._load_config():
            return self._finish(False)
        self._verify_manifest()
        status_path = self.output / "status.json"
        status = _read_json(status_path) if status_path.is_file() else None
        if status is None and not self.partial:
            self.error("missing status.json")
        if status is not None and not self.partial and status.get("status") != "completed":
            self.error(f"run status is not completed: {status.get('status')!r}")

        present_pretrains = self._verify_pretrains()
        result_files = self._student_files()
        if status is not None:
            status_pretrains = status.get("completed_pretrains")
            status_students = status.get("completed_students")
            if status_pretrains is not None and int(status_pretrains) != len(present_pretrains):
                message = (f"status completed_pretrains={status_pretrains} but found "
                           f"{len(present_pretrains)} completion records")
                (self.error if not self.partial else self.warnings.append)(message)
            if status_students is not None and int(status_students) != len(result_files):
                message = (f"status completed_students={status_students} but found "
                           f"{len(result_files)} result files")
                (self.error if not self.partial else self.warnings.append)(message)
        expected_students = expected_student_cells(
            self.datasets, self.seeds, self.spaces, self.budgets, list(self.noise_rates)
        )
        unexpected = set(result_files) - expected_students
        if unexpected:
            self.error(f"found {len(unexpected)} student rows outside registered grid; first: {sorted(unexpected)[:5]}")
        missing_students = expected_students - set(result_files)
        if missing_students and not self.partial:
            self.error(f"missing {len(missing_students)} student results; first: {sorted(missing_students)[:5]}")
        for key, result_path in sorted(result_files.items()):
            self._audit_student(key, result_path)

        expected_pretrains = {
            (dataset, seed, arm)
            for dataset in self.datasets for seed in self.seeds for arm in self.arms
        }
        complete = not missing_students and not (expected_pretrains - present_pretrains)
        if complete and status is not None and status.get("status") != "completed":
            complete = False
        audited_summary = self._summaries()
        self._compare_runner_summary(audited_summary, complete)
        return self._finish(complete, status, expected_students, expected_pretrains,
                            result_files, present_pretrains, audited_summary)

    def _finish(self, complete: bool, status: dict[str, Any] | None = None,
                expected_students: set[tuple[Any, ...]] | None = None,
                expected_pretrains: set[tuple[str, int, str]] | None = None,
                result_files: dict[tuple[Any, ...], Path] | None = None,
                present_pretrains: set[tuple[str, int, str]] | None = None,
                audited_summary: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        expected_students = expected_students or set()
        expected_pretrains = expected_pretrains or set()
        result_files = result_files or {}
        present_pretrains = present_pretrains or set()
        audited_summary = audited_summary or self._summaries()
        report = {
            "audit": "context_gb_two_seed",
            "output": str(self.output),
            "mode": "partial" if self.partial else "complete",
            "coverage_complete": bool(complete),
            "run_status": status.get("status") if status else None,
            "expected_students": len(expected_students),
            "present_student_results": len(result_files),
            "audited_students": self.checked["students"],
            "missing_students": len(expected_students - set(result_files)),
            "unexpected_students": len(set(result_files) - expected_students),
            "expected_pretrains": len(expected_pretrains),
            "present_pretrains": len(present_pretrains),
            "checked": self.checked,
            "split_notes": self.split_notes,
            "mean_std_groups": len(audited_summary),
            "knn_kmeans_comparators": "excluded",
            "errors": self.errors,
            "warnings": self.warnings,
        }
        audit_dir = self.output / "audit"
        _write_json(audit_dir / "report.json", report)
        _write_csv(audit_dir / "verified_students.csv", self.rows)
        _write_csv(audit_dir / "audited_summary.csv", audited_summary)
        _write_csv(audit_dir / "split_notes.csv", self.split_notes)
        return report


def audit_output(output: str | Path, partial: bool = False) -> dict[str, Any]:
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    return Auditor(path.resolve(), partial).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs", help="run directory name or path (default: runs)")
    parser.add_argument("--partial", action="store_true",
                        help="audit existing completed artifacts without requiring full grid coverage")
    args = parser.parse_args(argv)
    report = audit_output(args.output, args.partial)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
