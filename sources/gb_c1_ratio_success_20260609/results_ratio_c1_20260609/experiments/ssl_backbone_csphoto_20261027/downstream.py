"""Frozen-R2 alignment and paired CE/SUPPORT32 student fits for each SSL H."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from artifacts import ROOT, WORKSPACE, CODE, PROTOCOL, read, save, sha
from data_bridge import class_count, import_file, load_features, load_supervised
from data_bridge import seed_all, state_hash

CONFIG = dict(PROTOCOL, r2=PROTOCOL['downstream'], support_tree=PROTOCOL['downstream']['tree'],
              arms=['grace','gca_degree','progcl_weight_degree','graphecl','grace200'])
_CORE = import_file("representation_supervision_core", WORKSPACE / "experiments" / "representation_supervision_20261021" / "core.py")
_GENERATOR = import_file("neighbor_preserving_ball_generator", WORKSPACE / "experiments" / "neighbor_preserving_ball_20261022" / "generator.py")
_SUPPORT = import_file("neighbor_support_ablation_kernels", WORKSPACE / "experiments" / "neighbor_support_ablation_20261022" / "kernels.py")


def select_best_epoch(rows: list[dict]) -> dict:
    """Return the earliest epoch attaining the maximum clean-validation count."""
    if not rows:
        raise ValueError("No validation records")
    best = None
    best_correct = -1
    for row in rows:
        epoch = int(row["epoch"])
        correct = int(row["validation_correct"])
        if epoch < 1 or correct < 0:
            raise ValueError("Invalid validation record")
        if correct > best_correct:
            best, best_correct = row, correct
    return best


def build_support32_targets(observed: torch.Tensor, assignment: torch.Tensor, *,
                            classes: int, beta: float = 0.5,
                            min_votes: int = 2) -> torch.Tensor:
    """Create LOO soft targets from observed training labels and train assignments."""
    observed = torch.as_tensor(observed).long().cpu()
    assignment = torch.as_tensor(assignment).long().cpu()
    if observed.ndim != 1 or assignment.shape != observed.shape or len(observed) == 0:
        raise ValueError("Observed labels and train-node assignments must be matching vectors")
    targets, _ = _CORE.make_targets(
        observed, assignment, classes, leave_one_out=True, tie_safe=True,
        beta=beta, min_votes=min_votes,
    )
    return targets.float().cpu()


def _inside(path: Path) -> Path:
    return Path(path).resolve()


def _save_tensor(path: Path, value) -> None:
    path = _inside(path)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Tensor outputs must stay inside the experiment root")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_h(path: str | Path, expected_nodes: int, expected_dim: int) -> torch.Tensor:
    source = Path(path).resolve()
    stored = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(stored, dict):
        values = stored.get("H")
    else:
        values = stored
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"Invalid H artifact: {source}")
    values = F.normalize(values.float().cpu(), dim=1, eps=1e-12).detach()
    if values.shape[0] != expected_nodes or values.shape[1] < 1 or not torch.isfinite(values).all():
        raise ValueError(f"H shape/content mismatch: {source}")
    return values


def _source_hashes() -> dict[str, str]:
    import gblcss.data as gblcss_data
    import gblcss.models as gblcss_models

    paths = {
        "downstream": CODE / "downstream.py",
        "data_bridge": CODE / "data_bridge.py",
        "r2_core": Path(_CORE.__file__),
        "ball_generator": Path(_GENERATOR.__file__),
        "support_kernels": Path(_SUPPORT.__file__),
        "gblcss_data": Path(gblcss_data.__file__),
        "gblcss_models": Path(gblcss_models.__file__),
    }
    return {str(path.resolve()): sha(path) for path in paths.values()}


def _output_base(name: str, seed: int, arm: str, smoke: bool) -> Path:
    if name not in CONFIG["datasets"]:
        raise ValueError(f"Unregistered dataset: {name}")
    if int(seed) not in CONFIG["seeds"]:
        raise ValueError(f"Unregistered seed: {seed}")
    if arm not in CONFIG["arms"]:
        raise ValueError(f"Unregistered arm: {arm}")
    prefix = ROOT / "engineering" / "downstream" if smoke else ROOT / "downstream"
    return prefix / name / f"seed{int(seed)}" / arm


def _device(smoke: bool) -> torch.device:
    requested = CONFIG.get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if smoke:
        return torch.device("cpu")
    raise RuntimeError("Full downstream fits require the registered CUDA device")


def _existing(folder: Path, expected: dict, files: tuple[str, ...]):
    result_path = folder / "result.json"
    if not result_path.exists():
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Incomplete downstream fit must be retained and reviewed: {folder}")
        return None
    result = read(result_path)
    if result.get("status") != "complete":
        raise RuntimeError(f"Incomplete downstream fit must be retained and reviewed: {folder}")
    for filename in ("config.json", "epochs.jsonl"):
        if result.get(filename + "_sha256") != sha(folder / filename):
            raise RuntimeError(f"Completed log/config changed: {folder / filename}")
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"Completed fit configuration mismatch at {folder}: {key}")
    digest_fields = {
        "encoder.pt": "checkpoint_sha256",
        "student.pt": "student_sha256",
        "probabilities.pt": "probabilities_sha256",
        "target.pt": "target_file_sha256",
    }
    for filename in files:
        expected_sha = result.get(digest_fields.get(filename, f"{filename.removesuffix('.pt')}_sha256"))
        if not expected_sha or sha(folder / filename) != expected_sha:
            raise RuntimeError(f"Completed artifact changed: {folder / filename}")
    return result


def _fit_alignment(folder: Path, x: torch.Tensor, h: torch.Tensor, *,
                   h_path: Path, name: str, seed: int, arm: str,
                   classes: int, hidden: int, dropout: float, epochs: int,
                   device: torch.device) -> tuple[dict, dict[str, torch.Tensor]]:
    expected = {"dataset": name, "seed": int(seed), "arm": arm,
                "classes": int(classes), "epochs": epochs, "h_sha256": sha(h_path), "receipt_sha256": sha(ROOT / "receipt.json"), "protocol": CONFIG["r2"]}
    done = _existing(folder, expected, ("encoder.pt",))
    if done is not None:
        return done, _torch_load(folder / "encoder.pt")
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    initial_rng = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        initial_rng["gpu"] = torch.cuda.get_rng_state(device)
    initial_rng_hash = state_hash(initial_rng)
    model = _CORE.AlignedMLP(x.shape[1], classes, h.shape[1], hidden, dropout).to(device)
    initial_hash = state_hash(model.state_dict())
    save(folder / "config.json", {
        **expected, "hidden": hidden, "dropout": dropout,
        "learning_rate": float(CONFIG["r2"]["learning_rate"]),
        "weight_decay": float(CONFIG["r2"]["weight_decay"]),
        "initial_hash": initial_hash, "initial_rng_hash": initial_rng_hash,
        "selection": "minimum full-graph feature alignment loss; first strict minimum",
    })
    x_device, h_device = x.to(device), h.to(device)
    optimizer = torch.optim.Adam(
        model.encoder_parameters() + list(model.projection.parameters()),
        lr=float(CONFIG["r2"]["learning_rate"]),
        weight_decay=float(CONFIG["r2"]["weight_decay"]),
    )
    best_loss, best_epoch, best_state = float("inf"), 0, None
    log_path = folder / "epochs.jsonl"
    if log_path.exists():
        raise RuntimeError(f"Incomplete R2 alignment log must be retained and reviewed: {log_path}")
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _CORE.alignment_loss(model.projection(model.encode(x_device)), h_device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite R2 loss at {folder}/epoch{epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite R2 gradient at {folder}/epoch{epoch}")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluation = float(_CORE.alignment_loss(
                    model.projection(model.encode(x_device)), h_device
                ))
            if not np.isfinite(evaluation):
                raise FloatingPointError(f"Nonfinite R2 evaluation at {folder}/epoch{epoch}")
            if evaluation < best_loss:
                best_loss, best_epoch = evaluation, epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            log.write(json.dumps({"epoch": epoch, "training_loss": float(loss.detach()),
                                  "evaluation_loss": evaluation}) + "\n")
            log.flush()
    if best_state is None:
        raise RuntimeError(f"R2 alignment produced no checkpoint: {folder}")
    _save_tensor(folder / "encoder.pt", best_state)
    result = {
        **expected,
        "initial_hash": initial_hash,
        "initial_rng_hash": initial_rng_hash,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_alignment_loss": best_loss,
        "seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha(folder / "encoder.pt"),
        "selection": "minimum full-graph feature alignment loss; first strict minimum",
        "source_hashes": _source_hashes(),
        "config.json_sha256": sha(folder / "config.json"),
        "epochs.jsonl_sha256": sha(folder / "epochs.jsonl"),
    }
    save(folder / "result.json", result)
    return result, best_state


def _copy_r2_encoder(model, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    copied = 0
    for key in current:
        if key.startswith(("backbone.layers.0.", "backbone.batch_norms.", "projection.")):
            if key not in state or current[key].shape != state[key].shape:
                raise RuntimeError(f"R2 encoder tensor mismatch: {key}")
            current[key] = state[key]
            copied += 1
    if copied == 0:
        raise RuntimeError("No R2 encoder tensors copied into classifier")
    model.load_state_dict(current)


def _support_assignment(folder: Path, h: torch.Tensor, train: torch.Tensor,
                        name: str, seed: int, support: int) -> torch.Tensor:
    target_path = folder / "support32.pt"
    meta_path = folder / "support32.json"
    h_hash = state_hash({"H": h})
    if target_path.exists() and meta_path.exists():
        metadata = read(meta_path)
        if metadata.get("h_hash") != h_hash or metadata.get("train_count") != len(train):
            raise RuntimeError(f"SUPPORT32 assignment cache mismatch: {folder}")
        assignment = _torch_load(target_path).long().cpu()
        if assignment.shape != (len(h),) or (assignment < 0).any():
            raise RuntimeError(f"Invalid SUPPORT32 assignment cache: {target_path}")
        if metadata.get("assignment_sha256") != state_hash({"assignment": assignment}):
            raise RuntimeError(f"SUPPORT32 assignment artifact changed: {target_path}")
        return assignment
    if target_path.exists() or meta_path.exists():
        raise RuntimeError(f"Incomplete SUPPORT32 cache must be retained and reviewed: {folder}")
    tree = _GENERATOR.build_tree(
        h.numpy(), train.numpy(),
        max_leaves=int(CONFIG["support_tree"]["max_leaves"]),
        snapshot_leaves=int(CONFIG["support_tree"]["snapshot_leaves"]),
        min_nodes=int(CONFIG["support_tree"]["min_nodes"]),
        min_train=int(CONFIG["support_tree"]["min_train"]),
        seed=int(seed),
    )
    costs = _SUPPORT.support_costs(tree["nodes"], int(support))
    leaves, objective = _GENERATOR.select_cut(tree["nodes"], costs)
    assignment = torch.from_numpy(_GENERATOR.assignment(tree["nodes"], leaves, len(h))).long()
    if (assignment < 0).any() or len(torch.unique(assignment)) != len(leaves):
        raise RuntimeError(f"Invalid SUPPORT32 assignment for {name} seed{seed}")
    _save_tensor(target_path, assignment)
    save(meta_path, {
        "dataset": name, "seed": int(seed), "support": int(support), "objective": float(objective),
        "balls": len(leaves), "h_hash": h_hash, "train_count": int(len(train)),
        "assignment_sha256": state_hash({"assignment": assignment}),
        "generator": str(Path(_GENERATOR.__file__).resolve()),
        "generator_sha256": sha(_GENERATOR.__file__),
        "support_kernel": str(Path(_SUPPORT.__file__).resolve()),
        "support_kernel_sha256": sha(_SUPPORT.__file__),
    })
    return assignment


def _fit_classifier(folder: Path, *, x: torch.Tensor, train: torch.Tensor,
                    valid: torch.Tensor, observed: torch.Tensor, valid_y: torch.Tensor,
                    classes: int, encoder_state: dict[str, torch.Tensor],
                    target: torch.Tensor, name: str, seed: int, arm: str,
                    condition: str, head: str, h_path: Path, r2_sha: str,
                    hidden: int, dropout: float, epochs: int,
                    device: torch.device) -> dict:
    target_digest = state_hash({"target": target})
    expected = {"dataset": name, "seed": int(seed), "arm": arm, "condition": condition,
                "head": head, "epochs": epochs, "target_sha256": target_digest,
                "h_sha256": sha(h_path), "r2_checkpoint_sha256": r2_sha, "receipt_sha256": sha(ROOT / "receipt.json"), "protocol": CONFIG["r2"]}
    done = _existing(folder, expected, ("student.pt", "probabilities.pt", "target.pt"))
    if done is not None:
        return done
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    initial_rng = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        initial_rng["gpu"] = torch.cuda.get_rng_state(device)
    initial_rng_hash = state_hash(initial_rng)
    model = _CORE.AlignedMLP(x.shape[1], classes, encoder_state["projection.weight"].shape[0],
                             hidden, dropout).to(device)
    _copy_r2_encoder(model, encoder_state)
    head_initial_hash = state_hash(model.deployment_state())
    save(folder / "config.json", {
        **expected, "hidden": hidden, "dropout": dropout,
        "learning_rate": float(CONFIG["r2"]["learning_rate"]),
        "weight_decay": float(CONFIG["r2"]["weight_decay"]),
        "initial_hash": head_initial_hash, "initial_rng_hash": initial_rng_hash,
        "r2_encoder_hash": state_hash(model.encoder_state()),
        "selection": "first strict maximum clean validation correct",
        "optimizer": "Adam",
    })
    model.freeze_encoder()
    initial_encoder_hash = state_hash(model.encoder_state())
    x_device = x.to(device)
    train_device, valid_device = train.to(device), valid.to(device)
    target_device = target.to(device)
    observed_device, valid_labels = observed.to(device), valid_y.to(device)
    with torch.no_grad():
        frozen_hidden = model.encode(x_device).detach()
    optimizer = torch.optim.Adam(
        model.classifier_parameters(),
        lr=float(CONFIG["r2"]["learning_rate"]),
        weight_decay=float(CONFIG["r2"]["weight_decay"]),
    )
    best_correct, best_epoch, best_state = -1, 0, None
    log_path = folder / "epochs.jsonl"
    if log_path.exists():
        raise RuntimeError(f"Incomplete classifier log must be retained and reviewed: {log_path}")
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train_head()
            optimizer.zero_grad(set_to_none=True)
            logits = model.backbone.layers[-1](frozen_hidden)
            losses = -(target_device * F.log_softmax(logits[train_device], dim=1)).sum(1)
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite classifier loss at {folder}/epoch{epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"Nonfinite classifier gradient at {folder}/epoch{epoch}")
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_correct = int((model.backbone.layers[-1](frozen_hidden[valid_device]).argmax(1)
                                           == valid_labels).sum())
            row = {"epoch": epoch, "training_loss": float(loss.detach()),
                   "validation_correct": validation_correct, "validation_count": int(len(valid))}
            if validation_correct > best_correct:
                best_correct, best_epoch = validation_correct, epoch
                best_state = model.deployment_state()
            log.write(json.dumps(row) + "\n")
            log.flush()
    if state_hash(model.encoder_state()) != initial_encoder_hash:
        raise RuntimeError(f"Frozen encoder or batch-normalization state changed: {folder}")
    if best_state is None:
        raise RuntimeError(f"Classifier produced no checkpoint: {folder}")
    model.backbone.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model.backbone(x_device).softmax(1).detach().cpu()
    selected_validation_correct = int((probabilities[valid].argmax(1) == valid_y.cpu()).sum())
    if selected_validation_correct != best_correct:
        raise RuntimeError(f"Selected checkpoint validation replay differs: {folder}")
    deployed = _CORE.MLP(x.shape[1], hidden, classes, 2, dropout).to(device).eval()
    deployed.load_state_dict(best_state)
    with torch.no_grad():
        deployment_probabilities = deployed(x_device).softmax(1).detach().cpu()
    deployment_difference = float((probabilities - deployment_probabilities).abs().max())
    if deployment_difference > 1e-7:
        raise RuntimeError(f"X-only MLP deployment differs from the frozen R2 student: {folder}")
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(1), torch.ones(len(probabilities)), atol=1e-5, rtol=0
    ):
        raise RuntimeError(f"Invalid student probabilities: {folder}")
    _save_tensor(folder / "student.pt", best_state)
    _save_tensor(folder / "probabilities.pt", probabilities)
    _save_tensor(folder / "target.pt", target.cpu())
    result = {
        **expected,
        "status": "complete",
        "initial_hash": head_initial_hash,
        "initial_rng_hash": initial_rng_hash,
        "best_epoch": best_epoch,
        "valid_correct": best_correct,
        "valid_total": int(len(valid)),
        "validation_accuracy": best_correct / len(valid),
        "seconds": time.perf_counter() - started,
        "frozen_encoder_hash_before": initial_encoder_hash,
        "frozen_encoder_hash_after": state_hash(model.encoder_state()),
        "frozen_encoder_unchanged": True,
        "deployment_difference": deployment_difference,
        "target_path": str((folder / "target.pt").resolve()),
        "student_sha256": sha(folder / "student.pt"),
        "probabilities_sha256": sha(folder / "probabilities.pt"),
        "target_file_sha256": sha(folder / "target.pt"),
        "selection": "first strict maximum clean validation correct",
        "source_hashes": _source_hashes(),
        "config.json_sha256": sha(folder / "config.json"),
        "epochs.jsonl_sha256": sha(folder / "epochs.jsonl"),
    }
    save(folder / "result.json", result)
    return result


def run_downstream(name: str, seed: int, arm: str, h_path: str | Path,
                   smoke: bool = False) -> list[dict]:
    """Fit one label-free R2 adapter, then CE and SUPPORT32 heads for four conditions."""
    base = _output_base(name, seed, arm, smoke)
    features = load_features(name, int(seed))
    x, train = features["x"], features["train"]
    classes = class_count(name, int(seed))
    h_path = Path(h_path).resolve()
    h = _load_h(h_path, len(x), int(features["h_dim"]))
    if len(train) <= int(CONFIG["r2"]["support"]):
        raise ValueError("Training split is too small for the registered SUPPORT32 setting")
    device = _device(smoke)
    torch.set_num_threads(int(CONFIG.get("threads", 1)))
    r2_cfg = CONFIG["r2"]
    alignment_epochs = 2 if smoke else int(r2_cfg["alignment_epochs"])
    head_epochs = 2 if smoke else int(r2_cfg["head_epochs"])
    r2_folder = base / "r2"
    r2_result, encoder_state = _fit_alignment(
        r2_folder, x, h, h_path=h_path, name=name, seed=int(seed), arm=arm,
        classes=classes,
        hidden=int(r2_cfg["hidden"]), dropout=float(r2_cfg["dropout"]),
        epochs=alignment_epochs, device=device,
    )
    assignment = _support_assignment(
        r2_folder, h, train, name, int(seed), int(r2_cfg["support"])
    )
    results = [dict(stage="r2", **r2_result)]
    for condition in CONFIG["conditions"]:
        data = load_supervised(name, int(seed), condition)
        if int(data["classes"]) != classes:
            raise RuntimeError(f"{name}/{condition}: cached class count differs from R2 setup")
        if not torch.equal(data["train"], train):
            raise RuntimeError(f"{name}/{condition}: split changed after feature load")
        observed, valid_y = data["observed"], data["valid_y"]
        support_target = build_support32_targets(
            observed, assignment[train], classes=int(data["classes"]),
            beta=float(r2_cfg["beta"]), min_votes=int(r2_cfg["min_votes"]),
        )
        ce_target = F.one_hot(observed.long(), int(data["classes"])).float()
        for head, target in (("CE", ce_target), ("SUPPORT32", support_target)):
            folder = base / "conditions" / condition / head
            result = _fit_classifier(
                folder, x=x, train=train, valid=data["valid"], observed=observed,
                valid_y=valid_y, classes=int(data["classes"]), encoder_state=encoder_state,
                target=target, name=name, seed=int(seed), arm=arm, condition=condition,
                head=head, h_path=h_path, r2_sha=r2_result["checkpoint_sha256"],
                hidden=int(r2_cfg["hidden"]), dropout=float(r2_cfg["dropout"]),
                epochs=head_epochs, device=device,
            )
            results.append(dict(stage="classifier", **result))
    return results

