"""Run the five-dataset R2 + SUPPORT + LCSS grid."""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
OLD = WORKSPACE / "experiments" / "representation_supervision_20261021"
FUSION = WORKSPACE / "experiments" / "gce_lcss_fusion_20261022"
SUPPORT = WORKSPACE / "experiments" / "neighbor_support_ablation_20261022"
GENERATOR_DIR = WORKSPACE / "experiments" / "neighbor_preserving_ball_20261022"
COMMON_DIR = WORKSPACE / "experiments" / "new_graph_mlp_seed0_20261009"
MATRIX = WORKSPACE / "experiments" / "main_protocol_replay_20260922" / "matrix"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(WORKSPACE / "GB-LCSS_code" / "src"))
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(COMMON_DIR))
from core import AlignedMLP, alignment_loss, make_targets, state_hash
from common import DATA_FILES, raw_data, fingerprint
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed
from gblcss.models import MLP

sys.path.insert(0, str(GENERATOR_DIR))
import generator

kernel_spec = importlib.util.spec_from_file_location("support_kernels_frozen", SUPPORT / "kernels.py")
support_kernels = importlib.util.module_from_spec(kernel_spec)
kernel_spec.loader.exec_module(support_kernels)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def initialize():
    torch.set_num_threads(CONFIG["model"]["threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("The verified GPU environment is required")


def encoder_paths(name, seed):
    if seed == 0:
        folder = OLD / "pretraining" / name
    else:
        folder = FUSION / "encoders" / name / f"seed{seed}"
    return folder / "model.pt", folder / "result.json"


def input_dir(name, seed):
    return ROOT / "inputs" / name / f"seed{seed}"


def old_run_dir(name, seed, condition):
    return SUPPORT / "runs" / name / f"seed{seed}" / condition / "R2_SUPPORT"


def new_run_dir(name, seed, condition):
    return ROOT / "runs" / name / f"seed{seed}" / condition / "R2_SUPPORT"


def numpy_targets(z, assignment, classes, beta, min_votes):
    z = np.asarray(z, dtype=np.int64)
    assignment = np.asarray(assignment, dtype=np.int64)
    output = np.zeros((len(z), classes), dtype=np.float64)
    for ball in np.unique(assignment):
        ids = np.flatnonzero(assignment == ball)
        counts = np.bincount(z[ids], minlength=classes).astype(np.float64)
        for row in ids:
            votes = counts.copy()
            votes[z[row]] -= 1
            support = votes.sum()
            own = np.eye(classes, dtype=np.float64)[z[row]]
            if support < min_votes:
                output[row] = own
                continue
            distribution = votes / support
            maximum = distribution.max()
            alpha = np.clip((classes * maximum - 1) / (classes - 1), 0, 1)
            winners = np.flatnonzero(distribution == maximum)
            consensus = (distribution if len(winners) > 1 else
                         (1 - beta) * distribution + beta * np.eye(classes)[winners[0]])
            output[row] = (1 - alpha) * own + alpha * consensus
    return output


def load_h(name, raw):
    meta_path = MATRIX / name / "default" / "seed0.json"
    meta = read(meta_path)
    h_path = Path(meta["ball_embedding_path"])
    stored = torch.load(h_path, map_location="cpu", weights_only=False)
    values = stored["H"] if isinstance(stored, dict) and "H" in stored else stored
    if isinstance(stored, dict) and "labels" in stored:
        if not torch.equal(stored["labels"].long().cpu(), raw.y.long().cpu()):
            raise RuntimeError(f"{name}: GRACE node order differs from raw labels")
    h = F.normalize(values.float(), dim=1, eps=1e-12).detach().cpu()
    if h.ndim != 2 or h.shape[0] != raw.x.shape[0] or not torch.isfinite(h).all():
        raise RuntimeError(f"{name}: invalid GRACE embedding")
    return h, h_path, meta_path


def prepare_new_data():
    geometry_rows, protocol_rows, source_paths = [], [], set()
    support_cfg = CONFIG["support"]
    for name in CONFIG["new_datasets"]:
        raw = raw_data(name)
        h, h_path, _ = load_h(name, raw)
        source_paths.update((DATA_FILES[name], h_path))
        h_np = h.numpy()
        classes = int(raw.y.max()) + 1
        for seed in CONFIG["seeds"]:
            indices = tuple(torch.as_tensor(v, dtype=torch.long) for v in percent_split_indices(len(raw.x), seed))
            meta_path = MATRIX / name / "default" / f"seed{seed}.json"
            meta = read(meta_path)
            split_hash = fingerprint(*indices)
            if split_hash != meta["split_hash"]:
                raise RuntimeError(f"{name} seed{seed}: historical split mismatch")
            source_paths.add(meta_path)
            train, valid, test = indices
            tree = generator.build_tree(
                h_np, train.numpy(), support_cfg["max_leaves"],
                support_cfg["snapshot_leaves"], support_cfg["min_nodes"],
                support_cfg["min_train"], seed=seed
            )
            nodes = tree["nodes"]
            k = min(support_cfg["reference_k"], len(train) - 1)
            costs = support_kernels.support_costs(nodes, k)
            leaves, objective = generator.select_cut(nodes, costs)
            assignment = generator.assignment(nodes, leaves, len(raw.x))
            if (assignment < 0).any() or len(np.unique(assignment)) != len(leaves):
                raise RuntimeError(f"{name} seed{seed}: invalid SUPPORT partition")
            directory = input_dir(name, seed)
            directory.mkdir(parents=True, exist_ok=True)
            geometry_path = directory / "geometry.pt"
            torch.save({
                "train": train, "valid": valid, "test": test,
                "assignment": torch.from_numpy(assignment).long(),
                "nodes": [{**node, "members": torch.from_numpy(node["members"]).long()} for node in nodes],
                "support_leaves": leaves, "k": k, "objective": objective,
                "split_hash": split_hash, "h_sha256": sha(h_path)
            }, geometry_path)
            source_paths.add(geometry_path)
            train_assignment = torch.from_numpy(assignment[train.numpy()]).long()
            support = np.bincount(assignment[train.numpy()])[assignment[train.numpy()]] - 1
            geometry_rows.append({
                "dataset": name, "seed": seed, "nodes": len(raw.x), "classes": classes,
                "balls": len(leaves), "train_nodes": len(train), "k": k,
                "support_mean": float(support.mean()), "support_min": int(support.min()),
                "support_max": int(support.max()), "split_hash": split_hash,
                "h_sha256": sha(h_path), "support_objective": objective
            })
            for condition, (noise_kind, rate) in CONFIG["conditions"].items():
                masks = []
                for ids in indices:
                    mask = torch.zeros(len(raw.x), dtype=torch.bool)
                    mask[ids] = True
                    masks.append(mask)
                graph = GraphData(
                    name, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                    train_index=train, val_index=valid, test_index=test
                )
                inject_train_label_noise(graph, noise_kind, rate, seed)
                if not torch.equal(graph.y[~masks[0]], graph.noisy_y[~masks[0]]):
                    raise RuntimeError(f"{name} seed{seed} {condition}: noise escaped training")
                observed = graph.noisy_y[train].long().cpu()
                if condition == "uniform0.4":
                    actual_noise_hash = fingerprint(graph.y, graph.noisy_y)
                    if actual_noise_hash != meta["noise_hash"]:
                        raise RuntimeError(f"{name} seed{seed}: historical uniform .4 noise mismatch")
                else:
                    actual_noise_hash = fingerprint(graph.y, graph.noisy_y)
                targets, alpha = make_targets(
                    observed, train_assignment, classes,
                    beta=support_cfg["beta"], min_votes=support_cfg["min_votes"]
                )
                reference = numpy_targets(
                    observed.numpy(), assignment[train.numpy()], classes,
                    support_cfg["beta"], support_cfg["min_votes"]
                )
                target_error = float(np.max(np.abs(reference - targets.numpy())))
                if target_error > 3e-7:
                    raise RuntimeError(f"{name} seed{seed} {condition}: independent LCSS target mismatch")
                cache_path = directory / f"{condition}.pt"
                torch.save({
                    "observed": observed, "target": targets,
                    "alpha": alpha, "valid_y": graph.y[valid].long().cpu()
                }, cache_path)
                record = {
                    "dataset": name, "seed": seed, "condition": condition,
                    "noise_type": noise_kind, "noise_rate": rate,
                    "split_hash": split_hash, "noise_hash": actual_noise_hash,
                    "modified_train": int((graph.y[train] != graph.noisy_y[train]).sum()),
                    "modified_valid": 0, "modified_test": 0,
                    "target_reference_max_error": target_error,
                    "target_hash": state_hash({"target": targets}),
                    "assignment_hash": state_hash({"assignment": torch.from_numpy(assignment).long()}),
                    "feature_source": str(DATA_FILES[name]), "h_source": str(h_path),
                    "labels_in_partition": False
                }
                write(directory / f"{condition}.json", record)
                protocol_rows.append(record)
                source_paths.add(cache_path)
                source_paths.add(directory / f"{condition}.json")
            print(f"PREPARED {name} seed{seed}: balls={len(leaves)}", flush=True)
    csv_write(ROOT / "geometry.csv", geometry_rows)
    write(ROOT / "protocol_checks.json", {
        "protocol_count": len(protocol_rows), "geometry_count": len(geometry_rows),
        "target_reference_max_error": max(r["target_reference_max_error"] for r in protocol_rows),
        "all_noise_train_only": all(r["modified_valid"] == r["modified_test"] == 0 for r in protocol_rows),
        "uniform04_historical_noise_matches": True
    })
    return protocol_rows, source_paths


def validate_reused_cells():
    records, source_paths = [], set()
    for name in CONFIG["reused_datasets"]:
        raw = raw_data(name)
        _, h_path, _ = load_h(name, raw)
        source_paths.add(h_path)
        for seed in CONFIG["seeds"]:
            old_geometry = SUPPORT / "inputs" / name / f"seed{seed}" / "geometry.pt"
            geo = torch.load(old_geometry, map_location="cpu", weights_only=True)
            expected = percent_split_indices(len(raw.x), seed)
            for key, value in zip(("train", "valid", "test"), expected):
                if not np.array_equal(geo[key].numpy(), value):
                    raise RuntimeError(f"Reuse split mismatch: {name} seed{seed} {key}")
            source_paths.add(old_geometry)
            for condition in CONFIG["conditions"]:
                folder = old_run_dir(name, seed, condition)
                record = read(folder / "result.json")
                condition_path = SUPPORT / "inputs" / name / f"seed{seed}" / f"{condition}.pt"
                cache = torch.load(condition_path, map_location="cpu", weights_only=True)
                for key, expected_value in {
                    "status": "ok", "dataset": name, "seed": seed,
                    "condition": condition, "route": "R2", "arm": "SUPPORT",
                    "epochs": CONFIG["model"]["head_epochs"], "smoke": False
                }.items():
                    if record.get(key) != expected_value:
                        raise RuntimeError(f"Invalid reused result {folder}: {key}")
                if record["target_hash"] != state_hash({"target": cache["targets"]["SUPPORT"]}):
                    raise RuntimeError(f"Reused target hash mismatch: {folder}")
                if record["input_sha256"] != sha(condition_path):
                    raise RuntimeError(f"Reused input hash mismatch: {folder}")
                for artifact, key in (("student.pt", "checkpoint_sha256"),
                                      ("probabilities.pt", "probabilities_sha256")):
                    if sha(folder / artifact) != record[key]:
                        raise RuntimeError(f"Reused artifact hash mismatch: {folder / artifact}")
                    source_paths.add(folder / artifact)
                epoch_log = folder / "epochs.jsonl"
                history = [json.loads(line) for line in epoch_log.read_text(encoding="utf-8").splitlines()]
                if len(history) != CONFIG["model"]["head_epochs"]:
                    raise RuntimeError(f"Reused epoch log incomplete: {folder}")
                best = max(row["validation_correct"] for row in history)
                first_best = next(row["epoch"] for row in history if row["validation_correct"] == best)
                if first_best != record["best_epoch"] or best != record["valid_correct"]:
                    raise RuntimeError(f"Reused selection mismatch: {folder}")
                encoder = Path(record["encoder_source"])
                if not encoder.is_file():
                    raise RuntimeError(f"Missing reused R2 encoder: {encoder}")
                records.append({
                    "dataset": name, "seed": seed, "condition": condition,
                    "source": str(folder), "target_hash": record["target_hash"],
                    "probabilities_sha256": record["probabilities_sha256"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "best_epoch": record["best_epoch"]
                })
                source_paths.update((condition_path, epoch_log, folder / "result.json",
                                     folder / "config.json", encoder))
        print(f"VERIFIED REUSE {name}: 9 R2+SUPPORT noisy cells", flush=True)
    csv_write(ROOT / "reused_cells.csv", records)
    return records, source_paths


def collect_frozen_sources(extra_paths):
    paths = set(extra_paths)
    paths.update({
        ROOT / "PLAN.md", ROOT / "config.json", ROOT / "run.py",
        OLD / "core.py", FUSION / "run.py", FUSION / "config.json",
        SUPPORT / "run.py", SUPPORT / "config.json", SUPPORT / "kernels.py",
        GENERATOR_DIR / "generator.py", COMMON_DIR / "common.py",
        WORKSPACE / "GB-LCSS_code" / "src" / "gblcss" / "data.py",
        WORKSPACE / "GB-LCSS_code" / "src" / "gblcss" / "models.py",
        WORKSPACE / "GB-LCSS_code" / "src" / "gblcss" / "targets.py"
    })
    for name in CONFIG["datasets"]:
        paths.add(DATA_FILES[name])
        for seed in CONFIG["seeds"]:
            paths.add(MATRIX / name / "default" / f"seed{seed}.json")
            if name in CONFIG["reused_datasets"]:
                encoder, result = encoder_paths(name, seed)
                paths.update((encoder, result))
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen source(s): " + ", ".join(missing))
    return {str(Path(p).resolve()): sha(p) for p in sorted(paths, key=lambda v: str(v))}


def prepare():
    initialize()
    if (ROOT / "frozen_inputs.json").exists():
        raise RuntimeError("Prepared inputs are frozen; do not overwrite")
    protocol_rows, new_paths = prepare_new_data()
    reused_rows, reused_paths = validate_reused_cells()
    if len(protocol_rows) != len(CONFIG["new_datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["conditions"]):
        raise RuntimeError("Unexpected new protocol cell count")
    if len(reused_rows) != CONFIG["expected_reused_cells"]:
        raise RuntimeError("Unexpected reused cell count")
    environment = {
        "python": sys.executable, "python_version": sys.version,
        "torch": str(torch.__version__), "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
        "threads": torch.get_num_threads()
    }
    write(ROOT / "environment.json", environment)
    frozen = collect_frozen_sources(new_paths | reused_paths | {ROOT / "environment.json", ROOT / "geometry.csv"})
    write(ROOT / "frozen_inputs.json", {
        "stage": "prepared and frozen before new model fitting",
        "config": CONFIG, "sha256": frozen,
        "new_protocol_cells": len(protocol_rows),
        "reused_cells": len(reused_rows)
    })
    write(ROOT / "progress.json", {
        "status": "prepared", "reused_completed": len(reused_rows),
        "new_fits_expected": CONFIG["expected_new_fits"],
        "alignments_expected": CONFIG["expected_new_alignments"]
    })
    print(f"PREPARED: {len(reused_rows)} verified reused cells; "
          f"{CONFIG['expected_new_fits']} new fits; GPU={environment['gpu']}", flush=True)


def verify_frozen():
    manifest = read(ROOT / "frozen_inputs.json")
    if manifest.get("config") != CONFIG:
        raise RuntimeError("Frozen configuration mismatch")
    for path, digest in manifest["sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen input/source changed: " + path)


def model_for(name, seed, x, h, classes):
    set_seed(seed)
    return AlignedMLP(
        x.shape[1], classes, h.shape[1],
        CONFIG["model"]["hidden"], CONFIG["model"]["dropout"]
    ).cuda()


def alignment_checkpoint(name, seed, x, h, classes):
    if name in CONFIG["reused_datasets"]:
        checkpoint, result_path = encoder_paths(name, seed)
        result = read(result_path)
        if result.get("status") != "ok" or result.get("labels_used") is not False:
            raise RuntimeError(f"Invalid cached label-free R2 alignment: {result_path}")
        digest = result.get("checkpoint_sha256", result.get("sha256"))
        if digest != sha(checkpoint):
            raise RuntimeError(f"Cached R2 encoder hash mismatch: {checkpoint}")
        return checkpoint, "reused:" + str(checkpoint), digest

    folder = ROOT / "pretraining" / name / f"seed{seed}"
    checkpoint, result_path = folder / "model.pt", folder / "result.json"
    if result_path.exists():
        result = read(result_path)
        if result.get("status") != "ok":
            raise RuntimeError(f"Partial/failed alignment preserved for review: {folder}")
        if sha(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"R2 checkpoint hash mismatch: {checkpoint}")
        return checkpoint, str(checkpoint), result["checkpoint_sha256"]
    if (folder / "epochs.jsonl").exists():
        raise RuntimeError(f"Partial alignment log exists; do not overwrite: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = AlignedMLP(
        x.shape[1], classes, h.shape[1],
        CONFIG["model"]["hidden"], CONFIG["model"]["dropout"]
    ).cuda()
    optimizer = torch.optim.Adam(
        model.encoder_parameters() + list(model.projection.parameters()),
        lr=CONFIG["model"]["learning_rate"],
        weight_decay=CONFIG["model"]["weight_decay"]
    )
    epochs = CONFIG["model"]["alignment_epochs"]
    best, selected, state = float("inf"), 0, None
    started = time.perf_counter()
    write(result_path, {
        "status": "running", "dataset": name, "seed": seed,
        "epochs": epochs, "labels_used": False,
        "h_sha256": sha(Path(read(MATRIX / name / "default" / "seed0.json")["ball_embedding_path"])),
        "initial_hash": state_hash(model.deployment_state())
    })
    try:
        with (folder / "epochs.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = alignment_loss(model.projection(model.encode(x)), h)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite R2 alignment loss")
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    evaluation = float(alignment_loss(model.projection(model.encode(x)), h))
                if not np.isfinite(evaluation):
                    raise FloatingPointError("Nonfinite R2 alignment evaluation")
                if evaluation < best:
                    best, selected, state = evaluation, epoch, deepcopy(model.state_dict())
                log.write(json.dumps({
                    "epoch": epoch, "train_alignment": float(loss.detach()),
                    "evaluation_alignment": evaluation
                }, allow_nan=False) + "\n")
                log.flush()
        model.load_state_dict(state)
        model.eval()
        torch.save({key: value.detach().cpu() for key, value in state.items()}, checkpoint)
        result = {
            "status": "ok", "dataset": name, "seed": seed, "epochs": epochs,
            "selected_epoch": selected, "selected_by": "minimum label-free alignment",
            "labels_used": False, "best_alignment": best,
            "checkpoint_sha256": sha(checkpoint), "seconds": time.perf_counter() - started,
            "h_sha256": sha(Path(read(MATRIX / name / "default" / "seed0.json")["ball_embedding_path"]))
        }
        write(result_path, result)
        print(f"ALIGN {name} seed{seed}: epoch={selected}, loss={best:.6f}", flush=True)
        return checkpoint, str(checkpoint), result["checkpoint_sha256"]
    except Exception:
        write(result_path, {
            "status": "failed", "dataset": name, "seed": seed,
            "traceback": traceback.format_exc()
        })
        raise


def load_fit_inputs(name, seed, condition):
    raw = raw_data(name)
    h, _, _ = load_h(name, raw)
    directory = input_dir(name, seed)
    geometry = torch.load(directory / "geometry.pt", map_location="cpu", weights_only=True)
    cache = torch.load(directory / f"{condition}.pt", map_location="cpu", weights_only=True)
    return raw.x.float().cuda(), h.cuda(), int(raw.y.max()) + 1, geometry, cache


def fit(name, seed, condition):
    folder = new_run_dir(name, seed, condition)
    result_path = folder / "result.json"
    if result_path.exists():
        prior = read(result_path)
        if prior.get("status") != "ok":
            raise RuntimeError(f"Failed/partial fit preserved for review: {folder}")
        if sha(folder / "student.pt") != prior["checkpoint_sha256"] or \
                sha(folder / "probabilities.pt") != prior["probabilities_sha256"]:
            raise RuntimeError(f"Completed fit artifact hash mismatch: {folder}")
        return prior
    if (folder / "epochs.jsonl").exists():
        raise RuntimeError(f"Partial fit log exists; do not overwrite: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    x, h, classes, geometry, cache = load_fit_inputs(name, seed, condition)
    train, valid = geometry["train"].cuda(), geometry["valid"].cuda()
    target, valid_y = cache["target"].cuda(), cache["valid_y"].cuda()
    set_seed(seed)
    model = model_for(name, seed, x, h, classes)
    encoder, encoder_source, encoder_hash = alignment_checkpoint(name, seed, x, h, classes)
    model.load_state_dict(torch.load(encoder, map_location="cuda", weights_only=True))
    model.freeze_encoder()
    frozen_hash = state_hash(model.encoder_state())
    with torch.no_grad():
        hidden = model.encode(x).detach()
    optimizer = torch.optim.Adam(
        model.classifier_parameters(),
        lr=CONFIG["model"]["learning_rate"],
        weight_decay=CONFIG["model"]["weight_decay"]
    )
    initial_hash = state_hash(model.deployment_state())
    initial_rng_hash = state_hash({
        "cpu": torch.get_rng_state(), "gpu": torch.cuda.get_rng_state()
    })
    input_hash = sha(input_dir(name, seed) / f"{condition}.pt")
    target_hash = state_hash({"target": cache["target"]})
    meta = {
        "dataset": name, "seed": seed, "condition": condition,
        "route": "R2", "arm": "SUPPORT", "epochs": CONFIG["model"]["head_epochs"],
        "alignment_epochs": CONFIG["model"]["alignment_epochs"],
        "initial_hash": initial_hash, "initial_rng_hash": initial_rng_hash,
        "input_sha256": input_hash, "target_hash": target_hash,
        "encoder_source": encoder_source, "encoder_sha256": encoder_hash,
        "support_assignment_sha256": state_hash({"assignment": geometry["assignment"]}),
        "smoke": False
    }
    write(folder / "config.json", meta)
    write(result_path, {**meta, "status": "running"})
    best, selected, state = -1, 0, None
    started = time.perf_counter()
    limit = CONFIG["model"]["timeout_seconds_per_fit"]
    torch.cuda.synchronize()
    try:
        with (folder / "epochs.jsonl").open("x", encoding="utf-8") as log:
            for epoch in range(1, CONFIG["model"]["head_epochs"] + 1):
                if time.perf_counter() - started > limit:
                    raise TimeoutError("Per-fit timeout")
                model.train_head()
                logits = model.backbone.layers[-1](hidden)
                loss = -(target * F.log_softmax(logits[train], dim=1)).sum(1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite LCSS loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    valid_logits = model.backbone.layers[-1](hidden[valid])
                    correct = int((valid_logits.argmax(1) == valid_y).sum())
                improved = correct > best
                if improved:
                    best, selected, state = correct, epoch, deepcopy(model.deployment_state())
                log.write(json.dumps({
                    "epoch": epoch, "loss": float(loss.detach()),
                    "validation_correct": correct, "checkpoint_selected": improved
                }, allow_nan=False) + "\n")
                log.flush()
        if state is None:
            raise RuntimeError("No validation checkpoint was selected")
        if state_hash(model.encoder_state()) != frozen_hash:
            raise RuntimeError("R2 encoder or batch-normalization state changed")
        model.backbone.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probabilities = model(x)[0].softmax(1).cpu()
        deployed = MLP(x.shape[1], CONFIG["model"]["hidden"], classes, 2,
                       CONFIG["model"]["dropout"]).cuda().eval()
        deployed.load_state_dict(state)
        with torch.no_grad():
            deployment_diff = float(
                (deployed(x).softmax(1).cpu() - probabilities).abs().max()
            )
        if deployment_diff != 0:
            raise RuntimeError("X-only deployed MLP does not match R2 predictions")
        torch.save(state, folder / "student.pt")
        torch.save(probabilities, folder / "probabilities.pt")
        torch.cuda.synchronize()
        result = {
            **meta, "status": "ok", "best_epoch": selected,
            "valid_correct": best, "valid_total": len(valid),
            "validation": best / len(valid), "seconds": time.perf_counter() - started,
            "frozen_encoder_unchanged": True, "deployment_difference": deployment_diff,
            "checkpoint_sha256": sha(folder / "student.pt"),
            "probabilities_sha256": sha(folder / "probabilities.pt"),
            "folder": str(folder)
        }
        write(result_path, result)
        print(f"FIT {name} seed{seed} {condition}: val={100*result['validation']:.2f}%, "
              f"epoch={selected}", flush=True)
        return result
    except Exception:
        write(result_path, {**meta, "status": "failed", "traceback": traceback.format_exc()})
        raise


def run():
    initialize()
    verify_frozen()
    alignments = []
    for name in CONFIG["new_datasets"]:
        raw = raw_data(name)
        h, _, _ = load_h(name, raw)
        x, h_device = raw.x.float().cuda(), h.cuda()
        classes = int(raw.y.max()) + 1
        for seed in CONFIG["seeds"]:
            checkpoint, source, digest = alignment_checkpoint(name, seed, x, h_device, classes)
            alignments.append({
                "dataset": name, "seed": seed, "checkpoint": str(checkpoint),
                "source": source, "checkpoint_sha256": digest
            })
    csv_write(ROOT / "new_alignments.csv", alignments)
    jobs = [
        (name, seed, condition)
        for name in CONFIG["new_datasets"]
        for seed in CONFIG["seeds"]
        for condition in CONFIG["conditions"]
    ]
    started = time.perf_counter()
    for index, job in enumerate(jobs, 1):
        record = fit(*job)
        write(ROOT / "progress.json", {
            "status": "running" if index < len(jobs) else "completed",
            "new_fits_completed": index, "new_fits_expected": len(jobs),
            "last_job": job, "elapsed_seconds": time.perf_counter() - started
        })
    verify_frozen()
    print(f"RUN COMPLETE: {len(jobs)} new fits; cached 27 reused cells remain frozen.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"prepare", "run"}:
        raise SystemExit("Usage: python run.py prepare|run")
    try:
        if sys.argv[1] == "prepare":
            prepare()
        else:
            run()
    except Exception:
        if sys.argv[1] == "prepare":
            write(ROOT / "prepare_failure.json", {"traceback": traceback.format_exc()})
        raise
