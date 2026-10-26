from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
COMMON = WORKSPACE / "experiments" / "new_graph_mlp_seed0_20261009"
MATRIX = WORKSPACE / "experiments" / "main_protocol_replay_20260922" / "matrix"
SUPPORT_OLD = WORKSPACE / "experiments" / "neighbor_support_ablation_20261022" / "inputs"
SUPPORT_NEW = WORKSPACE / "experiments" / "r2_support_5dataset_20261022" / "inputs"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub" / "src"))
sys.path.insert(0, str(COMMON))
from common import DATA_FILES, fingerprint, raw_data
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed
from gblcss.models import GCN, MLP
from gblcss.targets import build_leave_one_out_targets
from gblcss.training import soft_cross_entropy


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def support_folder(dataset: str, seed: int) -> Path:
    old = SUPPORT_OLD / dataset / f"seed{seed}"
    new = SUPPORT_NEW / dataset / f"seed{seed}"
    if (old / "geometry.pt").is_file():
        return old
    if (new / "geometry.pt").is_file():
        return new
    raise FileNotFoundError(f"No frozen SUPPORT geometry for {dataset} seed {seed}")


def load_geometry(dataset: str, seed: int):
    folder = support_folder(dataset, seed)
    geometry = torch.load(folder / "geometry.pt", map_location="cpu", weights_only=False)
    if "partitions" in geometry:
        support = torch.as_tensor(geometry["partitions"]["SUPPORT"], dtype=torch.long).cpu().numpy()
    else:
        support = torch.as_tensor(geometry["assignment"], dtype=torch.long).cpu().numpy()
    archive_path = MATRIX / dataset / "default" / f"seed{seed}.npz"
    with np.load(archive_path) as archive:
        old = np.asarray(archive["assignment"], dtype=np.int64).copy()
    return folder, geometry, old, support, archive_path


def cached_observed(folder: Path, condition: str):
    cache = torch.load(folder / f"{condition}.pt", map_location="cpu", weights_only=False)
    return cache


def prepare():
    torch.set_num_threads(CONFIG["model"]["threads"])
    checks, input_paths = [], []
    input_root = ROOT / "inputs"
    for dataset in CONFIG["datasets"]:
        raw = raw_data(dataset)
        n = int(raw.x.shape[0])
        classes = int(raw.y.max().item()) + 1
        for seed in CONFIG["seeds"]:
            indices = tuple(torch.as_tensor(x, dtype=torch.long) for x in percent_split_indices(n, seed))
            masks = []
            for index in indices:
                mask = torch.zeros(n, dtype=torch.bool)
                mask[index] = True
                masks.append(mask)
            data = GraphData(dataset, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                             train_index=indices[0], val_index=indices[1], test_index=indices[2])
            support_dir, geometry, old_assignment, support_assignment, archive_path = load_geometry(dataset, seed)
            split_hash = fingerprint(*indices)
            metadata_path = archive_path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if split_hash != metadata["split_hash"]:
                raise RuntimeError(f"{dataset} seed {seed}: split differs from the frozen main protocol")
            for name, assignment in (("OLD_LCSS", old_assignment), ("SUPPORT_LCSS", support_assignment)):
                if assignment.shape != (n,) or (assignment < 0).any():
                    raise RuntimeError(f"{dataset} seed {seed}: invalid {name} assignment")
            for key, expected in zip(("train", "valid", "test"), indices):
                if key in geometry and not torch.equal(torch.as_tensor(geometry[key]).long(), expected):
                    raise RuntimeError(f"{dataset} seed {seed}: SUPPORT geometry {key} index mismatch")
            cached_support = torch.load(support_dir / "geometry.pt", map_location="cpu", weights_only=False)
            if "partitions" in cached_support and not np.array_equal(
                    support_assignment, np.asarray(cached_support["partitions"]["SUPPORT"])):
                raise RuntimeError(f"{dataset} seed {seed}: SUPPORT partition source mismatch")

            for condition, (noise_type, noise_rate) in CONFIG["conditions"].items():
                inject_train_label_noise(data, noise_type, float(noise_rate), seed)
                cache = cached_observed(support_dir, condition)
                observed = torch.as_tensor(cache["observed"], dtype=torch.long)
                if not torch.equal(data.noisy_y[indices[0]], observed):
                    raise RuntimeError(f"{dataset} seed {seed} {condition}: observed training labels mismatch")
                if not torch.equal(data.noisy_y[indices[1]], data.y[indices[1]]) or not torch.equal(
                        data.noisy_y[indices[2]], data.y[indices[2]]):
                    raise RuntimeError(f"{dataset} seed {seed} {condition}: noise escaped training split")
                if "valid_y" in cache and not torch.equal(cache["valid_y"].long(), data.y[indices[1]]):
                    raise RuntimeError(f"{dataset} seed {seed} {condition}: cached clean validation labels differ")
                noise_hash = fingerprint(data.y, data.noisy_y)
                if condition == "uniform0.4" and noise_hash != metadata["noise_hash"]:
                    raise RuntimeError(f"{dataset} seed {seed}: uniform0.4 noise differs from historical protocol")

                ce_target = F.one_hot(observed, num_classes=classes).float()
                old_target = build_leave_one_out_targets(
                    data.noisy_y, old_assignment, indices[0], classes,
                    min_train_per_ball=CONFIG["model"]["min_remaining_votes"],
                    beta=CONFIG["model"]["target_beta"]
                ).soft_target[indices[0]].cpu()
                support_target = build_leave_one_out_targets(
                    data.noisy_y, support_assignment, indices[0], classes,
                    min_train_per_ball=CONFIG["model"]["min_remaining_votes"],
                    beta=CONFIG["model"]["target_beta"]
                ).soft_target[indices[0]].cpu()

                if "targets" in cache:
                    references = cache["targets"]
                    for arm, generated in (("OLD_LCSS", old_target), ("SUPPORT_LCSS", support_target)):
                        reference = torch.as_tensor(references["OLD64" if arm == "OLD_LCSS" else "SUPPORT"]).float()
                        error = float((generated - reference).abs().max())
                        if error > 3e-7:
                            raise RuntimeError(f"{dataset} seed {seed} {condition}: {arm} target cache mismatch {error}")
                elif "target" in cache:
                    error = float((support_target - cache["target"].float()).abs().max())
                    if error > 3e-7:
                        raise RuntimeError(f"{dataset} seed {seed} {condition}: SUPPORT target cache mismatch {error}")
                else:
                    raise RuntimeError(f"{dataset} seed {seed} {condition}: no cached target reference")

                out = input_root / dataset / f"seed{seed}" / f"{condition}.pt"
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "train_index": indices[0], "valid_index": indices[1], "test_index": indices[2],
                    "observed_train": observed,
                    "targets": {"CE": ce_target, "OLD_LCSS": old_target, "SUPPORT_LCSS": support_target},
                    "split_hash": split_hash, "noise_hash": noise_hash,
                    "old_ball_count": int(np.unique(old_assignment).size),
                    "support_ball_count": int(np.unique(support_assignment).size)
                }, out)
                input_paths.extend((out, support_dir / "geometry.pt", support_dir / f"{condition}.pt", archive_path, metadata_path))
                checks.append({
                    "dataset": dataset, "seed": seed, "condition": condition,
                    "split_hash": split_hash, "noise_hash": noise_hash,
                    "modified_train": int((data.y[indices[0]] != observed).sum()),
                    "modified_valid": 0, "modified_test": 0,
                    "old_ball_count": int(np.unique(old_assignment).size),
                    "support_ball_count": int(np.unique(support_assignment).size),
                    "old_target_hash": hashlib.sha256(old_target.numpy().tobytes()).hexdigest(),
                    "support_target_hash": hashlib.sha256(support_target.numpy().tobytes()).hexdigest(),
                    "target_cache_max_error": 0.0
                })
    csv_write(ROOT / "preflight.csv", checks)
    dependencies = [
        ROOT / "run.py", ROOT / "analyze.py", ROOT / "PLAN.md", ROOT / "config.json",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "data.py",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "models.py",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "targets.py",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "training.py",
        COMMON / "common.py",
        WORKSPACE / "experiments" / "robust_graph_seed0_20261006" / "run_screen.py",
        WORKSPACE / "experiments" / "neighbor_support_ablation_20261022" / "run.py",
        WORKSPACE / "experiments" / "r2_support_5dataset_20261022" / "run.py",
    ]
    dependencies.extend(DATA_FILES.values())
    dependencies.extend(input_paths)
    sources = {str(path): digest(path) for path in sorted(set(Path(p).resolve() for p in dependencies))}
    write_json(ROOT / "environment.json", {
        "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
        "hip": torch.version.hip, "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "threads": CONFIG["model"]["threads"]
    })
    write_json(ROOT / "frozen_manifest.json", {
        "stage": "inputs and sources frozen before teacher fitting",
        "expected_fits": len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["conditions"]) * len(CONFIG["arms"]),
        "source_sha256": sources,
        "config": CONFIG
    })
    print(f"PREPARED {len(checks)} protocol cells; target references verified; {len(sources)} files hashed", flush=True)


def verify_frozen():
    manifest_path = ROOT / "frozen_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Run --prepare before fitting")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG:
        raise RuntimeError("Configuration changed after freeze")
    for path, expected in manifest["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen source/input changed: {path}")


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, classes: int) -> float:
    encoded = y_true.long() * classes + y_pred.long()
    confusion = torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes).double()
    tp = confusion.diag()
    denom = 2 * tp + (confusion.sum(0) - tp) + (confusion.sum(1) - tp)
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom))
    return float(f1.mean().item())


def train_one(dataset: str, seed: int, condition: str, arm: str, epochs: int | None = None):
    device = torch.device("cuda")
    raw = raw_data(dataset)
    cache_path = ROOT / "inputs" / dataset / f"seed{seed}" / f"{condition}.pt"
    pack = torch.load(cache_path, map_location="cpu", weights_only=False)
    x = raw.x.float().to(device)
    edge_index = raw.edge_index.long().to(device)
    n = int(x.shape[0])
    adjacency = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.shape[1], dtype=x.dtype, device=device), (n, n)
    ).coalesce()
    train = pack["train_index"].to(device)
    valid = pack["valid_index"].to(device)
    test = pack["test_index"].to(device)
    clean = raw.y.long().to(device)
    observed_train = pack["observed_train"].to(device)
    target = pack["targets"][arm].float().to(device)
    if target.shape != (len(train), int(clean.max().item()) + 1):
        raise RuntimeError(f"Bad target shape for {dataset}/{seed}/{condition}/{arm}")

    set_seed(seed)
    # Match the historical teacher initialization sequence used by the reference screen.
    MLP(x.shape[1], CONFIG["model"]["hidden"], int(clean.max().item()) + 1,
        CONFIG["model"]["layers"], CONFIG["model"]["dropout"]).to(device)
    model = GCN(x.shape[1], CONFIG["model"]["hidden"], int(clean.max().item()) + 1,
                CONFIG["model"]["layers"], CONFIG["model"]["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["model"]["learning_rate"],
                                 weight_decay=CONFIG["model"]["weight_decay"])
    best_valid, best_epoch, best_state = -1.0, -1, None
    epoch_budget = int(epochs if epochs is not None else CONFIG["model"]["epochs"])
    run_dir = ROOT / ("smoke" if epochs is not None else "runs") / dataset / f"seed{seed}" / condition / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log_path = run_dir / "epochs.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epoch_budget + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, adjacency)
            loss = soft_cross_entropy(logits[train], target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {dataset}/{seed}/{condition}/{arm}")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_logits = model(x, adjacency)[valid]
                valid_acc = float((val_logits.argmax(1) == clean[valid]).float().mean().item())
            log.write(json.dumps({"epoch": epoch, "loss": float(loss.item()), "valid_accuracy": valid_acc}) + "\n")
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 25 == 0 or epoch == epoch_budget:
                print(f"{dataset} seed={seed} {condition} {arm} epoch={epoch}/{epoch_budget} val={valid_acc:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(x, adjacency).softmax(1)
        pred = probabilities.argmax(1)
        valid_acc = float((pred[valid] == clean[valid]).float().mean().item())
        test_acc = float((pred[test] == clean[test]).float().mean().item())
        train_observed_acc = float((pred[train] == observed_train).float().mean().item())
        test_f1 = macro_f1(clean[test], pred[test], int(clean.max().item()) + 1)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result = {
        "dataset": dataset, "seed": seed, "condition": condition, "arm": arm,
        "status": "complete", "best_epoch": best_epoch, "best_valid_accuracy": valid_acc,
        "test_accuracy": test_acc, "test_macro_f1": test_f1,
        "train_observed_accuracy": train_observed_acc, "epochs": epoch_budget,
        "seconds": seconds, "device": torch.cuda.get_device_name(0),
        "split_hash": pack["split_hash"], "noise_hash": pack["noise_hash"],
        "old_ball_count": pack["old_ball_count"], "support_ball_count": pack["support_ball_count"],
        "classes": int(clean.max().item()) + 1
    }
    torch.save({"state_dict": best_state, "result": result}, run_dir / "teacher.pt")
    torch.save(probabilities.detach().cpu(), run_dir / "probabilities.pt")
    write_json(run_dir / "result.json", result)
    return result


def completed_results():
    path = ROOT / "results.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_grid(smoke: bool = False):
    verify_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU fallback is forbidden")
    torch.set_num_threads(CONFIG["model"]["threads"])
    prior = completed_results() if not smoke else []
    prior_keys = {(r["dataset"], int(r["seed"]), r["condition"], r["arm"]): r for r in prior if r["status"] == "complete"}
    results = list(prior)
    jobs = [(d, s, c, a) for d in CONFIG["datasets"] for s in CONFIG["seeds"]
            for c in CONFIG["conditions"] for a in CONFIG["arms"]]
    if smoke:
        jobs = [("cora", 0, "uniform0.4", arm) for arm in CONFIG["arms"]]
    for index, (dataset, seed, condition, arm) in enumerate(jobs, start=1):
        key = (dataset, seed, condition, arm)
        if key in prior_keys:
            continue
        print(f"START {index}/{len(jobs)} {dataset} seed={seed} {condition} {arm}", flush=True)
        result = train_one(dataset, seed, condition, arm, epochs=3 if smoke else None)
        print(f"DONE {index}/{len(jobs)} test={result['test_accuracy']:.4f} best_epoch={result['best_epoch']} seconds={result['seconds']:.1f}", flush=True)
        if not smoke:
            results = [r for r in results if (r["dataset"], int(r["seed"]), r["condition"], r["arm"]) != key]
            results.append(result)
            csv_write(ROOT / "results.csv", results)
            done = len(prior_keys) + index
            write_json(ROOT / "progress.json", {"completed_fits": len(results), "expected_fits": len(jobs), "last_completed": result})
    if smoke:
        write_json(ROOT / "smoke" / "smoke_summary.json", {"status": "complete", "fits": len(jobs), "results": [json.loads((ROOT / "smoke" / "cora" / "seed0" / "uniform0.4" / arm / "result.json").read_text(encoding="utf-8")) for arm in CONFIG["arms"]]})
    else:
        write_json(ROOT / "progress.json", {"completed_fits": len(results), "expected_fits": len(jobs), "status": "complete"})
        print(f"GRID COMPLETE: {len(results)}/{len(jobs)} fits", flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.smoke:
        run_grid(smoke=True)
    else:
        run_grid(smoke=False)


if __name__ == "__main__":
    main()
