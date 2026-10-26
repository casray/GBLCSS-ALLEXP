from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
TEACHER_ROOT = WORKSPACE / "experiments" / "teacher_support_lcss_20261022"
COMMON = WORKSPACE / "experiments" / "new_graph_mlp_seed0_20261009"
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub" / "src"))
sys.path.insert(0, str(COMMON))

from common import DATA_FILES, raw_data  # noqa: E402
from gblcss.data import percent_split_indices, set_seed  # noqa: E402
from gblcss.models import MLP  # noqa: E402
from gblcss.training import soft_cross_entropy  # noqa: E402

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def rng_digest() -> str:
    h = hashlib.sha256()
    h.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def expected_keys() -> set[tuple[str, int, str, str]]:
    return {
        (dataset, seed, condition, arm)
        for dataset in CONFIG["datasets"]
        for seed in CONFIG["seeds"]
        for condition in CONFIG["conditions"]
        for arm in CONFIG["teacher_arms"]
    }


def teacher_cell(dataset: str, seed: int, condition: str, arm: str) -> Path:
    return TEACHER_ROOT / "runs" / dataset / f"seed{seed}" / condition / arm


def protocol_cell(dataset: str, seed: int, condition: str) -> Path:
    return TEACHER_ROOT / "inputs" / dataset / f"seed{seed}" / f"{condition}.pt"


def verify_probabilities(dataset: str, probability_path: Path, y: torch.Tensor, test: torch.Tensor,
                         expected_accuracy: float) -> tuple[torch.Tensor, dict]:
    probabilities = torch.load(probability_path, map_location="cpu", weights_only=True).float()
    classes = int(y.max().item()) + 1
    if probabilities.shape != (y.numel(), classes):
        raise RuntimeError(f"{dataset}: bad teacher probability shape {tuple(probabilities.shape)}")
    if not torch.isfinite(probabilities).all():
        raise RuntimeError(f"{dataset}: non-finite teacher probabilities in {probability_path}")
    row_error = float((probabilities.sum(1) - 1).abs().max().item())
    if row_error > 1e-5:
        raise RuntimeError(f"{dataset}: teacher probabilities do not sum to one; max error={row_error}")
    accuracy = float((probabilities[test].argmax(1) == y[test]).float().mean().item())
    if abs(accuracy - expected_accuracy) > 1e-6:
        raise RuntimeError(f"{dataset}: saved teacher metric mismatch {accuracy} vs {expected_accuracy}")
    return probabilities, {"row_sum_max_error": row_error, "test_accuracy_recomputed": accuracy}


def prepare() -> None:
    manifest_path = ROOT / "frozen_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("frozen_manifest.json already exists; refusing to replace a frozen experiment")
    teacher_progress = json.loads((TEACHER_ROOT / "progress.json").read_text(encoding="utf-8"))
    if teacher_progress.get("status") != "complete":
        raise RuntimeError(f"Teacher experiment is not complete: {teacher_progress}")
    teacher_results = read_csv(TEACHER_ROOT / "results.csv")
    teacher_map = {}
    for row in teacher_results:
        key = (row["dataset"], int(row["seed"]), row["condition"], row["arm"])
        if key in teacher_map:
            raise RuntimeError(f"Duplicate teacher result: {key}")
        if row["status"] != "complete":
            raise RuntimeError(f"Incomplete teacher result: {key} {row['status']}")
        teacher_map[key] = row
    if set(teacher_map) != expected_keys():
        missing = sorted(expected_keys() - set(teacher_map))
        extra = sorted(set(teacher_map) - expected_keys())
        raise RuntimeError(f"Teacher result grid mismatch; missing={missing[:5]}, extra={extra[:5]}")

    audit_rows, source_paths = [], []
    for dataset in CONFIG["datasets"]:
        raw = raw_data(dataset)
        y = raw.y.long().cpu()
        n = int(raw.x.shape[0])
        classes = int(y.max().item()) + 1
        for seed in CONFIG["seeds"]:
            for condition in CONFIG["conditions"]:
                cache_path = protocol_cell(dataset, seed, condition)
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                train = cache["train_index"].long().cpu()
                valid = cache["valid_index"].long().cpu()
                test = cache["test_index"].long().cpu()
                expected_split = tuple(torch.as_tensor(index, dtype=torch.long) for index in percent_split_indices(n, seed))
                if any(not torch.equal(actual.cpu(), expected) for actual, expected in zip((train, valid, test), expected_split)):
                    raise RuntimeError(f"{dataset} seed{seed} {condition}: split differs from the historical split function")
                if (torch.isin(train, valid).any() or torch.isin(train, test).any() or torch.isin(valid, test).any()):
                    raise RuntimeError(f"{dataset} seed{seed} {condition}: split indices overlap")
                if cache.get("split_hash") is None or cache.get("noise_hash") is None:
                    raise RuntimeError(f"Missing split/noise fingerprint: {dataset} seed{seed} {condition}")
                if cache["observed_train"].numel() != train.numel():
                    raise RuntimeError(f"Observed-label length mismatch: {dataset} seed{seed} {condition}")
                if not torch.equal(cache["observed_train"].long(), y[train]):
                    changed = int((cache["observed_train"].long() != y[train]).sum().item())
                else:
                    changed = 0
                source_paths.append(cache_path)
                for arm in CONFIG["teacher_arms"]:
                    key = (dataset, seed, condition, arm)
                    output = teacher_cell(*key)
                    probability_path = output / "probabilities.pt"
                    result_path = output / "result.json"
                    if not probability_path.is_file() or not result_path.is_file():
                        raise FileNotFoundError(f"Missing frozen teacher output for {key}")
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("status") != "complete" or result.get("split_hash") != cache["split_hash"] or result.get("noise_hash") != cache["noise_hash"]:
                        raise RuntimeError(f"Teacher protocol mismatch for {key}")
                    probabilities, check = verify_probabilities(
                        dataset, probability_path, y, test, float(result["test_accuracy"])
                    )
                    if probabilities.shape[1] != classes:
                        raise RuntimeError(f"Class count mismatch for {key}")
                    source_paths.extend((probability_path, result_path))
                    audit_rows.append({
                        "dataset": dataset, "seed": seed, "condition": condition, "teacher_arm": arm,
                        "nodes": n, "classes": classes, "train_nodes": train.numel(),
                        "valid_nodes": valid.numel(), "test_nodes": test.numel(),
                        "observed_train_mismatch_count": changed,
                        "split_hash": cache["split_hash"], "noise_hash": cache["noise_hash"],
                        "teacher_test_accuracy": result["test_accuracy"], **check,
                    })
    if len(audit_rows) != len(expected_keys()):
        raise RuntimeError(f"Unexpected preflight count: {len(audit_rows)}")

    source_paths.extend((
        ROOT / "PLAN.md", ROOT / "config.json", ROOT / "run.py", ROOT / "analyze.py",
        TEACHER_ROOT / "config.json", TEACHER_ROOT / "run.py", TEACHER_ROOT / "results.csv",
        TEACHER_ROOT / "progress.json", TEACHER_ROOT / "frozen_manifest.json",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "data.py",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "models.py",
        WORKSPACE / "GB-LCSS_GitHub" / "src" / "gblcss" / "training.py",
        COMMON / "common.py",
    ))
    source_paths.extend(Path(path) for path in DATA_FILES.values())
    resolved_sources = sorted({Path(path).resolve() for path in source_paths})
    missing_files = [str(path) for path in resolved_sources if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing frozen source files: {missing_files[:10]}")
    source_hashes = {str(path): digest(path) for path in resolved_sources}
    write_csv(ROOT / "preflight.csv", audit_rows)
    write_json(ROOT / "environment.json", {
        "python": sys.version, "executable": sys.executable, "torch": torch.__version__,
        "hip": torch.version.hip, "gpu_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "threads": 4,
    })
    write_json(manifest_path, {
        "status": "frozen before student fits", "expected_student_fits": len(expected_keys()),
        "teacher_experiment": str(TEACHER_ROOT), "config": CONFIG,
        "sha256": source_hashes,
    })
    print(f"PREPARED {len(audit_rows)} teacher cells; split/probability checks passed; {len(source_hashes)} files hashed", flush=True)


def verify_frozen() -> None:
    path = ROOT / "frozen_manifest.json"
    if not path.is_file():
        raise RuntimeError("Run --prepare before fitting")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG:
        raise RuntimeError("Configuration changed after freeze")
    for filename, expected in manifest["sha256"].items():
        if digest(Path(filename)) != expected:
            raise RuntimeError(f"Frozen input/source changed: {filename}")


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, classes: int) -> float:
    encoded = y_true.long() * classes + y_pred.long()
    confusion = torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes).double()
    tp = confusion.diag()
    denom = 2 * tp + (confusion.sum(0) - tp) + (confusion.sum(1) - tp)
    return float(torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(denom)).mean().item())


def train_one(dataset: str, seed: int, condition: str, arm: str, raw, smoke: bool = False) -> dict:
    device = torch.device("cuda")
    x = raw.x.float().to(device)
    y = raw.y.long().to(device)
    cache = torch.load(protocol_cell(dataset, seed, condition), map_location="cpu", weights_only=False)
    train = cache["train_index"].long().to(device)
    valid = cache["valid_index"].long().to(device)
    test = cache["test_index"].long().to(device)
    classes = int(y.max().item()) + 1
    teacher_dir = teacher_cell(dataset, seed, condition, arm)
    teacher_path = teacher_dir / "probabilities.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True).float().to(device)
    if teacher.shape != (x.shape[0], classes) or not torch.isfinite(teacher).all():
        raise RuntimeError(f"Invalid teacher probabilities for {dataset}/seed{seed}/{condition}/{arm}")

    set_seed(seed)
    student = MLP(x.shape[1], 64, classes, 2, 0.5).to(device)
    init_hash = tensor_digest(student.state_dict())
    init_rng_hash = rng_digest()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01, weight_decay=0.0005)
    best_valid, best_epoch, best_state = -1.0, -1, None
    epochs = 3 if smoke else int(CONFIG["student"]["epochs"])
    run_root = ROOT / ("smoke" if smoke else "runs")
    folder = run_root / dataset / f"seed{seed}" / condition / arm
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing fit: {folder}")
    result = {
        "dataset": dataset, "seed": seed, "condition": condition, "teacher_arm": arm,
        "status": "running", "student_seed": seed, "epochs": epochs,
        "teacher_probability_sha256": digest(teacher_path), "teacher_test_accuracy": float(
            json.loads((teacher_dir / "result.json").read_text(encoding="utf-8"))["test_accuracy"]
        ),
        "initial_state_sha256": init_hash, "initial_rng_sha256": init_rng_hash,
        "split_hash": cache["split_hash"], "noise_hash": cache["noise_hash"],
        "device": torch.cuda.get_device_name(0), "smoke": smoke,
    }
    write_json(result_path, result)
    started = time.perf_counter()
    try:
        with (folder / "epochs.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                student.train()
                optimizer.zero_grad(set_to_none=True)
                logits = student(x)
                loss = soft_cross_entropy(logits, teacher)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite student loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                student.eval()
                with torch.no_grad():
                    valid_prediction = student(x)[valid].argmax(1)
                    valid_accuracy = float((valid_prediction == y[valid]).float().mean().item())
                record = {"epoch": epoch, "loss": float(loss.item()), "valid_accuracy": valid_accuracy}
                log.write(json.dumps(record) + "\n")
                log.flush()
                if valid_accuracy > best_valid:
                    best_valid, best_epoch = valid_accuracy, epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
                if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                    print(f"{dataset} seed{seed} {condition} {arm} epoch {epoch}/{epochs} val={valid_accuracy:.4f}", flush=True)
        if best_state is None:
            raise RuntimeError("No student checkpoint selected")
        student.load_state_dict(best_state)
        student.eval()
        with torch.no_grad():
            probabilities = student(x).softmax(1)
            prediction = probabilities.argmax(1)
            selected_valid = float((prediction[valid] == y[valid]).float().mean().item())
            test_accuracy = float((prediction[test] == y[test]).float().mean().item())
            test_macro_f1 = macro_f1(y[test], prediction[test], classes)
        torch.cuda.synchronize()
        result.update({
            "status": "complete", "best_epoch": best_epoch,
            "best_valid_accuracy": selected_valid, "test_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1, "seconds": time.perf_counter() - started,
            "max_probability_row_sum_error": float((probabilities.sum(1) - 1).abs().max().item()),
        })
        torch.save({"state_dict": best_state, "result": result}, folder / "student.pt")
        torch.save(probabilities.detach().cpu(), folder / "probabilities.pt")
        write_json(result_path, result)
        return result
    except Exception as error:
        result.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                       "seconds": time.perf_counter() - started})
        write_json(result_path, result)
        raise


def key_for(row: dict) -> tuple[str, int, str, str]:
    return row["dataset"], int(row["seed"]), row["condition"], row["teacher_arm"]


def run_grid(smoke: bool = False) -> None:
    verify_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; CPU fallback is disabled")
    torch.set_num_threads(4)
    expected = expected_keys()
    rows = [] if smoke else read_csv(ROOT / "results.csv")
    completed = {key_for(row): row for row in rows if row.get("status") == "complete"}
    if len(completed) != len([row for row in rows if row.get("status") == "complete"]):
        raise RuntimeError("Duplicate completed fit keys in results.csv")
    jobs = [(d, s, c, a) for d in CONFIG["datasets"] for s in CONFIG["seeds"]
            for c in CONFIG["conditions"] for a in CONFIG["teacher_arms"]]
    if smoke:
        jobs = [("cora", 0, "uniform0.4", arm) for arm in CONFIG["teacher_arms"]]

    raw = None
    active_dataset = None
    for number, (dataset, seed, condition, arm) in enumerate(jobs, start=1):
        key = (dataset, seed, condition, arm)
        if not smoke and key in completed:
            continue
        output = (ROOT / ("smoke" if smoke else "runs") / dataset / f"seed{seed}" / condition / arm)
        if not smoke and output.exists():
            result_path = output / "result.json"
            if not result_path.is_file():
                raise RuntimeError(f"Incomplete output directory requires inspection; not retrying: {output}")
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("status") != "complete":
                raise RuntimeError(f"Previous fit is not complete; inspect before retrying: {output}")
            rows.append(saved)
            completed[key] = saved
            write_csv(ROOT / "results.csv", rows)
            continue
        if dataset != active_dataset:
            raw = raw_data(dataset)
            active_dataset = dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"START {number}/{len(jobs)} {dataset} seed{seed} {condition} {arm}", flush=True)
        result = train_one(dataset, seed, condition, arm, raw, smoke=smoke)
        print(f"DONE {number}/{len(jobs)} test={result['test_accuracy']:.4f} best_epoch={result['best_epoch']} seconds={result['seconds']:.1f}", flush=True)
        if not smoke:
            rows = [row for row in rows if key_for(row) != key]
            rows.append(result)
            completed[key] = result
            write_csv(ROOT / "results.csv", rows)
            write_json(ROOT / "progress.json", {"completed_fits": len(completed), "expected_fits": len(expected),
                                                  "last_completed": key})
        del result
    if smoke:
        print("SMOKE COMPLETE: three paired teacher arms", flush=True)
        return
    if set(completed) != expected:
        raise RuntimeError(f"Grid incomplete: {len(completed)}/{len(expected)}")
    write_json(ROOT / "progress.json", {"completed_fits": len(completed), "expected_fits": len(expected), "status": "complete"})
    print(f"GRID COMPLETE: {len(completed)}/{len(expected)} student fits", flush=True)


def main() -> None:
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
