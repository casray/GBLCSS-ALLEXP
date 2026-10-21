from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import torch
import torch_geometric

from common import DATASETS, ROOT, final_metrics, load_data, set_seed, write_json

METHODS = ["gssc", "tined", "dgd"]


def source_versions():
    versions = {}
    for method in METHODS:
        repo = ROOT / "vendor" / method
        versions[method] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return versions


def fit(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU fallback is disabled")
    torch.set_num_threads(4)
    mode = "smoke" if args.smoke else "runs"
    outdir = ROOT / mode / args.dataset / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "result.json").exists():
        print(f"reuse completed {outdir}", flush=True)
        return
    data, audit = load_data(args.dataset)
    write_json(outdir / "protocol.json", audit)
    config = dict(seed=0, epochs=2 if args.smoke else 200,
                  hidden=64, dropout=.5, learning_rate=.01, weight_decay=.0005,
                  noise="uniform0.4", validation_selection="clean validation accuracy; last tie",
                  device="cuda", smoke=args.smoke)
    write_json(outdir / "requested_config.json", config)
    write_json(outdir / "environment.json", dict(python=sys.executable,
               torch=torch.__version__, pyg=torch_geometric.__version__,
               gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
               author_commits=source_versions(), cpu_fallback=False))
    write_json(outdir / "adapter_provenance.json", {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in [f"{args.method}_adapter.py", "common.py", "run_screen.py"]})
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        set_seed(0)
        adapter = importlib.import_module(f"{args.method}_adapter")
        result = adapter.run(data, config, outdir)
        torch.cuda.synchronize()
        probabilities = torch.load(outdir / "probabilities.pt", weights_only=False, map_location="cpu")
        if isinstance(probabilities, dict):
            probabilities = probabilities["probabilities"]
        metrics = final_metrics(args.dataset, probabilities)
        if abs(metrics["valid"] - result["best_val"]) > 1e-6:
            raise ValueError("saved probability validation accuracy does not match selected checkpoint")
        result.update(audit)
        result.update(status="ok", method=args.method, smoke=args.smoke,
                      test=metrics["test"], valid=metrics["valid"],
                      wall_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(outdir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        write_json(outdir / "failure.json", dict(method=args.method, dataset=args.dataset,
                   status="failed", error=traceback.format_exc(), **config))
        raise


def aggregate():
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            folder = ROOT / "runs" / dataset / method
            if (folder / "result.json").exists():
                value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                rows.append({key: value.get(key) for key in ["dataset", "method", "status", "seed", "valid", "test", "best_epoch", "parameters", "wall_seconds", "peak_allocated_gib", "split_hash", "noise_hash"]})
            elif (folder / "blocked.json").exists():
                rows.append(dict(dataset=dataset, method=method, status="blocked"))
            else:
                rows.append(dict(dataset=dataset, method=method,
                            status="failed" if (folder / "failure.json").exists() else "pending"))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="cora")
    parser.add_argument("--method", choices=METHODS, default="gssc")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate(), indent=2))
    elif args.grid:
        for method in args.methods:
            for dataset in args.datasets:
                folder = ROOT / "runs" / dataset / method
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(ROOT / "run_screen.py"), "--method", method, "--dataset", dataset]
                print("RUN", " ".join(command), flush=True)
                with (folder / "console.log").open("a", encoding="utf-8") as handle:
                    completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
                print(dataset, method, "exit", completed.returncode, flush=True)
                aggregate()
    else:
        fit(args)


if __name__ == "__main__":
    main()
