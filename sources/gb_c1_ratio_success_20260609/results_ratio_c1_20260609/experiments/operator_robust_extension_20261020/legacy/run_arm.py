"""One process, one arm: original-node validation, explicit costs and provenance."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score, recall_score
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from granularity import select_cut
from mechanisms import build_targets, random_size_matched_assignment
from runtime import Checkpoints, Stage, array_hash, atomic_json, sha256_file, validation_due

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub" / "src"))
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed


class GCN(nn.Module):
    def __init__(self, din, hidden, classes, dropout):
        super().__init__()
        self.l1, self.l2 = nn.Linear(din, hidden), nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x, p):
        z = F.relu(torch.sparse.mm(p, self.l1(x)))
        z = F.dropout(z, self.dropout, self.training)
        return torch.sparse.mm(p, self.l2(z))


class CachedMLP(nn.Module):
    def __init__(self, din, hidden, classes, dropout):
        super().__init__()
        self.l1, self.l2 = nn.Linear(din, hidden), nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x):
        z = F.dropout(F.relu(self.l1(x)), self.dropout, self.training)
        return self.l2(z)


def sparse_tensor(matrix, device):
    coo = matrix.tocoo()
    ij = torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64))
    return torch.sparse_coo_tensor(ij, torch.from_numpy(coo.data), coo.shape).coalesce().to(device)


def protocol_labels(y, seed, noise, classes):
    n = len(y)
    tr, va, te = percent_split_indices(n, seed)
    masks = []
    for ids in (tr, va, te):
        mask = np.zeros(n, dtype=bool)
        mask[ids] = True
        masks.append(torch.from_numpy(mask))
    data = GraphData("protocol", torch.empty(n, 0), torch.empty(2, 0, dtype=torch.long),
                     torch.from_numpy(y.copy()), *masks, train_index=torch.from_numpy(tr),
                     val_index=torch.from_numpy(va), test_index=torch.from_numpy(te))
    kind, rate = ("uniform", 0.) if noise == "clean" else (("uniform", float(noise[1:])) if noise.startswith("u") else ("pair", float(noise[4:])))
    inject_train_label_noise(data, kind, rate, seed)
    assert torch.equal(data.noisy_y[data.val_mask], data.y[data.val_mask])
    assert torch.equal(data.noisy_y[data.test_mask], data.y[data.test_mask])
    # Downstream target/score routines receive no held-out label values.
    permitted = np.zeros(n, dtype=np.int64)
    permitted[tr] = data.noisy_y.numpy()[tr]
    return tr, va, te, permitted


def assignment_for(job, folder, y, permitted, train, config, times):
    n = len(y)
    if job["arm"] == "A":
        return np.arange(n), {"actual_M": n, "cut_cold_s": 0., "rule": "none"}
    budget = round(job["ratio"] * n)
    rule = job["rule"]
    name = f"{rule}_M{budget}" + (f"_s{job['seed']}_{job['noise']}" if rule != "R0" else "")
    prefix = folder / "cuts" / name
    prefix.parent.mkdir(exist_ok=True)
    if prefix.with_suffix(".json").exists():
        info = json.loads(prefix.with_suffix(".json").read_text())
        assignment = np.load(prefix.with_suffix(".npy"))
        if array_hash(assignment) != info.get("assignment_sha256"):
            raise RuntimeError("Cached assignment hash mismatch")
        if rule != "R0":
            expected = y[train] if rule == "O2" else permitted[train]
            if array_hash(expected) != info.get("score_training_label_hash"):
                raise RuntimeError("Cached cut label source mismatch")
    else:
        with Stage(times, "cut_select_s"):
            with open(folder / "tree.pkl", "rb") as f:
                tree = pickle.load(f)
            mask = np.zeros(n, dtype=bool)
            mask[train] = True
            allowed = permitted.copy()
            if rule == "O2":
                allowed[train] = y[train]
            assignment, info = select_cut(tree, budget, allowed, mask, int(y.max()) + 1,
                                          rule=rule, beta=config["beta_cut"])
        info["cut_cold_s"] = times["cut_select_s"]
        info["rule"] = rule
        info["score_training_label_hash"] = array_hash(allowed[train]) if rule != "R0" else None
        info["assignment_sha256"] = array_hash(assignment)
        np.save(prefix.with_suffix(".npy"), assignment)
        atomic_json(prefix.with_suffix(".json"), info)
    if job["arm"] == "D":
        original_sizes = np.bincount(assignment)
        assignment = random_size_matched_assignment(assignment, 900000 + job["seed"])
        assert np.array_equal(np.bincount(assignment), original_sizes)
    if int(assignment.max()) + 1 != budget:
        raise RuntimeError("Actual budget does not match the paired design")
    return assignment, info


def run(config_path, job_path, root):
    start = time.perf_counter()
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    job = json.loads(Path(job_path).read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-compatible PyTorch GPU interface is required")
    device = torch.device("cuda")
    sync = torch.cuda.synchronize
    process = psutil.Process()
    resource = {"rss_peak_bytes": process.memory_info().rss}
    stop_sample = threading.Event()

    def sample():
        while not stop_sample.wait(.1):
            resource["rss_peak_bytes"] = max(resource["rss_peak_bytes"], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    times = {}
    with Stage(times, "load_cache_s"):
        folder = root / "cache" / job["dataset"]
        meta = json.loads((folder / "metadata.json").read_text())
        if meta["config_sha256"] != sha256_file(config_path):
            raise RuntimeError("Prepared cache configuration mismatch")
        for name, expected in meta["cache_files"].items():
            if (folder / name).stat().st_size != expected["bytes"]:
                raise RuntimeError(f"Cache size mismatch: {name}")
        y = np.load(folder / "y.npy")
        x = np.load(folder / "x.npy", mmap_mode="r")
        features = np.load(folder / "features.npy", mmap_mode="r") if job["backbone"] == "cached_mlp" else None
        p = sp.load_npz(folder / "p.npz") if job["backbone"] == "gcn" else None
        tr, va, te, permitted = protocol_labels(y, job["seed"], job["noise"], meta["C"])
    with Stage(times, "cut_and_targets_s"):
        assignment, cut_info = assignment_for(job, folder, y, permitted, tr, config, times)
        target_mode = job["arm"] if job["arm"] in ("A", "B", "Bprime", "LCSS") else "B"
        targets, q, support, target_info = build_targets(permitted, assignment, tr, meta["C"], target_mode,
                                                        target_beta=config["beta_target"])
    coarse = job["arm"] in ("C", "D")
    torch.cuda.reset_peak_memory_stats()
    free_before, memory_total = torch.cuda.mem_get_info()
    with Stage(times, "materialize_s", sync):
        if job["backbone"] == "gcn":
            p_full = sparse_tensor(p, device)
            x_full = torch.from_numpy(np.array(x)).to(device)
        m = len(support)
        selected = np.flatnonzero(support > 0)
        if coarse:
            sizes = np.bincount(assignment, minlength=m)
            h = sp.csr_matrix((np.ones(len(y), np.float32), (np.arange(len(y)), assignment)), shape=(len(y), m))
            r = (sp.diags(1 / sizes) @ h.T).tocsr().astype(np.float32)
            if job["backbone"] == "gcn":
                pc = (r @ p @ h).tocsr().astype(np.float32)
                fit_p = sparse_tensor(pc, device)
                fit_x = torch.from_numpy((r @ x).astype(np.float32)).to(device)
                fit_idx = torch.from_numpy(selected).to(device)
                fit_y = torch.from_numpy(q[selected]).to(device)
            else:
                coarse_features = (r @ features).astype(np.float32)
                fit_x = torch.from_numpy(coarse_features[selected]).to(device)
                fit_y = torch.from_numpy(q[selected]).to(device)
            weights = torch.from_numpy((support[selected] / support.sum()).astype(np.float32)).to(device)
        else:
            if job["backbone"] == "gcn":
                fit_x, fit_p = x_full, p_full
                fit_idx = torch.from_numpy(tr).to(device)
            else:
                fit_x = torch.from_numpy(np.array(features[tr])).to(device)
            fit_y = torch.from_numpy(targets).to(device)
            weights = torch.full((len(tr),), 1 / len(tr), device=device)
        din = fit_x.shape[1]
        set_seed(job["seed"])
        cls = GCN if job["backbone"] == "gcn" else CachedMLP
        model = cls(din, config["hidden"], meta["C"], config["dropout"]).to(device)

    def fit_logits():
        return model(fit_x, fit_p)[fit_idx] if job["backbone"] == "gcn" else model(fit_x)

    def original_predictions(indices):
        model.eval()
        with torch.no_grad():
            if job["backbone"] == "gcn":
                logits = model(x_full, p_full)
                return logits[torch.as_tensor(indices, device=device)].argmax(1).cpu().numpy()
            output = []
            for lo in range(0, len(indices), config["eval_batch_size"]):
                batch = torch.from_numpy(np.array(features[indices[lo:lo + config["eval_batch_size"]]])).to(device)
                output.append(model(batch).argmax(1).cpu().numpy())
            return np.concatenate(output)

    def lifted_predictions(indices):
        model.eval()
        with torch.no_grad():
            if job["backbone"] == "gcn":
                predictions = model(fit_x, fit_p).argmax(1).cpu().numpy()
            else:
                predictions = np.empty(m, dtype=np.int64)
                for lo in range(0, m, config["eval_batch_size"]):
                    batch = torch.from_numpy(coarse_features[lo:lo + config["eval_batch_size"]]).to(device)
                    predictions[lo:lo + len(batch)] = model(batch).argmax(1).cpu().numpy()
            return predictions[assignment[indices]]

    with Stage(times, "warmup_s", sync):
        model.train()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            (-(fit_y * F.log_softmax(fit_logits(), dim=1)).sum(1) * weights).sum().backward()
        model.zero_grad(set_to_none=True)
    # Identical initial model for paired arms; warmup must not advance its RNG.
    set_seed(job["seed"])
    del model
    model = cls(din, config["hidden"], meta["C"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    bank = Checkpoints(config["patience_checks"])
    history = []
    with Stage(times, "training_s", sync):
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = fit_logits()
            loss = (-(fit_y * F.log_softmax(logits, dim=1)).sum(1) * weights).sum()
            loss.backward()
            optimizer.step()
            if validation_due(epoch, config["epochs"], config["eval_every"]):
                with Stage(times, "validation_s", sync):
                    primary = float(np.mean(original_predictions(va) == y[va]))
                    secondary = float(np.mean(lifted_predictions(va) == y[va])) if coarse else None
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bank.observe(epoch, primary, secondary, state)
                    history.append({"epoch": epoch, "primary_val": primary, "lift_val": secondary,
                                    "train_loss": float(loss.detach().cpu())})
                if bank.stop:
                    break
    train_peak = torch.cuda.max_memory_allocated() / 2**30
    train_reserved = torch.cuda.max_memory_reserved() / 2**30
    with Stage(times, "transition_cleanup_s", sync):
        del logits, loss, optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    model.load_state_dict(bank.primary_state)
    torch.cuda.reset_peak_memory_stats()
    with Stage(times, "primary_inference_s", sync):
        prediction = original_predictions(te)
    primary_peak = torch.cuda.max_memory_allocated() / 2**30
    lift_accuracy = None
    lift_prediction = None
    if coarse:
        model.load_state_dict(bank.secondary_state)
        with Stage(times, "secondary_inference_s", sync):
            lift_prediction = lifted_predictions(te)
            lift_accuracy = float(np.mean(lift_prediction == y[te]))
    infer_peak = torch.cuda.max_memory_allocated() / 2**30
    free_after, _ = torch.cuda.mem_get_info()
    stop_sample.set()
    sampler.join(timeout=1)
    elapsed = time.perf_counter() - start
    shared = (meta["times"]["load_data_s"] + meta["times"]["adjacency_s"]
              + meta.get("preparation_common_overhead_s", meta["times"].get("provenance_s", 0.)))
    if job["backbone"] == "cached_mlp":
        shared += meta["times"]["cached_features_s"]
    if job["arm"] != "A":
        shared += meta["times"]["coordinates_s"] + meta["times"]["tree_s"]
    cut_credit = max(0., cut_info.get("cut_cold_s", 0.) - times.get("cut_select_s", 0.))
    cold_total = elapsed + shared + cut_credit
    actual_noisy_rate = float(np.mean(permitted[tr] != y[tr]))
    recalls = recall_score(y[te], prediction, labels=np.arange(meta["C"]), average=None, zero_division=0)
    clean_train_onehot = np.eye(meta["C"], dtype=np.float32)[y[tr]]
    target_brier_clean = float(np.square(targets - clean_train_onehot).sum(1).mean())
    clean_hist = np.zeros((m, meta["C"]), dtype=np.float64)
    np.add.at(clean_hist, (assignment[tr], y[tr]), 1)
    clean_q = np.divide(clean_hist, support[:, None], out=np.zeros_like(clean_hist), where=support[:, None] > 0)
    clean_ball_gini = float(np.sum(support * (1 - np.square(clean_q).sum(1))) / len(tr))
    result = {
        "status": "completed", "job": job, "config_sha256": sha256_file(config_path),
        "source_sha256": protocol["source_sha256"], "source_hashes": protocol.get("source_hashes"),
        "test_accuracy": float(np.mean(prediction == y[te])),
        "macro_f1": float(f1_score(y[te], prediction, labels=np.arange(meta["C"]), average="macro", zero_division=0)),
        "worst_class_recall": float(recalls.min()), "per_class_recall": recalls.tolist(),
        "secondary_lift_test_accuracy": lift_accuracy,
        "best_primary_val": bank.primary, "best_primary_epoch": bank.primary_epoch,
        "best_secondary_val": bank.secondary if coarse else None, "best_secondary_epoch": bank.secondary_epoch,
        "epochs": epoch, "validation_history": history, "times": times,
        "run_total_s": elapsed, "cold_total_s": cold_total, "shared_cold_s": shared,
        "cold_cost_scope": "Reconstructed from required preparation stages plus common process/provenance overhead and the complete worker process wall time; raw preparation wall time also retained.",
        "preparation_process_wall_s": meta.get("preparation_process_wall_s"),
        "preparation_host_peak_gib": meta.get("preparation_host_peak_gib"),
        "cut_cache_credit_s": cut_credit,
        "gpu_peak_allocated_gib": max(train_peak, infer_peak),
        "gpu_training_peak_gib": train_peak, "gpu_primary_inference_peak_gib": primary_peak,
        "gpu_inference_peak_gib": infer_peak, "gpu_training_reserved_gib": train_reserved,
        "gpu_free_before_gib": free_before / 2**30, "gpu_free_after_gib": free_after / 2**30,
        "gpu_total_gib": memory_total / 2**30, "gpu": torch.cuda.get_device_name(0),
        "host_rss_peak_gib": resource["rss_peak_bytes"] / 2**30,
        "N": len(y), "C": meta["C"], "L": len(tr), "M": m,
        "M_plus": int((support > 0).sum()), "actual_ratio": m / len(y),
        "coarse_edges": int(pc.nnz) if coarse and job["backbone"] == "gcn" else None,
        "actual_noise_rate": actual_noisy_rate, "split_sha256": array_hash(tr, va, te),
        "noise_sha256": array_hash(permitted[tr]), "assignment_sha256": array_hash(assignment),
        "source_data_sha256": meta["source_sha256"], "target_diagnostics": target_info,
        "target_brier_clean_diagnostic": target_brier_clean,
        "clean_training_ball_gini_diagnostic": clean_ball_gini,
        "cut_diagnostics": cut_info, "torch_version": torch.__version__,
        "interpretation": "Screening run; primary checkpoint uses original-node validation. No test-based selection.",
    }
    prediction_path = root / "results" / f"{job['id']}_predictions.npz"
    payload = {"indices": te, "prediction": prediction}
    if lift_prediction is not None:
        payload["secondary_lift_prediction"] = lift_prediction
    np.savez(prediction_path, **payload)
    result["prediction_file"] = prediction_path.name
    result["prediction_sha256"] = sha256_file(prediction_path)
    atomic_json(root / "results" / f"{job['id']}.json", result)
    print(json.dumps({"id": job["id"], "accuracy": result["test_accuracy"], "cold_total_s": cold_total,
                      "M": m, "epochs": epoch}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    Path(args.run_root, "results").mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(args.config, args.job, args.run_root)
