"""Additive audit of completed frozen-target runs on an explicit device.

All generated files live under audit/. Existing runs and their source files are
read only. Deep reconstruction defaults to protocol seed 0 for each dataset and
representation. Other completed seeds receive an artifact/protocol census.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import diagnostic_v2 as d
from gblcss.targets import build_leave_one_out_targets

OUT = ROOT / "audit"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=2e-6):
    require(actual.shape == expected.shape, message + ": shape")
    require(np.allclose(actual, expected, atol=tolerance, rtol=0, equal_nan=True), message)


def reference_vote(w, labels, classes):
    """Independent vectorized formula; zero support falls back to uniform q."""
    n = len(labels)
    counts = np.diff(w.indptr)
    totals = np.asarray(w.sum(axis=1)).ravel()
    hist = np.column_stack([np.asarray(w @ (labels == c)).ravel() for c in range(classes)])
    r = np.full((n, classes), 1.0 / classes)
    np.divide(hist, totals[:, None], out=r, where=totals[:, None] > 0)
    max_r = r.max(axis=1)
    alpha = np.clip((max_r - 1.0 / classes) / (1.0 - 1.0 / classes), 0, 1)
    alpha[counts < d.CONFIG["min_votes"]] = 0
    tied = np.isclose(r, max_r[:, None], rtol=0, atol=1e-12).sum(axis=1) > 1
    q = r.copy()
    q[~tied] *= 1 - d.CONFIG["beta"]
    q[np.flatnonzero(~tied), r[~tied].argmax(axis=1)] += d.CONFIG["beta"]
    square = np.asarray(w.power(2).sum(axis=1)).ravel()
    eff = np.divide(totals ** 2, square, out=np.zeros_like(totals), where=square > 0)
    return r, q, alpha, counts, eff


def reconstruction(folder, h, train, device):
    """Reconstruct from frozen assignments, never refit partitions."""
    relations = {}
    with np.load(folder / "assignments.npz", allow_pickle=False) as assignments:
        for budget in (64, 128):
            for method in ("GB", "KM"):
                arm = method + str(budget)
                relations[arm] = d.group_weights(assignments[arm][train])
            counts = np.diff(relations["GB" + str(budget)].indptr)
            relations["KNN" + str(budget)] = d.knn_weights(h[train], counts, device=device)
        relations["GB128_weighted"] = d.group_weights(assignments["GB128"][train], h[train])
    original = read(folder / "relations_audit.json")
    out = OUT / folder.relative_to(ROOT / "runs")
    out.mkdir(parents=True, exist_ok=True)
    for arm, w in relations.items():
        require(not np.any(w.diagonal()), arm + ": self vote")
        require(w.nnz == original[arm]["nonzeros"], arm + ": nonzero count changed")
        require(d.fingerprint(w.indptr, w.indices, w.data) == original[arm]["relation_fingerprint"],
                arm + ": reconstructed relation hash differs (CPU/CUDA rounding is possible for kNN)")
        sparse.save_npz(out / (arm + "_relations.npz"), w)
    for budget in (64, 128):
        require(np.array_equal(np.diff(relations["GB" + str(budget)].indptr),
                               np.diff(relations["KNN" + str(budget)].indptr)),
                "kNN unique support mismatch")
    return relations


def audit_run(folder, deep, device):
    seed = int(folder.name.removeprefix("seed"))
    representation = folder.parent.name
    name = folder.parent.parent.name
    output = OUT / folder.relative_to(ROOT / "runs")
    output.mkdir(parents=True, exist_ok=True)
    graph = d.load_graph(name, seed)
    train = graph.train_index.numpy()
    with np.load(folder / "split.npz", allow_pickle=False) as split:
        for key, expected in (("train", train), ("val", graph.val_index.numpy()),
                              ("test", graph.test_index.numpy())):
            require(np.array_equal(split[key], expected), key + ": split mismatch")
    source_manifest = read(folder.parent / "source_manifest.json")
    for path, digest in source_manifest.items():
        require(d.sha(path) == digest, "source changed since run: " + path)
    config = read(folder.parent / "config.json")
    for key, value in d.CONFIG.items():
        require(config.get(key) == value, "reconstruction config differs: " + key)
    relations = None
    if deep:
        h = d.load_frozen_representation(config["h_path"], graph.num_nodes).numpy()
        relations = reconstruction(folder, h, train, device)
    metric_rows = read(folder / "metrics.json")
    results = []
    for kind, rate in d.CONFIG["noise_conditions"].items():
        condition = folder / kind
        protocol = read(condition / "protocol.json")
        graph = d.inject_train_label_noise(graph, "uniform" if kind == "clean" else kind, rate, seed)
        observed = graph.noisy_y[train].numpy()
        clean = graph.y[train].numpy()
        require(np.array_equal(observed, np.load(condition / "observed_train.npy", allow_pickle=False)),
                kind + ": observed labels mismatch")
        require(protocol["noisy_train_fingerprint"] == d.fingerprint(observed), "noise fingerprint mismatch")
        require(protocol["split_fingerprint"] == d.fingerprint(train, graph.val_index.numpy(), graph.test_index.numpy()),
                "split fingerprint mismatch")
        require(torch.equal(graph.y[graph.val_mask], graph.noisy_y[graph.val_mask]), "validation label changed")
        require(torch.equal(graph.y[graph.test_mask], graph.noisy_y[graph.test_mask]), "test label changed")
        require(protocol["clean_val_unchanged"] and protocol["clean_test_unchanged"], "recorded clean-label check failed")
        arrays = {}
        frozen = read(condition / "frozen_targets.json")
        require(len(frozen) == 14, "missing target arm")
        for arm, digest in frozen.items():
            require(d.sha(condition / (arm + ".npz")) == digest, arm + ": frozen hash changed")
            with np.load(condition / (arm + ".npz"), allow_pickle=False) as saved:
                a = {key: saved[key].copy() for key in saved.files}
            arrays[arm] = a
            for key in ("q", "target"):
                require(a[key].shape == (len(train), graph.num_classes), arm + ": shape mismatch")
                require(np.isfinite(a[key]).all() and a[key].min() >= 0, arm + ": invalid probabilities")
                close(a[key].sum(axis=1), np.ones(len(train)), arm + ": normalization")
            require(np.isfinite(a["alpha"]).all() and np.all((a["alpha"] >= 0) & (a["alpha"] <= 1)), "invalid alpha")
            close(a["target"], d.mix(a["q"], a["alpha"], observed), arm + ": saved target mixture")
            rows = [r for r in metric_rows if r["noise"] == kind and r["arm"] == arm]
            require(len(rows) == 1, arm + ": missing/duplicate metric row")
            metrics = d.target_metrics(a["target"], a["q"], a["alpha"], clean, observed)
            # Float32 storage may affect exact argmax ties; aggregate Brier is stable.
            for key in ("target_brier", "q_brier", "mean_alpha", "q_true_probability", "clean_brier", "noisy_brier"):
                if metrics[key] is not None:
                    require(abs(metrics[key] - rows[0][key]) < 2e-6, arm + ": aggregate " + key)
            results.append(dict(noise=kind, arm=arm, zero_vote_rate=float(np.mean(a["support"] == 0)),
                insufficient_vote_rate=float(np.mean(a["support"] < 2)),
                mean_effective=float(a["effective"].mean()), deep_reconstruction=deep))
        if relations is None:
            continue
        reference = {arm: reference_vote(w, observed, graph.num_classes) for arm, w in relations.items()}
        r, q, alpha, count, eff = reference["GB128"]
        coarse, _, _, coarse_count, _ = reference["GB64"]
        both = (count >= 2) & (coarse_count >= 2)
        midpoint = (r + coarse) / 2
        divergence = .5 * np.sum(r * np.log(np.maximum(r, 1e-300) / np.maximum(midpoint, 1e-300)) +
            coarse * np.log(np.maximum(coarse, 1e-300) / np.maximum(midpoint, 1e-300)), axis=1)
        js = np.where(both, divergence, np.nan)
        js_factor = np.where(both, np.exp(-divergence / d.CONFIG["js_tau"]), 1.)
        ess_factor = eff / (eff + d.CONFIG["ess_kappa"])
        factor = ess_factor * js_factor
        scale = np.dot(alpha, factor) / alpha.sum() if alpha.sum() else 0.
        alpha_map = dict(half=alpha * .5, ess=alpha * ess_factor, js=alpha * js_factor,
            full=alpha * factor, mean_matched=alpha * scale,
            shuffled=alpha * np.random.RandomState(10000 + seed).permutation(factor))
        for label, values in alpha_map.items():
            arm = "GB128_" + label
            close(arrays[arm]["q"], q, arm + ": changed q")
            close(arrays[arm]["alpha"], values, arm + ": alpha formula")
        require(abs(arrays["GB128_full"]["alpha"].mean() - arrays["GB128_mean_matched"]["alpha"].mean()) < 1e-6,
                "mean-matched alpha mismatch")
        dest = output / kind
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / "reliability_nodes.npz", train_node_id=train, js=js, js_missing=~both,
            js_factor=js_factor, ess_factor=ess_factor, full_factor=factor)
        for arm, (r, q, alpha, count, eff) in reference.items():
            for key, values in (("q", q), ("alpha", alpha), ("support", count), ("effective", eff)):
                close(arrays[arm][key], values, arm + ": reconstructed " + key)
            close(arrays[arm]["target"], d.mix(q, alpha, observed), arm + ": reconstructed target")
            true_r = reference_vote(relations[arm], clean, graph.num_classes)[0]
            other = true_r.copy()
            other[np.arange(len(train)), clean] = -np.inf
            margin = true_r[np.arange(len(train)), clean] - other.max(axis=1)
            margin[count < 2] = np.nan
            np.savez_compressed(dest / (arm + "_nodes.npz"), train_node_id=train, r=r,
                clean_margin=margin, clean_margin_missing=count < 2, initially_clean=observed == clean,
                brier=np.square(arrays[arm]["target"] - np.eye(graph.num_classes)[clean]).sum(axis=1),
                positive_clean_margin=margin > 0)
        # Original implementation is an independent scalar-loop oracle for GB.
        with np.load(folder / "assignments.npz", allow_pickle=False) as assignments:
            for arm in ("GB64", "GB128"):
                formal = build_leave_one_out_targets(graph.noisy_y, assignments[arm], graph.train_index,
                    graph.num_classes, min_train_per_ball=2, beta=.5)
                close(arrays[arm]["target"], formal.soft_target[train].numpy(), arm + ": original target parity")
    d.write(output / "audit.json", dict(dataset=name, representation=representation, seed=seed,
        status="passed", deep_reconstruction=deep, rows=results,
        margin_definition="Clean training voters, excluding root; positive fraction denominator is all training roots; mean uses eligible roots only."))
    return dict(dataset=name, representation=representation, seed=seed, status="passed", deep_reconstruction=deep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--device", default="cuda", help="Match original kNN device for exact relation hashes")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    runs = sorted((ROOT / "runs").glob("*/*/seed*/metrics.json"))
    results = []
    with threadpool_limits(limits=4):
        for metrics in runs:
            folder = metrics.parent
            deep = int(folder.name.removeprefix("seed")) in args.deep_seeds
            print("AUDIT", folder.relative_to(ROOT), "deep=" + str(deep), flush=True)
            try:
                results.append(audit_run(folder, deep, args.device))
            except Exception as exc:
                results.append(dict(run=str(folder.relative_to(ROOT)), status="failed",
                    error_type=type(exc).__name__, error=str(exc), deep_reconstruction=deep))
    report = dict(script_sha256=d.sha(__file__), completed_runs_at_start=len(runs),
        deep_seeds=args.deep_seeds, device=args.device, runs=results,
        coverage="All completed runs: input/frozen hashes, split/noise, probability/mixture/Brier checks. Deep seeds additionally: relation CSR hashes, original GB target parity, independent vote/control formula, per-node r/margin/JS.",
        limitations=["Partitions are loaded, not refitted.", "Source manifests shared across seeds cannot establish historical provenance after a prior overwrite.",
                     "A CPU/CUDA kNN boundary mismatch is flagged as a failure rather than silently accepted.",
                     "An audit is post-hoc verification; it does not convert exploratory diagnostics into independent test evidence."])
    d.write(OUT / "report.json", report)
    require(all(r["status"] == "passed" for r in results), "audit failures; see audit/report.json")
    print("PASS", len(results), "completed runs; report:", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
