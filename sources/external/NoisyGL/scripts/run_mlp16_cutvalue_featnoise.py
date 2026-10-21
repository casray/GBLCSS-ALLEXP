from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import (  # noqa: E402
    _apply_config_overrides,
    _config_id,
    _load_dataset,
    _noise_fingerprint,
    _noise_stats,
    _parse_int_list,
    _parse_str_list,
    _split_fingerprint,
    predictor_class_for,
)
from utils.labelnoise import label_process  # noqa: E402
from utils.tools import load_conf, setup_seed  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp16_cutvalue_featnoise_20260706"
DEFAULT_DATASETS = ("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
DEFAULT_SEEDS = tuple(range(5))
DEFAULT_ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
DISTILL_DATASETS = {"cora", "citeseer", "pubmed"}

OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MLP_CE = "mlp_ce"
SMALL_LOSS = "mlp_small_loss"
JOCOR = "mlp_jocor"
KNN = "knn_mlp_softdist_beta05"
DISTILL = "mlp_distill_softdist_xh"
GLNN = "mlp_distill_gcn"

P0_METHODS = (OURS, MLP_CE, SHUFFLED, KNN, DISTILL, GLNN)
P1_METHODS = (OURS, MLP_CE, SMALL_LOSS, JOCOR, SHUFFLED)

METHOD_LABELS = {
    OURS: "Ours beta=0.5 train-only",
    SHUFFLED: "Shuffled local consensus",
    MLP_CE: "MLP+CE",
    SMALL_LOSS: "Small-loss",
    JOCOR: "JoCoR",
    KNN: "kNN@matched beta=0.5",
    DISTILL: "Ours-distill T3",
    GLNN: "GLNN-style T1",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _as_index(mask, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype == torch.bool:
        return torch.nonzero(mask, as_tuple=False).flatten().long()
    return mask.long().flatten()


def _dense_features(features: torch.Tensor) -> torch.Tensor:
    return features.to_dense().float() if getattr(features, "is_sparse", False) else features.float()


def _load_best_weights(predictor) -> None:
    weights = getattr(predictor, "weights", None)
    if weights is None:
        return
    if isinstance(weights, dict) and "model" in weights:
        predictor.model.load_state_dict(weights["model"])
        if getattr(predictor, "model2", None) is not None and weights.get("model2") is not None:
            predictor.model2.load_state_dict(weights["model2"])
    else:
        predictor.model.load_state_dict(weights)


def _model_probs(predictor, features: torch.Tensor | None = None) -> torch.Tensor:
    _load_best_weights(predictor)
    predictor.model.eval()
    if getattr(predictor, "model2", None) is not None:
        predictor.model2.eval()
    with torch.no_grad():
        logits = predictor.model(predictor.feats if features is None else features, predictor.adj)
        return F.softmax(logits, dim=1)


def _test_acc_from_probs(predictor, probs: torch.Tensor) -> float:
    idx = _as_index(predictor.test_mask, probs.device)
    labels = predictor.clean_label[idx].detach().cpu().numpy()
    return float(predictor.metric(labels, probs[idx].detach().cpu().numpy()))


def cut_value_from_probs(probs: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> float:
    """Compute tr(Y^T A Y) / tr(Y^T D Y) on an unweighted symmetrized graph."""
    if probs.ndim != 2:
        raise ValueError("probs must be a 2-D tensor")
    edge_index = edge_index.to(probs.device).long()
    if edge_index.numel() == 0:
        return float("nan")
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]
    if edge_index.numel() == 0:
        return float("nan")
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    values = torch.ones(edge_index.shape[1], device=probs.device)
    adj = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes), device=probs.device).coalesce()
    idx = adj.indices()
    # Coalescing sums duplicate directed edges. Reset to unit weights so the
    # metric is invariant to whether the loader already stores both directions.
    vals = torch.ones(idx.shape[1], device=probs.device, dtype=probs.dtype)
    edge_agreement = (probs[idx[0]] * probs[idx[1]]).sum(dim=1)
    numerator = (vals * edge_agreement).sum()
    degree = torch.zeros(n_nodes, device=probs.device, dtype=probs.dtype)
    degree.scatter_add_(0, idx[0], vals)
    denominator = (degree * probs.square().sum(dim=1)).sum().clamp_min(1e-12)
    return float((numerator / denominator).detach().cpu())


def perturb_test_features(
    features: torch.Tensor,
    test_mask,
    alpha: float,
    seed: int,
    dataset: str = "",
    method: str = "",
) -> torch.Tensor:
    """Apply standardized Gaussian corruption only to test rows."""
    alpha = float(alpha)
    if math.isclose(alpha, 0.0, abs_tol=1e-12):
        return features
    dense = _dense_features(features).detach()
    out = dense.clone()
    mean = dense.mean(dim=0, keepdim=True)
    std = dense.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = (dense - mean) / std
    stable_payload = f"{dataset}:{method}".encode("utf-8")
    stable = int(hashlib.sha256(stable_payload).hexdigest()[:12], 16) % 1_000_003
    noise_seed = int(seed) * 1_000_003 + int(round(alpha * 1000.0)) * 9176 + stable
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    eps = torch.randn(z.shape, generator=generator, dtype=torch.float32).to(device=z.device, dtype=z.dtype)
    idx = _as_index(test_mask, z.device)
    out[idx] = ((1.0 - alpha) * z[idx] + alpha * eps[idx]) * std + mean
    return out


def _config_exists(method: str, dataset: str) -> bool:
    return (REPO_ROOT / "config" / method / f"{method}_{dataset}.yaml").exists()


def _method_allowed_for_dataset(method: str, dataset: str) -> bool:
    return method not in {DISTILL, GLNN} or dataset in DISTILL_DATASETS


def _needed_methods(stage: str, dataset: str) -> list[str]:
    methods: list[str] = []
    if stage in {"p0", "all"}:
        methods.extend(P0_METHODS)
    if stage in {"p1", "all"}:
        methods.extend(P1_METHODS)
    unique = []
    for method in methods:
        if method in unique:
            continue
        if not _method_allowed_for_dataset(method, dataset):
            continue
        unique.append(method)
    return unique


def _existing_model_keys(run_dir: Path) -> set[tuple[str, str, int]]:
    keys = set()
    for row in _read_csv(run_dir / "mlp16_model_raw.csv"):
        if row.get("status") == "ok":
            keys.add((row["dataset"], row["method"], int(row["seed"])))
    return keys


def _completed_keys(run_dir: Path, stage: str, alphas: tuple[float, ...]) -> set[tuple[str, str, int]]:
    model_keys = _existing_model_keys(run_dir)
    cut_keys = {
        (row["dataset"], row["method"], int(row["seed"]))
        for row in _read_csv(run_dir / "mlp16_cut_value_raw.csv")
    }
    alpha_keys = defaultdict(set)
    for row in _read_csv(run_dir / "mlp16_feature_noise_raw.csv"):
        alpha_keys[(row["dataset"], row["method"], int(row["seed"]))].add(float(row["alpha"]))
    completed = set()
    required_alphas = {float(alpha) for alpha in alphas}
    for key in model_keys:
        _, method, _ = key
        if stage in {"p0", "all"} and method in P0_METHODS and key not in cut_keys:
            continue
        if stage in {"p1", "all"} and method in P1_METHODS and not required_alphas.issubset(alpha_keys.get(key, set())):
            continue
        completed.add(key)
    return completed


def _train_predictor(args, dataset_name: str, method: str, seed: int):
    setup_seed(seed)
    data = _load_dataset(
        dataset_name,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    data.noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    conf = load_conf(None, method, dataset_name)
    _apply_config_overrides(conf, args.config_overrides)
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes
    conf.training["debug"] = bool(args.debug)
    predictor_cls = predictor_class_for(method)
    predictor = predictor_cls(conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = seed
    start = time.perf_counter()
    result = predictor.train()
    wall = time.perf_counter() - start
    return data, predictor, result, modified_mask, conf, wall


def _model_row(dataset, method, seed, result, data, predictor, modified_mask, conf, wall_time):
    return {
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": int(seed),
        "noise_type": predictor.noise_type,
        "noise_rate": float(predictor.noise_rate),
        "noise_scope": "train",
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, data.noisy_label),
        "config_id": _config_id(method, dataset, conf),
        **_noise_stats(
            data.labels,
            data.noisy_label,
            modified_mask,
            data.train_masks,
            data.val_masks,
            data.test_masks,
        ),
        "status": "ok",
        "error": "",
        "wall_time_seconds": float(wall_time),
        "train": float(result.get("train", 0.0)),
        "valid": float(result.get("valid", 0.0)),
        "test": float(result.get("test", 0.0)),
    }


def _evaluate_trained(args, run_dir: Path, dataset: str, method: str, seed: int) -> None:
    data, predictor, result, modified_mask, conf, wall = _train_predictor(args, dataset, method, seed)
    model_row = _model_row(dataset, method, seed, result, data, predictor, modified_mask, conf, wall)
    _append_csv(run_dir / "mlp16_model_raw.csv", model_row)

    if args.stage in {"p0", "all"} and method in P0_METHODS:
        probs = _model_probs(predictor)
        cv = cut_value_from_probs(probs, predictor.edge_index, predictor.n_nodes)
        _append_csv(run_dir / "mlp16_cut_value_raw.csv", {
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed": int(seed),
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "cut_value": cv,
            "test": float(result.get("test", 0.0)),
            "n_nodes": int(predictor.n_nodes),
            "n_classes": int(predictor.n_classes),
        })

    if args.stage in {"p1", "all"} and method in P1_METHODS:
        base_test = float(result.get("test", 0.0))
        for alpha in args.alphas:
            features = perturb_test_features(predictor.feats, predictor.test_mask, alpha, seed, dataset, method)
            probs = _model_probs(predictor, features)
            test_acc = _test_acc_from_probs(predictor, probs)
            _append_csv(run_dir / "mlp16_feature_noise_raw.csv", {
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": int(seed),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "alpha": float(alpha),
                "test": float(test_acc),
                "anchor_test": base_test,
                "anchor_abs_diff": abs(test_acc - base_test) if math.isclose(float(alpha), 0.0, abs_tol=1e-12) else "",
            })


def _aggregate(rows: list[dict[str, str]], value_key: str, group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(value_key, "") == "":
            continue
        key = tuple(row[k] for k in group_keys)
        groups[key].append(float(row[value_key]))
    out = []
    for key, values in sorted(groups.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item.update({
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
        out.append(item)
    return out


def _branch_rows(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cv_rows = _read_csv(run_dir / "mlp16_cut_value_raw.csv")
    fn_rows = _read_csv(run_dir / "mlp16_feature_noise_raw.csv")

    cv = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["cut_value"])
        for row in cv_rows
    }
    cv_branches = []
    for dataset in DEFAULT_DATASETS:
        for left, right, label in [
            (OURS, MLP_CE, "cv_ours_minus_ce"),
            (OURS, SHUFFLED, "cv_ours_minus_shuffled"),
            (OURS, KNN, "cv_ours_minus_knn"),
            (DISTILL, OURS, "cv_distill_minus_ours"),
            (DISTILL, GLNN, "cv_distill_minus_glnn"),
        ]:
            deltas = []
            for seed in DEFAULT_SEEDS:
                a = cv.get((dataset, left, seed))
                b = cv.get((dataset, right, seed))
                if a is not None and b is not None:
                    deltas.append(a - b)
            if deltas:
                cv_branches.append({
                    "comparison": label,
                    "dataset": dataset,
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(d > 0 for d in deltas),
                    "seed_deltas": " ".join(f"{d:.6f}" for d in deltas),
                })

    fn_values = {
        (row["dataset"], row["method"], int(row["seed"]), float(row["alpha"])): float(row["test"])
        for row in fn_rows
    }
    fn_slopes = []
    fn_deltas = []
    for dataset in DEFAULT_DATASETS:
        for method in P1_METHODS:
            drops = []
            for seed in DEFAULT_SEEDS:
                zero = fn_values.get((dataset, method, seed, 0.0))
                high = fn_values.get((dataset, method, seed, 0.9))
                if zero is not None and high is not None:
                    drops.append(zero - high)
            if drops:
                fn_slopes.append({
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "n": len(drops),
                    "drop_alpha0_to_0p9": statistics.mean(drops),
                    "std": statistics.stdev(drops) if len(drops) > 1 else 0.0,
                })
        for alpha in DEFAULT_ALPHAS:
            for baseline in (MLP_CE, SMALL_LOSS, JOCOR, SHUFFLED):
                deltas = []
                for seed in DEFAULT_SEEDS:
                    a = fn_values.get((dataset, OURS, seed, float(alpha)))
                    b = fn_values.get((dataset, baseline, seed, float(alpha)))
                    if a is not None and b is not None:
                        deltas.append(a - b)
                if deltas:
                    fn_deltas.append({
                        "dataset": dataset,
                        "baseline": baseline,
                        "alpha": float(alpha),
                        "n": len(deltas),
                        "mean_delta": statistics.mean(deltas),
                        "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        "positive_count": sum(d > 0 for d in deltas),
                    })
    return cv_branches, fn_slopes, fn_deltas


def _completeness(run_dir: Path, datasets: tuple[str, ...], seeds: tuple[int, ...], stage: str) -> list[dict[str, object]]:
    model_rows = _read_csv(run_dir / "mlp16_model_raw.csv")
    observed = {(row["dataset"], row["method"], int(row["seed"])) for row in model_rows if row.get("status") == "ok"}
    rows = []
    for dataset in datasets:
        for method in _needed_methods(stage, dataset):
            expected = [seed for seed in seeds if _config_exists(method, dataset)]
            got = [seed for seed in expected if (dataset, method, seed) in observed]
            rows.append({
                "dataset": dataset,
                "method": method,
                "expected": len(expected),
                "observed": len(got),
                "missing": " ".join(str(seed) for seed in expected if seed not in got),
            })
    return rows


def summarize(run_dir: Path, datasets: tuple[str, ...], seeds: tuple[int, ...], stage: str) -> None:
    cv_raw = _read_csv(run_dir / "mlp16_cut_value_raw.csv")
    fn_raw = _read_csv(run_dir / "mlp16_feature_noise_raw.csv")
    model_raw = _read_csv(run_dir / "mlp16_model_raw.csv")
    cv_agg = _aggregate(cv_raw, "cut_value", ("dataset", "method", "method_label"))
    fn_agg = _aggregate(fn_raw, "test", ("dataset", "method", "method_label", "alpha"))
    cv_branches, fn_slopes, fn_deltas = _branch_rows(run_dir)
    complete = _completeness(run_dir, datasets, seeds, stage)
    anchor = [
        row for row in fn_raw
        if math.isclose(float(row["alpha"]), 0.0, abs_tol=1e-12)
    ]
    anchor_rows = []
    for row in anchor:
        diff = float(row["anchor_abs_diff"]) if row.get("anchor_abs_diff", "") != "" else float("nan")
        anchor_rows.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "alpha0_test": float(row["test"]),
            "anchor_test": float(row["anchor_test"]),
            "anchor_abs_diff": diff,
            "passes": int(diff <= 1e-10),
        })
    _write_csv(run_dir / "mlp16_cut_value_aggregate.csv", cv_agg)
    _write_csv(run_dir / "mlp16_feature_noise_aggregate.csv", fn_agg)
    _write_csv(run_dir / "mlp16_cut_value_branches.csv", cv_branches)
    _write_csv(run_dir / "mlp16_feature_noise_slopes.csv", fn_slopes)
    _write_csv(run_dir / "mlp16_feature_noise_deltas.csv", fn_deltas)
    _write_csv(run_dir / "mlp16_anchor_check.csv", anchor_rows)
    _write_csv(run_dir / "mlp16_completeness.csv", complete)
    _write_report(run_dir / "mlp16_report.md", cv_agg, fn_agg, cv_branches, fn_slopes, fn_deltas, anchor_rows, complete, model_raw)


def _fmt_pct(value) -> str:
    if value in {"", None}:
        return "NA"
    return f"{100.0 * float(value):.2f}"


def _fmt(value) -> str:
    if value in {"", None}:
        return "NA"
    return f"{float(value):.4f}"


def _write_report(path: Path, cv_agg, fn_agg, cv_branches, fn_slopes, fn_deltas, anchor_rows, complete, model_raw) -> None:
    missing = [row for row in complete if int(row["observed"]) != int(row["expected"])]
    anchor_fail = [row for row in anchor_rows if int(row["passes"]) != 1]
    lines = [
        "# MLP-16 Cut Value + Feature Noise Report",
        "",
        "## Material Passport",
        "- Role: writing-stage mechanism quantification and graph-free-family feature-noise defense.",
        "- Protocol: train-only uniform label noise, rate 0.4, percent split 0.2/0.1/0.7.",
        "- P0 Cut Value is post-hoc only; P1 perturbs test features only.",
        f"- Trained/evaluated model rows: {len(model_raw)}",
        f"- Missing model groups: {len(missing)}",
        f"- Alpha=0 anchor failures: {len(anchor_fail)}",
        "",
        "## Figures",
        "- Feature-noise curves: `mlp16_feature_noise_curves.png`",
        "",
        "## P0 Cut Value Aggregate",
        "| Dataset | Method | n | CV mean | CV std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in cv_agg:
        lines.append(f"| {row['dataset']} | {row['method_label']} | {row['n']} | {_fmt(row['mean'])} | {_fmt(row['std'])} |")
    lines.extend([
        "",
        "## P0 Pre-registered Comparisons",
        "| Comparison | Dataset | n | Delta | Pos |",
        "|---|---|---:|---:|---:|",
    ])
    for row in cv_branches:
        lines.append(
            f"| {row['comparison']} | {row['dataset']} | {row['n']} | {_fmt(row['mean_delta'])} | {row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## P1 Feature Noise Curves",
        "| Dataset | Method | alpha | n | Test mean | Test std |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in fn_agg:
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | {float(row['alpha']):.1f} | {row['n']} | {_fmt_pct(row['mean'])} | {_fmt_pct(row['std'])} |"
        )
    lines.extend([
        "",
        "## P1 Drop From alpha=0 to alpha=0.9",
        "| Dataset | Method | n | Drop | std |",
        "|---|---|---:|---:|---:|",
    ])
    for row in fn_slopes:
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | {row['n']} | {_fmt_pct(row['drop_alpha0_to_0p9'])}pp | {_fmt_pct(row['std'])} |"
        )
    lines.extend([
        "",
        "## P1 Ours vs Baselines by alpha",
        "| Dataset | Baseline | alpha | n | Delta | Pos |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in fn_deltas:
        lines.append(
            f"| {row['dataset']} | {row['baseline']} | {float(row['alpha']):.1f} | {row['n']} | {_fmt_pct(row['mean_delta'])}pp | {row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## Alpha=0 Anchor Check",
        "| Dataset | Method | seed | alpha0 | anchor | abs diff | pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in anchor_rows[:200]:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | {_fmt_pct(row['alpha0_test'])} | "
            f"{_fmt_pct(row['anchor_test'])} | {float(row['anchor_abs_diff']):.3e} | {row['passes']} |"
        )
    lines.extend([
        "",
        "## Completeness",
        "| Dataset | Method | expected | observed | missing |",
        "|---|---|---:|---:|---|",
    ])
    for row in complete:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['expected']} | {row['observed']} | {row['missing']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = tuple(_parse_str_list(args.datasets))
    seeds = tuple(_parse_int_list(args.seeds))
    existing = _completed_keys(run_dir, args.stage, args.alphas)
    for dataset in datasets:
        for method in _needed_methods(args.stage, dataset):
            if not _config_exists(method, dataset):
                raise FileNotFoundError(f"missing config for method={method} dataset={dataset}")
            for seed in seeds:
                key = (dataset, method, seed)
                if key in existing and not args.force:
                    print(f"MLP16_SKIP dataset={dataset} method={method} seed={seed}")
                    continue
                print(f"MLP16_RUN dataset={dataset} method={method} seed={seed} stage={args.stage}")
                try:
                    _evaluate_trained(args, run_dir, dataset, method, seed)
                    existing.add(key)
                    summarize(run_dir, datasets, seeds, args.stage)
                except Exception as exc:
                    _append_csv(run_dir / "mlp16_model_raw.csv", {
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "seed": int(seed),
                        "noise_type": args.noise_type,
                        "noise_rate": float(args.noise_rate),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    summarize(run_dir, datasets, seeds, args.stage)
                    if args.stop_on_error:
                        raise
                    print(f"MLP16_FAILED dataset={dataset} method={method} seed={seed} error={type(exc).__name__}: {exc}")
    summarize(run_dir, datasets, seeds, args.stage)
    print(f"MLP16_DONE output={run_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run MLP-16 Cut Value and test-time feature-noise evaluation.")
    parser.add_argument("--stage", choices=["p0", "p1", "all", "summarize"], default="all")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", choices=["train", "all"], default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--config_overrides", default="{}")
    parser.add_argument("--output_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.alphas = tuple(float(item) for item in _parse_str_list(args.alphas))
    args.config_overrides = json.loads(args.config_overrides)
    if args.stage == "summarize":
        summarize(Path(args.output_dir), tuple(_parse_str_list(args.datasets)), tuple(_parse_int_list(args.seeds)), "all")
        return
    run(args)


if __name__ == "__main__":
    main()
