from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.NRGNN_Predictor import nrgnn_Predictor
from scripts.run_mlp46_cf_nagb import _as_numpy, _mask_indices, target_quality_metrics
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    build_knn_soft_targets_from_features,
    load_frozen_embeddings,
    prepare_ball_features,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.teacher_verified_gb import size_matched_shuffled_assignment
from utils.tools import load_conf, setup_seed


EXPECTED_DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SEED = 0
FORMAL_GB_CONFIG = "frozengrace_gb_mlp_softdist_beta05_loo"
EXPECTED_METHODS = (
    "formal_grace_h_gb",
    "nrgnn_direct_teacher",
    "nrgnn_z_gb",
    "nrgnn_z_knn",
    "nrgnn_z_gb_shuffled",
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def extract_best_nrgnn_state(predictor) -> tuple[torch.Tensor, torch.Tensor]:
    if predictor.weights is None:
        raise RuntimeError("NRGNN did not record validation-selected weights")
    if predictor.best_model_index is None or predictor.best_graph is None:
        raise RuntimeError("NRGNN did not record a validation-selected graph")
    predictor.model.load_state_dict(predictor.weights)
    predictor.model.eval()
    reformed_adj = torch.sparse_coo_tensor(
        predictor.best_model_index,
        predictor.best_graph,
        [predictor.n_nodes, predictor.n_nodes],
        device=predictor.feats.device,
    ).coalesce()
    with torch.no_grad():
        logits, hidden = predictor.model.forward_with_hidden(
            predictor.feats, reformed_adj
        )
    if logits.ndim != 2 or hidden.ndim != 2:
        raise RuntimeError("NRGNN logits and hidden representation must be matrices")
    if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
        raise RuntimeError("NRGNN produced non-finite logits or hidden representation")
    return logits.detach(), hidden.detach()


def exact_knn_loo_targets(
    *,
    features: torch.Tensor,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_indices: np.ndarray,
    n_classes: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = build_knn_soft_targets_from_features(
        ball_features=features,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=train_indices,
        n_classes=int(n_classes),
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        random_state=SEED,
        matched_k=int(k),
        tie_mode="distribution_only",
        exclude_self=True,
    )
    return _as_numpy(result.soft_target, np.float64), _as_numpy(result.alpha, np.float64)


def _formal_embedding(dataset: str, expected_nodes: int) -> tuple[torch.Tensor, str, str]:
    conf = load_conf(None, FORMAL_GB_CONFIG, dataset)
    raw_path = Path(str(conf.gb["embedding_path"]))
    path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    features = prepare_ball_features(
        load_frozen_embeddings(path, expected_nodes=expected_nodes)
    ).cpu()
    return features, str(path), _tensor_sha256(features)


def _matched_k(assignment: np.ndarray, train: np.ndarray) -> int:
    counts = [
        int(np.sum(assignment[train] == group))
        for group in np.unique(assignment)
        if int(np.sum(assignment[train] == group)) > 0
    ]
    if not counts:
        return 0
    return max(1, int(round(float(np.mean(counts)))))


def _target_row(common, method, targets, alpha, noisy, clean, train, **extra):
    target_np = _as_numpy(targets, np.float64)
    alpha_np = _as_numpy(alpha, np.float64)
    return {
        **common,
        "method": method,
        **target_quality_metrics(target_np, noisy, clean, train),
        "mean_alpha": float(np.mean(alpha_np[train])),
        "alpha_coverage": float(np.mean(alpha_np[train] > 0.0)),
        "target_simplex_max_error": float(
            np.max(np.abs(target_np.sum(axis=1) - 1.0))
        ),
        "status": "ok",
        **extra,
    }


def _cell_key(dataset: str, rate: float) -> tuple[str, float]:
    return str(dataset).lower(), round(float(rate), 6)


def evaluate_phase_a_gate(rows: list[dict]) -> dict:
    expected_cells = {
        _cell_key(dataset, rate)
        for dataset in EXPECTED_DATASETS
        for rate in NOISE_RATES
    }
    by_cell: dict[tuple[str, float], dict[str, dict]] = {}
    duplicates = []
    for row in rows:
        key = _cell_key(row["dataset"], row["noise_rate"])
        method = str(row["method"])
        if method in by_cell.setdefault(key, {}):
            duplicates.append(f"{key}:{method}")
        by_cell[key][method] = row
    missing = {
        f"{dataset}:{rate}": sorted(
            set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {}))
        )
        for dataset, rate in sorted(expected_cells)
        if set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {}))
    }
    if set(by_cell) != expected_cells or missing or duplicates:
        return {
            "pass": False,
            "reason": "incomplete",
            "missing": missing,
            "duplicates": duplicates,
        }

    wins_vs_formal = 0
    pareto_wins_vs_knn = 0
    wins_vs_shuffled = 0
    direct_teacher_wins = 0
    integrity_failures = []
    per_cell = {}
    for key in sorted(expected_cells):
        methods = by_cell[key]
        split_hashes = {str(row.get("split_hash", "")) for row in methods.values()}
        noise_hashes = {str(row.get("noise_hash", "")) for row in methods.values()}
        if len(split_hashes) != 1 or "" in split_hashes:
            integrity_failures.append(f"{key}:split_hash")
        if len(noise_hashes) != 1 or "" in noise_hashes:
            integrity_failures.append(f"{key}:noise_hash")
        for method, row in methods.items():
            if str(row.get("status", "")) != "ok":
                integrity_failures.append(f"{key}:{method}:status")
            if int(row.get("modified_count", -1)) != int(
                row.get("modified_train_count", -2)
            ):
                integrity_failures.append(f"{key}:{method}:noise_scope")
            if int(row.get("modified_val_count", -1)) != 0 or int(
                row.get("modified_test_count", -1)
            ) != 0:
                integrity_failures.append(f"{key}:{method}:label_leakage")
            if float(row.get("target_simplex_max_error", 1.0)) > 1e-6:
                integrity_failures.append(f"{key}:{method}:simplex")
            if int(row.get("representation_finite", 0)) != 1:
                integrity_failures.append(f"{key}:{method}:representation")
            if not str(row.get("representation_hash", "")):
                integrity_failures.append(f"{key}:{method}:representation_hash")
        formal = methods["formal_grace_h_gb"]
        direct = methods["nrgnn_direct_teacher"]
        candidate = methods["nrgnn_z_gb"]
        knn = methods["nrgnn_z_knn"]
        shuffled = methods["nrgnn_z_gb_shuffled"]
        same_z_hashes = {
            str(methods[name]["representation_hash"])
            for name in (
                "nrgnn_direct_teacher",
                "nrgnn_z_gb",
                "nrgnn_z_knn",
                "nrgnn_z_gb_shuffled",
            )
        }
        if len(same_z_hashes) != 1:
            integrity_failures.append(f"{key}:nrgnn_representation_mismatch")
        formal_win = float(candidate["brier"]) < float(formal["brier"]) - 1e-12
        pareto_win = (
            float(candidate["brier"]) < float(knn["brier"]) - 1e-12
            and float(candidate["target_accuracy"])
            >= float(knn["target_accuracy"]) - 0.005 - 1e-12
        )
        shuffled_win = (
            float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        )
        direct_win = float(direct["brier"]) < float(formal["brier"]) - 1e-12
        wins_vs_formal += int(formal_win)
        pareto_wins_vs_knn += int(pareto_win)
        wins_vs_shuffled += int(shuffled_win)
        direct_teacher_wins += int(direct_win)
        per_cell[f"{key[0]}:{key[1]}"] = {
            "z_gb_brier_delta_vs_formal": float(candidate["brier"])
            - float(formal["brier"]),
            "z_gb_brier_delta_vs_z_knn": float(candidate["brier"])
            - float(knn["brier"]),
            "z_gb_accuracy_delta_vs_z_knn": float(candidate["target_accuracy"])
            - float(knn["target_accuracy"]),
            "direct_brier_delta_vs_formal": float(direct["brier"])
            - float(formal["brier"]),
        }
    if integrity_failures:
        reason = "integrity_failed"
    elif wins_vs_formal < 4:
        reason = "formal_grace_h_gate_failed"
    elif pareto_wins_vs_knn < 4:
        reason = "same_space_knn_gate_failed"
    elif wins_vs_shuffled < 6:
        reason = "shuffled_gate_failed"
    elif direct_teacher_wins < 4:
        reason = "teacher_information_gate_failed"
    else:
        reason = "pass"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "z_gb_brier_wins_vs_formal": wins_vs_formal,
        "z_gb_brier_wins_vs_z_knn": pareto_wins_vs_knn,
        "z_gb_brier_wins_vs_shuffled": wins_vs_shuffled,
        "direct_teacher_brier_wins_vs_formal": direct_teacher_wins,
        "integrity_failures": integrity_failures,
        "per_cell": per_cell,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _train_nrgnn(data, noisy_tensor, dataset, noise_rate, device):
    conf = load_conf(None, "nrgnn", dataset)
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes
    conf.training["debug"] = False
    predictor = nrgnn_Predictor(conf, data, device)
    predictor.method_name = "nrgnn"
    predictor.noise_type = "uniform"
    predictor.noise_rate = float(noise_rate)
    predictor.seed = SEED
    predictor.noisy_label = noisy_tensor
    started = time.perf_counter()
    result = predictor.train()
    wall_time = time.perf_counter() - started
    logits, hidden = extract_best_nrgnn_state(predictor)
    return predictor, result, logits, hidden, wall_time


def run_cell(dataset: str, noise_rate: float, device: str, artifact_dir: Path):
    setup_seed(SEED)
    data = _load_dataset(
        dataset,
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_tensor, modified = _build_noisy_labels(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=float(noise_rate),
        seed=SEED,
        noise_scope="train",
        train_mask=data.train_masks,
        val_mask=data.val_masks,
        debug=False,
    )
    data.noisy_label = noisy_tensor
    clean = _as_numpy(data.labels, np.int64)
    noisy = _as_numpy(noisy_tensor, np.int64)
    train = _mask_indices(data.train_masks)
    val = _mask_indices(data.val_masks)
    test = _mask_indices(data.test_masks)
    modified = _as_numpy(modified, np.int64).reshape(-1)
    if np.any(noisy[val] != clean[val]) or np.any(noisy[test] != clean[test]):
        raise RuntimeError("validation/test labels were modified")
    if np.any(~np.isin(modified, train)):
        raise RuntimeError("noise_scope=train was violated")

    predictor, teacher_result, logits, hidden, teacher_wall = _train_nrgnn(
        data, noisy_tensor, dataset, noise_rate, device
    )
    z = prepare_ball_features(hidden).cpu()
    z_hash = _tensor_sha256(z)
    teacher_probs = F.softmax(logits.detach().cpu(), dim=1)
    if not torch.isfinite(teacher_probs).all():
        raise RuntimeError("NRGNN teacher probabilities are non-finite")
    if not torch.allclose(
        teacher_probs.sum(dim=1), torch.ones(teacher_probs.shape[0]), atol=1e-6
    ):
        raise RuntimeError("NRGNN teacher probabilities do not sum to one")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{dataset}_uniform{noise_rate:.1f}_seed0.pt"
    torch.save(
        {
            "dataset": dataset,
            "noise_rate": float(noise_rate),
            "seed": SEED,
            "split_hash": _split_fingerprint(
                data.train_masks, data.val_masks, data.test_masks
            ),
            "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
            "hidden": z,
            "teacher_probs": teacher_probs,
            "teacher_test": float(teacher_result["test"]),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in predictor.weights.items()
            },
            "reformed_edge_index": predictor.best_model_index.detach().cpu(),
            "reformed_edge_weight": predictor.best_graph.detach().cpu(),
        },
        artifact_path,
    )

    formal_h, formal_path, formal_hash = _formal_embedding(dataset, int(data.n_nodes))
    formal_partition = build_ordinary_granular_balls(
        formal_h,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=SEED,
    )
    formal_target, formal_alpha, _ = apply_ball_soft_targets(
        noisy_label=noisy_tensor.detach().cpu(),
        assignment=formal_partition.assignment,
        train_mask=train,
        n_classes=int(data.n_classes),
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )

    z_partition = build_ordinary_granular_balls(
        z,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=SEED,
    )
    z_target, z_alpha, _ = apply_ball_soft_targets(
        noisy_label=noisy_tensor.detach().cpu(),
        assignment=z_partition.assignment,
        train_mask=train,
        n_classes=int(data.n_classes),
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )
    matched_k = _matched_k(z_partition.assignment, train)
    knn_target, knn_alpha = exact_knn_loo_targets(
        features=z,
        noisy_label=noisy_tensor.detach().cpu(),
        clean_label=data.labels.detach().cpu(),
        train_indices=train,
        n_classes=int(data.n_classes),
        k=matched_k,
    )
    shuffled_assignment = size_matched_shuffled_assignment(
        z_partition.assignment,
        random_state=53000 + int(round(noise_rate * 100)),
    )
    shuffled_target, shuffled_alpha, _ = apply_ball_soft_targets(
        noisy_label=noisy_tensor.detach().cpu(),
        assignment=shuffled_assignment,
        train_mask=train,
        n_classes=int(data.n_classes),
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )

    common = {
        "dataset": dataset,
        "seed": SEED,
        "noise_type": "uniform",
        "noise_rate": float(noise_rate),
        "noise_scope": "train",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "split_hash": _split_fingerprint(
            data.train_masks, data.val_masks, data.test_masks
        ),
        "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
        "modified_count": int(modified.size),
        "modified_train_count": int(np.isin(modified, train).sum()),
        "modified_val_count": int(np.isin(modified, val).sum()),
        "modified_test_count": int(np.isin(modified, test).sum()),
        "tie_mode": "distribution_only",
        "leave_one_out": 1,
        "teacher_test_accuracy": float(teacher_result["test"]),
        "teacher_wall_time_seconds": float(teacher_wall),
        "teacher_artifact": str(artifact_path),
    }
    direct_alpha = np.zeros(int(data.n_nodes), dtype=np.float64)
    rows = [
        _target_row(
            common,
            "formal_grace_h_gb",
            formal_target,
            formal_alpha,
            noisy,
            clean,
            train,
            representation_source="frozen_grace_h",
            representation_dim=int(formal_h.shape[1]),
            representation_hash=formal_hash,
            representation_finite=1,
            embedding_path=formal_path,
            num_balls=int(formal_partition.summary["num_balls"]),
            matched_k="",
            knn_exclude_self="",
        ),
        _target_row(
            common,
            "nrgnn_direct_teacher",
            teacher_probs,
            direct_alpha,
            noisy,
            clean,
            train,
            representation_source="nrgnn_penultimate_hidden",
            representation_dim=int(z.shape[1]),
            representation_hash=z_hash,
            representation_finite=1,
            embedding_path=str(artifact_path),
            num_balls="",
            matched_k="",
            knn_exclude_self="",
        ),
        _target_row(
            common,
            "nrgnn_z_gb",
            z_target,
            z_alpha,
            noisy,
            clean,
            train,
            representation_source="nrgnn_penultimate_hidden",
            representation_dim=int(z.shape[1]),
            representation_hash=z_hash,
            representation_finite=1,
            embedding_path=str(artifact_path),
            num_balls=int(z_partition.summary["num_balls"]),
            matched_k=matched_k,
            knn_exclude_self="",
        ),
        _target_row(
            common,
            "nrgnn_z_knn",
            knn_target,
            knn_alpha,
            noisy,
            clean,
            train,
            representation_source="nrgnn_penultimate_hidden",
            representation_dim=int(z.shape[1]),
            representation_hash=z_hash,
            representation_finite=1,
            embedding_path=str(artifact_path),
            num_balls=int(z_partition.summary["num_balls"]),
            matched_k=matched_k,
            knn_exclude_self=1,
        ),
        _target_row(
            common,
            "nrgnn_z_gb_shuffled",
            shuffled_target,
            shuffled_alpha,
            noisy,
            clean,
            train,
            representation_source="nrgnn_penultimate_hidden",
            representation_dim=int(z.shape[1]),
            representation_hash=z_hash,
            representation_finite=1,
            embedding_path=str(artifact_path),
            num_balls=int(z_partition.summary["num_balls"]),
            matched_k="",
            knn_exclude_self="",
        ),
    ]
    return rows


def _report(rows: list[dict], decision: dict) -> str:
    lines = [
        "# MLP-53 NRGNN latent-space granular-ball report",
        "",
        "## Frozen protocol",
        "",
        "- NRGNN is retrained independently per dataset/noise cell; the validation-selected model and graph are used.",
        "- The candidate changes only the grouping representation. GB parameters and the tie-safe LOO target are unchanged.",
        "- The same normalized NRGNN representation is used by exact self-excluding, vote-matched kNN.",
        "- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.",
        "",
        "## Target quality",
        "",
        "| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Balls | k | Teacher test |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["method"])
    ):
        balls = row.get("num_balls", "")
        k = row.get("matched_k", "")
        lines.append(
            f"| {row['dataset']} | {float(row['noise_rate']):.1f} | {row['method']} | "
            f"{float(row['target_accuracy']):.4f} | {float(row['brier']):.4f} | "
            f"{float(row['ncr']):.4f} | {float(row['cdr']):.4f} | {balls} | {k} | "
            f"{float(row['teacher_test_accuracy']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- PASS: `{int(bool(decision.get('pass', False)))}`.",
            f"- Reason: `{decision.get('reason', 'unknown')}`.",
            f"- Z-GB Brier wins vs formal GRACE-H GB: `{decision.get('z_gb_brier_wins_vs_formal', 0)}/6`.",
            f"- Z-GB Pareto wins vs same-space kNN: `{decision.get('z_gb_brier_wins_vs_z_knn', 0)}/6`.",
            f"- Z-GB Brier wins vs shuffled Z-GB: `{decision.get('z_gb_brier_wins_vs_shuffled', 0)}/6`.",
            f"- Direct-teacher Brier wins vs formal GRACE-H GB: `{decision.get('direct_teacher_brier_wins_vs_formal', 0)}/6`.",
            f"- Integrity failures: `{decision.get('integrity_failures', [])}`.",
            "",
            "## Decision",
            "",
            (
                "Phase A passed. Downstream training requires a separate preregistration."
                if decision.get("pass")
                else "Phase A failed. This is the third non-positive round, so the autonomous search stops without downstream training or another mechanism variant."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLP-53 NRGNN latent-space granular-ball diagnostics."
    )
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument(
        "--noise_rates", default=",".join(str(rate) for rate in NOISE_RATES)
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "mlp53_nrgnn_latent_balls_20260806",
    )
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    rates = tuple(float(item.strip()) for item in args.noise_rates.split(",") if item.strip())
    if datasets != EXPECTED_DATASETS or rates != NOISE_RATES or args.seed != SEED:
        raise ValueError("formal MLP-53 protocol is frozen; datasets/rates/seed cannot change")
    rows = []
    for dataset in datasets:
        for rate in rates:
            print(f"MLP53 dataset={dataset} noise_rate={rate} seed={SEED}", flush=True)
            rows.extend(
                run_cell(
                    dataset=dataset,
                    noise_rate=rate,
                    device=args.device,
                    artifact_dir=args.run_dir / "teacher_artifacts",
                )
            )
    decision = evaluate_phase_a_gate(rows)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.run_dir / "phase_a_raw.csv", rows)
    (args.run_dir / "phase_a_gate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run_dir / "mlp53_report.md").write_text(
        _report(rows, decision), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "pass": decision["pass"], "reason": decision["reason"]}))


if __name__ == "__main__":
    main()

