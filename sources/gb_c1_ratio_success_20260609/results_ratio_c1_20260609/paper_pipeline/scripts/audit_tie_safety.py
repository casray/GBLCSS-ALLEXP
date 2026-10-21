from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
RESULT_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = RESULT_ROOT.parents[1]
NOISYGL_ROOT = PROJECT_ROOT / "external" / "NoisyGL"
if str(NOISYGL_ROOT) not in sys.path:
    sys.path.insert(0, str(NOISYGL_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from scripts.run_standard_downstream import _noise_fingerprint, _split_fingerprint
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    build_frozen_grace_soft_targets,
)
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


METHOD = "frozengrace_gb_mlp_softdist_beta05"
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

# Frozen before inspecting audit results. A row is materially exposed when at
# least 1% of train targets or at least 1% of total train-node alpha mass is
# affected by the implementation's tie break.
MATERIAL_NODE_RATIO = 0.01
MATERIAL_ALPHA_SHARE = 0.01
# Float32 builder and the independent expression can differ by one or two ULPs
# solely because the arithmetic is associated differently.
TARGET_TOLERANCE = 1e-6

FORMAL_RAW_FILES = {
    "mlp10": NOISYGL_ROOT / "runs" / "mlp10_baseline_refresh_20260702" / "mlp10_combined_raw.csv",
    "mlp12": NOISYGL_ROOT / "runs" / "mlp12_amazonphoto_20260702" / "mlp12_combined_raw.csv",
    "mlp21": NOISYGL_ROOT / "runs" / "mlp21_dataset_replacement_20260710" / "mlp21_raw.csv",
}
PHASE4_SELECTED = (
    NOISYGL_ROOT / "runs" / "phase4_faithful_claims_20260712" / "phase4_selected_raw.csv"
)
MLP36_RAW = NOISYGL_ROOT / "runs" / "mlp36_review_defense_20260715" / "mlp36_raw.csv"
EMBEDDING_ROOT = PROJECT_ROOT / "gb_graph_aware_minival_6_1" / "checkpoints" / "embeddings_v2"
FORMULA_SOURCE = NOISYGL_ROOT / "utils" / "frozen_grace_targets.py"
BALL_SOURCE = NOISYGL_ROOT / "utils" / "ordinary_gb.py"


FIELDNAMES = [
    "audit_status",
    "gap_reason",
    "dataset",
    "setting",
    "noise_type",
    "noise_rate",
    "seed",
    "method",
    "num_classes",
    "train_nodes",
    "num_balls",
    "eligible_balls",
    "tie_balls",
    "target_active_tie_balls",
    "affected_train_nodes",
    "affected_train_node_ratio",
    "total_train_alpha_sum",
    "affected_train_alpha_sum",
    "affected_train_alpha_share",
    "min_index_selected_class_1based_ball_counts_json",
    "min_index_selected_class_1based_node_counts_json",
    "permutation",
    "permutation_changed_train_nodes",
    "permutation_changed_train_node_ratio",
    "permutation_train_l1_sum",
    "permutation_train_l1_mean",
    "permutation_train_l1_max",
    "permutation_equivariant",
    "expected_permutation_l1_sum",
    "permutation_l1_formula_abs_error",
    "material_node_ratio_threshold",
    "material_alpha_share_threshold",
    "material_flag",
    "reconstruction_strength",
    "reconstructed_split_hash",
    "formal_split_hash",
    "split_hash_match",
    "reconstructed_noise_hash",
    "formal_noise_hash",
    "noise_hash_match",
    "reconstructed_modified_train_count",
    "formal_modified_train_count",
    "modified_train_count_match",
    "reconstructed_actual_train_noise_rate",
    "formal_actual_train_noise_rate",
    "actual_train_noise_rate_abs_error",
    "anchor_num_balls",
    "anchor_num_balls_match",
    "anchor_mean_alpha_train",
    "anchor_mean_alpha_abs_error",
    "independent_target_max_abs_error",
    "builder_alpha_max_abs_error",
    "formal_source",
    "formal_source_sha256",
    "embedding_path",
    "embedding_sha256",
    "method_config_path",
    "method_config_sha256",
    "dataset_config_path",
    "dataset_config_sha256",
    "formula_source",
    "formula_source_sha256",
    "ball_source",
    "ball_source_sha256",
    "audit_script_sha256",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _norm_dataset(value: str) -> str:
    return str(value).strip().lower().replace("-", "")


def _key(dataset: str, noise_type: str, noise_rate: float, seed: int) -> tuple[str, str, float, int]:
    return (_norm_dataset(dataset), str(noise_type).lower(), round(float(noise_rate), 8), int(seed))


def _float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def _int_or_none(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _build_formal_index() -> dict[tuple[str, str, float, int], dict]:
    result: dict[tuple[str, str, float, int], dict] = {}
    allowed = set(DATASETS)
    for path in FORMAL_RAW_FILES.values():
        for row in _read_csv(path):
            if _norm_dataset(row.get("dataset", "")) not in allowed:
                continue
            if row.get("method") != METHOD:
                continue
            if row.get("status", "ok") not in ("", "ok"):
                continue
            item = dict(row)
            item["_source_path"] = str(path.resolve())
            result[_key(row["dataset"], row["noise_type"], row["noise_rate"], row["seed"])] = item

    expected = {
        _key(dataset, noise_type, noise_rate, seed)
        for dataset in DATASETS
        for noise_type, noise_rate in SETTINGS
        for seed in SEEDS
    }
    missing = expected - set(result)
    extra = set(result) - expected
    if missing or extra:
        raise RuntimeError(f"formal index mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return result


def _build_phase4_source_index() -> dict[tuple[str, str, float, int], str]:
    index = {}
    for row in _read_csv(PHASE4_SELECTED):
        if row.get("method") != METHOD:
            continue
        key = _key(row["dataset"], row["noise_type"], row["noise_rate"], row["seed"])
        source = Path(str(row["source"]).replace("\\", "/"))
        index[key] = str((NOISYGL_ROOT / source).resolve())
    return index


def _build_uniform_anchor_index() -> dict[tuple[str, int], dict]:
    index = {}
    for row in _read_csv(MLP36_RAW):
        if row.get("method") != METHOD:
            continue
        if row.get("noise_type") != "uniform" or not math.isclose(float(row["noise_rate"]), 0.4):
            continue
        dataset = _norm_dataset(row["dataset"])
        if dataset in DATASETS:
            index[(dataset, int(row["seed"]))] = row
    return index


def _method_config_path(dataset: str) -> Path:
    return (
        NOISYGL_ROOT
        / "config"
        / METHOD
        / f"{METHOD}_{dataset}.yaml"
    )


def _mask_to_indices(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy().astype(np.int64)
        return mask.detach().cpu().numpy().astype(np.int64)
    array = np.asarray(mask)
    if array.dtype == np.bool_:
        return np.flatnonzero(array).astype(np.int64)
    return array.astype(np.int64)


def _independent_targets(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
    target_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.one_hot(noisy_label.long(), num_classes=n_classes).float()
    alpha = torch.zeros_like(noisy_label, dtype=torch.float)
    zero_point = 1.0 / n_classes
    for ball_id in sorted(np.unique(assignment).tolist()):
        members = train_idx[assignment[train_idx] == ball_id]
        if members.size < min_train_per_ball:
            continue
        member_tensor = torch.as_tensor(members, dtype=torch.long)
        counts = torch.bincount(noisy_label[member_tensor].long(), minlength=n_classes)
        total = int(counts.sum().item())
        margin = float(counts.max().item() / total)
        ball_alpha = max(0.0, min(1.0, (margin - zero_point) / (1.0 - zero_point)))
        if ball_alpha <= 0:
            continue
        majority = int(torch.argmax(counts).item())
        distribution = counts.float() / total
        local = (
            (1.0 - target_beta) * distribution
            + target_beta * F.one_hot(torch.tensor(majority), num_classes=n_classes).float()
        )
        target[member_tensor] = (
            (1.0 - ball_alpha) * target[member_tensor]
            + ball_alpha * local.unsqueeze(0)
        )
        alpha[member_tensor] = ball_alpha
    return target, alpha


def _tie_metrics(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
    target_beta: float,
    soft_target: torch.Tensor,
    alpha: torch.Tensor,
) -> dict:
    tie_balls = 0
    active_tie_balls = 0
    affected = np.zeros(int(noisy_label.shape[0]), dtype=bool)
    selected_ball_counts: Counter[int] = Counter()
    selected_node_counts: Counter[int] = Counter()
    zero_point = 1.0 / n_classes

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = train_idx[assignment[train_idx] == ball_id]
        if members.size < min_train_per_ball:
            continue
        labels = noisy_label[torch.as_tensor(members, dtype=torch.long)].cpu().numpy()
        counts = np.bincount(labels, minlength=n_classes)
        maximum = int(counts.max())
        winners = np.flatnonzero(counts == maximum)
        if winners.size < 2:
            continue
        tie_balls += 1
        selected = int(winners.min())
        margin = maximum / int(counts.sum())
        ball_alpha = max(0.0, min(1.0, (margin - zero_point) / (1.0 - zero_point)))
        if ball_alpha <= 0.0 or target_beta <= 0.0:
            continue
        active_tie_balls += 1
        affected[members] = True
        selected_ball_counts[selected + 1] += 1
        selected_node_counts[selected + 1] += int(members.size)

    train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
    affected_idx = np.flatnonzero(affected)
    affected_tensor = torch.as_tensor(affected_idx, dtype=torch.long)
    total_alpha = float(alpha[train_tensor].sum().item())
    affected_alpha = float(alpha[affected_tensor].sum().item()) if affected_idx.size else 0.0

    # Reverse class indices. For every top-count tie, this maps the original
    # smallest tied class to the largest transformed index and the original
    # largest tied class to the smallest transformed index. It is therefore a
    # deterministic witness for every active argmax tie.
    permutation = torch.arange(n_classes - 1, -1, -1, dtype=torch.long)
    permuted_labels = permutation[noisy_label.long()]
    permuted_target, permuted_alpha, _ = apply_ball_soft_targets(
        noisy_label=permuted_labels,
        assignment=assignment,
        train_mask=train_idx,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=target_beta,
        leave_one_out=False,
    )
    restored_target = permuted_target[:, permutation]
    per_node_l1 = torch.abs(soft_target - restored_target).sum(dim=1)
    train_l1 = per_node_l1[train_tensor]
    changed = train_l1 > TARGET_TOLERANCE
    builder_alpha_error = float(torch.max(torch.abs(alpha - permuted_alpha)).item())
    expected_l1 = 2.0 * target_beta * affected_alpha
    l1_sum = float(train_l1.sum().item())

    affected_ratio = float(affected_idx.size / max(1, train_idx.size))
    alpha_share = None if total_alpha <= 0 else float(affected_alpha / total_alpha)
    material = affected_ratio >= MATERIAL_NODE_RATIO or (
        alpha_share is not None and alpha_share >= MATERIAL_ALPHA_SHARE
    )
    return {
        "eligible_balls": int(
            sum(
                1
                for ball_id in np.unique(assignment)
                if int(np.sum(assignment[train_idx] == ball_id)) >= min_train_per_ball
            )
        ),
        "tie_balls": int(tie_balls),
        "target_active_tie_balls": int(active_tie_balls),
        "affected_train_nodes": int(affected_idx.size),
        "affected_train_node_ratio": affected_ratio,
        "total_train_alpha_sum": total_alpha,
        "affected_train_alpha_sum": affected_alpha,
        "affected_train_alpha_share": "" if alpha_share is None else alpha_share,
        "min_index_selected_class_1based_ball_counts_json": json.dumps(
            dict(sorted(selected_ball_counts.items())), separators=(",", ":")
        ),
        "min_index_selected_class_1based_node_counts_json": json.dumps(
            dict(sorted(selected_node_counts.items())), separators=(",", ":")
        ),
        "permutation": "reverse_class_order",
        "permutation_changed_train_nodes": int(changed.sum().item()),
        "permutation_changed_train_node_ratio": float(changed.float().mean().item()),
        "permutation_train_l1_sum": l1_sum,
        "permutation_train_l1_mean": float(train_l1.mean().item()),
        "permutation_train_l1_max": float(train_l1.max().item()),
        "permutation_equivariant": int(int(changed.sum().item()) == 0),
        "expected_permutation_l1_sum": expected_l1,
        "permutation_l1_formula_abs_error": abs(l1_sum - expected_l1),
        "material_node_ratio_threshold": MATERIAL_NODE_RATIO,
        "material_alpha_share_threshold": MATERIAL_ALPHA_SHARE,
        "material_flag": int(material),
        "builder_alpha_max_abs_error": builder_alpha_error,
    }


def _formal_modified_count(row: dict) -> int | None:
    for key in ("modified_train_count", "modified_count"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _empty_row(dataset: str, noise_type: str, noise_rate: float, seed: int) -> dict:
    return {
        field: ""
        for field in FIELDNAMES
    } | {
        "audit_status": "GAP",
        "dataset": dataset,
        "setting": f"{noise_type}_{noise_rate:.1f}",
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "seed": seed,
        "method": METHOD,
        "material_node_ratio_threshold": MATERIAL_NODE_RATIO,
        "material_alpha_share_threshold": MATERIAL_ALPHA_SHARE,
    }


def run(output: Path) -> list[dict]:
    formal_index = _build_formal_index()
    phase4_sources = _build_phase4_source_index()
    anchors = _build_uniform_anchor_index()
    formula_hash = _sha256(FORMULA_SOURCE)
    ball_hash = _sha256(BALL_SOURCE)
    script_hash = _sha256(SCRIPT_PATH)
    rows: list[dict] = []

    for dataset in DATASETS:
        config_path = _method_config_path(dataset)
        dataset_config_path = NOISYGL_ROOT / "config" / "_dataset" / f"{dataset}.yaml"
        embedding_path = embedding_path_for(EMBEDDING_ROOT, dataset, "grace").resolve()
        try:
            conf = load_conf(None, METHOD, dataset)
            gb = conf.gb
            max_balls = int(gb["max_balls"])
            min_ball_size = int(gb["min_ball_size"])
            split_improvement = float(gb["split_improvement"])
            min_train_per_ball = int(gb["min_train_per_ball"])
            alpha_baseline = str(gb["alpha_baseline"])
            target_mode = str(gb["target_mode"])
            target_beta = float(gb["target_beta"])
            if (alpha_baseline, target_mode) != ("multiclass", "distribution"):
                raise RuntimeError(
                    f"unexpected target definition: alpha={alpha_baseline}, mode={target_mode}"
                )
            if not math.isclose(target_beta, 0.5):
                raise RuntimeError(f"unexpected target_beta={target_beta}")
            shared_hashes = {
                "embedding_path": str(embedding_path),
                "embedding_sha256": _sha256(embedding_path),
                "method_config_path": str(config_path.resolve()),
                "method_config_sha256": _sha256(config_path),
                "dataset_config_path": str(dataset_config_path.resolve()),
                "dataset_config_sha256": _sha256(dataset_config_path),
                "formula_source": str(FORMULA_SOURCE.resolve()),
                "formula_source_sha256": formula_hash,
                "ball_source": str(BALL_SOURCE.resolve()),
                "ball_source_sha256": ball_hash,
                "audit_script_sha256": script_hash,
            }
        except Exception as exc:
            for noise_type, noise_rate in SETTINGS:
                for seed in SEEDS:
                    row = _empty_row(dataset, noise_type, noise_rate, seed)
                    row["gap_reason"] = f"asset/config load failed: {type(exc).__name__}: {exc}"
                    rows.append(row)
            continue

        for seed in SEEDS:
            assignment = None
            for noise_type, noise_rate in SETTINGS:
                row = _empty_row(dataset, noise_type, noise_rate, seed)
                key = _key(dataset, noise_type, noise_rate, seed)
                formal = formal_index[key]
                formal_source = Path(phase4_sources.get(key, formal["_source_path"]))
                row.update(shared_hashes)
                row.update(
                    {
                        "formal_source": str(formal_source),
                        "formal_source_sha256": _sha256(formal_source),
                    }
                )
                try:
                    setup_seed(seed)
                    data = _load_dataset(
                        dataset,
                        device="cpu",
                        split_type="percent",
                        train_percent=0.2,
                        val_percent=0.1,
                        test_percent=0.7,
                    )
                    noisy_label, modified_mask = label_process(
                        labels=data.labels,
                        features=data.feats,
                        n_classes=data.n_classes,
                        noise_type=noise_type,
                        noise_rate=noise_rate,
                        random_seed=seed,
                        eligible_mask=data.train_masks,
                        debug=False,
                    )
                    train_idx = _mask_to_indices(data.train_masks)
                    if assignment is None:
                        built = build_frozen_grace_soft_targets(
                            embedding_path=embedding_path,
                            noisy_label=noisy_label,
                            clean_label=data.labels,
                            train_mask=data.train_masks,
                            n_classes=data.n_classes,
                            max_balls=max_balls,
                            min_ball_size=min_ball_size,
                            split_improvement=split_improvement,
                            min_train_per_ball=min_train_per_ball,
                            alpha_baseline=alpha_baseline,
                            target_mode=target_mode,
                            target_beta=target_beta,
                            random_state=seed,
                            random_control=False,
                        )
                        assignment = np.asarray(built.assignment, dtype=np.int64)
                        soft_target = built.soft_target.cpu()
                        alpha = built.alpha.cpu()
                    else:
                        soft_target, alpha, _ = apply_ball_soft_targets(
                            noisy_label=noisy_label,
                            assignment=assignment,
                            train_mask=data.train_masks,
                            n_classes=data.n_classes,
                            min_train_per_ball=min_train_per_ball,
                            alpha_baseline=alpha_baseline,
                            target_mode=target_mode,
                            target_beta=target_beta,
                            leave_one_out=False,
                        )
                        soft_target = soft_target.cpu()
                        alpha = alpha.cpu()

                    independent_target, independent_alpha = _independent_targets(
                        noisy_label=noisy_label.cpu(),
                        assignment=assignment,
                        train_idx=train_idx,
                        n_classes=int(data.n_classes),
                        min_train_per_ball=min_train_per_ball,
                        target_beta=target_beta,
                    )
                    target_error = float(torch.max(torch.abs(soft_target - independent_target)).item())
                    alpha_error = float(torch.max(torch.abs(alpha - independent_alpha)).item())
                    if target_error > TARGET_TOLERANCE or alpha_error > TARGET_TOLERANCE:
                        raise RuntimeError(
                            f"independent target mismatch target={target_error} alpha={alpha_error}"
                        )

                    split_hash = _split_fingerprint(
                        data.train_masks, data.val_masks, data.test_masks
                    )
                    noise_hash = _noise_fingerprint(data.labels, noisy_label)
                    # label_process returns the global indices whose labels were
                    # changed, not a boolean mask. Noise scope is train-only here;
                    # intersect explicitly instead of treating the array as a mask.
                    reconstructed_modified = int(
                        np.intersect1d(
                            np.asarray(modified_mask, dtype=np.int64),
                            train_idx,
                            assume_unique=False,
                        ).size
                    )
                    reconstructed_actual = float(
                        (noisy_label[train_idx] != data.labels[train_idx]).float().mean().item()
                    )
                    formal_modified = _formal_modified_count(formal)
                    formal_actual = _float_or_none(formal.get("actual_train_noise_rate"))
                    formal_split = formal.get("split_hash", "")
                    formal_noise = formal.get("noise_hash", "")
                    anchor = anchors.get((dataset, seed)) if (
                        noise_type == "uniform" and math.isclose(noise_rate, 0.4)
                    ) else None
                    anchor_num_balls = _int_or_none(anchor.get("num_balls")) if anchor else None
                    anchor_mean_alpha = _float_or_none(anchor.get("mean_alpha_train")) if anchor else None
                    num_balls = int(np.unique(assignment).size)

                    if formal_split and formal_noise:
                        reconstruction_strength = "reconstructed_hash_verified"
                    elif anchor is not None:
                        reconstruction_strength = "reconstructed_uniform_anchor_verified"
                    elif formal_modified is not None:
                        reconstruction_strength = "reconstructed_modified_count_verified_no_archived_hash"
                    else:
                        reconstruction_strength = "reconstructed_no_formal_identity_anchor"

                    row.update(
                        {
                            "audit_status": "OK",
                            "gap_reason": "",
                            "num_classes": int(data.n_classes),
                            "train_nodes": int(train_idx.size),
                            "num_balls": num_balls,
                            "reconstruction_strength": reconstruction_strength,
                            "reconstructed_split_hash": split_hash,
                            "formal_split_hash": formal_split,
                            "split_hash_match": "" if not formal_split else int(split_hash == formal_split),
                            "reconstructed_noise_hash": noise_hash,
                            "formal_noise_hash": formal_noise,
                            "noise_hash_match": "" if not formal_noise else int(noise_hash == formal_noise),
                            "reconstructed_modified_train_count": reconstructed_modified,
                            "formal_modified_train_count": "" if formal_modified is None else formal_modified,
                            "modified_train_count_match": ""
                            if formal_modified is None
                            else int(reconstructed_modified == formal_modified),
                            "reconstructed_actual_train_noise_rate": reconstructed_actual,
                            "formal_actual_train_noise_rate": "" if formal_actual is None else formal_actual,
                            "actual_train_noise_rate_abs_error": ""
                            if formal_actual is None
                            else abs(reconstructed_actual - formal_actual),
                            "anchor_num_balls": "" if anchor_num_balls is None else anchor_num_balls,
                            "anchor_num_balls_match": ""
                            if anchor_num_balls is None
                            else int(num_balls == anchor_num_balls),
                            "anchor_mean_alpha_train": ""
                            if anchor_mean_alpha is None
                            else anchor_mean_alpha,
                            "anchor_mean_alpha_abs_error": ""
                            if anchor_mean_alpha is None
                            else abs(float(alpha[train_idx].mean().item()) - anchor_mean_alpha),
                            "independent_target_max_abs_error": target_error,
                        }
                    )
                    row.update(
                        _tie_metrics(
                            noisy_label=noisy_label.cpu(),
                            assignment=assignment,
                            train_idx=train_idx,
                            n_classes=int(data.n_classes),
                            min_train_per_ball=min_train_per_ball,
                            target_beta=target_beta,
                            soft_target=soft_target,
                            alpha=alpha,
                        )
                    )
                except Exception as exc:
                    row["audit_status"] = "GAP"
                    row["gap_reason"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only reconstruction audit for argmax tie exposure in formal GB-LCSS targets."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_ROOT / "paper_pipeline" / "00_evidence" / "tie_safety_audit.csv",
    )
    args = parser.parse_args()
    rows = run(args.output.resolve())
    ok = sum(row["audit_status"] == "OK" for row in rows)
    gaps = len(rows) - ok
    print(f"WROTE {args.output.resolve()}")
    print(f"ROWS={len(rows)} OK={ok} GAP={gaps}")


if __name__ == "__main__":
    main()
