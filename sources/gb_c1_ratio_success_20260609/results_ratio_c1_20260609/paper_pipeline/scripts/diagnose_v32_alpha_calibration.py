from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

# Match the single-threaded environment frozen by the immutable full-grid run.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("PYTHONHASHSEED", "0")

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
from utils.frozen_grace_targets import build_frozen_grace_soft_targets
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


METHOD = "frozengrace_gb_mlp_softdist_beta05"
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SEEDS = tuple(range(5))
ALPHA_EDGES = np.asarray((0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001))
EMBEDDING_ROOT = (
    PROJECT_ROOT
    / "gb_graph_aware_minival_6_1"
    / "checkpoints"
    / "embeddings_v2"
)
FORMAL_RAW = (
    NOISYGL_ROOT
    / "runs"
    / "tie_safe_full_grid_20260727_v2_immutable"
    / "raw_combined.csv"
)
DEFAULT_OUTPUT = (
    RESULT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v32_alpha_calibration.csv"
)
DEFAULT_REPORT = (
    RESULT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v32_alpha_calibration_report.md"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mask_to_indices(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return (
                torch.nonzero(mask, as_tuple=False)
                .flatten()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
        return mask.detach().cpu().numpy().astype(np.int64)
    array = np.asarray(mask)
    if array.dtype == np.bool_:
        return np.flatnonzero(array).astype(np.int64)
    return array.astype(np.int64)


def formal_index(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows = read_csv(path)
    selected = {}
    for row in rows:
        if row.get("method") != METHOD:
            continue
        if row.get("noise_type") != "uniform":
            continue
        if not math.isclose(float(row["noise_rate"]), 0.4):
            continue
        key = (row["dataset"].lower(), int(row["seed"]))
        selected[key] = row
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise RuntimeError(
            f"formal uniform0.4 index mismatch: missing={missing} extra={extra}"
        )
    return selected


def summarize_bins(
    dataset: str,
    records: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    alpha = np.asarray(records["alpha"], dtype=float)
    if any(np.asarray(values).shape != alpha.shape for values in records.values()):
        raise ValueError("all diagnostic arrays must have identical shapes")
    rows = []
    for index in range(len(ALPHA_EDGES) - 1):
        lower = ALPHA_EDGES[index]
        upper = ALPHA_EDGES[index + 1]
        selected = (alpha >= lower) & (alpha < upper)
        if not np.any(selected):
            continue
        label_upper = 1.0 if index == len(ALPHA_EDGES) - 2 else upper
        rows.append(
            {
                "dataset": dataset,
                "alpha_bin": f"[{lower:.1f},{label_upper:.1f})",
                "count": int(selected.sum()),
                "mean_alpha": float(alpha[selected].mean()),
                "target_argmax_accuracy": float(
                    np.asarray(records["target_correct"])[selected].mean()
                ),
                "clean_label_mass": float(
                    np.asarray(records["clean_mass"])[selected].mean()
                ),
                "brier": float(np.asarray(records["brier"])[selected].mean()),
                "mean_train_votes": float(
                    np.asarray(records["train_votes"])[selected].mean()
                ),
                "tie_rate": float(np.asarray(records["tie"])[selected].mean()),
            }
        )
    return rows


def concatenate_records(
    items: Iterable[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    items = list(items)
    return {
        key: np.concatenate([item[key] for item in items])
        for key in items[0]
    }


def reconstruct_one(
    dataset: str,
    seed: int,
    anchor: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    setup_seed(seed)
    data = _load_dataset(
        dataset,
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, modified = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    conf = load_conf(None, METHOD, dataset)
    gb = conf.gb
    embedding_path = embedding_path_for(
        EMBEDDING_ROOT, dataset, "grace"
    ).resolve()
    built = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=int(gb["max_balls"]),
        min_ball_size=int(gb["min_ball_size"]),
        split_improvement=float(gb["split_improvement"]),
        min_train_per_ball=int(gb["min_train_per_ball"]),
        alpha_baseline=str(gb["alpha_baseline"]),
        target_mode=str(gb["target_mode"]),
        target_beta=float(gb["target_beta"]),
        tie_mode="distribution_only",
        random_state=seed,
        random_control=False,
    )

    split_hash = _split_fingerprint(
        data.train_masks, data.val_masks, data.test_masks
    )
    noise_hash = _noise_fingerprint(data.labels, noisy_label)
    assignment_hash = str(built.diagnostics["ball_assignment_hash"])
    checks = {
        "split_hash_match": split_hash == anchor["split_hash"],
        "noise_hash_match": noise_hash == anchor["noise_hash"],
        "assignment_hash_match": (
            assignment_hash == anchor["ball_assignment_hash"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"identity mismatch dataset={dataset} seed={seed}: {checks}"
        )

    train_idx = mask_to_indices(data.train_masks)
    assignment = np.asarray(built.assignment, dtype=np.int64)
    train_votes_by_ball: dict[int, int] = {}
    tie_by_ball: dict[int, bool] = {}
    for ball_id in np.unique(assignment):
        members = train_idx[assignment[train_idx] == ball_id]
        train_votes_by_ball[int(ball_id)] = int(members.size)
        if members.size < int(gb["min_train_per_ball"]):
            tie_by_ball[int(ball_id)] = False
            continue
        labels = noisy_label[
            torch.as_tensor(members, dtype=torch.long)
        ].cpu()
        counts = torch.bincount(labels, minlength=int(data.n_classes))
        tie_by_ball[int(ball_id)] = bool(
            int((counts == counts.max()).sum().item()) > 1
        )

    train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
    targets = built.soft_target[train_tensor].cpu()
    clean = data.labels[train_tensor].long().cpu()
    clean_onehot = F.one_hot(
        clean, num_classes=int(data.n_classes)
    ).float()
    node_balls = assignment[train_idx]
    records = {
        "alpha": built.alpha[train_tensor].cpu().numpy(),
        "target_correct": (
            targets.argmax(dim=1) == clean
        ).float().numpy(),
        "clean_mass": targets[
            torch.arange(targets.shape[0]), clean
        ].numpy(),
        "brier": ((targets - clean_onehot) ** 2).sum(dim=1).numpy(),
        "train_votes": np.asarray(
            [train_votes_by_ball[int(ball)] for ball in node_balls],
            dtype=float,
        ),
        "tie": np.asarray(
            [tie_by_ball[int(ball)] for ball in node_balls],
            dtype=float,
        ),
    }
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "split_hash": split_hash,
        "noise_hash": noise_hash,
        "ball_assignment_hash": assignment_hash,
        "modified_count": int(np.asarray(modified).size),
        "train_nodes": int(train_idx.size),
    }
    return records, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def monotonic_accuracy(rows: list[dict]) -> bool:
    ordered = sorted(
        rows,
        key=lambda row: float(row["mean_alpha"]),
    )
    values = [float(row["target_argmax_accuracy"]) for row in ordered]
    return all(
        right + 1e-12 >= left
        for left, right in zip(values, values[1:])
    )


def build_report(
    rows: list[dict],
    metadata: list[dict],
) -> str:
    overall = [row for row in rows if row["dataset"] == "ALL"]
    per_dataset = {
        dataset: monotonic_accuracy(
            [row for row in rows if row["dataset"] == dataset]
        )
        for dataset in DATASETS
    }
    overall_monotonic = monotonic_accuracy(overall)
    table_lines = [
        "| alpha 区间 | 节点数 | 平均 alpha | 目标 argmax 正确率 | clean-label mass | Brier | 平均训练票数 | 并列率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        table_lines.append(
            "| {alpha_bin} | {count} | {mean_alpha:.4f} | "
            "{target_argmax_accuracy:.4f} | {clean_label_mass:.4f} | "
            "{brier:.4f} | {mean_train_votes:.2f} | {tie_rate:.4f} |".format(
                **row
            )
        )
    verdict = (
        "整体分箱正确率随 alpha 单调不降。"
        if overall_monotonic
        else "整体分箱正确率并未随 alpha 严格单调不降。"
    )
    return "\n".join(
        (
            "# V32 alpha 诊断",
            "",
            "该诊断仅重建 uniform 0.4 下五个核心数据集、五个种子的 "
            "tie-safe `distribution_only` 软目标，不训练分类器。clean "
            "label 只用于事后评估目标质量。",
            "",
            f"- 身份核验：{len(metadata)}/{len(metadata)} 个重建项的 "
            "split、noise 与 ball-assignment 哈希全部匹配。",
            f"- 总体判定：{verdict}",
            "- 数据集内单调性："
            + json.dumps(per_dataset, ensure_ascii=False, sort_keys=True),
            "- 解释边界：alpha 仍是启发式归一化共识权重，不应称为校准置信度。",
            "",
            *table_lines,
            "",
        )
    )


def run(
    *,
    formal_raw: Path = FORMAL_RAW,
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> tuple[list[dict], list[dict]]:
    anchors = formal_index(formal_raw)
    records_by_dataset: dict[str, list[dict[str, np.ndarray]]] = {
        dataset: [] for dataset in DATASETS
    }
    metadata = []
    for dataset in DATASETS:
        for seed in SEEDS:
            records, item_metadata = reconstruct_one(
                dataset, seed, anchors[(dataset, seed)]
            )
            records_by_dataset[dataset].append(records)
            metadata.append(item_metadata)

    rows = []
    all_records = []
    for dataset, items in records_by_dataset.items():
        combined = concatenate_records(items)
        all_records.append(combined)
        rows.extend(summarize_bins(dataset, combined))
    rows.extend(
        summarize_bins("ALL", concatenate_records(all_records))
    )
    write_csv(output, rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(rows, metadata), encoding="utf-8")
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_raw", type=Path, default=FORMAL_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows, metadata = run(
        formal_raw=args.formal_raw.resolve(),
        output=args.output.resolve(),
        report=args.report.resolve(),
    )
    print(f"ROWS={len(rows)} RECONSTRUCTIONS={len(metadata)}")
    print(f"WROTE {args.output.resolve()}")
    print(f"WROTE {args.report.resolve()}")


if __name__ == "__main__":
    main()
