from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls import build_granular_balls, build_graph_aware_balls
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.diagnostics.class_failure import (
    class_confusion_matrix,
    node_profile_stats,
    per_class_target_diagnostics,
    summarize_class_rows,
)
from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets, train_noisy_linear_teacher
from noisy_label_gb_grace.smoothing.targets import CONFLICT_GATE_REPAIR_CONFIGS, REPAIR_CONTROL_CONFIGS
from noisy_label_gb_grace.utils import PROJECT_ROOT, get_device, save_csv, save_json


DEFAULT_METHODS = [
    "gb",
    "gagb",
    "khop_only",
    "gagb_qb_conflict_gate_alpha03",
    "gagb_qb_conflict_gate_alpha06",
    "gagb_qb_conflict_gate_alpha07",
    "gagb_qb_conflict_gate_alpha08",
    "gagb_qb_conflict_gate_alpha09",
    "gagb_qb_conflict_gate_replace_qmix",
    "gagb_class_capped_alpha09",
    "gagb_class_balanced_alpha09",
    "gagb_boundary_safe_alpha09",
    "khop_only_alpha09",
    "khop_only_replace",
    "gagb_replay_random_same_count",
]


def embedding_path(dataset: str, split_seed: int, pretrain_seed: int, split_mode: str, train_ratio: float | None, val_ratio: float | None) -> Path:
    split_token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "embeddings" / dataset / f"{split_token}_pretrain{pretrain_seed}" / "H.pt"


def _method_alpha(method: str, default_alpha: float) -> float:
    key = method.lower()
    config = CONFLICT_GATE_REPAIR_CONFIGS.get(key)
    if config is None and key in REPAIR_CONTROL_CONFIGS:
        config = CONFLICT_GATE_REPAIR_CONFIGS[REPAIR_CONTROL_CONFIGS[key]["base_method"]]
    return float(config["alpha_max"]) if config is not None else float(default_alpha)


def _method_grouping(method: str, gb_grouping: dict, gagb_grouping: dict, khop_grouping: dict) -> dict:
    key = method.lower()
    if key in {"gagb", "gagb_ca_ce"} | set(CONFLICT_GATE_REPAIR_CONFIGS):
        return gagb_grouping
    if key in {"khop_only", "khop_only_alpha09", "khop_only_replace"}:
        return khop_grouping
    return gb_grouping


def _degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    neighbors = [set() for _ in range(num_nodes)]
    edge_index = edge_index.detach().cpu().long()
    for src, dst in edge_index.t().tolist():
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return torch.tensor([len(item) for item in neighbors], dtype=torch.float32)


def _distribution(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    counts = torch.bincount(labels[indices].long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


def _entropy(probs: torch.Tensor) -> float:
    probs = probs.detach().cpu().float()
    if probs.numel() == 0 or float(probs.sum().item()) <= 0.0:
        return math.nan
    probs = probs / probs.sum().clamp_min(1e-12)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())


def _top1_margin(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.detach().cpu().float()
    if probs.dim() != 2 or probs.size(1) == 0:
        return torch.zeros(probs.size(0), dtype=torch.float32)
    if probs.size(1) == 1:
        return probs[:, 0]
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def qmix_by_node(
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    grouping: dict,
    teacher_probs: torch.Tensor,
    *,
    num_classes: int,
    min_train_per_group: int,
    qb_pred_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    probs = teacher_probs.detach().cpu().float()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    qmix = torch.zeros((labels.numel(), num_classes), dtype=torch.float32)
    valid = torch.zeros(labels.numel(), dtype=torch.bool)
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        if train_in_group.numel() < int(min_train_per_group) or idx.numel() == 0:
            continue
        q_label = _distribution(labels, train_in_group, num_classes)
        q_pred = probs[idx].mean(dim=0)
        q_pred = q_pred / q_pred.sum().clamp_min(1e-12)
        q_group = (1.0 - float(qb_pred_beta)) * q_label + float(qb_pred_beta) * q_pred
        q_group = q_group / q_group.sum().clamp_min(1e-12)
        qmix[train_in_group] = q_group
        valid[train_in_group] = True
    return qmix, valid


def ball_profile_values(
    grouping: dict,
    qmix: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    num_nodes = int(clean_labels.numel())
    group_ids = grouping["group_ids"].detach().cpu().long()
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    purity = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
    entropy = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
    train_count = torch.zeros(num_nodes, dtype=torch.float32)
    size = torch.zeros(num_nodes, dtype=torch.float32)
    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        if idx.numel() == 0:
            continue
        train_in_group = idx[train_mask[idx]]
        size[idx] = float(idx.numel())
        train_count[idx] = float(train_in_group.numel())
        if train_in_group.numel() == 0:
            continue
        clean_counts = torch.bincount(clean[train_in_group], minlength=num_classes).float()
        purity[idx] = clean_counts.max() / clean_counts.sum().clamp_min(1.0)
        entropy[idx] = _entropy(_distribution(noisy, train_in_group, num_classes))
    return {
        "ball_clean_purity": purity,
        "ball_noisy_entropy": entropy,
        "ball_train_count": train_count,
        "ball_size": size,
        "qmix_conf": qmix.max(dim=1).values,
        "qmix_margin": _top1_margin(qmix),
        "group_id": group_ids.float(),
    }


def per_class_accuracy(predictions: torch.Tensor, clean_labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> dict[int, float]:
    pred = predictions.detach().cpu().long()
    clean = clean_labels.detach().cpu().long()
    mask = mask.detach().cpu().bool()
    out = {}
    for class_id in range(num_classes):
        class_mask = mask & (clean == class_id)
        denom = int(class_mask.sum().item())
        out[class_id] = float(((pred == clean) & class_mask).float().sum().item() / denom) if denom else math.nan
    return out


def diagnostic_case(args: argparse.Namespace, dataset: str, train_ratio: float, noise_rate: float, seed: int, device: torch.device) -> tuple[list[dict], list[dict]]:
    split_seed = int(args.split_seed)
    artifact = embedding_path(dataset, split_seed, seed, args.split_mode, train_ratio, args.val_ratio)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {artifact}")
    payload = torch.load(artifact, map_location="cpu")
    h = payload["H"].float()
    h_norm = F.normalize(h, p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")
    if edge_index is None:
        _, data = load_dataset(
            dataset,
            device=None,
            split_mode=args.split_mode,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=args.val_ratio,
        )
        edge_index = data.edge_index.detach().cpu().long()
    split_token = split_name(args.split_mode, split_seed, train_ratio, args.val_ratio)
    noisy_labels, _, _ = load_or_create_noisy_labels(
        clean_labels,
        masks["train_mask"],
        num_classes,
        dataset=dataset,
        noise_type=args.noise_type,
        noise_rate=noise_rate,
        split_seed=split_seed,
        noise_seed=seed,
        split_name=split_token,
    )

    gb_grouping = build_granular_balls(
        h_norm,
        min_ball_size=args.gb_min_ball_size,
        min_radius_gain=args.gb_min_radius_gain,
        seed=seed,
    )
    gagb_grouping = build_graph_aware_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=args.gagb_lambda_h,
        lambda_t=args.gagb_lambda_t,
        max_ball_size=args.gagb_max_ball_size,
        seed=seed,
    )
    khop_grouping = build_graph_aware_balls(
        h_norm,
        edge_index,
        max_hops=args.gagb_max_hops,
        min_ball_size=args.gb_min_ball_size,
        tau_add=args.gagb_tau_add,
        lambda_h=0.0,
        lambda_t=1.0,
        max_ball_size=args.gagb_max_ball_size,
        seed=seed,
    )
    teacher_probs, _ = train_noisy_linear_teacher(
        h_norm,
        noisy_labels,
        masks["train_mask"],
        num_classes,
        seed=seed,
        epochs=args.teacher_epochs,
        learning_rate=args.teacher_lr,
        weight_decay=args.teacher_weight_decay,
        temperature=args.teacher_temperature,
        device=device,
    )
    degree = _degrees(edge_index, int(clean_labels.numel()))
    teacher_pred = teacher_probs.argmax(dim=1)
    teacher_conf = teacher_probs.max(dim=1).values

    class_rows = []
    confusion_rows = []
    for method in args.methods:
        grouping = _method_grouping(method, gb_grouping, gagb_grouping, khop_grouping)
        alpha = _method_alpha(method, args.alpha)
        soft_targets, target_stats = build_method_targets(
            method,
            h_norm,
            noisy_labels,
            masks["train_mask"],
            num_classes,
            alpha_max=alpha,
            gb_grouping=grouping,
            min_train_per_ball=args.min_train_per_ball,
            eps=args.eps,
            prior_gamma=args.prior_gamma,
            seed=seed,
            teacher_probs=teacher_probs,
            qb_pred_beta=args.qb_pred_beta,
            qb_teacher_gamma=args.qb_teacher_gamma,
            conflict_noisy_support_max=args.conflict_noisy_support_max,
            conflict_target_conf_min=args.conflict_target_conf_min,
        )
        target_pred = soft_targets.argmax(dim=1)
        gate_mask = torch.tensor(target_stats.get("target_update_gate_mask", [False] * clean_labels.numel()), dtype=torch.bool)
        qmix, qmix_valid = qmix_by_node(
            noisy_labels,
            masks["train_mask"],
            grouping,
            teacher_probs,
            num_classes=num_classes,
            min_train_per_group=args.min_train_per_ball,
            qb_pred_beta=args.qb_pred_beta,
        )
        profile_values = ball_profile_values(
            grouping,
            qmix,
            clean_labels,
            noisy_labels,
            masks["train_mask"],
            num_classes=num_classes,
        )
        profile_values.update({"degree": degree, "teacher_conf": teacher_conf})
        eval_stats = train_frozen_classifier(
            h,
            clean_labels,
            soft_targets,
            masks["train_mask"],
            masks["val_mask"],
            masks["test_mask"],
            seed=seed,
            epochs=args.classifier_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            return_logits=True,
        )
        predictions = eval_stats["predictions"]
        rows = per_class_target_diagnostics(
            clean_labels,
            noisy_labels,
            target_pred,
            masks["train_mask"],
            num_classes=num_classes,
            gated_mask=gate_mask,
            teacher_pred=teacher_pred,
            test_predictions=predictions,
            test_mask=masks["test_mask"],
            values_by_name=profile_values,
        )
        train_noisy = masks["train_mask"] & (clean_labels != noisy_labels)
        train_clean = masks["train_mask"] & (clean_labels == noisy_labels)
        corrected = train_noisy & (target_pred == clean_labels)
        damaged = train_clean & (target_pred != clean_labels)
        failed_gated = gate_mask & train_noisy & (target_pred != clean_labels)
        profile_summary = {}
        profile_summary.update(node_profile_stats(profile_values, corrected, prefix="overall_corrected"))
        profile_summary.update(node_profile_stats(profile_values, damaged, prefix="overall_damaged"))
        profile_summary.update(node_profile_stats(profile_values, failed_gated, prefix="overall_failed_gated"))
        test_acc_by_class = per_class_accuracy(predictions, clean_labels, masks["test_mask"], num_classes)
        for row in rows:
            class_id = int(row["class_id"])
            row.update({
                "dataset": dataset,
                "noise_rate": float(noise_rate),
                "seed": int(seed),
                "method": method,
                "alpha_max": float(alpha),
                "best_val_acc": float(eval_stats["best_val_acc"]),
                "test_acc_at_best_val": float(eval_stats["test_acc_at_best_val"]),
                "test_acc_by_class": test_acc_by_class[class_id],
                "gated_train_count": int(gate_mask.sum().item()),
                **profile_summary,
            })
            class_rows.append(row)
        confusion_rows.append({
            "dataset": dataset,
            "noise_rate": float(noise_rate),
            "seed": int(seed),
            "method": method,
            "qmix_argmax_confusion_matrix": class_confusion_matrix(clean_labels, qmix.argmax(dim=1), gate_mask & qmix_valid, num_classes=num_classes),
            "target_argmax_confusion_matrix": class_confusion_matrix(clean_labels, target_pred, masks["train_mask"], num_classes=num_classes),
            "clean_damage_confusion_matrix": class_confusion_matrix(clean_labels, target_pred, damaged, num_classes=num_classes),
        })
        print(
            "(D) | "
            f"dataset={dataset}, noise={noise_rate:g}, seed={seed}, method={method}, "
            f"test={eval_stats['test_acc_at_best_val']:.4f}, gated={int(gate_mask.sum().item())}"
        )
    return class_rows, confusion_rows


def _aggregate_matrices(confusions: list[dict]) -> list[dict]:
    groups = {}
    for row in confusions:
        key = (row["dataset"], row["noise_rate"], row["method"])
        entry = groups.setdefault(key, {
            "dataset": row["dataset"],
            "noise_rate": row["noise_rate"],
            "method": row["method"],
            "n": 0,
            "qmix_argmax_confusion_matrix": None,
            "target_argmax_confusion_matrix": None,
            "clean_damage_confusion_matrix": None,
        })
        entry["n"] += 1
        for field in ("qmix_argmax_confusion_matrix", "target_argmax_confusion_matrix", "clean_damage_confusion_matrix"):
            matrix = row[field]
            if entry[field] is None:
                entry[field] = [[0 for _ in line] for line in matrix]
            for i, line in enumerate(matrix):
                for j, value in enumerate(line):
                    entry[field][i][j] += int(value)
    return sorted(groups.values(), key=lambda item: (item["dataset"], item["noise_rate"], item["method"]))


def _fmt(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if math.isnan(number) else f"{number:.4f}"


def write_v23_report(path: Path, summary_rows: list[dict], confusion_summary: list[dict]) -> None:
    cora = [row for row in summary_rows if row["dataset"] == "Cora" and float(row["noise_rate"]) == 0.4]
    methods = [
        "gb",
        "gagb_qb_conflict_gate_alpha03",
        "gagb_qb_conflict_gate_alpha06",
        "gagb_qb_conflict_gate_alpha07",
        "gagb_qb_conflict_gate_alpha08",
        "gagb_qb_conflict_gate_alpha09",
        "gagb_qb_conflict_gate_replace_qmix",
        "gagb_class_capped_alpha09",
        "gagb_class_balanced_alpha09",
        "gagb_boundary_safe_alpha09",
        "khop_only_alpha09",
        "khop_only_replace",
        "gagb_replay_random_same_count",
    ]
    lines = [
        "# GAGB v2.3 Class-Safe Target Calibration Report",
        "",
        "## Purpose",
        "",
        "Test whether Cora noise=0.4 failure is caused by over-strong target updates, class-imbalanced correction, or indirect decision-boundary movement. Clean labels are used only for diagnostics/evaluation.",
        "",
        "## Protocol Boundary",
        "",
        "- GRACE embeddings, GAGB construction, teacher training, split, noise protocol, and classifier protocol are unchanged from v2.2.",
        "- Train-time target construction uses only noisy labels, train mask, qmix, teacher probabilities, and grouping.",
        "- `gagb_replay_same_class_count` is intentionally not implemented as a trainable method because it would require clean class counts.",
        "",
        "## Per-class Summary",
        "",
        "| Method | Class | Test Acc | NCR | CDR | Changed | Gated Noisy | Gated Clean | Teacher Gated Acc | Corrected Purity | Damaged Purity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        for row in sorted([item for item in cora if item["method"] == method], key=lambda item: int(item["class_id"])):
            lines.append(
                "| "
                f"{method} | {int(row['class_id'])} | "
                f"{_fmt(row.get('test_acc_by_class_mean'))} | "
                f"{_fmt(row.get('correction_by_class_ncr_mean'))} | "
                f"{_fmt(row.get('clean_damage_by_class_cdr_mean'))} | "
                f"{_fmt(row.get('target_changed_by_clean_class_mean'))} | "
                f"{_fmt(row.get('gated_noisy_by_class_mean'))} | "
                f"{_fmt(row.get('gated_clean_by_class_mean'))} | "
                f"{_fmt(row.get('teacher_clean_acc_by_class_on_gated_nodes_mean'))} | "
                f"{_fmt(row.get('corrected_ball_clean_purity_mean_mean'))} | "
                f"{_fmt(row.get('damaged_ball_clean_purity_mean_mean'))} |"
            )

    by_method_class = {(row["method"], int(row["class_id"])): row for row in cora}
    lines.extend(["", "## Largest Cora 0.4 Class-level Accuracy Drops", ""])
    for method, baseline in [
        ("gagb_qb_conflict_gate_alpha06", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_qb_conflict_gate_alpha07", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_qb_conflict_gate_alpha08", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_class_capped_alpha09", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_class_balanced_alpha09", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_boundary_safe_alpha09", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_replay_random_same_count", "gagb_qb_conflict_gate_alpha09"),
        ("gagb_qb_conflict_gate_alpha09", "gb"),
        ("gagb_qb_conflict_gate_replace_qmix", "gb"),
        ("gagb_qb_conflict_gate_alpha09", "gagb_qb_conflict_gate_alpha03"),
        ("gagb_qb_conflict_gate_replace_qmix", "gagb_qb_conflict_gate_alpha03"),
    ]:
        deltas = []
        for class_id in range(64):
            cur = by_method_class.get((method, class_id))
            base = by_method_class.get((baseline, class_id))
            if cur is None or base is None:
                continue
            delta = float(cur.get("test_acc_by_class_mean", math.nan)) - float(base.get("test_acc_by_class_mean", math.nan))
            if not math.isnan(delta):
                deltas.append((delta, class_id, cur, base))
        if deltas:
            delta, class_id, cur, _ = sorted(deltas, key=lambda item: item[0])[0]
            lines.append(
                f"- `{method}` vs `{baseline}` worst class `{class_id}`: delta `{delta:.4f}`, "
                f"NCR `{_fmt(cur.get('correction_by_class_ncr_mean'))}`, CDR `{_fmt(cur.get('clean_damage_by_class_cdr_mean'))}`, "
                f"gated noisy `{_fmt(cur.get('gated_noisy_by_class_mean'))}`, gated clean `{_fmt(cur.get('gated_clean_by_class_mean'))}`."
            )

    lines.extend(["", "## Class 1 / Class 6 Delta Focus", ""])
    for method in methods:
        if method in {"gb", "gagb_qb_conflict_gate_alpha03", "gagb_qb_conflict_gate_alpha09"}:
            continue
        for class_id in (1, 6):
            cur = by_method_class.get((method, class_id))
            alpha09 = by_method_class.get(("gagb_qb_conflict_gate_alpha09", class_id))
            gb = by_method_class.get(("gb", class_id))
            if cur is None:
                continue
            delta_alpha09 = (
                float(cur.get("test_acc_by_class_mean", math.nan)) - float(alpha09.get("test_acc_by_class_mean", math.nan))
                if alpha09 is not None
                else math.nan
            )
            delta_gb = (
                float(cur.get("test_acc_by_class_mean", math.nan)) - float(gb.get("test_acc_by_class_mean", math.nan))
                if gb is not None
                else math.nan
            )
            lines.append(
                f"- `{method}` class `{class_id}`: acc `{_fmt(cur.get('test_acc_by_class_mean'))}`, "
                f"delta vs alpha09 `{_fmt(delta_alpha09)}`, delta vs GB `{_fmt(delta_gb)}`, "
                f"NCR `{_fmt(cur.get('correction_by_class_ncr_mean'))}`, CDR `{_fmt(cur.get('clean_damage_by_class_cdr_mean'))}`."
            )

    lines.extend(["", "## Confusion Matrix Pointers", ""])
    for item in confusion_summary:
        if item["dataset"] == "Cora" and float(item["noise_rate"]) == 0.4 and item["method"] in methods:
            lines.append(f"- `{item['method']}` aggregate matrices are in `confusion_matrices.json`.")

    lines.extend(["", "## Required Decision Questions", ""])
    lines.append("1. Does alpha06/07/08 recover Cora 0.4 accuracy while keeping NCR > 0?")
    lines.append("2. Does class_capped_alpha09 reduce class 1 and class 6 drops?")
    lines.append("3. Does class_balanced_alpha09 reduce correction concentration?")
    lines.append("4. Does boundary_safe_alpha09 reduce CDR and indirect boundary damage?")
    lines.append("5. Is GAGB still better than khop_only repair after calibration?")
    lines.append("6. Does replay control suggest that failure is due to intervention strength rather than GAGB target content?")
    lines.append("7. Which variant should be kept for the next fair-control run?")

    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.gpu_id)
    class_rows = []
    confusion_rows = []
    for dataset in args.datasets:
        for train_ratio in args.train_ratios:
            for noise_rate in args.noise_rates:
                for seed in args.seeds:
                    rows, matrices = diagnostic_case(args, dataset, float(train_ratio), float(noise_rate), int(seed), device)
                    class_rows.extend(rows)
                    confusion_rows.extend(matrices)
    summary_rows = summarize_class_rows(class_rows)
    confusion_summary = _aggregate_matrices(confusion_rows)
    save_csv(output_dir / "per_seed_class_diagnostics.csv", class_rows)
    save_csv(output_dir / "per_setting_class_summary.csv", summary_rows)
    save_csv(output_dir / "summary.csv", summary_rows)
    save_csv(output_dir / "per_class_summary.csv", summary_rows)
    save_json(output_dir / "confusion_matrices.json", {"per_seed": confusion_rows, "summary": confusion_summary})
    write_v23_report(output_dir / "v2_3_report.md", summary_rows, confusion_summary)
    save_json(output_dir / "run_config.json", vars(args))
    print(f"(S) | Wrote {len(class_rows)} per-class seed rows to {output_dir / 'per_seed_class_diagnostics.csv'}")
    print(f"(S) | Wrote {len(summary_rows)} per-class summary rows to {output_dir / 'per_setting_class_summary.csv'}")
    print(f"(S) | Wrote report to {output_dir / 'v2_3_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    parser.add_argument("--split-mode", choices=["ratio", "planetoid"], default="ratio")
    parser.add_argument("--train-ratios", nargs="+", type=float, default=[0.2])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    parser.add_argument("--noise-rates", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--min-train-per-ball", type=int, default=3)
    parser.add_argument("--gb-min-ball-size", type=int, default=5)
    parser.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    parser.add_argument("--gagb-max-hops", type=int, default=2)
    parser.add_argument("--gagb-tau-add", type=float, default=0.55)
    parser.add_argument("--gagb-lambda-h", type=float, default=0.7)
    parser.add_argument("--gagb-lambda-t", type=float, default=0.3)
    parser.add_argument("--gagb-max-ball-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--prior-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-lr", type=float, default=0.01)
    parser.add_argument("--teacher-weight-decay", type=float, default=5e-4)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--qb-pred-beta", type=float, default=0.5)
    parser.add_argument("--qb-teacher-gamma", type=float, default=2.0)
    parser.add_argument("--conflict-noisy-support-max", type=float, default=0.35)
    parser.add_argument("--conflict-target-conf-min", type=float, default=0.55)
    parser.add_argument("--classifier-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
