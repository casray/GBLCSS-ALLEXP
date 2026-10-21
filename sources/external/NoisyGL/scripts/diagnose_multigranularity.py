from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloader import Dataset
from utils.frozen_grace_targets import apply_ball_soft_targets, load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


SCALE_SPECS = {
    "fine": {"max_balls": 256, "min_ball_size": 8},
    "mid": {"max_balls": 64, "min_ball_size": 32},
    "coarse": {"max_balls": 16, "min_ball_size": 128},
}
RULES = ("all_consistent", "coarse_mid_consistent", "weighted_agreement")


def _conf_value(conf, key, default):
    if isinstance(conf, dict):
        value = conf.get(key, default)
    else:
        value = getattr(conf, key, default)
    return default if value is None else value


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _mask_to_numpy(mask, n_nodes: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            arr = mask.detach().cpu().numpy().astype(bool)
        else:
            idx = mask.detach().cpu().numpy().astype(np.int64)
            arr = np.zeros(int(n_nodes), dtype=bool)
            arr[idx] = True
    else:
        raw = np.asarray(mask)
        if raw.dtype == np.bool_:
            arr = raw.astype(bool)
        else:
            arr = np.zeros(int(n_nodes), dtype=bool)
            arr[raw.astype(np.int64)] = True
    if arr.shape[0] != int(n_nodes):
        raise ValueError("mask length must match node count")
    return arr


def _mask_to_tensor(mask, n_nodes: int, device) -> torch.Tensor:
    return torch.as_tensor(_mask_to_numpy(mask, n_nodes), dtype=torch.bool, device=device)


def _load_dataset(dataset_name, device, split_type, train_percent, val_percent, test_percent):
    data_conf = load_conf(str(REPO_ROOT / "config" / "_dataset" / f"{dataset_name}.yaml"))
    if split_type is not None:
        data_conf.split["split_type"] = split_type
    if train_percent is not None:
        data_conf.split["train_percent"] = train_percent
    if val_percent is not None:
        data_conf.split["val_percent"] = val_percent
    if test_percent is not None:
        data_conf.split["test_percent"] = test_percent
    return Dataset(
        dataset_name,
        path=str(REPO_ROOT / "data") + "/",
        feat_norm=data_conf.norm["feat_norm"],
        adj_norm=data_conf.norm["adj_norm"],
        train_size=data_conf.split["train_size"],
        val_size=data_conf.split["val_size"],
        test_size=data_conf.split["test_size"],
        train_percent=data_conf.split["train_percent"],
        val_percent=data_conf.split["val_percent"],
        test_percent=data_conf.split["test_percent"],
        train_examples_per_class=data_conf.split["train_examples_per_class"],
        val_examples_per_class=data_conf.split["val_examples_per_class"],
        test_examples_per_class=data_conf.split["test_examples_per_class"],
        add_self_loop=data_conf.modify["add_self_loop"],
        from_npz=data_conf.modify["from_npz_largest_component"],
        device=device,
        split_type=data_conf.split["split_type"],
    )


def majority_votes_for_assignment(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each node's train-only ball majority vote, or -1 when abstaining."""
    assignment = np.asarray(assignment, dtype=np.int64)
    n_nodes = int(noisy_label.shape[0])
    if assignment.ndim != 1 or assignment.shape[0] != n_nodes:
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    is_train = _mask_to_numpy(train_mask, n_nodes)
    labels_np = noisy_label.detach().cpu().long().numpy()
    vote_np = np.full(n_nodes, -1, dtype=np.int64)
    train_count_np = np.zeros(n_nodes, dtype=np.int64)
    margin_np = np.zeros(n_nodes, dtype=np.float32)
    min_train_per_ball = int(min_train_per_ball)

    for ball_id in sorted(np.unique(assignment).tolist()):
        member_idx = np.flatnonzero(assignment == ball_id)
        train_idx = member_idx[is_train[member_idx]]
        train_count_np[member_idx] = int(train_idx.size)
        if train_idx.size < min_train_per_ball:
            continue
        counts = np.bincount(labels_np[train_idx], minlength=int(n_classes))
        total = max(1, int(counts.sum()))
        vote_np[member_idx] = int(np.argmax(counts))
        margin_np[member_idx] = float(counts.max() / total)

    device = noisy_label.device
    return (
        torch.as_tensor(vote_np, dtype=torch.long, device=device),
        torch.as_tensor(train_count_np, dtype=torch.long, device=device),
        torch.as_tensor(margin_np, dtype=torch.float, device=device),
    )


def coarse_mid_agreement_summary(candidate_mask: torch.Tensor, mid_vote: torch.Tensor, coarse_vote: torch.Tensor) -> dict:
    candidate_mask = candidate_mask.bool()
    both = candidate_mask & (mid_vote >= 0) & (coarse_vote >= 0)
    agree = both & (mid_vote == coarse_vote)
    both_count = int(both.sum().item())
    agree_count = int(agree.sum().item())
    return {
        "coarse_mid_both_vote_count": both_count,
        "coarse_mid_agree_count": agree_count,
        "coarse_mid_agreement_rate": float(agree_count / both_count) if both_count else 0.0,
    }


def _hard_rule_metrics(
    rule: str,
    candidate_mask: torch.Tensor,
    execute_mask: torch.Tensor,
    noisy_flag: torch.Tensor,
    clean_flag: torch.Tensor,
    pseudo_to_clean: torch.Tensor,
    pseudo_damages_clean: torch.Tensor,
    noisy_count: int,
    clean_count: int,
    baseline_ncr_count: int,
    baseline_cdr_count: int,
) -> dict:
    protected = candidate_mask & ~execute_mask
    rule_ncr_count = int((execute_mask & noisy_flag & pseudo_to_clean).sum().item())
    rule_cdr_count = int((execute_mask & clean_flag & pseudo_damages_clean).sum().item())
    baseline_ncr = float(baseline_ncr_count / noisy_count) if noisy_count else 0.0
    rule_ncr = float(rule_ncr_count / noisy_count) if noisy_count else 0.0
    baseline_cdr = float(baseline_cdr_count / clean_count) if clean_count else 0.0
    rule_cdr = float(rule_cdr_count / clean_count) if clean_count else 0.0
    protected_count = int(protected.sum().item())
    return {
        "rule": rule,
        "execute_count": int(execute_mask.sum().item()),
        "execute_mass": float(execute_mask.float().sum().item()),
        "protected_count": protected_count,
        "protected_mass": float(protected_count),
        "protection_precision": float((protected & clean_flag).sum().item() / protected_count) if protected_count else 0.0,
        "baseline_ncr": baseline_ncr,
        "rule_ncr": rule_ncr,
        "ncr_relative_loss": float((baseline_ncr - rule_ncr) / baseline_ncr) if baseline_ncr > 0 else 0.0,
        "baseline_cdr": baseline_cdr,
        "rule_cdr": rule_cdr,
        "cdr_relative_reduction": float((baseline_cdr - rule_cdr) / baseline_cdr) if baseline_cdr > 0 else 0.0,
        "baseline_ncr_count": baseline_ncr_count,
        "rule_ncr_count": rule_ncr_count,
        "baseline_cdr_count": baseline_cdr_count,
        "rule_cdr_count": rule_cdr_count,
    }


def _weighted_rule_metrics(
    candidate_mask: torch.Tensor,
    alpha_scale: torch.Tensor,
    noisy_flag: torch.Tensor,
    clean_flag: torch.Tensor,
    pseudo_to_clean: torch.Tensor,
    pseudo_damages_clean: torch.Tensor,
    noisy_count: int,
    clean_count: int,
    baseline_ncr_count: int,
    baseline_cdr_count: int,
) -> dict:
    candidate_float = candidate_mask.double()
    scale = torch.clamp(alpha_scale.double(), min=0.0, max=1.0) * candidate_float
    protected_mass_vec = (1.0 - scale) * candidate_float
    execute_mass = float(scale.sum().item())
    protected_mass = float(protected_mass_vec.sum().item())
    rule_ncr_count = float((scale * (noisy_flag & pseudo_to_clean).double()).sum().item())
    rule_cdr_count = float((scale * (clean_flag & pseudo_damages_clean).double()).sum().item())
    baseline_ncr = float(baseline_ncr_count / noisy_count) if noisy_count else 0.0
    rule_ncr = float(rule_ncr_count / noisy_count) if noisy_count else 0.0
    baseline_cdr = float(baseline_cdr_count / clean_count) if clean_count else 0.0
    rule_cdr = float(rule_cdr_count / clean_count) if clean_count else 0.0
    protected_clean_mass = float((protected_mass_vec * clean_flag.double()).sum().item())
    return {
        "rule": "weighted_agreement",
        "execute_count": int(((candidate_mask) & (scale > 0)).sum().item()),
        "execute_mass": execute_mass,
        "protected_count": int(((candidate_mask) & (scale < 1.0)).sum().item()),
        "protected_mass": protected_mass,
        "protection_precision": float(protected_clean_mass / protected_mass) if protected_mass > 0 else 0.0,
        "baseline_ncr": baseline_ncr,
        "rule_ncr": rule_ncr,
        "ncr_relative_loss": float((baseline_ncr - rule_ncr) / baseline_ncr) if baseline_ncr > 0 else 0.0,
        "baseline_cdr": baseline_cdr,
        "rule_cdr": rule_cdr,
        "cdr_relative_reduction": float((baseline_cdr - rule_cdr) / baseline_cdr) if baseline_cdr > 0 else 0.0,
        "baseline_ncr_count": float(baseline_ncr_count),
        "rule_ncr_count": rule_ncr_count,
        "baseline_cdr_count": float(baseline_cdr_count),
        "rule_cdr_count": rule_cdr_count,
    }


def diagnose_multigranularity_rules(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    alpha: torch.Tensor,
    mid_pseudo_label: torch.Tensor,
    votes_by_scale: dict[str, torch.Tensor],
) -> dict[str, dict]:
    n_nodes = int(noisy_label.shape[0])
    train_mask_t = _mask_to_tensor(train_mask, n_nodes, noisy_label.device)
    candidate_mask = train_mask_t & (alpha.float() > 0) & (mid_pseudo_label.long() != noisy_label.long())
    noisy_flag = train_mask_t & (noisy_label.long() != clean_label.long())
    clean_flag = train_mask_t & ~noisy_flag
    pseudo_to_clean = mid_pseudo_label.long() == clean_label.long()
    pseudo_damages_clean = mid_pseudo_label.long() != clean_label.long()
    noisy_count = int(noisy_flag.sum().item())
    clean_count = int(clean_flag.sum().item())
    baseline_ncr_count = int((candidate_mask & noisy_flag & pseudo_to_clean).sum().item())
    baseline_cdr_count = int((candidate_mask & clean_flag & pseudo_damages_clean).sum().item())
    baseline_changed = int(candidate_mask.sum().item())

    fine_vote = votes_by_scale["fine"].long()
    mid_vote = votes_by_scale["mid"].long()
    coarse_vote = votes_by_scale["coarse"].long()
    vote_stack = torch.stack([fine_vote, mid_vote, coarse_vote], dim=0)
    non_abstain = vote_stack >= 0
    mid_available = mid_vote >= 0
    all_non_abstain_match_mid = ((vote_stack == mid_vote.unsqueeze(0)) | ~non_abstain).all(dim=0)
    all_consistent_execute = candidate_mask & mid_available & all_non_abstain_match_mid
    coarse_mid_execute = candidate_mask & mid_available & (coarse_vote == mid_vote)

    agreement_count = (
        ((fine_vote >= 0) & (fine_vote == mid_vote)).long()
        + (mid_available).long()
        + ((coarse_vote >= 0) & (coarse_vote == mid_vote)).long()
    )
    alpha_scale = agreement_count.float() / 3.0

    rows = {
        "all_consistent": _hard_rule_metrics(
            "all_consistent",
            candidate_mask,
            all_consistent_execute,
            noisy_flag,
            clean_flag,
            pseudo_to_clean,
            pseudo_damages_clean,
            noisy_count,
            clean_count,
            baseline_ncr_count,
            baseline_cdr_count,
        ),
        "coarse_mid_consistent": _hard_rule_metrics(
            "coarse_mid_consistent",
            candidate_mask,
            coarse_mid_execute,
            noisy_flag,
            clean_flag,
            pseudo_to_clean,
            pseudo_damages_clean,
            noisy_count,
            clean_count,
            baseline_ncr_count,
            baseline_cdr_count,
        ),
        "weighted_agreement": _weighted_rule_metrics(
            candidate_mask,
            alpha_scale,
            noisy_flag,
            clean_flag,
            pseudo_to_clean,
            pseudo_damages_clean,
            noisy_count,
            clean_count,
            baseline_ncr_count,
            baseline_cdr_count,
        ),
    }

    candidate_count = max(1, baseline_changed)
    scale_stats = {
        "baseline_changed_train_count": baseline_changed,
        "noisy_train_count": noisy_count,
        "clean_train_count": clean_count,
        "fine_abstain_rate": float(((candidate_mask) & (fine_vote < 0)).sum().item() / candidate_count),
        "mid_abstain_rate": float(((candidate_mask) & (mid_vote < 0)).sum().item() / candidate_count),
        "coarse_abstain_rate": float(((candidate_mask) & (coarse_vote < 0)).sum().item() / candidate_count),
        "agreement_count_mean": float(agreement_count[candidate_mask].float().mean().item()) if baseline_changed else 0.0,
        **coarse_mid_agreement_summary(candidate_mask, mid_vote, coarse_vote),
    }
    for row in rows.values():
        row.update(scale_stats)
    return rows


def _diagnose_one(args, dataset_name: str, seed: int) -> list[dict]:
    setup_seed(seed)
    data = _load_dataset(
        dataset_name=dataset_name,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    model_conf = load_conf(None, args.method_config, dataset_name)
    gb_conf = getattr(model_conf, "gb", {})
    embedding_path = _conf_value(gb_conf, "embedding_path", None)
    if embedding_path is None:
        raise ValueError(f"gb.embedding_path missing in {args.method_config}/{dataset_name}")
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.n_nodes))
    split_improvement = _conf_value(gb_conf, "split_improvement", 0.05)
    min_train_per_ball = int(_conf_value(gb_conf, "min_train_per_ball", args.min_train_per_ball))

    assignments = {}
    summaries = {}
    for scale_name, spec in SCALE_SPECS.items():
        gb_result = build_ordinary_granular_balls(
            embeddings,
            max_balls=spec["max_balls"],
            min_ball_size=spec["min_ball_size"],
            split_improvement=split_improvement,
            random_state=seed,
        )
        assignments[scale_name] = gb_result.assignment
        summaries[scale_name] = gb_result.summary

    _, alpha, mid_pseudo = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignments["mid"],
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=_conf_value(gb_conf, "alpha_baseline", "multiclass"),
        target_mode=_conf_value(gb_conf, "target_mode", "distribution"),
    )
    votes_by_scale = {}
    train_counts_by_scale = {}
    margins_by_scale = {}
    for scale_name, assignment in assignments.items():
        vote, train_count, margin = majority_votes_for_assignment(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            min_train_per_ball=min_train_per_ball,
        )
        votes_by_scale[scale_name] = vote
        train_counts_by_scale[scale_name] = train_count
        margins_by_scale[scale_name] = margin

    rows_by_rule = diagnose_multigranularity_rules(
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        alpha=alpha,
        mid_pseudo_label=mid_pseudo,
        votes_by_scale=votes_by_scale,
    )
    rows = []
    candidate_mask = _mask_to_tensor(data.train_masks, int(data.n_nodes), noisy_label.device) & (alpha > 0) & (mid_pseudo != noisy_label)
    for rule, diag in rows_by_rule.items():
        row = {
            "dataset": dataset_name,
            "seed": int(seed),
            "rule": rule,
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "split_type": args.split_type,
            "train_percent": float(args.train_percent),
            "val_percent": float(args.val_percent),
            "test_percent": float(args.test_percent),
            "min_train_per_ball": min_train_per_ball,
            "fine_num_balls": int(summaries["fine"]["num_balls"]),
            "mid_num_balls": int(summaries["mid"]["num_balls"]),
            "coarse_num_balls": int(summaries["coarse"]["num_balls"]),
            "fine_mean_ball_size": float(summaries["fine"]["mean_ball_size"]),
            "mid_mean_ball_size": float(summaries["mid"]["mean_ball_size"]),
            "coarse_mean_ball_size": float(summaries["coarse"]["mean_ball_size"]),
            "fine_mean_train_votes_changed": _mean_on_mask(train_counts_by_scale["fine"], candidate_mask),
            "mid_mean_train_votes_changed": _mean_on_mask(train_counts_by_scale["mid"], candidate_mask),
            "coarse_mean_train_votes_changed": _mean_on_mask(train_counts_by_scale["coarse"], candidate_mask),
            "fine_mean_margin_changed": _mean_on_mask(margins_by_scale["fine"], candidate_mask),
            "mid_mean_margin_changed": _mean_on_mask(margins_by_scale["mid"], candidate_mask),
            "coarse_mean_margin_changed": _mean_on_mask(margins_by_scale["coarse"], candidate_mask),
            **diag,
        }
        rows.append(row)
    return rows


def _mean_on_mask(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    return float(values[mask].float().mean().item())


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["rule"])].append(row)
    aggregate = []
    for (dataset, rule), group in sorted(groups.items()):
        out = {"dataset": dataset, "rule": rule, "num_seeds": len(group)}
        keys = sorted({key for row in group for key in row.keys()})
        for key in keys:
            if key in out or key in {"dataset", "rule", "noise_type", "split_type"}:
                continue
            values = [row.get(key) for row in group]
            if values and all(isinstance(value, (int, float)) for value in values):
                out[key] = float(sum(float(value) for value in values) / len(values))
        out["passes_cdr_reduction"] = int(out.get("cdr_relative_reduction", 0.0) >= 0.30)
        out["passes_ncr_loss"] = int(out.get("ncr_relative_loss", 1.0) <= 0.10)
        out["passes_protection_precision"] = int(out.get("protection_precision", 0.0) >= 0.70)
        out["passes_all_gates"] = int(
            out["passes_cdr_reduction"]
            and out["passes_ncr_loss"]
            and out["passes_protection_precision"]
        )
        aggregate.append(out)
    return aggregate


def rule_passes_both_datasets(aggregate_rows_: list[dict]) -> dict[str, bool]:
    by_rule = defaultdict(dict)
    for row in aggregate_rows_:
        by_rule[row["rule"]][row["dataset"]] = bool(row.get("passes_all_gates", 0))
    return {rule: all(by_rule[rule].get(dataset, False) for dataset in ("cora", "citeseer")) for rule in RULES}


def write_csv(path: Path, rows: list[dict], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(preferred_fields or [])
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, raw_rows: list[dict], aggregate: list[dict]) -> None:
    passed = rule_passes_both_datasets(aggregate)
    any_pass = any(passed.values())
    lines = [
        "# Multigranularity Consistency Diagnostic",
        "",
        "## Protocol",
        "",
        "- Dataset: Cora, CiteSeer",
        "- Split: percent 0.2/0.1/0.7",
        "- Noise: uniform 0.4",
        "- Seeds: 0, 1, 2",
        "- Balls: label-blind geometric GRACE-H balls, fine 256/8, mid 64/32, coarse 16/128",
        "- Clean labels are used only for diagnostics.",
        "",
        "## Structural Difference From Closed Directions",
        "",
        "- vs E1 adaptive granularity: construction is purely geometric and label-blind; labels only vote after balls are fixed.",
        "- vs binary gate: the gate signal is cross-scale consistency, not same-scale vote count or margin.",
        "- vs dynamic alpha: no model prediction feedback loop; this is an offline diagnostic.",
        "- vs unlabeled expansion: only train-node correction candidates are evaluated; no unlabeled supervision is added.",
        "",
        "## Predeclared Risks",
        "",
        "- Gate-class mechanisms may repeat the same NCR/CDR tradeoff failure.",
        "- Nested geometric correlation may inflate agreement and provide little protection.",
        "- CiteSeer may degenerate into almost no useful correction.",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Rule | Baseline NCR | Rule NCR | NCR loss | Baseline CDR | Rule CDR | CDR reduction | Protection precision | Protected | Execute mass | Coarse-mid agree | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {dataset} | {rule} | {baseline_ncr:.4f} | {rule_ncr:.4f} | {ncr_relative_loss:.4f} | "
            "{baseline_cdr:.4f} | {rule_cdr:.4f} | {cdr_relative_reduction:.4f} | "
            "{protection_precision:.4f} | {protected_count:.1f} | {execute_mass:.1f} | "
            "{coarse_mid_agreement_rate:.4f} | {passes_all_gates} |".format(**row)
        )
    lines.extend([
        "",
        "## Coarse-Mid Agreement Summary",
        "",
        "| Dataset | Rule | Both-vote count | Agree count | Agreement rate | Coarse abstain | Mid abstain |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in aggregate:
        lines.append(
            "| {dataset} | {rule} | {coarse_mid_both_vote_count:.1f} | {coarse_mid_agree_count:.1f} | "
            "{coarse_mid_agreement_rate:.4f} | {coarse_abstain_rate:.4f} | {mid_abstain_rate:.4f} |".format(**row)
        )
    lines.extend([
        "",
        "## Gate Decision",
        "",
    ])
    if any_pass:
        passed_rules = ", ".join(rule for rule, ok in passed.items() if ok)
        lines.append(f"At least one rule passes both datasets: {passed_rules}. Training-version follow-up is allowed by the preregistered gate.")
    else:
        lines.append("No rule passes all three gates on both datasets. By the preregistered rule, the multigranularity literature-mining line is closed and no training version should be run.")
    lines.extend([
        "",
        "## Required Experiment Record",
        "",
        "- Hypothesis: cross-scale agreement protects class-boundary clean labels while preserving stable-region noisy-label correction.",
        "- Granular-ball role: label-blind local semantic neighborhoods and cross-scale confidence signal.",
        "- Code change: added `scripts/diagnose_multigranularity.py` and unit tests for voting/rule diagnostics.",
        "- Config change: none; existing SoftDist XH anchor behavior is untouched.",
        "- Success standard: CDR reduction >= 30%, NCR loss <= 10%, protection precision >= 70% on both datasets for one rule.",
        "- Failure standard: any gate violation blocks training implementation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose multigranularity consistency arbitration for SoftDist corrections.")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--method_config", default="frozengrace_gb_gcn_softdist_xh")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "multigranularity_diag_20260612" / "multigranularity_diag_raw_20260612.csv"))
    parser.add_argument("--aggregate_output", default=None)
    parser.add_argument("--report_output", default=None)
    args = parser.parse_args()
    args.datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.seeds = _parse_int_list(args.seeds)

    rows = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            print(f"MULTIGRANULARITY_DIAG {dataset_name} seed={seed}")
            rows.extend(_diagnose_one(args, dataset_name, seed))

    output_path = Path(args.output)
    aggregate_path = Path(args.aggregate_output) if args.aggregate_output else output_path.with_name("multigranularity_diag_aggregate_20260612.csv")
    report_path = Path(args.report_output) if args.report_output else output_path.with_name("multigranularity_diag_report_20260612.md")
    preferred = [
        "dataset", "seed", "rule", "noise_type", "noise_rate", "split_type",
        "train_percent", "val_percent", "test_percent", "min_train_per_ball",
        "baseline_changed_train_count", "execute_count", "execute_mass", "protected_count", "protected_mass",
        "protection_precision", "baseline_ncr", "rule_ncr", "ncr_relative_loss",
        "baseline_cdr", "rule_cdr", "cdr_relative_reduction",
        "baseline_ncr_count", "rule_ncr_count", "baseline_cdr_count", "rule_cdr_count",
        "fine_abstain_rate", "mid_abstain_rate", "coarse_abstain_rate", "agreement_count_mean",
        "coarse_mid_both_vote_count", "coarse_mid_agree_count", "coarse_mid_agreement_rate",
        "fine_num_balls", "mid_num_balls", "coarse_num_balls",
        "fine_mean_ball_size", "mid_mean_ball_size", "coarse_mean_ball_size",
        "fine_mean_train_votes_changed", "mid_mean_train_votes_changed", "coarse_mean_train_votes_changed",
        "fine_mean_margin_changed", "mid_mean_margin_changed", "coarse_mean_margin_changed",
    ]
    write_csv(output_path, rows, preferred_fields=preferred)
    aggregate = aggregate_rows(rows)
    write_csv(aggregate_path, aggregate, preferred_fields=["dataset", "rule", "num_seeds"])
    write_report(report_path, rows, aggregate)
    print(f"WROTE {output_path}")
    print(f"WROTE {aggregate_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
