from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import setup_seed


@dataclass
class NodewiseTargetResult:
    soft_target: torch.Tensor
    alpha: torch.Tensor
    pseudo_label: torch.Tensor
    ball_dist: torch.Tensor
    distance_ratio: torch.Tensor


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def _alpha_from_margin(margin: float, n_classes: int, alpha_baseline: str) -> float:
    if str(alpha_baseline) == "multiclass":
        zero_point = 1.0 / max(1, int(n_classes))
    else:
        zero_point = 0.5
    if zero_point >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (float(margin) - zero_point) / (1.0 - zero_point)))


def _normalized_embeddings(embeddings: torch.Tensor, device) -> torch.Tensor:
    embeddings = embeddings.to(device=device, dtype=torch.float)
    return F.normalize(embeddings, p=2, dim=1)


def _ball_distance_ratio(z: torch.Tensor, member_tensor: torch.Tensor) -> torch.Tensor:
    center = z[member_tensor].mean(dim=0, keepdim=True)
    distances = torch.norm(z[member_tensor] - center, p=2, dim=1)
    radius = distances.max().clamp_min(1e-12)
    return distances / radius


def apply_nodewise_soft_targets(
    embeddings: torch.Tensor,
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    vote_mode: str = "uniform",
    alpha_mode: str = "ball",
    tau_sim: float = 0.5,
    gamma: float = 1.0,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
) -> NodewiseTargetResult:
    """Construct train-only nodewise soft targets from fixed geometric balls.

    Default `vote_mode='uniform', alpha_mode='ball'` matches the current
    SoftDist formula. Only train labels vote; val/test labels never enter.
    """
    device = noisy_label.device
    n_nodes = int(noisy_label.shape[0])
    n_classes = int(n_classes)
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or int(assignment.shape[0]) != n_nodes:
        raise ValueError("assignment must be a 1-D array matching noisy_label length")
    if vote_mode not in {"uniform", "sim_weighted"}:
        raise ValueError("vote_mode must be 'uniform' or 'sim_weighted'")
    if alpha_mode not in {"ball", "distance"}:
        raise ValueError("alpha_mode must be 'ball' or 'distance'")
    if target_mode not in {"distribution", "majority"}:
        raise ValueError("target_mode must be 'distribution' or 'majority'")

    z = _normalized_embeddings(embeddings, device=device)
    is_train = _mask_to_numpy(train_mask, n_nodes)
    train_idx = np.flatnonzero(is_train)
    soft_target = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    alpha = torch.zeros(n_nodes, dtype=torch.float, device=device)
    ball_dist = torch.zeros(n_nodes, n_classes, dtype=torch.float, device=device)
    distance_ratio = torch.zeros(n_nodes, dtype=torch.float, device=device)
    min_train_per_ball = int(min_train_per_ball)

    source_onehot = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id)
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        member_tensor = torch.as_tensor(members, dtype=torch.long, device=device)
        train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        target_train_np = ball_train
        target_tensor = train_tensor

        counts = torch.bincount(noisy_label[train_tensor].long(), minlength=n_classes)
        total = max(1, int(counts.sum().item()))
        margin = float(counts.max().item() / total)
        ball_alpha = _alpha_from_margin(margin, n_classes, alpha_baseline)
        if ball_alpha <= 0.0:
            continue

        dist = counts.float() / float(total)
        ball_dist[member_tensor] = dist
        member_distance = _ball_distance_ratio(z, member_tensor)
        distance_ratio[member_tensor] = member_distance
        target_distance = distance_ratio[target_tensor]

        if vote_mode == "uniform":
            if target_mode == "majority":
                majority = int(torch.argmax(counts).item())
                vote_target = F.one_hot(
                    torch.full((target_tensor.numel(),), majority, dtype=torch.long, device=device),
                    num_classes=n_classes,
                ).float()
            else:
                vote_target = dist.unsqueeze(0).expand(target_tensor.numel(), -1)
        else:
            tau_sim = max(float(tau_sim), 1e-6)
            logits = torch.mm(z[target_tensor], z[train_tensor].t()) / tau_sim
            weights = torch.softmax(logits, dim=1)
            vote_target = torch.mm(weights, source_onehot[train_tensor])

        if alpha_mode == "distance":
            node_alpha = float(ball_alpha) * torch.exp(-float(gamma) * target_distance)
            node_alpha = torch.clamp(node_alpha, min=0.0, max=1.0)
        else:
            node_alpha = torch.full((target_tensor.numel(),), float(ball_alpha), dtype=torch.float, device=device)

        soft_target[target_tensor] = (
            (1.0 - node_alpha.unsqueeze(1)) * soft_target[target_tensor]
            + node_alpha.unsqueeze(1) * vote_target
        )
        alpha[target_tensor] = node_alpha

    pseudo_label = soft_target.argmax(dim=1)
    return NodewiseTargetResult(
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
        ball_dist=ball_dist.detach(),
        distance_ratio=distance_ratio.detach(),
    )


def target_metrics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    pseudo_label: torch.Tensor,
    alpha: torch.Tensor,
    train_mask,
) -> dict:
    n_nodes = int(noisy_label.shape[0])
    train_bool = torch.as_tensor(_mask_to_numpy(train_mask, n_nodes), dtype=torch.bool, device=noisy_label.device)
    changed = train_bool & (alpha > 0) & (pseudo_label.long() != noisy_label.long())
    noisy_flag = train_bool & (noisy_label.long() != clean_label.long())
    clean_flag = train_bool & ~noisy_flag
    pseudo_to_clean = pseudo_label.long() == clean_label.long()
    pseudo_damages_clean = pseudo_label.long() != clean_label.long()
    noisy_count = int(noisy_flag.sum().item())
    clean_count = int(clean_flag.sum().item())
    ncr_count = int((changed & noisy_flag & pseudo_to_clean).sum().item())
    cdr_count = int((changed & clean_flag & pseudo_damages_clean).sum().item())
    return {
        "changed_train_count": int(changed.sum().item()),
        "ncr_count": ncr_count,
        "cdr_count": cdr_count,
        "noisy_train_count": noisy_count,
        "clean_train_count": clean_count,
        "ncr": float(ncr_count / noisy_count) if noisy_count else 0.0,
        "cdr": float(cdr_count / clean_count) if clean_count else 0.0,
        "mean_alpha_train": float(alpha[train_bool].mean().item()) if int(train_bool.sum().item()) else 0.0,
    }


def distance_bin_target_metrics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    pseudo_label: torch.Tensor,
    alpha: torch.Tensor,
    train_mask,
    distance_ratio: torch.Tensor,
    n_bins: int = 4,
) -> dict[str, dict]:
    n_nodes = int(noisy_label.shape[0])
    train_np = _mask_to_numpy(train_mask, n_nodes)
    distance_np = distance_ratio.detach().cpu().numpy()
    train_dist = distance_np[train_np]
    out = {"all": target_metrics(noisy_label, clean_label, pseudo_label, alpha, train_mask)}
    if train_dist.size == 0:
        return out
    quantiles = np.quantile(train_dist, np.linspace(0.0, 1.0, int(n_bins) + 1))
    for bin_id in range(int(n_bins)):
        lo = quantiles[bin_id]
        hi = quantiles[bin_id + 1]
        if bin_id == int(n_bins) - 1:
            bin_np = train_np & (distance_np >= lo) & (distance_np <= hi)
        else:
            bin_np = train_np & (distance_np >= lo) & (distance_np < hi)
        if not bin_np.any():
            continue
        bin_mask = torch.as_tensor(bin_np, dtype=torch.bool, device=noisy_label.device)
        metrics = target_metrics(noisy_label, clean_label, pseudo_label, alpha, bin_mask)
        metrics.update({
            "bin": int(bin_id),
            "distance_min": float(lo),
            "distance_max": float(hi),
            "train_count": int(bin_np.sum()),
        })
        out[f"bin_{bin_id}"] = metrics
    return out


def mean_kl_to_ball_distribution(soft_target: torch.Tensor, ball_dist: torch.Tensor, train_mask) -> float:
    n_nodes = int(soft_target.shape[0])
    train_bool = torch.as_tensor(_mask_to_numpy(train_mask, n_nodes), dtype=torch.bool, device=soft_target.device)
    valid = train_bool & (ball_dist.sum(dim=1) > 0)
    if int(valid.sum().item()) == 0:
        return 0.0
    p = soft_target[valid].clamp_min(1e-12)
    q = ball_dist[valid].clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum(dim=1).mean().item())


def _relative_rows(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]))
        if row["variant"] == "baseline":
            by_key[key] = row
    out = []
    for row in rows:
        baseline = by_key[(row["dataset"], int(row["seed"]))]
        base_ncr = float(baseline["ncr"])
        base_cdr = float(baseline["cdr"])
        row = dict(row)
        row["baseline_ncr"] = base_ncr
        row["baseline_cdr"] = base_cdr
        row["ncr_relative_loss"] = float((base_ncr - float(row["ncr"])) / base_ncr) if base_ncr > 0 else 0.0
        row["cdr_relative_reduction"] = float((base_cdr - float(row["cdr"])) / base_cdr) if base_cdr > 0 else 0.0
        out.append(row)
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], row["tau_sim"], row["gamma"])].append(row)
    out = []
    for (dataset, variant, tau_sim, gamma), group in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "variant": variant,
            "tau_sim": tau_sim,
            "gamma": gamma,
            "seeds": ";".join(str(int(row["seed"])) for row in group),
        }
        for metric in ["ncr", "cdr", "ncr_relative_loss", "cdr_relative_reduction", "mean_alpha_train", "mean_kl_to_ball_dist"]:
            vals = [float(row[metric]) for row in group]
            mean, std = _mean_std(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        item["pass_gate"] = int(item["cdr_relative_reduction_mean"] >= 0.30 and item["ncr_relative_loss_mean"] <= 0.10)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _diagnose_dataset_seed(args, dataset_name: str, seed: int) -> tuple[list[dict], list[dict]]:
    setup_seed(seed)
    dataset_key = dataset_name.lower()
    data = _load_dataset(
        dataset_key,
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
    embedding_path = embedding_path_for(args.embedding_root, dataset_name, args.ssl_method)
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.labels.shape[0]))
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=args.max_balls,
        min_ball_size=args.min_ball_size,
        split_improvement=args.split_improvement,
        random_state=seed,
    )
    assignment = gb_result.assignment
    specs = [("baseline", None, None, "uniform", "ball")]
    for tau in args.tau_grid:
        specs.append(("sim_weighted", tau, None, "sim_weighted", "ball"))
    for gamma in args.gamma_grid:
        specs.append(("distance_alpha", None, gamma, "uniform", "distance"))
    for tau in args.tau_grid:
        for gamma in args.gamma_grid:
            specs.append(("sim_weighted_distance", tau, gamma, "sim_weighted", "distance"))

    rows = []
    bin_rows = []
    for variant, tau, gamma, vote_mode, alpha_mode in specs:
        result = apply_nodewise_soft_targets(
            embeddings=embeddings,
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            vote_mode=vote_mode,
            alpha_mode=alpha_mode,
            tau_sim=0.5 if tau is None else tau,
            gamma=1.0 if gamma is None else gamma,
            min_train_per_ball=args.min_train_per_ball,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        metrics = target_metrics(noisy_label, data.labels, result.pseudo_label, result.alpha, data.train_masks)
        row = {
            "dataset": dataset_key,
            "seed": int(seed),
            "variant": variant,
            "tau_sim": "" if tau is None else float(tau),
            "gamma": "" if gamma is None else float(gamma),
            "embedding_path": str(embedding_path),
            "num_balls": int(gb_result.summary["num_balls"]),
            "mean_ball_size": float(gb_result.summary["mean_ball_size"]),
            **metrics,
            "mean_kl_to_ball_dist": mean_kl_to_ball_distribution(result.soft_target, result.ball_dist, data.train_masks),
        }
        rows.append(row)
        for bin_name, bin_metric in distance_bin_target_metrics(
            noisy_label=noisy_label,
            clean_label=data.labels,
            pseudo_label=result.pseudo_label,
            alpha=result.alpha,
            train_mask=data.train_masks,
            distance_ratio=result.distance_ratio,
            n_bins=args.distance_bins,
        ).items():
            bin_rows.append({
                "dataset": dataset_key,
                "seed": int(seed),
                "variant": variant,
                "tau_sim": "" if tau is None else float(tau),
                "gamma": "" if gamma is None else float(gamma),
                "bin": bin_name,
                **bin_metric,
            })
    return _relative_rows(rows), bin_rows


def _format_float(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{float(value):.4f}"


def write_report(path: Path, raw_rows: list[dict], aggregate: list[dict], bin_rows: list[dict]) -> None:
    passed = [row for row in aggregate if int(row["pass_gate"]) == 1 and row["variant"] != "baseline"]
    lines = [
        "# Phase 12 Nodewise Target Diagnostic Report",
        "",
        "## Protocol",
        "",
        "- Split: percent 0.2/0.1/0.7.",
        "- Noise: uniform 0.4.",
        "- Seeds: 0, 1, 2 unless overridden.",
        "- Balls: fixed geometric anchor, max_balls=64, min_ball_size=32.",
        "- Clean labels are used only for diagnostics.",
        "",
        "## Aggregate",
        "",
        "| Dataset | Variant | tau | gamma | NCR | CDR | CDR reduction | NCR loss | KL | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {dataset} | {variant} | {tau} | {gamma} | {ncr} | {cdr} | {cdr_red} | {ncr_loss} | {kl} | {passed} |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                tau=row["tau_sim"],
                gamma=row["gamma"],
                ncr=_format_float(row["ncr_mean"]),
                cdr=_format_float(row["cdr_mean"]),
                cdr_red=_format_float(row["cdr_relative_reduction_mean"]),
                ncr_loss=_format_float(row["ncr_relative_loss_mean"]),
                kl=_format_float(row["mean_kl_to_ball_dist_mean"]),
                passed=row["pass_gate"],
            )
        )
    lines.extend([
        "",
        "## Gate Decision",
        "",
    ])
    if passed:
        for row in passed:
            lines.append(
                f"- PASS: {row['dataset']} {row['variant']} tau={row['tau_sim']} gamma={row['gamma']} "
                f"CDR reduction={_format_float(row['cdr_relative_reduction_mean'])}, "
                f"NCR loss={_format_float(row['ncr_relative_loss_mean'])}."
            )
    else:
        lines.append("- No variant passes the preregistered gate on both datasets.")
    lines.extend([
        "",
        "## Distance-Bin Diagnostic",
        "",
        "The raw distance-bin CDR/NCR table is written to `distance_bins.csv`. It directly measures whether clean-label damage concentrates near ball boundaries.",
        "",
        "## Research Record",
        "",
        "- Hypothesis: ball-level SoftDist loses information because all train nodes in a ball receive the same vote distribution and alpha.",
        "- Granular-ball role: unchanged support boundary; only within-ball vote weighting and per-node alpha are diagnosed.",
        "- Code change: added `scripts/diagnose_nodewise_targets.py` and unit tests.",
        "- Config change: none to anchor SoftDist [X||H].",
        "- Success standard: CDR reduction >= 30% and NCR loss <= 10%.",
        f"- Result: {len(passed)} aggregate rows pass the single-dataset gate; downstream Phase 13 requires a variant that passes both datasets.",
        "",
        "## Difference From Closed Directions",
        "",
        "- Unlike C1_topk/kNN, the support set remains restricted to the geometric ball.",
        "- Unlike previous gates, distance alpha tests geometric membership confidence rather than noisy-label purity.",
        "- Unlike dynamic alpha, there is no model prediction feedback loop.",
        "- Unlike unlabeled expansion, only train-node targets are changed.",
        "- Unlike adaptive granularity, ball construction is unchanged.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose nodewise within-ball soft target variants without training.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--tau_grid", default="0.1,0.5,1.0")
    parser.add_argument("--gamma_grid", default="0.5,1.0,2.0")
    parser.add_argument("--max_balls", type=int, default=64)
    parser.add_argument("--min_ball_size", type=int, default=32)
    parser.add_argument("--split_improvement", type=float, default=0.05)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--distance_bins", type=int, default=4)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "nodewise_diag_20260612"))
    args = parser.parse_args()
    args.tau_grid = _parse_float_list(args.tau_grid)
    args.gamma_grid = _parse_float_list(args.gamma_grid)

    raw_rows = []
    bin_rows = []
    for dataset_name in _parse_str_list(args.datasets):
        for seed in _parse_int_list(args.seeds):
            print(f"NODEWISE_DIAG dataset={dataset_name} seed={seed}")
            rows, bins = _diagnose_dataset_seed(args, dataset_name, seed)
            raw_rows.extend(rows)
            bin_rows.extend(bins)

    aggregate = aggregate_rows(raw_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "raw.csv", raw_rows)
    write_csv(out_dir / "aggregate.csv", aggregate)
    write_csv(out_dir / "distance_bins.csv", bin_rows)
    write_report(out_dir / "nodewise_diag_report.md", raw_rows, aggregate, bin_rows)
    print(f"WROTE {out_dir}")


if __name__ == "__main__":
    main()
