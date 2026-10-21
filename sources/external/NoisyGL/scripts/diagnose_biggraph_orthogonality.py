from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import GCN
from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings, partition_purity_diagnostics
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


BIG_BALL_PARAMS = {
    "cora": {"max_balls": 64, "min_ball_size": 32, "split_improvement": 0.05},
    "citeseer": {"max_balls": 64, "min_ball_size": 32, "split_improvement": 0.05},
    "dblp": {"max_balls": 256, "min_ball_size": 32, "split_improvement": 0.001},
    "pubmed": {"max_balls": 256, "min_ball_size": 32, "split_improvement": 0.005},
    "amazonpho": {"max_balls": 256, "min_ball_size": 32, "split_improvement": 0.001},
    "lastfmasia": {"max_balls": 256, "min_ball_size": 32, "split_improvement": 0.001},
}


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _dataset_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    aliases = {
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
    }
    return aliases.get(key, key)


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values) -> float:
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if len(values) <= 1:
        return 0.0 if values else float("nan")
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size <= 1 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.flatnonzero(mask)
    return mask.astype(np.int64)


def _edge_homophily(labels: torch.Tensor, adj: torch.Tensor) -> float:
    adj = adj.coalesce()
    row, col = adj.indices()
    non_self = row != col
    if not bool(non_self.any()):
        return float("nan")
    row = row[non_self]
    col = col[non_self]
    weights = adj.values()[non_self].float()
    same_class = labels[row] == labels[col]
    return float(weights[same_class].sum().item() / weights.sum().item())


def _prepare_xh_features(features, embeddings, device):
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    features = features.float().to(device)
    embeddings = F.normalize(embeddings.float().to(device), p=2, dim=1)
    return torch.cat([features, embeddings], dim=1)


def _warmup_gcn_probs(data, noisy_label, embeddings, args):
    features = _prepare_xh_features(data.feats, embeddings, data.feats.device)
    conf = load_conf(None, args.method_config, data.name)
    model = GCN(
        in_channels=int(features.shape[1]),
        hidden_channels=conf.model["n_hidden"],
        out_channels=data.n_classes,
        n_layers=conf.model["n_layer"],
        dropout=conf.model["dropout"],
        norm_info=conf.model["norm_info"],
        act=conf.model["act"],
        input_layer=conf.model["input_layer"],
        output_layer=conf.model["output_layer"],
    ).to(data.feats.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.training["lr"],
        weight_decay=conf.training["weight_decay"],
    )
    train_mask = data.train_masks
    snapshots = []
    for epoch in range(1, int(args.warmup_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.adj)
        loss = F.cross_entropy(logits[train_mask], noisy_label[train_mask].long())
        loss.backward()
        optimizer.step()
        if int(args.window_start) <= epoch <= int(args.window_end):
            model.eval()
            with torch.no_grad():
                snapshots.append(F.softmax(model(features, data.adj), dim=1).detach().cpu())
    if not snapshots:
        raise ValueError("warmup window produced no snapshots")
    return torch.stack(snapshots, dim=0).mean(dim=0)


def _ball_suspicion_scores(noisy_label, assignment, train_idx, n_classes, min_train_per_ball):
    noisy_np = noisy_label.detach().cpu().long().numpy()
    assignment = np.asarray(assignment, dtype=np.int64)
    scores = np.zeros(train_idx.shape[0], dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float64)
        total = float(max(1.0, counts.sum()))
        dist = counts / total
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            scores[pos] = 1.0 - float(dist[int(noisy_np[node])])
            eligible[pos] = True
    return scores, eligible


def _knn_suspicion_scores(embeddings, noisy_label, train_idx, k):
    noisy_np = noisy_label.detach().cpu().long().numpy()
    train_idx = np.asarray(train_idx, dtype=np.int64)
    k = min(int(k), max(1, int(train_idx.size) - 1))
    if k <= 0:
        return np.zeros(train_idx.shape[0], dtype=np.float64)
    h = F.normalize(embeddings.float(), p=2, dim=1)[torch.as_tensor(train_idx, dtype=torch.long)]
    sim = h @ h.t()
    sim.fill_diagonal_(-float("inf"))
    nn_idx = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()
    train_labels = noisy_np[train_idx]
    scores = np.zeros(train_idx.shape[0], dtype=np.float64)
    for pos in range(train_idx.size):
        neighbor_labels = train_labels[nn_idx[pos]]
        agree = float(np.mean(neighbor_labels == train_labels[pos]))
        scores[pos] = 1.0 - agree
    return scores


def _quantile_mask(scores, clean_fraction):
    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(scores, float(clean_fraction)))
    return scores <= threshold


def _effective_noise_rate(noisy_flag, weights) -> float:
    noisy_flag = np.asarray(noisy_flag, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(weights.sum())
    if denom <= 1e-12:
        return float("nan")
    return float(weights[noisy_flag].sum() / denom)


def _offline_rows(dataset, seed, noisy_flag, s_ball, s_gcn, args):
    rows = []
    base_noise = float(np.mean(noisy_flag))
    weights = np.clip(1.0 - np.asarray(s_ball, dtype=np.float64), 0.0, 1.0)
    eff_noise = _effective_noise_rate(noisy_flag, weights)
    rows.append({
        "dataset": dataset,
        "seed": int(seed),
        "usage": "reweight",
        "param": "1_minus_s_ball",
        "base_noise_rate": base_noise,
        "effective_noise_rate": eff_noise,
        "noise_reduction": base_noise - eff_noise if np.isfinite(eff_noise) else float("nan"),
        "retention": 1.0,
        "selected_precision_noise": float("nan"),
        "dropped_clean_ratio": 0.0,
    })
    for top_k in args.filter_topk:
        top_k = float(top_k)
        suspicious = np.asarray(s_ball) >= float(np.quantile(s_ball, 1.0 - top_k))
        kept = ~suspicious
        kept_noise = float(np.mean(noisy_flag[kept])) if kept.any() else float("nan")
        selected_precision = float(np.mean(noisy_flag[suspicious])) if suspicious.any() else float("nan")
        clean_flag = ~noisy_flag
        dropped_clean_ratio = float((suspicious & clean_flag).sum() / max(1, clean_flag.sum()))
        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "usage": "filter",
            "param": f"top_{top_k:g}",
            "base_noise_rate": base_noise,
            "effective_noise_rate": kept_noise,
            "noise_reduction": base_noise - kept_noise if np.isfinite(kept_noise) else float("nan"),
            "retention": float(np.mean(kept)),
            "selected_precision_noise": selected_precision,
            "dropped_clean_ratio": dropped_clean_ratio,
        })
    for keep in args.clean_keep_rates:
        keep = float(keep)
        ball_clean = _quantile_mask(s_ball, keep)
        gcn_clean = _quantile_mask(s_gcn, keep)
        both_clean = ball_clean & gcn_clean
        both_noise = float(np.mean(noisy_flag[both_clean])) if both_clean.any() else float("nan")
        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "usage": "co_teaching_intersection",
            "param": f"keep_{keep:g}_each",
            "base_noise_rate": base_noise,
            "effective_noise_rate": both_noise,
            "noise_reduction": base_noise - both_noise if np.isfinite(both_noise) else float("nan"),
            "retention": float(np.mean(both_clean)),
            "selected_precision_noise": float("nan"),
            "dropped_clean_ratio": float(((~both_clean) & (~noisy_flag)).sum() / max(1, (~noisy_flag).sum())),
        })
    return rows


def diagnose_one(args, dataset_name: str, seed: int):
    dataset_key = _dataset_key(dataset_name)
    setup_seed(seed)
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
    h_path = embedding_path_for(args.embedding_root, dataset_key, args.ssl_method)
    embeddings = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes))
    params = dict(BIG_BALL_PARAMS.get(dataset_key, BIG_BALL_PARAMS["cora"]))
    if args.max_balls is not None:
        params["max_balls"] = int(args.max_balls)
    if args.min_ball_size is not None:
        params["min_ball_size"] = int(args.min_ball_size)
    if args.split_improvement is not None:
        params["split_improvement"] = float(args.split_improvement)
    gb = build_ordinary_granular_balls(
        embeddings=embeddings,
        max_balls=params["max_balls"],
        min_ball_size=params["min_ball_size"],
        split_improvement=params["split_improvement"],
        random_state=seed,
    )
    purity = partition_purity_diagnostics(
        assignment=gb.assignment,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=args.min_train_per_ball,
    )
    train_idx = _mask_to_index_array(data.train_masks)
    noisy_train = noisy_label.detach().cpu().long().numpy()[train_idx]
    clean_train = data.labels.detach().cpu().long().numpy()[train_idx]
    noisy_flag = noisy_train != clean_train
    s_ball, eligible = _ball_suspicion_scores(
        noisy_label=noisy_label,
        assignment=gb.assignment,
        train_idx=train_idx,
        n_classes=data.n_classes,
        min_train_per_ball=args.min_train_per_ball,
    )
    early_probs = _warmup_gcn_probs(data, noisy_label, embeddings, args)
    node_idx = torch.as_tensor(train_idx, dtype=torch.long)
    s_gcn = (1.0 - early_probs[node_idx, torch.as_tensor(noisy_train, dtype=torch.long)]).cpu().numpy()
    s_knn = _knn_suspicion_scores(embeddings, noisy_label, train_idx, args.knn_k)

    gcn_clean50 = _quantile_mask(s_gcn, 0.5)
    ball_clean50 = _quantile_mask(s_ball, 0.5)
    knn_clean50 = _quantile_mask(s_knn, 0.5)
    row = {
        "dataset": dataset_key,
        "seed": int(seed),
        "n_train": int(train_idx.size),
        "train_noise_rate": float(np.mean(noisy_flag)),
        "edge_homophily": _edge_homophily(data.labels, data.adj),
        "embedding_path": str(h_path),
        "max_balls": int(params["max_balls"]),
        "min_ball_size": int(params["min_ball_size"]),
        "split_improvement": float(params["split_improvement"]),
        "num_balls": int(gb.summary["num_balls"]),
        "mean_ball_size": float(gb.summary["mean_ball_size"]),
        "true_all_purity_weighted": float(purity.get("true_all_purity_weighted", 0.0)),
        "noisy_train_purity_weighted": float(purity.get("noisy_train_purity_weighted", 0.0)),
        "mean_train_per_ball": float(purity.get("mean_train_per_ball", 0.0)),
        "eligible_train_ratio": float(np.mean(eligible)),
        "ball_auc": _safe_auc(noisy_flag, s_ball),
        "gcn_auc": _safe_auc(noisy_flag, s_gcn),
        "knn_auc": _safe_auc(noisy_flag, s_knn),
        "ball_auc_in_gcn_clean50": _safe_auc(noisy_flag[gcn_clean50], s_ball[gcn_clean50]),
        "gcn_auc_in_ball_clean50": _safe_auc(noisy_flag[ball_clean50], s_gcn[ball_clean50]),
        "ball_auc_in_knn_clean50": _safe_auc(noisy_flag[knn_clean50], s_ball[knn_clean50]),
        "corr_ball_gcn": _safe_corr(s_ball, s_gcn),
        "corr_ball_knn": _safe_corr(s_ball, s_knn),
        "gcn_clean50_noise_rate": float(np.mean(noisy_flag[gcn_clean50])) if gcn_clean50.any() else float("nan"),
        "ball_clean50_noise_rate": float(np.mean(noisy_flag[ball_clean50])) if ball_clean50.any() else float("nan"),
        "knn_clean50_noise_rate": float(np.mean(noisy_flag[knn_clean50])) if knn_clean50.any() else float("nan"),
        "warmup_epochs": int(args.warmup_epochs),
        "window_start": int(args.window_start),
        "window_end": int(args.window_end),
    }
    offline = _offline_rows(dataset_key, seed, noisy_flag, s_ball, s_gcn, args)
    return row, offline


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


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    skip = set(group_keys) | {"embedding_path"}
    out_rows = []
    for key, group in sorted(groups.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out["n"] = len(group)
        numeric_keys = []
        for row in group:
            for metric, value in row.items():
                if metric in skip:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_keys.append(metric)
        for metric in sorted(set(numeric_keys)):
            vals = [float(row[metric]) for row in group if metric in row and np.isfinite(float(row[metric]))]
            if vals:
                out[f"{metric}_mean"] = _mean(vals)
                out[f"{metric}_std"] = _std(vals)
        out_rows.append(out)
    return out_rows


def _format_float(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_report(path: Path, args, summary: list[dict], offline_summary: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 33: Big-Graph Ball Orthogonality Diagnostic",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{args.datasets}`",
        f"- Seeds: `{args.seeds}`",
        f"- Noise: `{args.noise_type}` `{args.noise_rate}`",
        f"- Split: percent {args.train_percent} / {args.val_percent} / {args.test_percent}",
        f"- Warmup GCN: `{args.warmup_epochs}` epochs, probability window `{args.window_start}-{args.window_end}`",
        "- Clean labels are used only for diagnostic metrics.",
        "- No downstream model is trained by this script.",
        "",
        "## Phase 33a Summary",
        "",
        "| Dataset | n | homophily | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        cond = float(row.get("ball_auc_in_gcn_clean50_mean", float("nan")))
        if cond >= float(args.pass_auc):
            decision = "possible orthogonal signal"
        elif 0.45 <= cond <= 0.55:
            decision = "redundant / no orthogonal gain"
        else:
            decision = "weak or unstable"
        lines.append(
            "| {dataset} | {n} | {homophily} | {noise} | {balls} | {purity} | {ball_auc} | {gcn_auc} | {cond} | {sym} | {corr} | {decision} |".format(
                dataset=row["dataset"],
                n=row["n"],
                homophily=_format_float(row.get("edge_homophily_mean")),
                noise=_format_float(row.get("train_noise_rate_mean")),
                balls=_format_float(row.get("num_balls_mean"), 2),
                purity=_format_float(row.get("true_all_purity_weighted_mean")),
                ball_auc=_format_float(row.get("ball_auc_mean")),
                gcn_auc=_format_float(row.get("gcn_auc_mean")),
                cond=_format_float(row.get("ball_auc_in_gcn_clean50_mean")),
                sym=_format_float(row.get("gcn_auc_in_ball_clean50_mean")),
                corr=_format_float(row.get("corr_ball_gcn_mean")),
                decision=decision,
            )
        )
    lines.extend([
        "",
        "## Phase 33b Offline Usage Summary",
        "",
        "| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in offline_summary:
        lines.append(
            "| {dataset} | {usage} | {param} | {n} | {eff} | {red} | {ret} | {prec} |".format(
                dataset=row["dataset"],
                usage=row["usage"],
                param=row["param"],
                n=row["n"],
                eff=_format_float(row.get("effective_noise_rate_mean")),
                red=_format_float(row.get("noise_reduction_mean")),
                ret=_format_float(row.get("retention_mean")),
                prec=_format_float(row.get("selected_precision_noise_mean")),
            )
        )
    big_rows = [row for row in summary if row["dataset"] not in {"cora", "citeseer"}]
    big_pass = [row for row in big_rows if float(row.get("ball_auc_in_gcn_clean50_mean", float("nan"))) >= float(args.pass_auc)]
    small_rows = [row for row in summary if row["dataset"] in {"cora", "citeseer"}]
    lines.extend(["", "## Decision", ""])
    if big_pass:
        lines.append(
            "At least one big graph passes the pre-registered conditional orthogonality gate. "
            "Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above."
        )
    else:
        lines.append(
            "No big graph passes the conditional orthogonality gate. Ball suspicion may have marginal signal, "
            "but it does not provide enough information inside the GCN-clean subset to justify a big-graph training variant."
        )
    if small_rows:
        small_cond = _mean([float(row.get("ball_auc_in_gcn_clean50_mean", float("nan"))) for row in small_rows])
        big_cond = _mean([float(row.get("ball_auc_in_gcn_clean50_mean", float("nan"))) for row in big_rows])
        lines.append(
            f"Small-graph mean conditional ball AUC: `{_format_float(small_cond)}`; "
            f"big-graph mean conditional ball AUC: `{_format_float(big_cond)}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose conditional orthogonality between ball and GCN noise scores.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="cora,citeseer,dblp,pubmed,amazonpho")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--method_config", default="gcn_xh_ce")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--window_start", type=int, default=5)
    parser.add_argument("--window_end", type=int, default=10)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--filter_topk", default="0.2,0.3,0.4")
    parser.add_argument("--clean_keep_rates", default="0.5,0.6,0.7")
    parser.add_argument("--pass_auc", type=float, default=0.6)
    parser.add_argument("--max_balls", type=int, default=None)
    parser.add_argument("--min_ball_size", type=int, default=None)
    parser.add_argument("--split_improvement", type=float, default=None)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "biggraph_orthogonality_diag_20260616"))
    args = parser.parse_args()
    args.datasets = ",".join(_dataset_key(item) for item in _parse_str_list(args.datasets))
    args.dataset_list = _parse_str_list(args.datasets)
    args.seeds = ",".join(str(seed) for seed in _parse_int_list(args.seeds))
    args.seed_list = _parse_int_list(args.seeds)
    args.filter_topk = _parse_float_list(args.filter_topk)
    args.clean_keep_rates = _parse_float_list(args.clean_keep_rates)

    raw_rows = []
    offline_rows = []
    for dataset in args.dataset_list:
        for seed in args.seed_list:
            print(f"ORTHO_DIAG dataset={dataset} seed={seed}")
            row, offline = diagnose_one(args, dataset, seed)
            print(
                "RESULT dataset={dataset} seed={seed} ball_auc={ball:.4f} gcn_auc={gcn:.4f} cond_auc={cond:.4f}".format(
                    dataset=row["dataset"],
                    seed=seed,
                    ball=row["ball_auc"],
                    gcn=row["gcn_auc"],
                    cond=row["ball_auc_in_gcn_clean50"],
                )
            )
            raw_rows.append(row)
            offline_rows.extend(offline)
            output_dir = Path(args.output_dir)
            write_csv(output_dir / "orthogonality_raw.csv", raw_rows)
            write_csv(output_dir / "offline_usage_raw.csv", offline_rows)

    summary = summarize(raw_rows, ["dataset"])
    offline_summary = summarize(offline_rows, ["dataset", "usage", "param"])
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "orthogonality_summary.csv", summary)
    write_csv(output_dir / "offline_usage_summary.csv", offline_summary)
    write_report(output_dir / "biggraph_orthogonality_report.md", args, summary, offline_summary)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
