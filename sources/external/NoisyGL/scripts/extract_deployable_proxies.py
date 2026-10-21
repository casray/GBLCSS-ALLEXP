from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import GCN
from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


DATASET_ORDER = [
    "cora",
    "citeseer",
    "pubmed",
    "amazonpho",
    "dblp",
    "coauthorcs",
    "amazoncom",
    "coauthorph",
    "wikics",
]

SOFTDIST_TARGETS = ["softdist_vs_gcn_xh_mean", "softdist_vs_shuffled_mean"]
GBSELECT_TARGETS = ["gbselect_vs_gcn_only_mean", "gbselect_vs_shuffled_mean"]

DEPLOYABLE_PROXIES = [
    "noisy_train_purity_weighted",
    "noisy_train_purity_unweighted",
    "mean_vote_margin_weighted",
    "mean_vote_margin_unweighted",
    "mean_vote_entropy_weighted",
    "mean_vote_entropy_unweighted",
    "eligible_train_ratio",
    "eligible_ball_ratio",
    "num_balls",
    "mean_ball_size",
    "mean_train_per_ball",
    "ball_gcn_disagree_train",
    "ball_gcn_disagree_confident_train",
    "ball_gcn_original_label_disagree_train",
]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _dataset_key(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
        "coauthor-cs": "coauthorcs",
        "coauthorphysics": "coauthorph",
        "coauthor-physics": "coauthorph",
        "wikics": "wikics",
        "wiki-cs": "wikics",
    }
    return aliases.get(key, key)


def _safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(x, y).correlation
        return float(corr) if corr is not None and math.isfinite(float(corr)) else math.nan
    except Exception:
        return _pearson(_rankdata(x), _rankdata(y))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.flatnonzero(mask)
    return mask.astype(np.int64)


def _conf_value(conf, key: str, default=None):
    if conf is None:
        return default
    if isinstance(conf, dict):
        return conf.get(key, default)
    return getattr(conf, key, default)


def _load_ball_params(method_config: str, dataset: str) -> dict | None:
    try:
        conf = load_conf(None, method_config, dataset)
    except Exception:
        return None
    gb_conf = getattr(conf, "gb", None)
    if gb_conf is None:
        return None
    return {
        "max_balls": int(_conf_value(gb_conf, "max_balls", 64)),
        "min_ball_size": int(_conf_value(gb_conf, "min_ball_size", 32)),
        "split_improvement": float(_conf_value(gb_conf, "split_improvement", 0.05)),
        "min_train_per_ball": int(_conf_value(gb_conf, "min_train_per_ball", 2)),
        "warmup_epochs": int(_conf_value(gb_conf, "warmup_epochs", 10)),
        "warmup_window_start": int(_conf_value(gb_conf, "warmup_window_start", 5)),
        "warmup_window_end": int(_conf_value(gb_conf, "warmup_window_end", 10)),
    }


def _prepare_xh_features(features, embeddings, device):
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    features = features.float().to(device)
    embeddings = F.normalize(embeddings.float().to(device), p=2, dim=1)
    return torch.cat([features, embeddings], dim=1)


def _warmup_gcn_probs(data, noisy_label, embeddings, args, warmup_epochs: int, window_start: int, window_end: int):
    features = _prepare_xh_features(data.feats, embeddings, data.feats.device)
    conf = load_conf(None, args.warmup_method_config, data.name)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.training["lr"], weight_decay=conf.training["weight_decay"])
    snapshots = []
    for epoch in range(1, int(warmup_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.adj)
        loss = F.cross_entropy(logits[data.train_masks], noisy_label[data.train_masks].long())
        loss.backward()
        optimizer.step()
        if int(window_start) <= epoch <= int(window_end):
            model.eval()
            with torch.no_grad():
                snapshots.append(F.softmax(model(features, data.adj), dim=1).detach().cpu())
    if not snapshots:
        raise ValueError("warmup window produced no snapshots")
    return torch.stack(snapshots, dim=0).mean(dim=0)


def _entropy(dist: np.ndarray) -> float:
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist[dist > 0]
    if dist.size == 0:
        return 0.0
    return float(-(dist * np.log(dist)).sum())


def _ball_vote_proxies(
    assignment: np.ndarray,
    noisy_label: torch.Tensor,
    train_mask,
    n_classes: int,
    min_train_per_ball: int,
    gcn_probs: torch.Tensor | None,
) -> dict[str, object]:
    train_idx = _mask_to_index_array(train_mask)
    noisy_np = noisy_label.detach().cpu().long().numpy()
    assignment = np.asarray(assignment, dtype=np.int64)
    n_balls = int(np.unique(assignment).size)
    n_classes = int(n_classes)

    eligible_balls = 0
    eligible_train_total = 0
    purity_weighted = 0.0
    margin_weighted = 0.0
    entropy_weighted = 0.0
    purities = []
    margins = []
    entropies = []
    train_counts = []
    gcn_disagree = []
    gcn_confident_disagree = []
    gcn_original_label_disagree = []

    if gcn_probs is not None:
        gcn_pred = gcn_probs.argmax(dim=1).cpu().numpy()
        gcn_conf = gcn_probs.max(dim=1).values.cpu().numpy()
    else:
        gcn_pred = None
        gcn_conf = None

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=n_classes).astype(np.float64)
        total = float(max(1.0, counts.sum()))
        dist = counts / total
        sorted_counts = np.sort(counts)
        majority = int(np.argmax(counts))
        purity = float(counts.max() / total)
        second = float(sorted_counts[-2]) if sorted_counts.size >= 2 else 0.0
        margin = float((counts.max() - second) / total)
        entropy = _entropy(dist) / math.log(max(2, n_classes))
        eligible_balls += 1
        eligible_train_total += int(ball_train.size)
        purity_weighted += purity * int(ball_train.size)
        margin_weighted += margin * int(ball_train.size)
        entropy_weighted += entropy * int(ball_train.size)
        purities.append(purity)
        margins.append(margin)
        entropies.append(entropy)
        train_counts.append(int(ball_train.size))

        if gcn_pred is not None and gcn_conf is not None:
            pred = gcn_pred[ball_train]
            original = noisy_np[ball_train]
            disagree = pred != majority
            original_disagree = pred != original
            gcn_disagree.extend(disagree.astype(np.float64).tolist())
            confident = gcn_conf[ball_train] >= 0.7
            if confident.any():
                gcn_confident_disagree.extend(disagree[confident].astype(np.float64).tolist())
            gcn_original_label_disagree.extend(original_disagree.astype(np.float64).tolist())

    denom = max(1, eligible_train_total)
    return {
        "num_balls": int(n_balls),
        "eligible_balls": int(eligible_balls),
        "eligible_ball_ratio": float(eligible_balls / max(1, n_balls)),
        "eligible_train_nodes": int(eligible_train_total),
        "eligible_train_ratio": float(eligible_train_total / max(1, train_idx.size)),
        "noisy_train_purity_weighted": float(purity_weighted / denom),
        "noisy_train_purity_unweighted": _mean(purities),
        "mean_vote_margin_weighted": float(margin_weighted / denom),
        "mean_vote_margin_unweighted": _mean(margins),
        "mean_vote_entropy_weighted": float(entropy_weighted / denom),
        "mean_vote_entropy_unweighted": _mean(entropies),
        "mean_train_per_ball": _mean([float(v) for v in train_counts]),
        "ball_gcn_disagree_train": _mean([float(v) for v in gcn_disagree]),
        "ball_gcn_disagree_confident_train": _mean([float(v) for v in gcn_confident_disagree]),
        "ball_gcn_original_label_disagree_train": _mean([float(v) for v in gcn_original_label_disagree]),
    }


def extract_one_dataset_seed(args, dataset: str, seed: int) -> list[dict[str, object]]:
    setup_seed(seed)
    data = _load_dataset(
        dataset,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    h_path = embedding_path_for(args.embedding_root, dataset, args.ssl_method)
    if not h_path.exists():
        return [
            {
                "dataset": dataset,
                "seed": int(seed),
                "proxy_config": config_name,
                "status": "missing_h",
                "embedding_path": str(h_path),
                "error": f"missing H: {h_path}",
            }
            for config_name in args.proxy_config_list
        ]

    embeddings = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes))
    warmup_params = _load_ball_params(args.warmup_param_config, dataset) or {
        "warmup_epochs": 10,
        "warmup_window_start": 5,
        "warmup_window_end": 10,
    }
    warmup_epochs = int(args.warmup_epochs if args.warmup_epochs is not None else warmup_params.get("warmup_epochs", 10))
    window_start = int(
        args.warmup_window_start if args.warmup_window_start is not None else warmup_params.get("warmup_window_start", 5)
    )
    window_end = int(args.warmup_window_end if args.warmup_window_end is not None else warmup_params.get("warmup_window_end", 10))
    gcn_probs = _warmup_gcn_probs(data, noisy_label, embeddings, args, warmup_epochs, window_start, window_end)

    train_idx = _mask_to_index_array(data.train_masks)
    modified = np.asarray(modified_mask, dtype=np.int64)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    rows = []
    for config_name in args.proxy_config_list:
        params = _load_ball_params(config_name, dataset)
        if params is None:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "proxy_config": config_name,
                    "status": "missing_config",
                    "embedding_path": str(h_path),
                    "error": f"missing config for {config_name}/{dataset}",
                }
            )
            continue
        gb = build_ordinary_granular_balls(
            embeddings=embeddings,
            max_balls=params["max_balls"],
            min_ball_size=params["min_ball_size"],
            split_improvement=params["split_improvement"],
            random_state=seed,
        )
        proxies = _ball_vote_proxies(
            assignment=gb.assignment,
            noisy_label=noisy_label,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            min_train_per_ball=params["min_train_per_ball"],
            gcn_probs=gcn_probs,
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "proxy_config": config_name,
                "status": "ok",
                "embedding_path": str(h_path),
                "n_nodes": int(data.n_nodes),
                "n_classes": int(data.n_classes),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "modified_count": int(modified.size),
                "actual_noise_rate": float(modified.size / max(1, int(data.n_nodes))),
                "modified_train_count": int(modified_train.size),
                "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
                "max_balls": int(params["max_balls"]),
                "min_ball_size": int(params["min_ball_size"]),
                "split_improvement": float(params["split_improvement"]),
                "min_train_per_ball": int(params["min_train_per_ball"]),
                "warmup_epochs": int(warmup_epochs),
                "warmup_window_start": int(window_start),
                "warmup_window_end": int(window_end),
                "mean_ball_size": float(gb.summary["mean_ball_size"]),
                **proxies,
            }
        )
    return rows


def summarize_proxy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("dataset", "")), str(row.get("proxy_config", "")))].append(row)
    out_rows = []
    skip = {"dataset", "seed", "proxy_config", "status", "embedding_path", "error", "noise_type"}
    for (dataset, proxy_config), group in sorted(groups.items(), key=lambda kv: (DATASET_ORDER.index(kv[0][0]) if kv[0][0] in DATASET_ORDER else 999, kv[0][1])):
        ok_group = [row for row in group if row.get("status") == "ok"]
        out = {
            "dataset": dataset,
            "proxy_config": proxy_config,
            "n": len(ok_group),
            "status": "ok" if ok_group else str(group[0].get("status", "missing")),
            "error": "" if ok_group else str(group[0].get("error", "")),
        }
        numeric_keys = []
        for row in ok_group:
            for key, value in row.items():
                if key in skip:
                    continue
                if _safe_float(value) is not None:
                    numeric_keys.append(key)
        for key in sorted(set(numeric_keys)):
            values = [_safe_float(row.get(key)) for row in ok_group]
            values = [float(value) for value in values if value is not None]
            if values:
                out[f"{key}_mean"] = _mean(values)
                out[f"{key}_std"] = _std(values)
        out_rows.append(out)
    return out_rows


def _target_rows(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for row in _read_rows(path):
        dataset = _dataset_key(row.get("dataset", ""))
        if not dataset:
            continue
        rows[dataset] = {key: value for key, value in row.items()}
    return rows


def _select_targets_for_proxy_config(proxy_config: str) -> list[str]:
    if "gbselect" in proxy_config:
        return GBSELECT_TARGETS
    return SOFTDIST_TARGETS


def correlation_rows(summary_rows: list[dict[str, object]], target_path: Path) -> list[dict[str, object]]:
    targets = _target_rows(target_path)
    rows = []
    for proxy_config in sorted({str(row.get("proxy_config", "")) for row in summary_rows}):
        config_rows = [row for row in summary_rows if row.get("proxy_config") == proxy_config and row.get("status") == "ok"]
        for target_name in _select_targets_for_proxy_config(proxy_config):
            for proxy_name in DEPLOYABLE_PROXIES:
                x = []
                y = []
                datasets = []
                for row in config_rows:
                    dataset = str(row["dataset"])
                    proxy_value = _safe_float(row.get(f"{proxy_name}_mean"))
                    target_value = _safe_float(targets.get(dataset, {}).get(target_name))
                    if proxy_value is None or target_value is None:
                        continue
                    x.append(proxy_value)
                    y.append(target_value)
                    datasets.append(dataset)
                rows.append(
                    {
                        "proxy_config": proxy_config,
                        "target": target_name,
                        "proxy": proxy_name,
                        "n": len(x),
                        "spearman": _spearman(x, y),
                        "pearson": _pearson(x, y),
                        "datasets": " ".join(datasets),
                    }
                )
    return rows


def _fmt(value, digits=4) -> str:
    value = _safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _strong_proxy_rows(corr_rows: list[dict[str, object]], min_n: int, min_abs_spearman: float) -> list[dict[str, object]]:
    out = []
    for row in corr_rows:
        spearman = _safe_float(row.get("spearman"))
        n = int(row.get("n", 0))
        if spearman is not None and n >= int(min_n) and abs(spearman) >= float(min_abs_spearman):
            out.append(row)
    return out


def write_report(path: Path, summary_rows: list[dict[str, object]], corr_rows: list[dict[str, object]], args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_summary_rows = [row for row in summary_rows if row.get("status") == "ok"]
    noisy_purities = [
        float(value)
        for value in (_safe_float(row.get("noisy_train_purity_weighted_mean")) for row in ok_summary_rows)
        if value is not None
    ]
    max_corr_n = max([int(row.get("n", 0)) for row in corr_rows], default=0)
    precheck_rows = [
        row
        for row in corr_rows
        if row["proxy"] == "noisy_train_purity_weighted" and row["target"] in {"softdist_vs_gcn_xh_mean", "gbselect_vs_gcn_only_mean"}
    ]
    strong = _strong_proxy_rows(corr_rows, args.min_corr_n, args.min_abs_spearman)
    lines = [
        "# Phase 37d Deployable Proxy Report",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{args.datasets}`",
        f"- Seeds: `{args.seeds}`",
        f"- Noise: `{args.noise_type}` `{args.noise_rate}`",
        f"- Split: percent `{args.train_percent}/{args.val_percent}/{args.test_percent}`",
        "- Proxy computation uses H geometry + noisy train labels only.",
        "- Clean labels/test outcomes are used only in target deltas already computed by Phase 37c.",
        "- Coauthor-Physics is expected to be skipped unless its GRACE H exists.",
        "",
        "## Proxy Summary",
        "",
        "| Dataset | Config | n | noisy purity | vote margin | entropy | eligible train | GCN-ball disagree | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {config} | {n} | {purity} | {margin} | {entropy} | {eligible} | {disagree} | {status} |".format(
                dataset=row.get("dataset", ""),
                config=row.get("proxy_config", ""),
                n=row.get("n", 0),
                purity=_fmt(row.get("noisy_train_purity_weighted_mean")),
                margin=_fmt(row.get("mean_vote_margin_weighted_mean")),
                entropy=_fmt(row.get("mean_vote_entropy_weighted_mean")),
                eligible=_fmt(row.get("eligible_train_ratio_mean")),
                disagree=_fmt(row.get("ball_gcn_disagree_train_mean")),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "## 37d-pre: Noisy Purity Check", ""])
    if noisy_purities:
        lines.append(
            "Noisy train purity is not numerically washed flat across the available H-backed rows: "
            f"min={min(noisy_purities):.4f}, max={max(noisy_purities):.4f}, range={max(noisy_purities) - min(noisy_purities):.4f}."
        )
        lines.append("")
    if precheck_rows:
        lines.extend(["| Proxy config | Target | n | Spearman | Pearson |", "|---|---|---:|---:|---:|"])
        for row in precheck_rows:
            lines.append(
                f"| {row['proxy_config']} | {row['target']} | {row['n']} | {_fmt(row['spearman'])} | {_fmt(row['pearson'])} |"
            )
    else:
        lines.append("No precheck correlation rows were available.")
    lines.extend(["", "## 37d-main: Top Deployable Proxy Correlations", ""])
    ranked = sorted(
        [row for row in corr_rows if int(row.get("n", 0)) >= 3 and _safe_float(row.get("spearman")) is not None],
        key=lambda row: abs(float(row["spearman"])),
        reverse=True,
    )[:20]
    lines.extend(["| Config | Target | Proxy | n | Spearman | Pearson |", "|---|---|---|---:|---:|---:|"])
    for row in ranked:
        lines.append(
            f"| {row['proxy_config']} | {row['target']} | {row['proxy']} | {row['n']} | {_fmt(row['spearman'])} | {_fmt(row['pearson'])} |"
        )
    lines.extend(["", "## 37d-judge", ""])
    if max_corr_n < int(args.min_corr_n):
        lines.append(
            f"The largest available correlation sample is n={max_corr_n}, below the pre-registered n >= {args.min_corr_n} bar. "
            "Therefore this run cannot legally pass or fail the adaptive-trigger criterion; treat it as underpowered evidence and fill more target deltas before implementing an adaptive layer."
        )
    elif strong:
        lines.append(
            f"At least one deployable proxy reaches |Spearman| >= {args.min_abs_spearman} with n >= {args.min_corr_n}. "
            "This is a legitimate but still small-sample basis for a future adaptive layer; expand to 12+ datasets before claiming a rule."
        )
    else:
        lines.append(
            f"No deployable proxy reaches |Spearman| >= {args.min_abs_spearman} with n >= {args.min_corr_n}. "
            "Under the pre-registered rule, the adaptive layer should not be implemented yet; keep it as future work unless more datasets change this."
        )
    lines.extend(
        [
            "",
            "Claim boundary: write only that evidence suggests ball-clusterability mediates ball gains. "
            "Do not write that a causal determining variable has been found.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deployable H+noisy-label proxies and correlate them with ball gains.")
    parser.add_argument("--embedding_root", default=str(REPO_ROOT / ".." / ".." / "gb_graph_aware_minival_6_1" / "checkpoints" / "embeddings_v2"))
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--proxy_configs", default="frozengrace_gb_gcn_softdist_xh,gbselect")
    parser.add_argument("--warmup_method_config", default="gcn_xh_ce")
    parser.add_argument("--warmup_param_config", default="gbselect")
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--warmup_window_start", type=int, default=None)
    parser.add_argument("--warmup_window_end", type=int, default=None)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--target_summary", default=str(REPO_ROOT / "runs" / "applicability_grid_20260617" / "applicability_grid_summary.csv"))
    parser.add_argument("--min_corr_n", type=int, default=7)
    parser.add_argument("--min_abs_spearman", type=float, default=0.7)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "deployable_proxy_20260617"))
    parser.add_argument("--postprocess_only", action="store_true", help="Reuse deployable_proxy_raw.csv in output_dir and regenerate summary/correlation/report.")
    args = parser.parse_args()
    args.dataset_list = [_dataset_key(item) for item in _parse_str_list(args.datasets)]
    args.proxy_config_list = _parse_str_list(args.proxy_configs)
    args.seed_list = _parse_int_list(args.seeds)
    args.datasets = ",".join(args.dataset_list)
    args.seeds = ",".join(str(seed) for seed in args.seed_list)

    output_dir = Path(args.output_dir)
    if args.postprocess_only:
        raw_rows = _read_rows(output_dir / "deployable_proxy_raw.csv")
        if not raw_rows:
            raise FileNotFoundError(f"no raw rows found at {output_dir / 'deployable_proxy_raw.csv'}")
    else:
        raw_rows: list[dict[str, object]] = []
        for dataset in args.dataset_list:
            for seed in args.seed_list:
                print(f"DEPLOYABLE_PROXY dataset={dataset} seed={seed}")
                rows = extract_one_dataset_seed(args, dataset, seed)
                for row in rows:
                    status = row.get("status")
                    print(f"RESULT dataset={dataset} seed={seed} config={row.get('proxy_config')} status={status}")
                raw_rows.extend(rows)
                _write_csv(output_dir / "deployable_proxy_raw.csv", raw_rows)

    summary_rows = summarize_proxy_rows(raw_rows)
    corr_rows = correlation_rows(summary_rows, Path(args.target_summary))
    _write_csv(output_dir / "deployable_proxy_summary.csv", summary_rows)
    _write_csv(output_dir / "deployable_proxy_correlations.csv", corr_rows)
    write_report(output_dir / "deployable_proxy_report.md", summary_rows, corr_rows, args)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
