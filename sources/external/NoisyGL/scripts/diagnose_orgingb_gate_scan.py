import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor.module.GNNs import GCN
from utils.dataloader import Dataset
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls, build_random_size_matched_assignment
from utils.soft_targets import build_gb_soft_targets
from utils.tools import load_conf, setup_seed


TOP_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
MIN_BALL_CONFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def conf_value(conf_section, key, default):
    if isinstance(conf_section, dict):
        value = conf_section.get(key, default)
    else:
        value = getattr(conf_section, key, default)
    return default if value is None else value


def dataset_kwargs_from_conf(data_conf, dataset_name, device):
    return {
        "data": dataset_name,
        "path": "./data/",
        "feat_norm": data_conf.norm["feat_norm"],
        "adj_norm": data_conf.norm["adj_norm"],
        "train_size": data_conf.split["train_size"],
        "val_size": data_conf.split["val_size"],
        "test_size": data_conf.split["test_size"],
        "train_percent": data_conf.split["train_percent"],
        "val_percent": data_conf.split["val_percent"],
        "test_percent": data_conf.split["test_percent"],
        "train_examples_per_class": data_conf.split["train_examples_per_class"],
        "val_examples_per_class": data_conf.split["val_examples_per_class"],
        "test_examples_per_class": data_conf.split["test_examples_per_class"],
        "add_self_loop": data_conf.modify["add_self_loop"],
        "from_npz": data_conf.modify["from_npz_largest_component"],
        "device": device,
        "split_type": data_conf.split["split_type"],
    }


def select_top_count_mask(scores, train_indices, count):
    scores = np.asarray(scores, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    mask = np.zeros(scores.shape[0], dtype=bool)
    count = int(count)
    if count <= 0 or train_indices.size == 0:
        return mask
    count = min(count, train_indices.size)
    order = np.lexsort((train_indices, -scores[train_indices]))
    chosen = train_indices[order[:count]]
    mask[chosen] = True
    return mask


def select_real_gate_mask(scores, node_ball_confidence, train_indices, top_ratio, min_ball_conf):
    scores = np.asarray(scores, dtype=np.float64)
    node_ball_confidence = np.asarray(node_ball_confidence, dtype=np.float64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    count = int(np.ceil(train_indices.size * float(top_ratio)))
    top_mask = select_top_count_mask(scores, train_indices, count)
    conf_mask = node_ball_confidence >= float(min_ball_conf)
    return top_mask & conf_mask


def build_strength_matched_controls(real_mask, train_indices, no_ball_scores, random_scores):
    real_mask = np.asarray(real_mask, dtype=bool)
    count = int(real_mask.sum())
    return {
        "no_ball_strength_matched": select_top_count_mask(no_ball_scores, train_indices, count),
        "random_strength_matched": select_top_count_mask(random_scores, train_indices, count),
    }


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def summarize_gate(mask, train_indices, noisy_label, clean_label):
    mask = np.asarray(mask, dtype=bool)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    noisy_label = np.asarray(noisy_label, dtype=np.int64)
    clean_label = np.asarray(clean_label, dtype=np.int64)

    train_mask = np.zeros(mask.shape[0], dtype=bool)
    train_mask[train_indices] = True
    gated_train = mask & train_mask
    noise_flag = noisy_label != clean_label
    clean_flag = ~noise_flag
    train_noise = noise_flag & train_mask
    train_clean = clean_flag & train_mask
    gated_noise = gated_train & noise_flag
    gated_clean = gated_train & clean_flag

    gated_count = int(gated_train.sum())
    return {
        "gated_count": gated_count,
        "gated_ratio": float(gated_count / max(train_indices.size, 1)),
        "gated_noisy_precision": float(gated_noise.sum() / gated_count) if gated_count else 0.0,
        "gated_noisy_recall": float(gated_noise.sum() / max(train_noise.sum(), 1)),
        "gated_clean_ratio": float(gated_clean.sum() / max(train_clean.sum(), 1)),
    }


def train_warmup_gcn(conf, data, device, warmup_epochs):
    model = GCN(
        in_channels=conf.model["n_feat"],
        hidden_channels=conf.model["n_hidden"],
        out_channels=conf.model["n_classes"],
        n_layers=conf.model["n_layer"],
        dropout=conf.model["dropout"],
        norm_info=conf.model["norm_info"],
        act=conf.model["act"],
        input_layer=conf.model["input_layer"],
        output_layer=conf.model["output_layer"],
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=conf.training["lr"], weight_decay=conf.training["weight_decay"])
    adj = data.adj if conf.dataset["sparse"] else data.adj.to_dense()
    for _ in range(int(warmup_epochs)):
        model.train()
        optim.zero_grad()
        logits = model(data.feats, adj)
        loss = F.cross_entropy(logits[data.train_masks], data.noisy_label[data.train_masks])
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        logits, hidden = model.forward_with_hidden(data.feats, adj)
        probs = F.softmax(logits, dim=1)
    return probs, hidden


def build_soft_result(data, probs, assignment, clean_label, target_mode="reweight", suspicion_mode="gb"):
    conf = load_conf(None, "orgingb_gcn_reweight", data.name)
    gb_conf = getattr(conf, "gb", {})
    return build_gb_soft_targets(
        noisy_label=data.noisy_label,
        clean_label=clean_label,
        probs=probs,
        assignment=assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        target_mode=target_mode,
        suspicion_mode=suspicion_mode,
        suspicion_top_ratio=1.0,
        min_ball_conf=0.0,
        beta=conf_value(gb_conf, "beta", 0.5),
        min_weight=conf_value(gb_conf, "min_weight", 0.3),
        alpha_max=conf_value(gb_conf, "alpha_max", 0.5),
        gamma=conf_value(gb_conf, "gamma", 0.5),
        eta_max=conf_value(gb_conf, "eta_max", 0.4),
        min_train_per_ball=conf_value(gb_conf, "min_train_per_ball", 2),
        full_train_per_ball=conf_value(gb_conf, "full_train_per_ball", 4),
        label_prior_smoothing=conf_value(gb_conf, "label_prior_smoothing", 1.0),
    )


def rows_for_setting(
    top_ratio,
    min_ball_conf,
    train_indices,
    noisy_label_np,
    clean_label_np,
    real_scores,
    no_ball_scores,
    random_scores,
    node_ball_confidence,
    real_auc,
    no_ball_auc,
    random_auc,
    ball_summary,
    mean_ball_conf,
):
    real_mask = select_real_gate_mask(real_scores, node_ball_confidence, train_indices, top_ratio, min_ball_conf)
    controls = build_strength_matched_controls(real_mask, train_indices, no_ball_scores, random_scores)
    real_metrics = summarize_gate(real_mask, train_indices, noisy_label_np, clean_label_np)
    no_ball_metrics = summarize_gate(controls["no_ball_strength_matched"], train_indices, noisy_label_np, clean_label_np)
    random_metrics = summarize_gate(controls["random_strength_matched"], train_indices, noisy_label_np, clean_label_np)
    gain_vs_no_ball = real_metrics["gated_noisy_precision"] - no_ball_metrics["gated_noisy_precision"]
    gain_vs_random = real_metrics["gated_noisy_precision"] - random_metrics["gated_noisy_precision"]

    base = {
        "suspicion_top_ratio": top_ratio,
        "min_ball_conf": min_ball_conf,
        "precision_gain_vs_no_ball": gain_vs_no_ball,
        "precision_gain_vs_random": gain_vs_random,
        "num_balls": ball_summary["num_balls"],
        "mean_ball_size": ball_summary["mean_ball_size"],
        "mean_ball_conf": mean_ball_conf,
    }
    rows = []
    for method, metrics, auc in [
        ("real_gb", real_metrics, real_auc),
        ("no_ball_strength_matched", no_ball_metrics, no_ball_auc),
        ("random_strength_matched", random_metrics, random_auc),
    ]:
        row = {
            **base,
            "method": method,
            "suspicion_auroc": auc,
            **metrics,
        }
        rows.append(row)
    return rows


def choose_best_passing(rows):
    real_rows = [row for row in rows if row["method"] == "real_gb"]
    passing = [
        row for row in real_rows
        if row["gated_noisy_precision"] >= 0.55
        and row["precision_gain_vs_no_ball"] >= 0.05
        and row["precision_gain_vs_random"] >= 0.10
        and 0.05 <= row["gated_ratio"] <= 0.25
        and row["gated_clean_ratio"] <= 0.20
    ]
    if not passing:
        return None, real_rows
    best = sorted(
        passing,
        key=lambda row: (
            -row["precision_gain_vs_no_ball"],
            -row["gated_noisy_precision"],
            row["gated_ratio"],
        ),
    )[0]
    return best, real_rows


def write_summary(path, rows):
    best, real_rows = choose_best_passing(rows)
    strongest = sorted(real_rows, key=lambda row: (-row["gated_noisy_precision"], row["gated_ratio"]))[:5]
    lines = [
        "# OrginGB Gate Scan Summary",
        "",
        "## Decision",
        "",
    ]
    if best is None:
        lines.extend([
            "FAIL: no setting passed the offline gate criteria.",
            "",
            "Decision: do not run formal 5-seed.",
            "",
            "Ordinary GB should pause as the main denoising mechanism unless a later diagnostic shows independent gate value over no-ball and random controls.",
        ])
    else:
        lines.extend([
            "PASS: at least one setting passed the offline gate criteria.",
            "",
            (
                "Best setting: "
                f"top_ratio={best['suspicion_top_ratio']}, "
                f"min_ball_conf={best['min_ball_conf']}, "
                f"precision={best['gated_noisy_precision']:.4f}, "
                f"gain_vs_no_ball={best['precision_gain_vs_no_ball']:.4f}, "
                f"gain_vs_random={best['precision_gain_vs_random']:.4f}."
            ),
        ])
    lines.extend([
        "",
        "## Top Real-GB Settings By Precision",
        "",
        "| top_ratio | min_ball_conf | gated_ratio | precision | recall | clean_ratio | gain_vs_no_ball | gain_vs_random |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in strongest:
        lines.append(
            f"| {row['suspicion_top_ratio']:.2f} | {row['min_ball_conf']:.1f} | "
            f"{row['gated_ratio']:.4f} | {row['gated_noisy_precision']:.4f} | "
            f"{row['gated_noisy_recall']:.4f} | {row['gated_clean_ratio']:.4f} | "
            f"{row['precision_gain_vs_no_ball']:.4f} | {row['precision_gain_vs_random']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gate_scan(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    data_conf = load_conf("./config/_dataset/" + args.dataset + ".yaml")
    conf = load_conf(None, "orgingb_gcn_reweight", args.dataset)
    data = Dataset(**dataset_kwargs_from_conf(data_conf, args.dataset, args.device))
    clean_label = data.labels
    data.noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=args.seed,
        debug=False,
    )
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes

    probs, hidden = train_warmup_gcn(conf, data, device, args.warmup_epochs)
    gb_conf = getattr(conf, "gb", {})
    train_count = len(data.train_masks)
    max_balls = conf_value(gb_conf, "max_balls", min(64, max(8, train_count // 3)))
    min_ball_size = conf_value(
        gb_conf,
        "min_ball_size",
        max(32, int(np.ceil(data.n_nodes / float(max_balls)))),
    )
    gb_result = build_ordinary_granular_balls(
        hidden,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=conf_value(gb_conf, "split_improvement", 0.05),
        random_state=int(torch.initial_seed() % (2 ** 32 - 1)),
    )
    random_assignment = build_random_size_matched_assignment(
        gb_result.assignment,
        random_state=int(torch.initial_seed() % (2 ** 32 - 1)) + 17,
    )

    real_soft = build_soft_result(data, probs, gb_result.assignment, clean_label, "reweight", "gb")
    random_soft = build_soft_result(data, probs, random_assignment, clean_label, "reweight", "gb")
    no_ball_soft = build_soft_result(data, probs, gb_result.assignment, clean_label, "reweight", "no_ball")

    train_indices = np.asarray(data.train_masks, dtype=np.int64)
    noisy_label_np = data.noisy_label.detach().cpu().numpy()
    clean_label_np = clean_label.detach().cpu().numpy()
    noise_flags = (noisy_label_np[train_indices] != clean_label_np[train_indices]).astype(np.int64)

    real_scores = real_soft.suspicion.detach().cpu().numpy()
    random_scores = random_soft.suspicion.detach().cpu().numpy()
    no_ball_scores = no_ball_soft.suspicion.detach().cpu().numpy()
    node_ball_confidence = real_soft.node_ball_confidence.detach().cpu().numpy()
    train_ball_conf = node_ball_confidence[train_indices]
    mean_ball_conf = float(train_ball_conf.mean()) if train_ball_conf.size else 0.0

    real_auc = safe_auc(noise_flags, real_scores[train_indices])
    random_auc = safe_auc(noise_flags, random_scores[train_indices])
    no_ball_auc = safe_auc(noise_flags, no_ball_scores[train_indices])

    rows = []
    for top_ratio in TOP_RATIOS:
        for min_ball_conf in MIN_BALL_CONFS:
            rows.extend(
                rows_for_setting(
                    top_ratio=top_ratio,
                    min_ball_conf=min_ball_conf,
                    train_indices=train_indices,
                    noisy_label_np=noisy_label_np,
                    clean_label_np=clean_label_np,
                    real_scores=real_scores,
                    no_ball_scores=no_ball_scores,
                    random_scores=random_scores,
                    node_ball_confidence=node_ball_confidence,
                    real_auc=real_auc,
                    no_ball_auc=no_ball_auc,
                    random_auc=random_auc,
                    ball_summary=gb_result.summary,
                    mean_ball_conf=mean_ball_conf,
                )
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gate_scan.csv"
    fieldnames = [
        "method", "suspicion_top_ratio", "min_ball_conf",
        "gated_count", "gated_ratio", "gated_noisy_precision", "gated_noisy_recall",
        "gated_clean_ratio", "precision_gain_vs_no_ball", "precision_gain_vs_random",
        "suspicion_auroc", "num_balls", "mean_ball_size", "mean_ball_conf",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(out_dir / "gate_scan_summary.md", rows)
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'gate_scan_summary.md'}")
    print(f"rows {len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora", choices=["cora"])
    parser.add_argument("--noise_type", default="uniform", choices=["uniform"])
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--output_dir", default="runs/orgingb_gate_scan_20260603")
    return parser.parse_args()


if __name__ == "__main__":
    run_gate_scan(parse_args())
