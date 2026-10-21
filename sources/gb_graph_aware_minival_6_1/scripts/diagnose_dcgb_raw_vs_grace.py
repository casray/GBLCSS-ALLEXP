"""S2-mini diagnostic: is graph-aware granular ball (feature+topology) genuinely better than a
feature-only ball in the RAW feature space, where topology is NOT yet absorbed by the embedding?

For each representation space (raw normalized features X vs frozen GRACE embedding H) we build the
same granular-ball variants and report classifier-free ball quality (clean purity / noise-detection
AUROC). Clean labels are used only for diagnostics. The decisive number is the gap

    AUROC(dcgb) - AUROC(dcgb_emb_only)

within each space: it isolates what graph topology adds over feature-only mutual-kNN. The whole
"graph-aware" innovation hinges on this gap being meaningfully positive in the raw space even though
DCGB v2.8 showed it is ~0 on the strong GRACE embedding.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from noisy_label_gb_grace.balls import build_dcgb_balls, build_granular_balls
from noisy_label_gb_grace.data import load_dataset, split_name
from noisy_label_gb_grace.noise import load_or_create_noisy_labels
from noisy_label_gb_grace.smoothing import build_method_targets
from noisy_label_gb_grace.utils import PROJECT_ROOT, save_csv, save_json
from run_noisy_classifier import _group_label_diagnostics, _target_noise_diagnostics


def embedding_path(dataset, split_seed, pretrain_seed, split_mode, train_ratio, val_ratio):
    token = split_name(split_mode, split_seed, train_ratio, val_ratio)
    return PROJECT_ROOT / "checkpoints" / "embeddings" / dataset / f"{token}_pretrain{pretrain_seed}" / "H.pt"


def _build_variants(emb_norm, edge_index, args, seed):
    return {
        "gb": build_granular_balls(emb_norm, min_ball_size=args.gb_min_ball_size, min_radius_gain=args.gb_min_radius_gain, seed=seed),
        "dcgb": build_dcgb_balls(emb_norm, edge_index, max_hops=args.gagb_max_hops, min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size, seed=seed, mode="dcgb"),
        "dcgb_emb_only": build_dcgb_balls(emb_norm, edge_index, max_hops=args.gagb_max_hops, min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size, seed=seed, mode="emb_only"),
        "dcgb_topo_only": build_dcgb_balls(emb_norm, edge_index, max_hops=args.gagb_max_hops, min_ball_size=args.gb_min_ball_size, max_ball_size=args.gagb_max_ball_size, seed=seed, mode="topo_only"),
    }


def diagnose_case(args, dataset, noise_rate, seed):
    artifact = embedding_path(dataset, args.split_seed, seed, args.split_mode, args.train_ratio, args.val_ratio)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing embedding artifact: {artifact}")
    payload = torch.load(artifact, map_location="cpu")
    h_norm = F.normalize(payload["H"].float(), p=2, dim=1)
    clean_labels = payload["labels"].long()
    masks = payload["masks"]
    num_classes = int(payload["num_classes"])
    edge_index = payload.get("edge_index")

    _, data = load_dataset(dataset, device=None, split_mode=args.split_mode, split_seed=args.split_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    x_norm = F.normalize(data.x.detach().cpu().float(), p=2, dim=1)
    if edge_index is None:
        edge_index = data.edge_index.detach().cpu().long()
    # Sanity: raw features and the GRACE artifact must describe the same node set.
    if x_norm.size(0) != h_norm.size(0):
        raise ValueError(f"node count mismatch raw={x_norm.size(0)} grace={h_norm.size(0)}")

    token = split_name(args.split_mode, args.split_seed, args.train_ratio, args.val_ratio)
    noisy_labels, _, noise_meta = load_or_create_noisy_labels(
        clean_labels, masks["train_mask"], num_classes, dataset=dataset, noise_type=args.noise_type,
        noise_rate=noise_rate, split_seed=args.split_seed, noise_seed=seed, split_name=token,
    )

    spaces = {"raw_x": x_norm, "grace_h": h_norm}
    rows = []
    for space_name, emb in spaces.items():
        variants = _build_variants(emb, edge_index, args, seed)
        for variant, grouping in variants.items():
            soft_targets, _ = build_method_targets(
                "dcgb" if variant.startswith("dcgb") else "gb",
                emb, noisy_labels, masks["train_mask"], num_classes,
                alpha_max=args.alpha_max, gb_grouping=grouping,
                min_train_per_ball=args.min_train_per_ball, eps=args.eps, seed=seed, edge_index=edge_index,
            )
            stats = {}
            stats.update(_group_label_diagnostics(grouping, clean_labels, noisy_labels, masks["train_mask"], num_classes))
            stats.update(_target_noise_diagnostics(soft_targets, None, clean_labels, noisy_labels, masks["train_mask"]))
            diag = grouping["diagnostics"]
            rows.append({
                "dataset": dataset, "noise_rate": noise_rate, "seed": seed,
                "space": space_name, "variant": variant,
                "ball_count": diag.get("ball_count"), "avg_ball_size": diag.get("avg_ball_size"),
                "eligible_ball_count": stats.get("effective_coverage"),
                "clean_purity": stats.get("clean_purity"),
                "noise_detection_auroc": stats.get("noise_detection_auroc"),
                "noise_correction_rate": stats.get("noise_correction_rate"),
                "clean_damage_rate": stats.get("clean_damage_rate"),
            })
            print(f"(G) | {dataset} n={noise_rate:g} s={seed} {space_name:8s} {variant:16s} purity={stats.get('clean_purity', float('nan')):.4f} auroc={stats.get('noise_detection_auroc', float('nan')):.4f}")
    return rows


def _mean_std(values):
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return math.nan, math.nan
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(var)


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["noise_rate"], r["space"], r["variant"])].append(r)
    out = []
    for key, group in sorted(grouped.items()):
        s = {"dataset": key[0], "noise_rate": key[1], "space": key[2], "variant": key[3], "n": len(group)}
        for field in ["clean_purity", "noise_detection_auroc", "ball_count", "avg_ball_size"]:
            mean, std = _mean_std([g[field] for g in group])
            s[f"{field}_mean"], s[f"{field}_std"] = mean, std
        out.append(s)
    return out


def run(args):
    rows = []
    for dataset in args.datasets:
        for noise_rate in args.noise_rates:
            for seed in args.seeds:
                rows.extend(diagnose_case(args, dataset, noise_rate, seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    save_csv(args.output_dir / "raw_vs_grace_per_seed.csv", rows)
    save_csv(args.output_dir / "raw_vs_grace_summary.csv", summary)
    save_json(args.output_dir / "run_config.json", vars(args))
    print(f"(S) | Wrote raw-vs-grace ball diagnostics to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--split-mode", choices=["planetoid", "ratio"], default="ratio")
    p.add_argument("--train-ratio", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--noise-type", choices=["symmetric"], default="symmetric")
    p.add_argument("--noise-rates", nargs="+", type=float, default=[0.2, 0.4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--alpha-max", type=float, default=0.3)
    p.add_argument("--min-train-per-ball", type=int, default=3)
    p.add_argument("--gb-min-ball-size", type=int, default=5)
    p.add_argument("--gb-min-radius-gain", type=float, default=0.01)
    p.add_argument("--gagb-max-hops", type=int, default=2)
    p.add_argument("--gagb-max-ball-size", type=int, default=128)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "dcgb_v2_8_raw_vs_grace_20260605")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
