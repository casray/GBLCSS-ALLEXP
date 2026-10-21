from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MINIVAL_ROOT = REPO_ROOT.parents[1] / "gb_graph_aware_minival_6_1"
MINIVAL_SRC = MINIVAL_ROOT / "src"
if str(MINIVAL_SRC) not in sys.path:
    sys.path.insert(0, str(MINIVAL_SRC))

from noisy_label_gb_grace.data import load_dataset as load_grace_dataset
from noisy_label_gb_grace.data import split_name
from noisy_label_gb_grace.upstream_grace import build_grace_model, extract_encoder_h, load_grace_config

from predictor.FrozenGraceGB_GCN_Predictor import concat_l2_normalized_embeddings
from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_standard_downstream import (
    _config_id,
    _noise_fingerprint,
    _noise_stats,
    _parameter_count,
    _parse_int_list,
    _parse_str_list,
    _split_fingerprint,
    _load_dataset,
)
from scripts.run_ssl_swap_downstream import _set_embedding_path, _set_gb_value, predictor_class_for
from utils.labelnoise import label_process
from utils.structure_noise import perturb_test_adj
from utils.tools import load_conf, setup_seed


DATASET_CANONICAL = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def checkpoint_path(dataset_key: str, pretrain_seed: int, split_seed: int, train_ratio: float, val_ratio: float) -> Path:
    dataset_name = DATASET_CANONICAL[dataset_key]
    split_token = split_name("ratio", split_seed, train_ratio, val_ratio)
    return MINIVAL_ROOT / "checkpoints" / "grace" / dataset_name / f"{split_token}_pretrain{pretrain_seed}" / "checkpoint.pt"


def load_grace_encoder(dataset_key: str, device: str, pretrain_seed: int, split_seed: int, train_ratio: float, val_ratio: float):
    dataset_name = DATASET_CANONICAL[dataset_key]
    dataset, data = load_grace_dataset(
        dataset_name,
        device=torch.device(device),
        split_mode="ratio",
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    config = load_grace_config(dataset_name)
    config["seed"] = int(pretrain_seed)
    model = build_grace_model(dataset, config, torch.device(device))
    ckpt_path = checkpoint_path(dataset_key, pretrain_seed, split_seed, train_ratio, val_ratio)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing GRACE encoder checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, data.x, str(ckpt_path)


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    sparse = adj.coalesce() if getattr(adj, "is_sparse", False) else adj.to_sparse().coalesce()
    return sparse.indices().long()


@torch.no_grad()
def encode_h(model, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    return extract_encoder_h(model, x, adj_to_edge_index(adj), normalize=True).detach()


def train_predictor(args, dataset_key: str, method: str, seed: int):
    setup_seed(seed)
    data = _load_dataset(
        dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    data.noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.label_noise_type,
        noise_rate=args.label_noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    model_conf = load_conf(None, method, dataset_key)
    embedding_path = embedding_path_for(args.embedding_root, dataset_key, args.ssl_method)
    if not embedding_path.exists():
        raise FileNotFoundError(f"missing clean H for dataset={dataset_key}: {embedding_path}")
    _set_embedding_path(model_conf, str(embedding_path))
    _set_gb_value(model_conf, "ball_embedding_path", str(embedding_path))
    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)

    predictor_cls = predictor_class_for(method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = method
    predictor.noise_type = args.label_noise_type
    predictor.noise_rate = args.label_noise_rate
    predictor.seed = seed
    started = time.perf_counter()
    train_result = predictor.train()
    wall_time = time.perf_counter() - started
    meta = {
        "dataset": dataset_key,
        "method": method,
        "seed": int(seed),
        "label_noise_type": args.label_noise_type,
        "label_noise_rate": float(args.label_noise_rate),
        "noise_scope": args.noise_scope,
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, data.noisy_label),
        "config_id": _config_id(method, dataset_key, model_conf),
        **_noise_stats(
            data.labels,
            data.noisy_label,
            modified_mask,
            data.train_masks,
            data.val_masks,
            data.test_masks,
        ),
        "train_wall_time_seconds": float(wall_time),
        "parameter_count": _parameter_count(predictor),
        "train": float(train_result.get("train", 0.0)),
        "valid": float(train_result.get("valid", 0.0)),
        "clean_h_path": str(embedding_path),
    }
    return data, predictor, meta


def evaluate_predictor(predictor, feats: torch.Tensor, mask) -> tuple[float, float]:
    old_feats = predictor.feats
    predictor.feats = feats
    try:
        loss, acc = predictor.test(mask)
    finally:
        predictor.feats = old_feats
    return float(loss.item() if hasattr(loss, "item") else loss), float(acc)


def run_one(args, dataset_key: str, method: str, seed: int) -> list[dict[str, object]]:
    data, predictor, meta = train_predictor(args, dataset_key, method, seed)
    base_x = data.feats.float()
    clean_feats = predictor.feats
    rows = []
    clean_loss, clean_acc = evaluate_predictor(predictor, clean_feats, data.test_masks)
    rows.append({
        **meta,
        "structure_mode": "clean",
        "structure_p": 0.0,
        "structure_seed": int(seed),
        "h_recomputed": 0,
        "grace_checkpoint": "",
        "status": "ok",
        "error": "",
        "test_loss": clean_loss,
        "test": clean_acc,
        "clean_reference_test": clean_acc,
        "delta_from_clean": 0.0,
    })

    model, grace_x, ckpt_path = load_grace_encoder(
        dataset_key,
        args.device,
        args.pretrain_seed,
        args.split_seed,
        args.train_percent,
        args.val_percent,
    )
    if grace_x.shape[0] != data.n_nodes:
        raise ValueError(f"GRACE data node count {grace_x.shape[0]} != NoisyGL node count {data.n_nodes}")

    for mode in _parse_str_list(args.structure_modes):
        for p in _parse_float_list(args.structure_ps):
            perturbed = perturb_test_adj(data.adj, mode=mode, p=p, seed=seed, test_mask=data.test_masks)
            if method in {"mlp_small_loss_xh", "mlp_jocor_xh", "mlp_xh_ce"}:
                h = encode_h(model, grace_x, perturbed)
                eval_feats = concat_l2_normalized_embeddings(base_x, h).to(predictor.device)
                h_recomputed = 1
            else:
                eval_feats = clean_feats
                h_recomputed = 0
            loss, acc = evaluate_predictor(predictor, eval_feats, data.test_masks)
            rows.append({
                **meta,
                "structure_mode": mode,
                "structure_p": float(p),
                "structure_seed": int(seed),
                "h_recomputed": h_recomputed,
                "grace_checkpoint": ckpt_path,
                "status": "ok",
                "error": "",
                "test_loss": loss,
                "test": acc,
                "clean_reference_test": clean_acc,
                "delta_from_clean": float(acc - clean_acc),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="MLP-7 seed0 H-input structure robustness with perturbed-GRACE H recomputation.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--methods", default="frozengrace_gb_mlp_softdist_beta05,mlp_small_loss_xh,mlp_jocor_xh")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--label_noise_type", default="uniform")
    parser.add_argument("--label_noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--structure_modes", default="edge_drop,isolated")
    parser.add_argument("--structure_ps", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--pretrain_seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / "runs" / "mlp7_baselines_20260701" / "hinput_structure_seed0.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = _parse_str_list(args.datasets)
    methods = _parse_str_list(args.methods)
    seeds = _parse_int_list(args.seeds)
    rows = []
    for dataset in datasets:
        dataset_key = dataset.lower()
        for method in methods:
            for seed in seeds:
                print(f"HINPUT_STRUCTURE dataset={dataset_key} method={method} seed={seed}")
                new_rows = run_one(args, dataset_key, method, seed)
                for row in new_rows:
                    print(
                        "HINPUT_STRUCTURE_RESULT dataset={dataset} method={method} seed={seed} "
                        "mode={mode} p={p} h_recomputed={h_recomputed} test={test:.4f} delta={delta:.4f}".format(
                            dataset=row["dataset"],
                            method=row["method"],
                            seed=row["seed"],
                            mode=row["structure_mode"],
                            p=row["structure_p"],
                            h_recomputed=row["h_recomputed"],
                            test=float(row["test"]),
                            delta=float(row["delta_from_clean"]),
                        )
                    )
                rows.extend(new_rows)
                write_csv(Path(args.output), rows)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
