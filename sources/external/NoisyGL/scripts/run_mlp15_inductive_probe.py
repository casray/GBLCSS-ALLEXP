from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT.parents[1] / "gb_graph_aware_minival_6_1"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UPSTREAM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT / "src"))

from predictor.module.GNNs import GCN, MLP
from utils.dataloader import Dataset
from utils.frozen_grace_targets import build_knn_soft_targets_from_features, build_soft_targets_from_features
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed

from noisy_label_gb_grace.upstream_grace import build_grace_model, extract_encoder_h, load_grace_config, train_grace_epoch
from noisy_label_gb_grace.utils import set_seed as set_grace_seed


DATASET_CANONICAL = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}

METHODS = ["mlp_ce", "softdist_trainonly", "softdist_shuffled", "distill_knn_xh"]


@dataclass
class InductiveContext:
    dataset: str
    seed: int
    data: Dataset
    noisy_label: torch.Tensor
    modified_mask: torch.Tensor
    observed_idx: np.ndarray
    inductive_idx: np.ndarray
    observed_mask: torch.Tensor
    local_train_idx: np.ndarray
    local_val_idx: np.ndarray
    local_observed_idx: np.ndarray
    global_to_local: torch.Tensor
    edge_index_obs: torch.Tensor
    h_obs: torch.Tensor
    grace_epochs: int
    grace_loss_last: float
    embedding_path: Path
    target_diag: dict
    shuffled_diag: dict
    knn_diag: dict
    soft_target_full: torch.Tensor
    shuffled_target_full: torch.Tensor
    knn_target_obs: torch.Tensor


def _parse_csv(value: str, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _dataset_config(dataset: str, args) -> Dataset:
    key = dataset.lower()
    data_conf = load_conf(str(REPO_ROOT / "config" / "_dataset" / f"{key}.yaml"))
    data_conf.split["split_type"] = args.split_type
    data_conf.split["train_percent"] = args.train_percent
    data_conf.split["val_percent"] = args.val_percent
    data_conf.split["test_percent"] = args.test_percent
    for size_key in [
        "train_size",
        "val_size",
        "test_size",
        "train_examples_per_class",
        "val_examples_per_class",
        "test_examples_per_class",
    ]:
        data_conf.split[size_key] = None
    return Dataset(
        key,
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
        device=args.device,
        split_type=data_conf.split["split_type"],
    )


def _accuracy(labels: torch.Tensor, logits: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((logits.argmax(dim=1) == labels.long()).float().mean().item())


def _soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _load_upstream_model_module():
    path = UPSTREAM_ROOT / "upstream" / "GRACE" / "model.py"
    spec = importlib.util.spec_from_file_location("mlp15_grace_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import GRACE model module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _filter_observed_edges(data: Dataset, observed_mask: torch.Tensor, global_to_local: torch.Tensor) -> torch.Tensor:
    if hasattr(data, "g") and hasattr(data.g, "edge_index"):
        edge_index = data.g.edge_index.to(observed_mask.device).long()
    else:
        edge_index = data.adj.coalesce().indices().to(observed_mask.device).long()
    keep = observed_mask[edge_index[0]] & observed_mask[edge_index[1]]
    edge_index = edge_index[:, keep]
    return global_to_local[edge_index].long().contiguous()


def _train_observed_grace(
    dataset: str,
    x_obs: torch.Tensor,
    edge_index_obs: torch.Tensor,
    n_classes: int,
    seed: int,
    args,
) -> tuple[torch.Tensor, int, float]:
    canonical = DATASET_CANONICAL[dataset.lower()]
    config = load_grace_config(canonical)
    if args.grace_epochs is not None:
        config["num_epochs"] = int(args.grace_epochs)
    config["seed"] = int(seed)
    set_grace_seed(seed)
    model_proxy = SimpleNamespace(num_features=int(x_obs.shape[1]), num_classes=int(n_classes))
    model = build_grace_model(model_proxy, config, torch.device(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_value = 0.0
    for epoch in range(1, int(config["num_epochs"]) + 1):
        loss_value = train_grace_epoch(model, optimizer, x_obs, edge_index_obs, config)
        if args.log_every and epoch % int(args.log_every) == 0:
            print(f"GRACE dataset={dataset} seed={seed} epoch={epoch:04d} loss={loss_value:.4f}")
    with torch.no_grad():
        h_obs = extract_encoder_h(model, x_obs, edge_index_obs, normalize=True).detach()
    return h_obs, int(config["num_epochs"]), float(loss_value)


def _save_embedding_payload(ctx: InductiveContext) -> None:
    ctx.embedding_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "H_observed": ctx.h_obs.detach().cpu(),
            "observed_idx": torch.as_tensor(ctx.observed_idx, dtype=torch.long),
            "inductive_idx": torch.as_tensor(ctx.inductive_idx, dtype=torch.long),
            "edge_index_observed": ctx.edge_index_obs.detach().cpu().long(),
            "dataset": ctx.dataset,
            "seed": int(ctx.seed),
            "protocol": "mlp15_inductive_observed_subgraph",
            "grace_epochs": int(ctx.grace_epochs),
            "grace_loss_last": float(ctx.grace_loss_last),
        },
        ctx.embedding_path,
    )


def _build_context(dataset: str, seed: int, args) -> InductiveContext:
    setup_seed(seed)
    data = _dataset_config(dataset, args)
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks,
        debug=False,
    )
    test_idx = np.asarray(data.test_masks, dtype=np.int64)
    rng = np.random.default_rng(seed + int(args.inductive_seed_offset))
    n_inductive = max(1, int(round(test_idx.size * float(args.inductive_test_ratio))))
    inductive_idx = np.sort(rng.choice(test_idx, size=n_inductive, replace=False)).astype(np.int64)
    observed_mask_np = np.ones(int(data.n_nodes), dtype=bool)
    observed_mask_np[inductive_idx] = False
    observed_idx = np.flatnonzero(observed_mask_np).astype(np.int64)

    device = torch.device(args.device)
    observed_mask = torch.as_tensor(observed_mask_np, dtype=torch.bool, device=device)
    global_to_local = torch.full((int(data.n_nodes),), -1, dtype=torch.long, device=device)
    global_to_local[torch.as_tensor(observed_idx, dtype=torch.long, device=device)] = torch.arange(
        observed_idx.size, dtype=torch.long, device=device
    )
    edge_index_obs = _filter_observed_edges(data, observed_mask, global_to_local)
    x_obs = data.feats[torch.as_tensor(observed_idx, dtype=torch.long, device=device)].float()

    embedding_path = Path(args.out_dir) / "p1_inductive_embeddings" / dataset.lower() / f"seed{seed}" / "H_observed.pt"
    if args.reuse_embeddings and embedding_path.exists():
        payload = torch.load(embedding_path, map_location=device)
        h_obs = payload["H_observed"].to(device).float()
        grace_epochs = int(payload.get("grace_epochs", -1))
        grace_loss_last = float(payload.get("grace_loss_last", np.nan))
    else:
        h_obs, grace_epochs, grace_loss_last = _train_observed_grace(
            dataset=dataset,
            x_obs=x_obs,
            edge_index_obs=edge_index_obs,
            n_classes=int(data.n_classes),
            seed=seed,
            args=args,
        )

    train_global = np.asarray(data.train_masks, dtype=np.int64)
    val_global = np.asarray(data.val_masks, dtype=np.int64)
    local_train_idx = global_to_local[torch.as_tensor(train_global, dtype=torch.long, device=device)].detach().cpu().numpy()
    local_val_idx = global_to_local[torch.as_tensor(val_global, dtype=torch.long, device=device)].detach().cpu().numpy()
    local_observed_idx = np.arange(observed_idx.size, dtype=np.int64)
    if np.any(local_train_idx < 0) or np.any(local_val_idx < 0):
        raise RuntimeError("train/val nodes must remain in the observed subgraph")

    noisy_obs = noisy_label[torch.as_tensor(observed_idx, dtype=torch.long, device=device)]
    clean_obs = data.labels[torch.as_tensor(observed_idx, dtype=torch.long, device=device)]
    target = build_soft_targets_from_features(
        ball_features=h_obs,
        noisy_label=noisy_obs,
        clean_label=clean_obs,
        train_mask=local_train_idx,
        n_classes=int(data.n_classes),
        max_balls=int(args.max_balls),
        min_ball_size=int(args.min_ball_size),
        split_improvement=float(args.split_improvement),
        min_train_per_ball=int(args.min_train_per_ball),
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=float(args.target_beta),
        random_state=seed,
        random_control=False,
    )
    shuffled = build_soft_targets_from_features(
        ball_features=h_obs,
        noisy_label=noisy_obs,
        clean_label=clean_obs,
        train_mask=local_train_idx,
        n_classes=int(data.n_classes),
        max_balls=int(args.max_balls),
        min_ball_size=int(args.min_ball_size),
        split_improvement=float(args.split_improvement),
        min_train_per_ball=int(args.min_train_per_ball),
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=float(args.target_beta),
        random_state=seed,
        random_control=True,
    )
    knn = build_knn_soft_targets_from_features(
        ball_features=h_obs,
        noisy_label=noisy_obs,
        clean_label=clean_obs,
        train_mask=local_train_idx,
        n_classes=int(data.n_classes),
        max_balls=int(args.max_balls),
        min_ball_size=int(args.min_ball_size),
        split_improvement=float(args.split_improvement),
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.0,
        random_state=seed,
        matched_k=None,
    )

    full_onehot = F.one_hot(noisy_label.long(), num_classes=int(data.n_classes)).float()
    soft_target_full = full_onehot.clone()
    shuffled_target_full = full_onehot.clone()
    obs_tensor = torch.as_tensor(observed_idx, dtype=torch.long, device=device)
    soft_target_full[obs_tensor] = target.soft_target.to(device)
    shuffled_target_full[obs_tensor] = shuffled.soft_target.to(device)

    ctx = InductiveContext(
        dataset=dataset.lower(),
        seed=int(seed),
        data=data,
        noisy_label=noisy_label,
        modified_mask=modified_mask,
        observed_idx=observed_idx,
        inductive_idx=inductive_idx,
        observed_mask=observed_mask,
        local_train_idx=local_train_idx,
        local_val_idx=local_val_idx,
        local_observed_idx=local_observed_idx,
        global_to_local=global_to_local,
        edge_index_obs=edge_index_obs,
        h_obs=h_obs,
        grace_epochs=grace_epochs,
        grace_loss_last=grace_loss_last,
        embedding_path=embedding_path,
        target_diag=target.diagnostics,
        shuffled_diag=shuffled.diagnostics,
        knn_diag=knn.diagnostics,
        soft_target_full=soft_target_full,
        shuffled_target_full=shuffled_target_full,
        knn_target_obs=knn.soft_target.to(device),
    )
    _save_embedding_payload(ctx)
    return ctx


def _train_mlp_classifier(ctx: InductiveContext, target, method: str, args) -> dict:
    data = ctx.data
    device = torch.device(args.device)
    model = MLP(
        in_channels=int(data.dim_feats),
        hidden_channels=int(args.hidden),
        out_channels=int(data.n_classes),
        n_layers=int(args.n_layer),
        dropout=float(args.dropout),
        use_bn=True,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_idx = torch.as_tensor(data.train_masks, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(data.val_masks, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(ctx.inductive_idx, dtype=torch.long, device=device)
    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_train = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optim.zero_grad()
        logits = model(data.feats)
        if isinstance(target, torch.Tensor) and target.ndim == 2:
            loss = _soft_ce(logits[train_idx], target[train_idx])
            train_acc = _accuracy(target[train_idx].argmax(dim=1), logits[train_idx])
        else:
            loss = F.cross_entropy(logits[train_idx], target[train_idx].long())
            train_acc = _accuracy(target[train_idx], logits[train_idx])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.feats)
            val_acc = _accuracy(ctx.noisy_label[val_idx], logits[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_train = train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits = model(data.feats)
        test_acc = _accuracy(data.labels[test_idx], logits[test_idx])
    return {
        "method": method,
        "train": float(best_train),
        "valid": float(best_val),
        "test": float(test_acc),
        "best_epoch": int(best_epoch),
    }


def _train_distill_knn_xh(ctx: InductiveContext, args) -> dict:
    data = ctx.data
    device = torch.device(args.device)
    obs_tensor = torch.as_tensor(ctx.observed_idx, dtype=torch.long, device=device)
    train_local = torch.as_tensor(ctx.local_train_idx, dtype=torch.long, device=device)
    val_local = torch.as_tensor(ctx.local_val_idx, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(ctx.inductive_idx, dtype=torch.long, device=device)

    x_obs = data.feats[obs_tensor].float()
    xh_obs = torch.cat([x_obs, F.normalize(ctx.h_obs.float(), p=2, dim=1)], dim=1)
    teacher = GCN(
        in_channels=int(xh_obs.shape[1]),
        hidden_channels=int(args.teacher_hidden),
        out_channels=int(data.n_classes),
        n_layers=int(args.teacher_n_layer),
        dropout=float(args.teacher_dropout),
        norm_info=None,
        act="F.relu",
        input_layer=False,
        output_layer=False,
    ).to(device)
    optim = torch.optim.Adam(teacher.parameters(), lr=float(args.teacher_lr), weight_decay=float(args.teacher_weight_decay))
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(1, int(args.teacher_epochs) + 1):
        teacher.train()
        optim.zero_grad()
        logits = teacher(xh_obs, ctx.edge_index_obs)
        loss = _soft_ce(logits[train_local], ctx.knn_target_obs[train_local])
        loss.backward()
        optim.step()
        teacher.eval()
        with torch.no_grad():
            logits = teacher(xh_obs, ctx.edge_index_obs)
            val_acc = _accuracy(ctx.noisy_label[obs_tensor][val_local], logits[val_local])
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
    if best_state is not None:
        teacher.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    teacher.eval()
    with torch.no_grad():
        teacher_probs_obs = F.softmax(teacher(xh_obs, ctx.edge_index_obs), dim=1).detach()

    student = MLP(
        in_channels=int(data.dim_feats),
        hidden_channels=int(args.hidden),
        out_channels=int(data.n_classes),
        n_layers=int(args.n_layer),
        dropout=float(args.dropout),
        use_bn=True,
    ).to(device)
    student_optim = torch.optim.Adam(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_student_val = -1.0
    best_student_train = -1.0
    best_student_epoch = -1
    best_student_state = None
    temperature = max(1e-6, float(args.distill_temperature))
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        student_optim.zero_grad()
        logits_full = student(data.feats)
        logits_obs = logits_full[obs_tensor]
        log_prob = F.log_softmax(logits_obs / temperature, dim=1)
        loss = -(teacher_probs_obs * log_prob).sum(dim=1).mean() * (temperature ** 2)
        loss.backward()
        student_optim.step()

        student.eval()
        with torch.no_grad():
            logits_full = student(data.feats)
            val_acc = _accuracy(ctx.noisy_label[torch.as_tensor(data.val_masks, dtype=torch.long, device=device)], logits_full[torch.as_tensor(data.val_masks, dtype=torch.long, device=device)])
            train_acc = _accuracy(ctx.noisy_label[torch.as_tensor(data.train_masks, dtype=torch.long, device=device)], logits_full[torch.as_tensor(data.train_masks, dtype=torch.long, device=device)])
        if val_acc > best_student_val:
            best_student_val = val_acc
            best_student_train = train_acc
            best_student_epoch = epoch
            best_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    if best_student_state is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_student_state.items()})
    student.eval()
    with torch.no_grad():
        logits_full = student(data.feats)
        test_acc = _accuracy(data.labels[test_idx], logits_full[test_idx])
    return {
        "method": "distill_knn_xh",
        "train": float(best_student_train),
        "valid": float(best_student_val),
        "test": float(test_acc),
        "best_epoch": int(best_student_epoch),
        "teacher_best_epoch": int(best_epoch),
        "teacher_best_val": float(best_val),
        "teacher_best_test": float(best_test),
    }


def _rows_for_context(ctx: InductiveContext, methods: list[str], args) -> list[dict]:
    rows = []
    started = time.perf_counter()
    for method in methods:
        method_started = time.perf_counter()
        if method == "mlp_ce":
            result = _train_mlp_classifier(ctx, ctx.noisy_label, "mlp_ce", args)
        elif method == "softdist_trainonly":
            result = _train_mlp_classifier(ctx, ctx.soft_target_full, "softdist_trainonly", args)
        elif method == "softdist_shuffled":
            result = _train_mlp_classifier(ctx, ctx.shuffled_target_full, "softdist_shuffled", args)
        elif method == "distill_knn_xh":
            result = _train_distill_knn_xh(ctx, args)
        else:
            raise ValueError(f"unknown P1 method: {method}")
        diag = ctx.target_diag if method != "softdist_shuffled" else ctx.shuffled_diag
        row = {
            "dataset": ctx.dataset,
            "method": result["method"],
            "seed": int(ctx.seed),
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "split_type": args.split_type,
            "train_percent": float(args.train_percent),
            "val_percent": float(args.val_percent),
            "test_percent": float(args.test_percent),
            "inductive_test_ratio": float(args.inductive_test_ratio),
            "observed_count": int(ctx.observed_idx.size),
            "inductive_count": int(ctx.inductive_idx.size),
            "observed_edge_count": int(ctx.edge_index_obs.shape[1]),
            "grace_epochs": int(ctx.grace_epochs),
            "grace_loss_last": float(ctx.grace_loss_last),
            "embedding_path": str(ctx.embedding_path),
            "train": result["train"],
            "valid": result["valid"],
            "test": result["test"],
            "best_epoch": result["best_epoch"],
            "status": "ok",
            "wall_time": float(time.perf_counter() - method_started),
            "total_context_time": float(time.perf_counter() - started),
            "num_balls": int(diag.get("num_balls", 0)),
            "mean_ball_size": float(diag.get("mean_ball_size", 0.0)),
            "mean_train_per_ball": float(diag.get("mean_train_per_ball", 0.0)),
            "true_train_purity_weighted": float(diag.get("true_train_purity_weighted", np.nan)),
            "true_all_purity_weighted": float(diag.get("true_all_purity_weighted", np.nan)),
            "noisy_train_purity_weighted": float(diag.get("noisy_train_purity_weighted", np.nan)),
            "alpha_positive_train_ratio": float(diag.get("alpha_positive_train_ratio", np.nan)),
            "mean_alpha_train": float(diag.get("mean_alpha_train", np.nan)),
            "knn_matched_k": int(ctx.knn_diag.get("matched_k", 0)),
            "knn_true_train_purity_weighted": float(ctx.knn_diag.get("true_train_purity_weighted", np.nan)),
            "teacher_best_epoch": int(result.get("teacher_best_epoch", -1)),
            "teacher_best_val": float(result.get("teacher_best_val", np.nan)),
            "teacher_best_test": float(result.get("teacher_best_test", np.nan)),
        }
        rows.append(row)
        print(
            "P1_RESULT "
            f"dataset={ctx.dataset} method={result['method']} seed={ctx.seed} "
            f"test={result['test']:.4f} val={result['valid']:.4f}"
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    agg = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("seed", "nunique"),
            mean=("test", "mean"),
            std=("test", "std"),
            valid_mean=("valid", "mean"),
            purity=("true_train_purity_weighted", "mean"),
            num_balls=("num_balls", "mean"),
        )
        .sort_values(["dataset", "method"])
    )
    agg["mean_std"] = agg.apply(lambda r: f"{r['mean'] * 100:.2f}±{r['std'] * 100:.2f}", axis=1)
    return agg


def _paired(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("softdist_trainonly", "mlp_ce"),
        ("softdist_trainonly", "softdist_shuffled"),
        ("distill_knn_xh", "mlp_ce"),
        ("distill_knn_xh", "softdist_trainonly"),
    ]
    for dataset, grp in raw.groupby("dataset"):
        for ours, baseline in comparisons:
            a = grp[grp["method"].eq(ours)]
            b = grp[grp["method"].eq(baseline)]
            if a.empty or b.empty:
                continue
            merged = a[["seed", "test"]].merge(b[["seed", "test"]], on="seed", suffixes=("_ours", "_baseline"))
            if merged.empty:
                continue
            delta = merged["test_ours"] - merged["test_baseline"]
            rows.append(
                {
                    "dataset": dataset,
                    "ours": ours,
                    "baseline": baseline,
                    "n": int(len(delta)),
                    "mean_delta_pp": float(delta.mean() * 100),
                    "positive_seeds": int((delta > 0).sum()),
                    "deltas_pp": " ".join(f"{v * 100:.2f}" for v in delta.tolist()),
                }
            )
    return pd.DataFrame(rows)


def _load_transductive_refs(args) -> pd.DataFrame:
    frames = []
    mapping = {
        "mlp_ce": ("mlp_ce", args.transductive_mlp10),
        "softdist_trainonly": ("frozengrace_gb_mlp_softdist_beta05", args.transductive_mlp10),
        "softdist_shuffled": ("frozengrace_gb_mlp_softdist_beta05_shuffled", args.transductive_mlp10),
        "distill_knn_xh": ("mlp_distill_knn_xh", args.transductive_mlp9),
    }
    for p in {Path(v[1]) for v in mapping.values() if v[1]}:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    if "status" in raw.columns:
        raw = raw[raw["status"].fillna("ok").eq("ok")].copy()
    raw["noise_rate"] = pd.to_numeric(raw["noise_rate"], errors="coerce")
    raw["test"] = pd.to_numeric(raw["test"], errors="coerce")
    rows = []
    for ours, (source_method, _) in mapping.items():
        sub = raw[
            raw["dataset"].astype(str).str.lower().isin(["cora", "citeseer", "pubmed"])
            & raw["method"].eq(source_method)
            & raw["noise_type"].eq(args.noise_type)
            & raw["noise_rate"].eq(float(args.noise_rate))
        ].copy()
        if "train_percent" in sub.columns:
            tp = pd.to_numeric(sub["train_percent"], errors="coerce")
            sub = sub[tp.isna() | np.isclose(tp, float(args.train_percent))]
        for dataset, grp in sub.groupby(sub["dataset"].astype(str).str.lower()):
            rows.append(
                {
                    "dataset": dataset,
                    "method": ours,
                    "transductive_mean": float(grp["test"].mean()),
                    "transductive_std": float(grp["test"].std(ddof=1)),
                    "transductive_n": int(grp["seed"].nunique()) if "seed" in grp.columns else int(len(grp)),
                }
            )
    return pd.DataFrame(rows)


def _write_summary(out_dir: Path, raw_path: Path, args) -> None:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return
    raw = pd.read_csv(raw_path)
    if "test" not in raw.columns:
        raw["test"] = np.nan
    if "valid" not in raw.columns:
        raw["valid"] = np.nan
    if "seed" not in raw.columns:
        raw["seed"] = np.nan
    raw["test"] = pd.to_numeric(raw["test"], errors="coerce")
    raw["valid"] = pd.to_numeric(raw["valid"], errors="coerce")
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce")
    result_raw = raw[raw["method"].ne("diagnostic")].copy()
    agg = _aggregate(result_raw)
    paired = _paired(result_raw)
    refs = _load_transductive_refs(args)
    if not refs.empty and not agg.empty:
        drop = agg.merge(refs, on=["dataset", "method"], how="left")
        drop["inductive_minus_transductive_pp"] = (drop["mean"] - drop["transductive_mean"]) * 100
    else:
        drop = pd.DataFrame()

    diag = (
        raw.groupby(["dataset", "seed"], as_index=False)
        .agg(
            true_train_purity_weighted=("true_train_purity_weighted", "first"),
            true_all_purity_weighted=("true_all_purity_weighted", "first"),
            noisy_train_purity_weighted=("noisy_train_purity_weighted", "first"),
            num_balls=("num_balls", "first"),
            observed_count=("observed_count", "first"),
            inductive_count=("inductive_count", "first"),
            grace_epochs=("grace_epochs", "first"),
            grace_loss_last=("grace_loss_last", "first"),
        )
        .sort_values(["dataset", "seed"])
    )
    agg.to_csv(out_dir / "p1_inductive_aggregate.csv", index=False)
    paired.to_csv(out_dir / "p1_inductive_paired_deltas.csv", index=False)
    drop.to_csv(out_dir / "p1_inductive_drop_vs_transductive.csv", index=False)
    diag.to_csv(out_dir / "p1_inductive_diagnostics.csv", index=False)

    lines = ["# MLP-15 P1 Inductive Probe\n"]
    lines.append(
        f"Protocol: observed-subgraph GRACE retraining, inductive_test_ratio={args.inductive_test_ratio}, "
        f"noise={args.noise_type}{args.noise_rate}, train split={args.train_percent}.\n"
    )
    if not agg.empty:
        pivot = agg.pivot(index="dataset", columns="method", values="mean_std").reset_index()
        cols = [c for c in ["dataset", "mlp_ce", "softdist_shuffled", "softdist_trainonly", "distill_knn_xh"] if c in pivot.columns]
        lines.append("## Inductive Accuracy\n")
        lines.append(pivot[cols].to_markdown(index=False))
        lines.append("\n")
    if not paired.empty:
        lines.append("## Paired Deltas\n")
        show = paired.copy()
        show["mean_delta_pp"] = show["mean_delta_pp"].map(lambda x: f"{x:+.2f}")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")
    if not drop.empty:
        lines.append("## Drop vs Transductive Reference\n")
        show = drop[["dataset", "method", "mean_std", "transductive_mean", "transductive_n", "inductive_minus_transductive_pp"]].copy()
        show["transductive_mean"] = show["transductive_mean"].map(lambda x: f"{x * 100:.2f}" if pd.notna(x) else "")
        show["inductive_minus_transductive_pp"] = show["inductive_minus_transductive_pp"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")
    if not diag.empty:
        lines.append("## GRACE Ball Diagnostics\n")
        show = diag.copy()
        for col in ["true_train_purity_weighted", "true_all_purity_weighted", "noisy_train_purity_weighted"]:
            show[col] = show[col].map(lambda x: f"{x:.3f}")
        lines.append(show.to_markdown(index=False))
        lines.append("\n")
    (out_dir / "p1_inductive_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / ("p1_inductive_seed0_diagnostic_raw.csv" if args.diagnostic_only else "p1_inductive_uniform0p4_seeds_0_1_2_3_4.csv")
    methods = [] if args.diagnostic_only else _parse_csv(args.methods, str)
    rows = []
    for dataset in _parse_csv(args.datasets, str):
        for seed in _parse_csv(args.seeds, int):
            print(f"P1_CONTEXT dataset={dataset} seed={seed}")
            ctx = _build_context(dataset.lower(), seed, args)
            diag_row = {
                "dataset": ctx.dataset,
                "method": "diagnostic",
                "seed": int(ctx.seed),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "split_type": args.split_type,
                "train_percent": float(args.train_percent),
                "val_percent": float(args.val_percent),
                "test_percent": float(args.test_percent),
                "inductive_test_ratio": float(args.inductive_test_ratio),
                "observed_count": int(ctx.observed_idx.size),
                "inductive_count": int(ctx.inductive_idx.size),
                "observed_edge_count": int(ctx.edge_index_obs.shape[1]),
                "grace_epochs": int(ctx.grace_epochs),
                "grace_loss_last": float(ctx.grace_loss_last),
                "embedding_path": str(ctx.embedding_path),
                "status": "diagnostic",
                "num_balls": int(ctx.target_diag.get("num_balls", 0)),
                "mean_ball_size": float(ctx.target_diag.get("mean_ball_size", 0.0)),
                "mean_train_per_ball": float(ctx.target_diag.get("mean_train_per_ball", 0.0)),
                "true_train_purity_weighted": float(ctx.target_diag.get("true_train_purity_weighted", np.nan)),
                "true_all_purity_weighted": float(ctx.target_diag.get("true_all_purity_weighted", np.nan)),
                "noisy_train_purity_weighted": float(ctx.target_diag.get("noisy_train_purity_weighted", np.nan)),
                "knn_matched_k": int(ctx.knn_diag.get("matched_k", 0)),
                "knn_true_train_purity_weighted": float(ctx.knn_diag.get("true_train_purity_weighted", np.nan)),
            }
            rows.append(diag_row)
            print(
                "P1_DIAG "
                f"dataset={ctx.dataset} seed={ctx.seed} purity={diag_row['true_train_purity_weighted']:.4f} "
                f"balls={diag_row['num_balls']}"
            )
            if not args.diagnostic_only:
                rows.extend(_rows_for_context(ctx, methods, args))
            _write_csv(raw_path, rows)
    _write_csv(raw_path, rows)
    _write_summary(out_dir, raw_path, args)
    print(f"WROTE {raw_path}")
    return raw_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="cora,citeseer,pubmed")
    parser.add_argument("--methods", default="mlp_ce,softdist_trainonly,softdist_shuffled,distill_knn_xh")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--inductive_test_ratio", type=float, default=0.2)
    parser.add_argument("--inductive_seed_offset", type=int, default=7919)
    parser.add_argument("--out_dir", default="runs/mlp15_review_defense_20260704")
    parser.add_argument("--diagnostic_only", action="store_true")
    parser.add_argument("--reuse_embeddings", action="store_true")
    parser.add_argument("--grace_epochs", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--max_balls", type=int, default=64)
    parser.add_argument("--min_ball_size", type=int, default=32)
    parser.add_argument("--split_improvement", type=float, default=0.05)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--target_beta", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--teacher_hidden", type=int, default=64)
    parser.add_argument("--teacher_n_layer", type=int, default=2)
    parser.add_argument("--teacher_dropout", type=float, default=0.5)
    parser.add_argument("--teacher_lr", type=float, default=1e-2)
    parser.add_argument("--teacher_weight_decay", type=float, default=5e-4)
    parser.add_argument("--teacher_epochs", type=int, default=200)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--transductive_mlp10", default="runs/mlp10_baseline_refresh_20260702/mlp10_combined_raw.csv")
    parser.add_argument("--transductive_mlp9", default="runs/mlp9_denoised_teacher_20260702/mlp9_combined_raw.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
