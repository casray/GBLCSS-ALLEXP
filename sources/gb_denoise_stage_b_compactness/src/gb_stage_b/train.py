from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from gb_stage_b.augment import make_two_views
from gb_stage_b.balls import (
    generate_angular_balls,
    generate_gain_split_balls,
    make_ball_batch,
    reliable_ball_mask,
    shuffled_assignments,
    size_matched_random_assignments,
)
from gb_stage_b.compactness import compactness_loss
from gb_stage_b.config import StageBConfig
from gb_stage_b.data import load_graph
from gb_stage_b.diagnostics import ball_diagnostics, representation_diagnostics
from gb_stage_b.eval import linear_eval
from gb_stage_b.losses import (
    infonce_loss,
    negative_free_loss,
    variance_loss,
    symmetric_cross_entropy_loss,
    ball_majority_vote,
    ball_label_smoothing,
    inter_ball_separation_loss,
    knn_majority_vote,
    knn_label_smoothing,
)
from gb_stage_b.models import StageBModel, LinearClassifier
from gb_stage_b.utils import ensure_dir, read_json, set_seed, write_json


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be auto, cuda, or cpu")


def _ssl_loss(cfg: StageBConfig, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    if cfg.ssl_objective == "infonce":
        return infonce_loss(z1, z2)
    loss = negative_free_loss(p1, z2, p2, z1)
    if cfg.variant == "negfree_var":
        loss = loss + cfg.variance_loss_weight * 0.5 * (variance_loss(z1) + variance_loss(z2))
    return loss


def _supervised_loss(cfg: StageBConfig, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if cfg.phase2_supervised_loss == "sce":
        return symmetric_cross_entropy_loss(
            logits,
            labels,
            alpha=cfg.phase2_sce_alpha,
            beta=cfg.phase2_sce_beta,
        )
    return F.cross_entropy(logits, labels.long())


def _view_rate(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _ball_generator(cfg: StageBConfig, embeddings: torch.Tensor):
    if cfg.ball_algorithm == "gain_split":
        return generate_gain_split_balls(
            embeddings,
            min_ball_size=cfg.gain_split_min_ball_size,
            max_depth=cfg.gain_split_max_depth,
            min_split_gain=cfg.gain_split_min_gain,
        )
    return generate_angular_balls(
        embeddings,
        min_size=cfg.reliable_ball_min_size,
        max_size=cfg.reliable_ball_max_size,
        quality_threshold=0.35,
    )


def _controlled_assignment(cfg: StageBConfig, assignment: torch.Tensor) -> torch.Tensor:
    if cfg.method == "B3" or cfg.assignment_control == "shuffled":
        return shuffled_assignments(assignment, seed=cfg.seed)
    if cfg.method == "B4" or cfg.assignment_control == "size_matched_random":
        return size_matched_random_assignments(assignment, seed=cfg.seed)
    return assignment


def _build_balls(cfg: StageBConfig, embeddings: torch.Tensor):
    balls = _ball_generator(cfg, embeddings.detach())
    assignment = _controlled_assignment(cfg, balls.assignment)
    if not torch.equal(assignment, balls.assignment):
        balls = make_ball_batch(embeddings.detach(), assignment)
    reliable = reliable_ball_mask(
        balls,
        min_size=cfg.reliable_ball_min_size,
        max_size=cfg.reliable_ball_max_size,
        ad_percentile=cfg.reliable_ball_ad_percentile,
    )
    return balls, reliable


def _update_ema(target: StageBModel, online: StageBModel, decay: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.mul_(decay).add_(online_param.detach(), alpha=1.0 - decay)
        for target_buffer, online_buffer in zip(target.buffers(), online.buffers()):
            target_buffer.copy_(online_buffer)


def _phase2_diagnostic(
    model: StageBModel,
    data,
    cfg: StageBConfig,
    balls,
    reliable,
) -> dict[str, float]:
    """Compute ball quality diagnostics using y_true (never enters loss)."""
    diag: dict[str, float] = {}
    y_true = data.y_true if hasattr(data, "y_true") else data.y
    noise_mask = data.noise_mask if hasattr(data, "noise_mask") else None

    diag.update(ball_diagnostics(balls, labels=y_true.cpu()))

    if noise_mask is not None and reliable is not None:
        reliable_bool = reliable.to(dtype=torch.bool, device=balls.assignment.device)
        node_reliable = reliable_bool[balls.assignment]
        if node_reliable.any():
            reliable_nodes = node_reliable.cpu()
            is_noisy = noise_mask.cpu()
            reliable_clean_rate = float((~is_noisy[reliable_nodes]).float().mean().item())
            diag["reliable_ball_clean_rate"] = reliable_clean_rate
    return diag


def run_phase2_finetune(
    cfg: StageBConfig,
    model: StageBModel,
    data,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    """Phase 2 fine-tuning with optional label correction and GB structure regularization.

    Supports control variants:
    - C1: real balls + majority vote + intra/inter (main method)
    - C1_noball: no balls, plain noisy-CE fine-tune
    - C1_topk: kNN consensus majority vote (no ball structure)
    - C1_shuffled: shuffled ball assignments
    - C2_*: no label correction; CE/SCE supervised loss + optional real/shuffled GB-reg
    """
    y_noisy = data.y if not hasattr(data, "y_noisy") else data.y_noisy
    y_true = data.y_true if hasattr(data, "y_true") else data.y
    num_classes = int(y_true.unique().numel())
    train_mask = data.train_mask
    use_label_correction = cfg.method.startswith("C1") and cfg.phase2_use_label_correction
    use_balls = cfg.method not in {"C1_noball", "C1_topk", "C2_ce_noball", "C2_sce_noball"}
    use_knn = cfg.method == "C1_topk"
    use_shuffled = cfg.method in {"C1_shuffled", "C2_sce_shuffled"}

    classifier = LinearClassifier(cfg.hidden_dim, num_classes).to(device)
    params = list(model.encoder.parameters()) + list(classifier.parameters())
    opt = Adam(params, lr=cfg.phase2_lr, weight_decay=cfg.weight_decay)

    model.train()
    balls = None
    reliable = None
    history: list[dict] = []
    frozen_purity: float | None = None
    matched_k: int = cfg.knn_k

    if use_knn:
        model.eval()
        with torch.no_grad():
            h_ref, _, _ = model(data.x, data.edge_index)
            h_ref_norm = F.normalize(h_ref, dim=-1, eps=1e-8)
            ref_balls, ref_reliable = _build_balls(cfg, h_ref_norm)
            ref_assign = ref_balls.assignment
            train_bool = train_mask.to(dtype=torch.bool, device=ref_assign.device)
            ball_train_sizes = []
            for b in range(int(ref_assign.max().item()) + 1):
                if ref_reliable is not None and not ref_reliable[b]:
                    continue
                members = (ref_assign == b).nonzero(as_tuple=True)[0]
                n_train = int(train_bool[members].sum().item())
                if n_train > 1:
                    ball_train_sizes.append(n_train)
            if ball_train_sizes:
                matched_k = max(2, int(sum(ball_train_sizes) / len(ball_train_sizes)) - 1)
        model.train()

    for epoch in range(cfg.phase2_epochs):
        if use_balls:
            should_refresh = balls is None or (epoch % cfg.phase2_ball_refresh == 0)
            if should_refresh:
                model.eval()
                with torch.no_grad():
                    h_full, _, _ = model(data.x, data.edge_index)
                    h_norm = F.normalize(h_full, dim=-1, eps=1e-8)
                    balls, reliable = _build_balls(cfg, h_norm)
                    if use_shuffled:
                        shuf_assign = shuffled_assignments(balls.assignment, seed=cfg.seed + epoch)
                        balls = make_ball_batch(h_norm, shuf_assign)
                        reliable = reliable_ball_mask(
                            balls,
                            min_size=cfg.reliable_ball_min_size,
                            max_size=cfg.reliable_ball_max_size,
                            ad_percentile=cfg.reliable_ball_ad_percentile,
                        )
                if frozen_purity is None:
                    diag0 = _phase2_diagnostic(model, data, cfg, balls, reliable)
                    frozen_purity = diag0.get("label_purity_diagnostic", 0.0)
                model.train()

        h, _, _ = model(data.x, data.edge_index)

        if use_label_correction and use_balls and balls is not None:
            corrected_labels = ball_majority_vote(
                balls.assignment.to(device), y_noisy.to(device), reliable.to(device), num_classes,
                train_mask=train_mask,
            )
            soft_targets = ball_label_smoothing(
                balls.assignment.to(device), corrected_labels, reliable.to(device),
                num_classes, smoothing=cfg.phase2_label_smoothing,
                train_mask=train_mask,
            )
            logits = classifier(h)
            log_probs = F.log_softmax(logits, dim=-1)
            loss_sce = -(soft_targets[train_mask] * log_probs[train_mask]).sum(dim=-1).mean()
        elif use_label_correction and use_knn:
            h_detached = h.detach()
            corrected_labels = knn_majority_vote(
                h_detached, y_noisy.to(device), num_classes, k=matched_k, train_mask=train_mask,
            )
            soft_targets = knn_label_smoothing(
                h_detached, corrected_labels, num_classes, k=matched_k,
                smoothing=cfg.phase2_label_smoothing, train_mask=train_mask,
            )
            logits = classifier(h)
            log_probs = F.log_softmax(logits, dim=-1)
            loss_sce = -(soft_targets[train_mask] * log_probs[train_mask]).sum(dim=-1).mean()
        else:
            corrected_labels = y_noisy.to(device)
            logits = classifier(h)
            loss_sce = _supervised_loss(cfg, logits[train_mask], y_noisy.to(device)[train_mask])

        loss_intra_val = 0.0
        loss_inter_val = 0.0
        intra_stats = {"covered_nodes": 0.0, "covered_ratio": 0.0, "mean_distance": 0.0}

        if use_balls and balls is not None and reliable is not None:
            loss_intra, intra_stats = compactness_loss(
                h, balls.assignment.to(device), balls.centers.to(device), reliable.to(device),
            )
            assign_dev = balls.assignment.to(device)
            num_balls = int(assign_dev.max().item()) + 1
            diff_centers = torch.zeros(num_balls, h.size(1), device=device)
            counts = torch.zeros(num_balls, 1, device=device)
            diff_centers.scatter_add_(0, assign_dev.unsqueeze(1).expand_as(h), h)
            counts.scatter_add_(0, assign_dev.unsqueeze(1), torch.ones(h.size(0), 1, device=device))
            counts = counts.clamp(min=1)
            diff_centers = diff_centers / counts
            loss_inter = inter_ball_separation_loss(
                diff_centers, reliable.to(device),
            )
            loss = (
                cfg.lambda_sce * loss_sce
                + cfg.lambda_intra * loss_intra
                + cfg.lambda_inter * loss_inter
            )
            loss_intra_val = float(loss_intra.item())
            loss_inter_val = float(loss_inter.item())
        else:
            loss = cfg.lambda_sce * loss_sce

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits[data.val_mask].argmax(dim=-1)
            val_true = y_true[data.val_mask] if hasattr(data, "y_true") else data.y[data.val_mask]
            val_acc = float((pred == val_true).float().mean().item())

            correction_acc = 0.0
            if hasattr(data, "noise_mask") and data.noise_mask.any():
                noisy_idx = data.noise_mask & train_mask
                if noisy_idx.any():
                    correction_acc = float((corrected_labels[noisy_idx] == y_true[noisy_idx]).float().mean().item())

        row = {
            "phase2_epoch": epoch,
            "loss_sce": float(loss_sce.item()),
            "loss_intra": loss_intra_val,
            "loss_inter": loss_inter_val,
            "loss_total": float(loss.item()),
            "val_acc": val_acc,
            "correction_acc": correction_acc,
            **intra_stats,
        }

        if use_balls and balls is not None and (epoch % cfg.phase2_ball_refresh == 0 or epoch == 0):
            diag = _phase2_diagnostic(model, data, cfg, balls, reliable)
            row.update({f"diag_{k}": v for k, v in diag.items()})

        history.append(row)

        if epoch == 49:
            val_improving = val_acc > 0.2
            if not val_improving:
                row["hard_stop"] = "G1_val_acc_not_improving"
                break

    model.eval()
    with torch.no_grad():
        h_final, _, _ = model(data.x, data.edge_index)
        h_final_norm = F.normalize(h_final, dim=-1, eps=1e-8)

    metrics: dict[str, float] = {}
    final_balls = _ball_generator(cfg, h_final_norm.detach())
    metrics.update(ball_diagnostics(final_balls, labels=y_true.cpu()))
    metrics.update(representation_diagnostics(h_final_norm))
    metrics["linear_eval_acc"] = linear_eval(
        h_final_norm, y_true, data.train_mask, data.val_mask, data.test_mask, seed=cfg.seed,
    )

    logits_final = classifier(h_final)
    pred_test = logits_final[data.test_mask].argmax(dim=-1)
    metrics["classifier_test_acc"] = float(
        (pred_test == y_true[data.test_mask]).float().mean().item()
    )
    metrics["frozen_purity"] = frozen_purity if frozen_purity is not None else 0.0
    metrics["phase2_epochs_ran"] = float(len(history))

    return metrics, history


def run_experiment(cfg: StageBConfig) -> Path:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    data = load_graph(cfg.dataset, seed=cfg.seed, noise_rate=cfg.noise_rate, config=cfg).to(device)
    model = StageBModel(data.x.size(1), cfg.hidden_dim, cfg.proj_dim, variant=cfg.variant).to(device)
    target_model = None
    if cfg.variant == "negfree_ema":
        target_model = copy.deepcopy(model).to(device)
        target_model.eval()
        for param in target_model.parameters():
            param.requires_grad_(False)
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    balls = None
    reliable = None
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        x1, e1, x2, e2 = make_two_views(
            data.x,
            data.edge_index,
            edge_drop_1=_view_rate(cfg.edge_drop_1, cfg.edge_drop),
            edge_drop_2=_view_rate(cfg.edge_drop_2, cfg.edge_drop),
            feat_mask_1=_view_rate(cfg.feat_mask_1, cfg.feat_mask),
            feat_mask_2=_view_rate(cfg.feat_mask_2, cfg.feat_mask),
            seed=cfg.seed + epoch * 2,
        )
        h1, z1, p1 = model(x1, e1)
        _, z2_online, p2 = model(x2, e2)
        if target_model is not None and cfg.ssl_objective == "negfree":
            target_model.eval()
            with torch.no_grad():
                _, z1_target, _ = target_model(x1, e1)
                _, z2_target, _ = target_model(x2, e2)
            loss_ssl = negative_free_loss(p1, z2_target, p2, z1_target)
        else:
            loss_ssl = _ssl_loss(cfg, p1, z1, p2, z2_online)
        loss_gb = loss_ssl.new_tensor(0.0)
        gb_stats = {"covered_nodes": 0.0, "covered_ratio": 0.0, "mean_distance": 0.0}

        should_use_gb = cfg.method in {"B1", "B2", "B3", "B4"} and epoch >= cfg.warmup_epochs
        should_refresh = balls is None or (
            cfg.method == "B2" and cfg.ball_refresh_interval > 0 and epoch % cfg.ball_refresh_interval == 0
        )
        if should_use_gb and should_refresh:
            with torch.no_grad():
                h_full, z_full, _ = model(data.x, data.edge_index)
                space = h_full if cfg.ball_space == "H" else z_full
                balls, reliable = _build_balls(cfg, F.normalize(space, dim=-1, eps=1e-8))
        if should_use_gb and balls is not None and reliable is not None:
            current = h1 if cfg.ball_space == "H" else z1
            loss_gb, gb_stats = compactness_loss(current, balls.assignment, balls.centers, reliable)

        loss = loss_ssl + cfg.lambda_gb * loss_gb
        opt.zero_grad()
        loss.backward()
        opt.step()
        if target_model is not None:
            _update_ema(target_model, model, cfg.ema_decay)
        history.append(
            {
                "epoch": epoch,
                "loss_ssl": float(loss_ssl.detach().item()),
                "loss_compact": float(loss_gb.detach().item()),
                "loss_compact_weighted": float((cfg.lambda_gb * loss_gb).detach().item()),
                **gb_stats,
            }
        )

    model.eval()

    if cfg.method.startswith(("C1", "C2")):
        phase2_metrics, phase2_history = run_phase2_finetune(cfg, model, data, device)
        with torch.no_grad():
            h, _, _ = model(data.x, data.edge_index)
            final_space = F.normalize(h, dim=-1, eps=1e-8)
        metrics = {}
        metrics.update(phase2_metrics)
        phase1_compact = [float(row["loss_compact"]) for row in history]
        phase1_covered = [float(row["covered_nodes"]) for row in history]
        metrics["phase1_compact_nonzero_epochs"] = float(sum(1 for v in phase1_compact if abs(v) > 1e-12))
        metrics["phase1_covered_nodes_max"] = float(max(phase1_covered) if phase1_covered else 0.0)
        metrics["dataset"] = cfg.dataset
        metrics["method"] = cfg.method
        metrics["seed"] = cfg.seed
        metrics["ssl_objective"] = cfg.ssl_objective
        metrics["variant"] = cfg.variant
        metrics["experiment_group"] = cfg.experiment_group
        metrics["ball_algorithm"] = cfg.ball_algorithm
        metrics["device"] = str(device)
        metrics["noise_rate"] = cfg.noise_rate
        metrics["phase2_supervised_loss"] = cfg.phase2_supervised_loss
        metrics["phase2_use_label_correction"] = float(cfg.phase2_use_label_correction)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = ensure_dir(Path(cfg.results_dir) / cfg.dataset / cfg.method / f"seed_{cfg.seed}" / run_id)
        write_json(out_dir / "config.json", asdict(cfg))
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "phase1_history.json", {"history": history})
        write_json(out_dir / "phase2_history.json", {"history": phase2_history})
        torch.save(final_space.detach().cpu(), out_dir / "embedding.pt")
        return out_dir

    with torch.no_grad():
        h, z, _ = model(data.x, data.edge_index)
        final_space = F.normalize(h if cfg.ball_space == "H" else z, dim=-1, eps=1e-8)
        final_balls = _ball_generator(cfg, final_space)

    metrics = {}
    metrics.update(ball_diagnostics(final_balls, labels=data.y))
    metrics.update(representation_diagnostics(final_space))
    metrics["linear_eval_acc"] = linear_eval(final_space, data.y, data.train_mask, data.val_mask, data.test_mask, seed=cfg.seed)
    compact_values = [float(row["loss_compact"]) for row in history]
    covered_values = [float(row["covered_nodes"]) for row in history]
    metrics["compact_nonzero_epochs"] = float(sum(1 for value in compact_values if abs(value) > 1e-12))
    metrics["covered_nodes_max"] = float(max(covered_values) if covered_values else 0.0)
    metrics["covered_nodes_last"] = float(covered_values[-1] if covered_values else 0.0)
    metrics["dataset"] = cfg.dataset
    metrics["method"] = cfg.method
    metrics["seed"] = cfg.seed
    metrics["ssl_objective"] = cfg.ssl_objective
    metrics["variant"] = cfg.variant
    metrics["experiment_group"] = cfg.experiment_group
    metrics["ball_algorithm"] = cfg.ball_algorithm
    metrics["device"] = str(device)
    metrics["gain_split_min_ball_size"] = float(cfg.gain_split_min_ball_size)
    metrics["gain_split_max_depth"] = float(cfg.gain_split_max_depth)
    metrics["gain_split_min_gain"] = float(cfg.gain_split_min_gain)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = ensure_dir(Path(cfg.results_dir) / cfg.dataset / cfg.method / f"seed_{cfg.seed}" / run_id)
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", {"history": history})
    torch.save(final_space.detach().cpu(), out_dir / "embedding.pt")
    return out_dir


def rebuild_balls_from_run(run_dir: str | Path, min_ball_size: int, max_depth: int, min_split_gain: float) -> Path:
    run_dir = Path(run_dir)
    cfg = StageBConfig(**read_json(run_dir / "config.json"))
    embedding = torch.load(run_dir / "embedding.pt", map_location="cpu")
    data = load_graph(cfg.dataset, seed=cfg.seed)
    balls = generate_gain_split_balls(
        embedding,
        min_ball_size=min_ball_size,
        max_depth=max_depth,
        min_split_gain=min_split_gain,
    )
    metrics = {}
    metrics.update(ball_diagnostics(balls, labels=data.y))
    metrics.update(representation_diagnostics(embedding))
    metrics["linear_eval_acc"] = linear_eval(embedding, data.y, data.train_mask, data.val_mask, data.test_mask, seed=cfg.seed)
    metrics["dataset"] = cfg.dataset
    metrics["method"] = cfg.method
    metrics["seed"] = cfg.seed
    metrics["ssl_objective"] = cfg.ssl_objective
    metrics["variant"] = cfg.variant
    metrics["experiment_group"] = cfg.experiment_group
    metrics["ball_algorithm"] = "gain_split"
    metrics["gain_split_min_ball_size"] = float(min_ball_size)
    metrics["gain_split_max_depth"] = float(max_depth)
    metrics["gain_split_min_gain"] = float(min_split_gain)
    token = str(min_split_gain).replace(".", "p")
    out = run_dir / "ball_ablation" / f"gain_split_mbs{min_ball_size}_depth{max_depth}_gain{token}.json"
    write_json(out, metrics)
    return out
