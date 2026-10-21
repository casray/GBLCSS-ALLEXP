"""CUDA-only, sparse-edge port of the author's GSSC training core."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions.utils import clamp_probs

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "gssc"
OFFICIAL_COMMIT = "197a7f55e67e8d683dcd88149b788b76c39f058f"
METHOD = "gssc_official_core"


def resolve_config(name: str, config: dict) -> dict:
    key = name.lower().replace("-", "").replace("_", "")
    mapping = {"cora": "cora", "citeseer": "citeseer", "corafull": "cora",
               "coauthorcs": "coauthor-cs", "photo": "cora", "amazonpho": "cora",
               "amazonphoto": "cora"}
    if key not in mapping:
        raise ValueError(f"Unsupported dataset name: {name}")
    source = mapping[key]
    param = json.loads((VENDOR / "param" / "best_parameters.json").read_text(encoding="utf-8"))[source]
    requested = dict(config)
    if requested.get("method", METHOD).lower().replace("-", "_") not in (METHOD, "gssc", "official_core"):
        raise ValueError("method must be GSSC official-core, not a generic MLP")
    fixed = {"seed": 0, "hidden": 64, "hidden_dim": 64, "hid_dim": 64,
             "dropout": .5, "lr": .01, "learning_rate": .01, "weight_decay": .0005,
             "n_layers": 2, "mixup_lr": .0001}
    for field, value in fixed.items():
        if field in requested and requested[field] != value:
            raise ValueError(f"Main protocol fixes {field}={value}")
    epochs = requested.get("epochs", 200)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    param.update(requested)
    param.update(method=METHOD, dataset=name, official_parameter_dataset=source,
                 official_commit=OFFICIAL_COMMIT, hidden=64, hid_dim=64,
                 hidden_dim=64, dropout=.5, lr=.01, learning_rate=.01,
                 weight_decay=.0005, seed=0, n_layers=2, mixup_lr=.0001,
                 epochs=epochs, optimizer="Adam", device="cuda",
                 validation_selection="clean validation accuracy; last tie",
                 early_stopping=False, cpu_fallback=False,
                 rng_reset="parent only", edge_chunk_size=requested.get("edge_chunk_size", 8192))
    if requested.get("device", "cuda").split(":")[0] != "cuda":
        raise ValueError("CUDA device is required")
    if requested.get("smoke", False):
        for field in ("ep_epochs", "cl_epochs"):
            if field not in requested:
                param[field] = min(param[field], epochs)
    for field in ("ep_epochs", "cl_epochs", "batch_size", "edge_chunk_size"):
        value = param[field]
        minimum = 0 if field in ("ep_epochs", "cl_epochs") else 2 if field == "batch_size" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"Invalid {field}: {value}")
    for field in ("scale", "alpha"):
        if not math.isfinite(float(param[field])) or not 0 <= float(param[field]) <= 1:
            raise ValueError(f"{field} must be in [0, 1]")
    if not math.isfinite(float(param["temp"])) or float(param["temp"]) <= 0:
        raise ValueError("temp must be positive")
    param["phase_budgets"] = {"edge_pretrain": param["ep_epochs"],
                              "classifier_pretrain": param["cl_epochs"], "main": epochs}
    param["supervised_epochs_total"] = param["cl_epochs"] + epochs
    param["parameter_overrides"] = {k: requested[k] for k in
                                    ("scale", "alpha", "temp", "batch_size", "ep_epochs", "cl_epochs")
                                    if k in requested}
    json.dumps(param, allow_nan=False)
    return param


class _EdgeScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, src, dst, chunk_size):
        ctx.save_for_backward(h, src, dst)
        ctx.chunk_size = chunk_size
        scores = h.new_empty(src.numel())
        for start in range(0, src.numel(), chunk_size):
            end = start + chunk_size
            scores[start:end] = (h[src[start:end]] * h[dst[start:end]]).sum(1)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        h, src, dst = ctx.saved_tensors
        grad_h = torch.zeros_like(h)
        # Exact dot-product adjoint; never retain an E x hidden activation.
        for start in range(0, src.numel(), ctx.chunk_size):
            end = start + ctx.chunk_size
            s, d = src[start:end], dst[start:end]
            g = grad_scores[start:end, None]
            grad_h.index_add_(0, s, g * h[d])
            grad_h.index_add_(0, d, g * h[s])
        return grad_h, None, None, None


def edge_scores(h: Tensor, src: Tensor, dst: Tensor, chunk_size: int) -> Tensor:
    return _EdgeScores.apply(h, src, dst, chunk_size)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft):
        return soft.round()

    @staticmethod
    def backward(ctx, grad):
        return grad


def straight_through(soft: Tensor) -> Tensor:
    return _StraightThrough.apply(soft)


def _fc(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm1d(in_dim), nn.LeakyReLU(),
                         nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


class NeighContrast(nn.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, config: dict):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden)])
        for _ in range(config["n_layers"] - 1):
            self.layers.append(_fc(hidden, hidden, config["dropout"]))
        self.out = _fc(hidden, classes, config["dropout"])
        self.inf = _fc(hidden, classes, config["dropout"])

    def encode(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encode(x)
        return self.out(h), self.inf(h)

    def update(self, x: Tensor) -> Tensor:
        # Like the author, update only the out head's BN state here.
        return self.out(self.encode(x)).argmax(1)


class MixupScale(nn.Module):
    def __init__(self, in_dim: int, hidden: int, config: dict):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.att = nn.Linear(hidden * 2, 1, bias=False)
        self.norm = nn.BatchNorm1d(hidden)
        self.scale = config["scale"]

    def forward(self, x: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        h = torch.cat((self.norm(self.linear(x[src])), self.norm(self.linear(x[dst]))), 1)
        return torch.sigmoid(self.att(h)) * (1 - self.scale) + self.scale


class EdgeSampler(nn.Module):
    def __init__(self, in_dim: int, hidden: int, config: dict):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.alpha = config["alpha"]
        self.temp = config["temp"]
        self.chunk_size = config["edge_chunk_size"]

    def forward(self, x: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        h = self.linear2(self.linear1(x))
        probs = edge_scores(h, src, dst, self.chunk_size).sigmoid() * (1 - self.alpha) + self.alpha
        soft = torch.distributions.RelaxedBernoulli(x.new_tensor(self.temp), probs=probs).rsample()
        return straight_through(clamp_probs(soft))


def self_contrast(y1: Tensor, y2: Tensor, z3: Tensor, z4: Tensor, z: Tensor) -> Tensor:
    return (F.mse_loss(y1, z3) + F.mse_loss(y2, z4)
            - F.mse_loss(y1.softmax(-1), z.softmax(-1))
            - F.mse_loss(y2.softmax(-1), z.softmax(-1)))


def endpoint_weights(y: Tensor, train_mask: Tensor, src: Tensor, dst: Tensor,
                     classes: int) -> tuple[Tensor, list[int]]:
    node_counts = torch.bincount(y[train_mask], minlength=classes).float()
    edge_counts = (torch.bincount(y[src[train_mask[src]]], minlength=classes)
                   + torch.bincount(y[dst[train_mask[dst]]], minlength=classes)).float()
    ratios = torch.where(edge_counts > 0, node_counts / edge_counts.clamp_min(1), 0.)
    missing = ((node_counts > 0) & (edge_counts == 0)).nonzero().flatten().tolist()
    return classes * F.normalize(ratios, p=1, dim=0), missing


def homophily_loss(pred: Tensor, src: Tensor, dst: Tensor,
                   weights: Tensor) -> tuple[Tensor | None, Tensor | None]:
    same = pred[src] == pred[dst]
    if not bool(same.any()) or not bool((~same).any()):
        return None, None
    pos, neg = weights[same], weights[~same]
    return neg.mean() / (pos + 1e-12).mean(), neg.sum() / (pos + 1e-12).sum()


def validate_data(data: Any) -> None:
    if not isinstance(data.x, Tensor) or not data.x.is_cuda:
        raise RuntimeError("GSSC requires CUDA data; CPU training/fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
    fields = ("edge_index", "y", "train_index", "val_index", "test_index",
              "train_mask", "val_mask", "test_mask", "val_y")
    for field in fields:
        value = getattr(data, field, None)
        if not isinstance(value, Tensor) or value.device != data.x.device:
            raise ValueError(f"{field} must be a tensor on {data.x.device}")
    n = data.x.shape[0]
    c = data.num_classes
    if data.x.ndim != 2 or n < 2 or data.x.shape[1] == 0 or not data.x.is_floating_point():
        raise ValueError("x must be a nonempty floating N x D matrix, N >= 2")
    if not isinstance(c, int) or c < 2 or not isinstance(data.name, str):
        raise ValueError("num_classes >= 2 and a string name are required")
    if not bool(torch.isfinite(data.x).all()):
        raise ValueError("Non-finite features")
    edge = data.edge_index
    if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2 or not edge.shape[1]:
        raise ValueError("edge_index must be nonempty long [2, E]")
    if bool(((edge < 0) | (edge >= n)).any()):
        raise ValueError("edge_index out of range")
    if data.y.dtype != torch.long or data.y.shape != (n,):
        raise ValueError("y must be long [N]")
    occupancy = torch.zeros(n, dtype=torch.int8, device=data.x.device)
    for split in ("train", "val", "test"):
        index, mask = getattr(data, split + "_index"), getattr(data, split + "_mask")
        if index.dtype != torch.long or index.ndim != 1:
            raise ValueError(f"{split}_index must be long [K]")
        if split != "test" and not index.numel():
            raise ValueError(f"{split}_index is empty")
        if bool(((index < 0) | (index >= n)).any()) or index.unique().numel() != index.numel():
            raise ValueError(f"Invalid or duplicated {split}_index")
        expected = torch.zeros(n, dtype=torch.bool, device=data.x.device)
        expected[index] = True
        if mask.dtype != torch.bool or mask.shape != (n,) or not torch.equal(mask, expected):
            raise ValueError(f"{split}_mask does not match its ordered index")
        occupancy += mask.to(torch.int8)
    if bool((occupancy > 1).any()):
        raise ValueError("Train/validation/test splits overlap")
    if bool((data.y[~data.train_mask] != -1).any()):
        raise ValueError("All non-training y must be -1; label isolation violated")
    train_y = data.y[data.train_index]
    if bool(((train_y < 0) | (train_y >= c)).any()):
        raise ValueError("Noisy training label out of range")
    if data.val_y.dtype != torch.long or data.val_y.shape != data.val_index.shape:
        raise ValueError("val_y must align with ordered val_index")
    if bool(((data.val_y < 0) | (data.val_y >= c)).any()):
        raise ValueError("Validation label out of range")


@torch.no_grad()
def validation_accuracy(model: NeighContrast, data: Any) -> float:
    model.eval()
    logits, _ = model(data.x)
    return float((logits[data.val_index].argmax(1) == data.val_y).double().mean())


def _finite(loss: Tensor, stage: str) -> None:
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError(f"Non-finite GSSC loss in {stage}")


def _train_classifier_epoch(model, mixup, data, src, dst, optimizer, scale_optimizer, cfg):
    model.train()
    mixup.train()
    weights, missing = endpoint_weights(data.y, data.train_mask, src, dst, data.num_classes)
    weights = weights.to(dtype=data.x.dtype)
    negative = torch.randint(data.x.shape[0], (src.numel(),), device=data.x.device)
    order = torch.randperm(src.numel(), device=data.x.device)
    sums = {"self_contrast": 0., "supervised": 0., "edge_total": 0.}
    batches, used, skipped = 0, 0, 0
    for start in range(0, src.numel(), cfg["batch_size"]):
        idx = order[start:start + cfg["batch_size"]]
        if idx.numel() <= 1:
            skipped += idx.numel()
            continue
        model.train()
        s, d, neg = src[idx], dst[idx], negative[idx]
        beta = mixup(data.x, s, d)
        _, z = model(data.x[neg])
        y1, _ = model(data.x[s])
        y2, _ = model(data.x[d])
        _, z3 = model(data.x[d] * beta + data.x[s] * (1 - beta))
        _, z4 = model(data.x[s] * beta + data.x[d] * (1 - beta))
        loss_ne = self_contrast(y1, y2, z3, z4, z)
        loss_cla = loss_ne.new_zeros(())
        for nodes, target_logits, virtual_logits in ((s, y1, z3), (d, y2, z4)):
            mask = data.train_mask[nodes]
            if bool(mask.any()):
                target = data.y[nodes[mask]]
                loss_cla = loss_cla + F.cross_entropy(target_logits[mask], target, weight=weights)
                loss_cla = loss_cla + F.cross_entropy(virtual_logits[mask], target, weight=weights)
        loss = loss_ne + loss_cla
        _finite(loss, "classifier")
        optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scale_optimizer.step()
        for key, value in (("self_contrast", loss_ne), ("supervised", loss_cla), ("edge_total", loss)):
            sums[key] += float(value.detach())
        batches += 1
        used += idx.numel()

    logits, _ = model(data.x)
    pred = logits.detach().argmax(1)
    center = logits.sum() * 0
    noisy_train = data.y[data.train_index]
    for label in range(data.num_classes):
        labeled = data.train_index[noisy_train == label]
        predicted = pred == label
        if labeled.numel() and bool(predicted.any()):
            center = center + F.mse_loss(logits[labeled].mean(0), logits[predicted].mean(0))
    _finite(center, "class_center")
    optimizer.zero_grad(set_to_none=True)
    center.backward()
    optimizer.step()
    return {**{k: v / max(batches, 1) for k, v in sums.items()},
            "center": float(center.detach()), "batches": batches, "processed_edges": used,
            "singleton_edges_skipped": skipped, "uncovered_train_classes": missing}


def _cpu_state(model, mixup, sampler):
    return {key: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for key, module in (("model", model), ("mixup", mixup), ("sampler", sampler))}


def run(data, config, outdir) -> dict:
    """Fit on noisy train labels; choose by val_y only; write best CPU probabilities."""
    validate_data(data)
    cfg = resolve_config(data.name, config)
    out = Path(outdir).resolve()
    artifacts = [out / filename for filename in ("epochs.jsonl", "checkpoint.pt", "probabilities.pt")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("GSSC artifacts already exist; use a fresh outdir")
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = data.x.device, data.x.dtype
    model = NeighContrast(data.x.shape[1], cfg["hidden"], data.num_classes, cfg).to(device=device, dtype=dtype)
    mixup = MixupScale(data.x.shape[1], cfg["hidden"], cfg).to(device=device, dtype=dtype)
    sampler = EdgeSampler(data.x.shape[1], cfg["hidden"], cfg).to(device=device, dtype=dtype)
    modules = (model, mixup, sampler)
    parameters_by_module = {name: sum(p.numel() for p in module.parameters())
                            for name, module in zip(("model", "mixup", "sampler"), modules)}
    src_all, dst_all = data.edge_index
    best_val, best_epoch, best_phase, best_phase_epoch = -math.inf, 0, "", 0
    best_state = None
    global_epoch, classifier_epoch = 0, 0
    diagnostics = {"official_commit": OFFICIAL_COMMIT, "test_labels_read": False,
                   "validation_labels_in_optimization": False, "cpu_training": False,
                   "num_nodes": data.x.shape[0], "num_raw_edges": src_all.numel(),
                   "parameters_by_module": parameters_by_module,
                   "parameter_fallback": cfg["official_parameter_dataset"] == "cora" and data.name.lower() != "cora",
                   "sampler_updates": 0, "homophily_updates": 0, "homophily_skipped": 0,
                   "empty_sampled_epochs": 0, "singleton_edges_skipped": 0,
                   "uncovered_train_classes": [], "phase_budgets": cfg["phase_budgets"],
                   "source_sha256": {str(path.relative_to(VENDOR)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (VENDOR / "code" / "main.py", VENDOR / "code" / "model.py",
                                                  VENDOR / "param" / "best_parameters.json")}}
    main_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    main_scale_optimizer = torch.optim.Adam(mixup.parameters(), lr=cfg["mixup_lr"], weight_decay=cfg["weight_decay"])
    main_edge_optimizer = torch.optim.Adam(sampler.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    with artifacts[0].open("w", encoding="utf-8") as log:
        def record(phase, phase_epoch, loss, details, eligible):
            nonlocal global_epoch, classifier_epoch, best_val, best_epoch, best_state, best_phase, best_phase_epoch
            global_epoch += 1
            if eligible:
                classifier_epoch += 1
            val = validation_accuracy(model, data)
            if not math.isfinite(val):
                raise FloatingPointError("Non-finite validation accuracy")
            selected = eligible and val >= best_val
            if selected:
                best_val, best_epoch = val, classifier_epoch
                best_phase, best_phase_epoch = phase, phase_epoch
                best_state = _cpu_state(model, mixup, sampler)
            row = dict(global_epoch=global_epoch, epoch=classifier_epoch if eligible else None,
                       phase=phase, phase_epoch=phase_epoch, loss=loss, val=val,
                       checkpoint_eligible=eligible, selected=selected, **details)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()

        # Separate pretraining optimizer moments match the author's main.py.
        edge_optimizer = torch.optim.Adam(sampler.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        for epoch in range(1, cfg["ep_epochs"] + 1):
            sampler.train()
            sampled = sampler(data.x, src_all, dst_all)
            loss = F.binary_cross_entropy_with_logits(sampled, torch.ones_like(sampled))
            _finite(loss, "edge_pretrain")
            edge_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            edge_optimizer.step()
            diagnostics["sampler_updates"] += 1
            record("edge_pretrain", epoch, float(loss.detach()), {}, False)
            del sampled, loss

        warm_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        warm_scale_optimizer = torch.optim.Adam(mixup.parameters(), lr=cfg["mixup_lr"], weight_decay=cfg["weight_decay"])
        for phase, budget, optimizer, scale_optimizer in (
                ("classifier_pretrain", cfg["cl_epochs"], warm_optimizer, warm_scale_optimizer),
                ("main", cfg["epochs"], main_optimizer, main_scale_optimizer)):
            for epoch in range(1, budget + 1):
                sampled, edge_weights = None, None
                if phase == "main":
                    sampler.train()
                    sampled = sampler(data.x, src_all, dst_all)
                    selected = sampled.detach() == 1
                    src, dst = src_all[selected], dst_all[selected]
                    edge_weights = sampled[selected]
                else:
                    src, dst = src_all, dst_all
                details = _train_classifier_epoch(model, mixup, data, src, dst, optimizer, scale_optimizer, cfg)
                details["sampled_edges"] = src.numel()
                diagnostics["singleton_edges_skipped"] += details["singleton_edges_skipped"]
                diagnostics["uncovered_train_classes"] = sorted(set(diagnostics["uncovered_train_classes"] + details["uncovered_train_classes"]))
                if phase == "main":
                    diagnostics["empty_sampled_epochs"] += int(src.numel() == 0)
                    # Argmax is the author's first-order outer target, not a val-label meta loss.
                    model.train()
                    with torch.no_grad():
                        pred = model.update(data.x)
                    hom_mean, hom_sum = homophily_loss(pred, src, dst, edge_weights)
                    if hom_mean is not None:
                        _finite(hom_mean, "homophily")
                        main_edge_optimizer.zero_grad(set_to_none=True)
                        hom_mean.backward()
                        main_edge_optimizer.step()
                        diagnostics["homophily_updates"] += 1
                        diagnostics["sampler_updates"] += 1
                    else:
                        diagnostics["homophily_skipped"] += 1
                    details.update(homophily_mean=None if hom_mean is None else float(hom_mean.detach()),
                                   homophily_sum=None if hom_sum is None else float(hom_sum.detach()))
                    del pred, hom_mean, hom_sum
                record(phase, epoch, details["edge_total"] + details["center"], details, True)
                del sampled, edge_weights

    if best_state is None:
        raise RuntimeError("No eligible classifier checkpoint")
    for name, module in zip(("model", "mixup", "sampler"), modules):
        module.load_state_dict(best_state[name])
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x)
        probabilities = logits.softmax(-1).detach().cpu()
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("Non-finite saved probabilities")
    diagnostics.update(completed_global_epochs=global_epoch, completed_classifier_epochs=classifier_epoch,
                       best_phase=best_phase, best_phase_epoch=best_phase_epoch,
                       inference_parameters=parameters_by_module["model"],
                       memory_order="edge sampler: O(N*hidden + E + edge_chunk_size*hidden), excluding raw features/parameters; no N^2 adjacency")
    torch.save(dict(state=best_state, config=cfg, best_epoch=best_epoch, best_val=best_val,
                    diagnostics=diagnostics), artifacts[1])
    torch.save(probabilities, artifacts[2])
    return dict(best_epoch=best_epoch, best_val=best_val, parameters=sum(parameters_by_module.values()),
                config=cfg, probabilities_file=str(artifacts[2]), diagnostics=diagnostics)
