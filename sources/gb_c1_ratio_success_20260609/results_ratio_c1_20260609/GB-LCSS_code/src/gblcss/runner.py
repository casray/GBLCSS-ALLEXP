from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

from gblcss.baselines import train_noisy_label_baseline
from gblcss.data import inject_train_label_noise, load_public_dataset, set_seed
from gblcss.experiment import load_config, run_gblcss
from gblcss.representation import load_frozen_representation
from gblcss.training import train_x_only_mlp


TABLE2_METHODS = ["mlp_ce", "small_loss", "coteaching", "jocor", "sce", "elr", "logitclip"]
STRUCTURE_METHODS = ["postel", "als", "sals", "graphmlp", "gsdn"]
TRANSFER_METHODS = ["glnn_style", "ffg2m", "krd", "vqgraph"]
ABLATION_METHODS = ["gblcss_self_vote", "uniform_smoothing", "prior_smoothing", "knn", "kmeans", "raw_features", "propagated_features"]
REPRESENTATION_METHODS = {"gblcss", "gblcss_shuffled", "gblcss_distill", "gblcss_self_vote", "knn", "kmeans"}
GROUPS = {
    "matched": ["gblcss", *TABLE2_METHODS],
    "table2": ["gblcss", *TABLE2_METHODS],
    "structure": ["gblcss", *STRUCTURE_METHODS],
    "distillation": ["gblcss", *TRANSFER_METHODS, "gblcss_distill"],
    "ablations": ["gblcss", "gblcss_shuffled", "gblcss_self_vote", "uniform_smoothing", "prior_smoothing"],
    "grouping": ["gblcss", "knn", "kmeans"],
    "representation": ["gblcss", "raw_features", "propagated_features"],
    "validation": ["gblcss", "gblcss_shuffled", "mlp_ce", "jocor", "postel", "gsdn"],
}
SUPPORTED_METHODS = {"gblcss", "gblcss_shuffled", "gblcss_distill", *TABLE2_METHODS,
                     *STRUCTURE_METHODS, *TRANSFER_METHODS, *ABLATION_METHODS}
FULL_SPECTRUM = [("uniform", 0.2), ("uniform", 0.4), ("uniform", 0.6), ("pair", 0.4), ("instance", 0.4)]


def expand_methods(value):
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    expanded = []
    for item in requested:
        entries = sorted(SUPPORTED_METHODS) if item == "all" else GROUPS.get(item, [item])
        for method in entries:
            if method not in SUPPORTED_METHODS:
                raise ValueError(f"unknown runnable method: {method}")
            if method not in expanded:
                expanded.append(method)
    if not expanded:
        raise ValueError("at least one method is required")
    return expanded


def validate_main_config(config):
    if config['method']['tie_mode'] != 'distribution_only' or config['method']['leave_one_out'] is not True:
        raise ValueError('the main method requires distribution_only ties and leave_one_out=true')
    expected_split = {'type': 'percent', 'train': .2, 'validation': .1, 'test': .7}
    if config.get('split') != expected_split:
        raise ValueError('this runner implements the reported percent 0.2/0.1/0.7 split')
    if int(config['training']['epochs']) < 1:
        raise ValueError('epochs must be positive')


def run_method(method, graph, representation, config, *, seed, noise_rate, device, adaptations=None):
    """Run one method from a fresh RNG state for paired-seed comparability."""
    set_seed(seed)
    if method in REPRESENTATION_METHODS and representation is None:
        raise ValueError(f"{method} requires a frozen representation")
    if method == "gblcss":
        output = run_gblcss(graph, representation, config, seed=seed)
        return output.result, output.diagnostics
    if method == "gblcss_shuffled":
        output = run_gblcss(graph, representation, config, seed=seed, shuffled=True)
        return output.result, output.diagnostics
    if method == "gblcss_distill":
        output = run_gblcss(graph, representation, config, seed=seed, distill=True)
        return output.result, output.diagnostics
    common = {
        "hidden": config["model"]["hidden"], "layers": config["model"]["layers"],
        "dropout": config["model"]["dropout"], "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"], "epochs": config["training"]["epochs"],
        "device": device,
    }
    if method in ABLATION_METHODS:
        from gblcss.ablation_targets import build_ablation_targets
        targets, diagnostics = build_ablation_targets(graph, representation, method, config, seed=seed)
        return train_x_only_mlp(graph, targets, **common), diagnostics
    if method in STRUCTURE_METHODS or method in TRANSFER_METHODS:
        if adaptations is None:
            adaptations = load_config(ROOT / "configs/main/protocol_adaptations.yaml")
        if method == "graphmlp":
            from gblcss.baselines.graph_mlp import train_graph_mlp
            settings = adaptations["structure_assisted"][method]
            result = train_graph_mlp(graph, settings, **common)
        elif method in STRUCTURE_METHODS:
            from gblcss.baselines.structure import train_structure_baseline
            settings = adaptations["structure_assisted"][method]
            result = train_structure_baseline(graph, method, settings, **common)
        else:
            from gblcss.baselines.graph_transfer import train_transfer_baseline
            settings = adaptations["graph_transfer"][method]
            result = train_transfer_baseline(graph, method, settings, **common)
        return result, {"implementation": settings.get("implementation", "project_implementation")}
    if method in TABLE2_METHODS:
        if adaptations is None:
            adaptations = load_config(ROOT / "configs/main/protocol_adaptations.yaml")
        result = train_noisy_label_baseline(
            graph,
            method,
            noise_rate=noise_rate,
            hidden=config["model"]["hidden"],
            layers=config["model"]["layers"],
            dropout=config["model"]["dropout"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            epochs=config["training"]["epochs"],
            device=device,
            settings=adaptations["matched_backbone"][method],
        )
        return result, {}
    raise ValueError(f"unknown runnable method: {method}")


