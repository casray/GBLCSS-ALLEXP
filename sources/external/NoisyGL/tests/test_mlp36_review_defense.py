import numpy as np
import pandas as pd
import pytest
import torch
import importlib
import importlib.util
from types import SimpleNamespace

from scripts import run_standard_downstream
from utils import frozen_grace_targets
from utils.tools import load_conf


def test_loo_ball_targets_exclude_the_current_node_vote():
    noisy = torch.tensor([0, 0, 1, 1])
    assignment = np.zeros(4, dtype=np.int64)
    train_mask = torch.ones(4, dtype=torch.bool)

    _, alpha, pseudo = frozen_grace_targets.apply_ball_soft_targets(
        noisy_label=noisy,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=2,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
    )

    assert torch.all(alpha > 0)
    assert pseudo.tolist() == [1, 1, 0, 0]


def test_loo_ball_targets_fall_back_when_external_support_is_too_small():
    noisy = torch.tensor([0, 1])
    assignment = np.zeros(2, dtype=np.int64)
    train_mask = torch.ones(2, dtype=torch.bool)

    soft_target, alpha, pseudo = frozen_grace_targets.apply_ball_soft_targets(
        noisy_label=noisy,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=2,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
    )

    assert torch.equal(soft_target, torch.eye(2))
    assert torch.equal(alpha, torch.zeros(2))
    assert torch.equal(pseudo, noisy)


def test_global_smoothing_uses_only_noisy_train_labels():
    builder = getattr(frozen_grace_targets, "build_global_smoothing_targets", None)
    assert callable(builder), "global smoothing target builder is not implemented"

    noisy = torch.tensor([0, 1, 1, 0])
    changed_holdout = torch.tensor([0, 1, 0, 1])
    train_mask = torch.tensor([True, True, False, False])

    target_a, alpha_a, _, diagnostics_a = builder(
        noisy_label=noisy,
        train_mask=train_mask,
        n_classes=2,
        epsilon=0.1,
        distribution="prior",
    )
    target_b, alpha_b, _, diagnostics_b = builder(
        noisy_label=changed_holdout,
        train_mask=train_mask,
        n_classes=2,
        epsilon=0.1,
        distribution="prior",
    )

    assert torch.allclose(target_a[train_mask], target_b[train_mask])
    assert torch.equal(alpha_a[train_mask], alpha_b[train_mask])
    assert diagnostics_a["smoothing_distribution"] == "prior"
    assert diagnostics_b["smoothing_distribution"] == "prior"


def test_uniform_smoothing_matches_the_declared_formula():
    builder = getattr(frozen_grace_targets, "build_global_smoothing_targets", None)
    assert callable(builder), "global smoothing target builder is not implemented"

    noisy = torch.tensor([0, 1, 0])
    train_mask = torch.tensor([True, True, False])
    target, alpha, pseudo, diagnostics = builder(
        noisy_label=noisy,
        train_mask=train_mask,
        n_classes=2,
        epsilon=0.1,
        distribution="uniform",
    )

    assert torch.allclose(target[0], torch.tensor([0.95, 0.05]))
    assert torch.allclose(target[1], torch.tensor([0.05, 0.95]))
    assert torch.allclose(alpha[train_mask], torch.tensor([0.1, 0.1]))
    assert torch.equal(pseudo, noisy)
    assert diagnostics["smoothing_epsilon"] == 0.1


def test_mlp36_methods_are_registered():
    expected = {
        "frozengrace_gb_mlp_softdist_beta05_loo",
        "frozengrace_gb_mlp_softdist_beta05_loo_shuffled",
        "mlp_uniform_smoothing",
        "mlp_prior_smoothing",
    }
    assert expected.issubset(run_standard_downstream.PREDICTOR_MODULES)


def test_target_quality_diagnostics_report_brier_and_mean_clean_mass_gain():
    noisy = torch.tensor([0, 1])
    clean = torch.tensor([0, 0])
    target = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
    alpha = torch.tensor([0.1, 0.4])
    pseudo = target.argmax(dim=1)
    train_mask = torch.ones(2, dtype=torch.bool)

    diagnostics = frozen_grace_targets.diagnostics_for_soft_targets(
        noisy, clean, target, alpha, pseudo, train_mask
    )

    assert abs(diagnostics["soft_target_brier"] - 0.37) < 1e-6
    assert abs(diagnostics["noisy_onehot_brier"] - 1.0) < 1e-6
    assert abs(diagnostics["soft_brier_gain"] - 0.63) < 1e-6
    assert abs(diagnostics["soft_clean_mass_delta_mean"] - 0.15) < 1e-6
    assert abs(diagnostics["soft_target_argmax_accuracy"] - 0.5) < 1e-6


def test_mlp36_configs_freeze_loo_and_smoothing_settings():
    datasets = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho", "pubmed")
    for dataset in datasets:
        loo = load_conf(method="frozengrace_gb_mlp_softdist_beta05_loo", dataset=dataset)
        assert loo.model["method"] == "frozengrace_gb_mlp_softdist_beta05_loo"
        assert loo.gb["leave_one_out"] is True
        assert loo.gb["target_beta"] == 0.5
        assert loo.gb["min_train_per_ball"] == 2

        for method, builder in (
            ("mlp_uniform_smoothing", "uniform_smoothing"),
            ("mlp_prior_smoothing", "prior_smoothing"),
        ):
            conf = load_conf(method=method, dataset=dataset)
            assert conf.model["method"] == method
            assert conf.gb["soft_target_builder"] == builder
            assert conf.gb["smoothing_epsilon"] == 0.1


def test_standard_runner_exports_gb_diagnostics(monkeypatch):
    data = SimpleNamespace(
        labels=torch.tensor([0, 1, 0]),
        feats=torch.eye(3),
        train_masks=np.array([0]),
        val_masks=np.array([1]),
        test_masks=np.array([2]),
        dim_feats=3,
        n_classes=2,
    )
    conf = SimpleNamespace(
        model={"n_feat": 3, "n_classes": 2},
        training={},
        gb={},
    )

    class FakePredictor:
        def __init__(self, *_args):
            self.gb_diagnostics = {"soft_target_brier": 0.25, "leave_one_out": 1}

        def train(self):
            return {"train": 0.5, "valid": 0.5, "test": 0.5}

    monkeypatch.setattr(run_standard_downstream, "setup_seed", lambda _seed: None)
    monkeypatch.setattr(run_standard_downstream, "_load_dataset", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(
        run_standard_downstream,
        "label_process",
        lambda **_kwargs: (data.labels.clone(), np.array([], dtype=np.int64)),
    )
    monkeypatch.setattr(run_standard_downstream, "load_conf", lambda *_args: conf)
    monkeypatch.setattr(run_standard_downstream, "predictor_class_for", lambda _method: FakePredictor)

    args = SimpleNamespace(
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        noise_type="uniform",
        noise_rate=0.4,
        noise_scope="train",
        debug=False,
        config_overrides={},
    )
    row = run_standard_downstream.run_one(args, "cora", "mlp_ce", 0)

    assert row["soft_target_brier"] == 0.25
    assert row["leave_one_out"] == 1


def test_mlp36_holm_adjustment_and_paired_statistics():
    spec = importlib.util.find_spec("scripts.summarize_mlp36_review_defense")
    assert spec is not None, "MLP-36 summary module is not implemented"
    summary = importlib.import_module("scripts.summarize_mlp36_review_defense")

    adjusted = summary.holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])

    stats = summary.paired_statistics(
        np.array([0.6, 0.7, 0.8, 0.9, 1.0]),
        np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    )
    assert abs(stats["mean_delta"] - 0.1) < 1e-12
    assert stats["positive_seeds"] == 5
    assert stats["n"] == 5

    aggregate = summary.aggregate(
        pd.DataFrame(
            {
                "dataset": ["cora"] * 5,
                "method": ["candidate"] * 5,
                "test": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
    )
    assert abs(aggregate.loc[0, "test_std"] - np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1)) < 1e-12


def test_primary_statistics_are_paired_and_holm_adjusted_within_family():
    spec = importlib.util.find_spec("scripts.summarize_primary_statistics_20260715")
    assert spec is not None, "primary statistics summary module is not implemented"
    summary = importlib.import_module("scripts.summarize_primary_statistics_20260715")

    rows = []
    for dataset, offset in (("a", 0.0), ("b", 0.02)):
        for seed in range(5):
            rows.extend(
                [
                    {"dataset": dataset, "noise_type": "uniform", "noise_rate": 0.4,
                     "seed": seed, "method": "lcss", "test": 0.7 + offset + 0.01 * seed},
                    {"dataset": dataset, "noise_type": "uniform", "noise_rate": 0.4,
                     "seed": seed, "method": "ce", "test": 0.6 + offset + 0.01 * seed},
                    {"dataset": dataset, "noise_type": "uniform", "noise_rate": 0.4,
                     "seed": seed, "method": "shuffled", "test": 0.55 + offset + 0.01 * seed},
                    {"dataset": dataset, "noise_type": "uniform", "noise_rate": 0.4,
                     "seed": seed, "method": "jocor", "test": 0.65 + offset + 0.01 * seed},
                ]
            )

    result = summary.compute_primary_statistics(pd.DataFrame(rows))

    assert len(result) == 6
    assert set(result["comparison"]) == {"lcss_vs_ce", "lcss_vs_shuffled", "lcss_vs_jocor"}
    assert set(result["n"]) == {5}
    assert result["holm_p"].notna().all()

    incomplete = pd.DataFrame(rows[:-1])
    with pytest.raises(ValueError, match="exactly 5 paired seeds"):
        summary.compute_primary_statistics(incomplete)


def test_latency_benchmark_reports_fullgraph_and_per_node_costs():
    spec = importlib.util.find_spec("scripts.benchmark_mlp36_latency")
    assert spec is not None, "MLP-36 latency benchmark module is not implemented"
    benchmark = importlib.import_module("scripts.benchmark_mlp36_latency")

    class FakeModel(torch.nn.Module):
        def forward(self, features, _adj):
            return features @ torch.ones(features.shape[1], 2)

    result = benchmark.benchmark_forward(
        FakeModel(),
        torch.ones(8, 4),
        None,
        warmup=1,
        repeats=3,
    )

    assert result["inference_repeats"] == 3
    assert result["fullgraph_latency_ms"] > 0
    assert result["per_node_latency_us"] > 0


def test_figure1_deployable_mlp_uses_layer_blocks_not_graph_like_edges():
    import matplotlib.pyplot as plt

    figure_module = importlib.import_module("scripts.make_fig1_lcss_pipeline_20260708")
    fig, ax = plt.subplots()
    line_count = len(ax.lines)

    figure_module.draw_mlp(ax, 0.1, 0.1, 0.8, 0.6)

    labels = " ".join(text.get_text().replace("\n", " ") for text in ax.texts)
    assert len(ax.lines) == line_count
    assert "Input X" in labels
    assert "Hidden 64" in labels
    assert "Output C" in labels
    plt.close(fig)
