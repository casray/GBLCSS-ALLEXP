import torch

from gb_stage_b.balls import generate_angular_balls
from gb_stage_b.data import load_graph
from gb_stage_b.diagnostics import ball_diagnostics, representation_diagnostics
from gb_stage_b.eval import linear_eval


def test_synthetic_graph_has_masks_edges_and_y():
    graph = load_graph("Synthetic", seed=0)

    assert graph.x.ndim == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.y.shape[0] == graph.x.shape[0]
    assert graph.y_true.shape[0] == graph.x.shape[0]
    assert graph.y_noisy.shape[0] == graph.x.shape[0]
    assert torch.equal(graph.y, graph.y_noisy)
    assert graph.train_mask.dtype is torch.bool
    assert graph.val_mask.sum().item() > 0
    assert graph.test_mask.sum().item() > 0


def test_diagnostics_return_expected_keys():
    graph = load_graph("Synthetic", seed=0)
    balls = generate_angular_balls(graph.x, min_size=3, quality_threshold=0.4)

    ball_stats = ball_diagnostics(balls, labels=graph.y)
    rep_stats = representation_diagnostics(graph.x)

    assert "num_balls" in ball_stats
    assert "label_purity_diagnostic" in ball_stats
    assert "effective_rank" in rep_stats
    assert rep_stats["feature_std_mean"] >= 0


def test_ball_diagnostics_can_omit_labels():
    graph = load_graph("Synthetic", seed=0)
    balls = generate_angular_balls(graph.x, min_size=3, quality_threshold=0.4)

    stats = ball_diagnostics(balls)

    assert "label_purity_diagnostic" not in stats
    assert stats["num_balls"] > 0


def test_representation_diagnostics_handles_single_node():
    stats = representation_diagnostics(torch.ones(1, 4))

    assert stats["mean_pairwise_cosine"] == 0.0
    assert stats["embedding_uniformity"] == 0.0


def test_linear_eval_runs_on_synthetic_graph():
    graph = load_graph("Synthetic", seed=0)

    acc = linear_eval(graph.x, graph.y, graph.train_mask, graph.val_mask, graph.test_mask, seed=0)

    assert 0.0 <= acc <= 1.0
