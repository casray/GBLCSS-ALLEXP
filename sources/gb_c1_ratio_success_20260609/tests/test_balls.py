import math

import numpy as np
import torch

from gb_stage_b.balls import (
    BallBatch,
    GranularBall,
    ball_metrics,
    build_balls,
    generate_gain_split_balls,
    generate_angular_balls,
    make_ball_batch,
    reliable_ball_mask,
    shuffled_assignments,
    size_matched_random_assignments,
)


def _clustered_data():
    embeddings = np.array(
        [
            [0.00, 0.00],
            [0.05, 0.00],
            [0.00, 0.05],
            [5.00, 5.00],
            [5.05, 5.00],
            [5.00, 5.05],
            [-5.00, 5.00],
            [-5.05, 5.00],
            [-5.00, 5.05],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    return embeddings, labels


def test_build_balls_covers_all_samples_once():
    embeddings, labels = _clustered_data()

    balls = build_balls(embeddings, labels, min_size=3, max_size=3, random_state=0)

    assert balls
    assert all(isinstance(ball, GranularBall) for ball in balls)
    covered = sorted(int(i) for ball in balls for i in ball.indices.tolist())
    assert covered == list(range(embeddings.shape[0]))


def test_pure_cluster_data_has_high_purity_and_finite_radius():
    embeddings, labels = _clustered_data()

    balls = build_balls(embeddings, labels, min_size=3, max_size=3, random_state=0)

    assert min(ball.purity for ball in balls) == 1.0
    assert all(math.isfinite(ball.radius) for ball in balls)
    assert all(ball.radius >= 0.0 for ball in balls)


def test_shuffled_labels_do_not_improve_weighted_purity():
    embeddings, labels = _clustered_data()
    shuffled = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

    true_purity = ball_metrics(build_balls(embeddings, labels, min_size=3, max_size=3, random_state=0), embeddings, labels)[
        "weighted_purity"
    ]
    shuffled_purity = ball_metrics(
        build_balls(embeddings, shuffled, min_size=3, max_size=3, random_state=0),
        embeddings,
        shuffled,
    )["weighted_purity"]

    assert shuffled_purity <= true_purity


def test_ball_metrics_recomputes_purity_from_supplied_labels():
    embeddings, labels = _clustered_data()
    shuffled = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    balls = build_balls(embeddings, labels, min_size=3, max_size=3, random_state=0)

    true_metrics = ball_metrics(balls, embeddings, labels)
    shuffled_metrics = ball_metrics(balls, embeddings, shuffled)

    assert true_metrics["weighted_purity"] > shuffled_metrics["weighted_purity"]
    assert true_metrics["mean_purity"] > shuffled_metrics["mean_purity"]


def test_make_ball_batch_recomputes_centers_radius_and_labels_from_assignment():
    embeddings, labels = _clustered_data()
    assignment = np.array([1, 1, 1, 0, 0, 0, 2, 2, 2], dtype=np.int64)

    balls = make_ball_batch(embeddings, labels, assignment)

    assert len(balls) == 3
    assert np.allclose(balls[0].center, embeddings[3:6].mean(axis=0))
    assert balls[0].majority_label == 1
    assert balls[0].label_hist == {1: 3}
    assert math.isclose(balls[0].purity, 1.0)
    assert balls[0].radius > 0.0


def test_ball_metrics_fields_are_finite():
    embeddings, labels = _clustered_data()
    balls = build_balls(embeddings, labels, min_size=3, max_size=3, random_state=0)

    metrics = ball_metrics(balls, embeddings, labels)

    for key in (
        "num_balls",
        "mean_radius",
        "mean_purity",
        "weighted_purity",
        "mean_intra_distance",
        "center_separation",
        "ball_size_mean",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_small_data_returns_single_ball():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    balls = build_balls(embeddings, labels, min_size=3, max_size=256, random_state=0)

    assert len(balls) == 1
    assert balls[0].indices.tolist() == [0, 1]


def test_build_balls_handles_duplicate_and_extreme_points_with_finite_metrics():
    embeddings = np.array(
        [[0.0, 0.0]] * 8 + [[1000.0, 1000.0]] + [[-1000.0, 1000.0]],
        dtype=np.float32,
    )
    labels = np.array([0] * 8 + [1, 2], dtype=np.int64)

    balls = build_balls(embeddings, labels, min_size=2, max_size=3, random_state=0)
    metrics = ball_metrics(balls, embeddings, labels)

    covered = sorted(int(i) for ball in balls for i in ball.indices.tolist())
    assert covered == list(range(embeddings.shape[0]))
    assert all(math.isfinite(value) for value in metrics.values())


def test_generate_angular_balls_covers_all_nodes():
    x = torch.randn(30, 8)

    balls = generate_angular_balls(x, min_size=3, quality_threshold=0.35)

    assert isinstance(balls, BallBatch)
    assert balls.assignment.shape == (30,)
    assert set(balls.assignment.tolist()) == set(range(balls.num_balls))
    assert balls.centers.shape[0] == balls.num_balls


def test_generate_gain_split_balls_covers_all_nodes_once():
    x = torch.cat([torch.randn(16, 4) - 4.0, torch.randn(16, 4) + 4.0], dim=0)

    balls = generate_gain_split_balls(x, min_ball_size=4, max_depth=4, min_split_gain=0.0)

    assert isinstance(balls, BallBatch)
    assert balls.assignment.shape == (32,)
    assert set(balls.assignment.tolist()) == set(range(balls.num_balls))
    assert int(torch.bincount(balls.assignment).sum().item()) == 32
    assert balls.sizes.min().item() >= 4


def test_generate_gain_split_balls_is_label_free():
    x = torch.cat([torch.randn(12, 4) - 3.0, torch.randn(12, 4) + 3.0], dim=0)
    labels_a = torch.tensor([0] * 12 + [1] * 12)
    labels_b = 1 - labels_a

    balls_a = generate_gain_split_balls(x, min_ball_size=4, max_depth=4, min_split_gain=0.0)
    balls_b = generate_gain_split_balls(x, min_ball_size=4, max_depth=4, min_split_gain=0.0)

    assert torch.equal(balls_a.assignment, balls_b.assignment)
    assert not torch.equal(labels_a, labels_b)


def test_generate_gain_split_balls_respects_depth_and_gain_threshold():
    x = torch.cat([torch.randn(12, 4) - 5.0, torch.randn(12, 4) + 5.0], dim=0)

    depth_limited = generate_gain_split_balls(x, min_ball_size=4, max_depth=0, min_split_gain=0.0)
    gain_limited = generate_gain_split_balls(x, min_ball_size=4, max_depth=4, min_split_gain=100.0)

    assert depth_limited.num_balls == 1
    assert gain_limited.num_balls == 1


def test_generate_angular_balls_splits_oversized_low_distance_ball():
    x = torch.ones(18, 4)

    balls = generate_angular_balls(x, min_size=3, max_size=6, quality_threshold=100.0)

    assert balls.num_balls > 1
    assert balls.sizes.max().item() <= 6
    assert balls.sizes.min().item() >= 3


def test_reliable_ball_mask_filters_by_size_and_ad():
    x = torch.randn(20, 4)
    balls = generate_angular_balls(x, min_size=2, quality_threshold=0.4)

    mask = reliable_ball_mask(balls, min_size=2, max_size=20, ad_percentile=100.0)

    assert mask.dtype is torch.bool
    assert mask.shape == (balls.num_balls,)
    assert mask.any()


def test_assignment_controls_preserve_shape_and_sizes():
    assignment = torch.tensor([0, 0, 1, 1, 1, 2])

    shuffled = shuffled_assignments(assignment, seed=0)
    random_assignment = size_matched_random_assignments(assignment, seed=0)

    assert shuffled.shape == assignment.shape
    assert random_assignment.shape == assignment.shape
    assert sorted(torch.bincount(random_assignment).tolist()) == sorted(torch.bincount(assignment).tolist())


def test_make_ball_batch_can_return_ball_batch_without_labels():
    x = torch.randn(12, 4)
    assignment = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    balls = make_ball_batch(x, assignment)

    assert isinstance(balls, BallBatch)
    assert balls.centers.shape == (4, 4)
    assert balls.sizes.tolist() == [3, 3, 3, 3]
