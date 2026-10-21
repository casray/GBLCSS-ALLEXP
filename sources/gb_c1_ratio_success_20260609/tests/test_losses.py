import pytest
import torch

from gb_stage_b.augment import drop_edges, make_two_views, mask_features
from gb_stage_b.data import load_graph
from gb_stage_b.losses import (
    infonce_loss,
    negative_free_loss,
    symmetric_cross_entropy_loss,
    variance_loss,
    ball_majority_vote,
    ball_label_smoothing,
    ball_reliability_weights,
    gated_ball_soft_correction,
    inter_ball_separation_loss,
    knn_majority_vote,
    knn_label_smoothing,
    weighted_soft_target_loss,
)
from gb_stage_b.models import StageBModel


def test_model_forward_shapes_on_synthetic_graph():
    data = load_graph("Synthetic", seed=0)
    model = StageBModel(in_dim=data.x.size(1), hidden_dim=16, proj_dim=8)

    h, z, p = model(data.x, data.edge_index)

    assert h.shape == (data.x.size(0), 16)
    assert z.shape == (data.x.size(0), 8)
    assert p.shape == (data.x.size(0), 8)


def test_drop_edges_zero_probability_returns_input_unchanged():
    data = load_graph("Synthetic", seed=0)

    out = drop_edges(data.edge_index, drop_prob=0.0, seed=123)

    assert out is data.edge_index
    assert torch.equal(out, data.edge_index)
    assert out.dtype is data.edge_index.dtype
    assert out.device == data.edge_index.device


def test_mask_features_zero_probability_returns_input_unchanged():
    data = load_graph("Synthetic", seed=0)

    out = mask_features(data.x, mask_prob=0.0, seed=123)

    assert out is data.x
    assert torch.equal(out, data.x)
    assert out.dtype is data.x.dtype
    assert out.device == data.x.device


def test_augmentations_preserve_shape_dtype_and_device_at_high_probability():
    data = load_graph("Synthetic", seed=0)

    edge_index = drop_edges(data.edge_index, drop_prob=1.0, seed=0)
    x = mask_features(data.x, mask_prob=1.0, seed=0)

    assert edge_index.shape == (2, 0)
    assert edge_index.dtype is data.edge_index.dtype
    assert edge_index.device == data.edge_index.device
    assert x.shape == data.x.shape
    assert x.dtype is data.x.dtype
    assert x.device == data.x.device
    assert torch.equal(x, torch.zeros_like(data.x))


def test_ssl_losses_return_finite_zero_dim_scalars():
    z1 = torch.randn(12, 8)
    z2 = torch.randn(12, 8)
    p1 = torch.randn(12, 8)
    p2 = torch.randn(12, 8)

    negfree = negative_free_loss(p1, z2, p2, z1)
    infonce = infonce_loss(z1, z2)

    assert negfree.ndim == 0
    assert infonce.ndim == 0
    assert torch.isfinite(negfree)
    assert torch.isfinite(infonce)
    assert infonce.item() >= 0.0


def test_negative_free_loss_is_lower_for_identical_embeddings_than_mismatched():
    z = torch.randn(16, 8)
    aligned = negative_free_loss(z, z, z, z)
    mismatched_z = torch.flip(z, dims=[0])
    mismatched = negative_free_loss(z, mismatched_z, z, mismatched_z)

    assert aligned.item() < mismatched.item()
    assert aligned.item() < -0.99


def test_infonce_rejects_non_positive_temperature():
    z1 = torch.randn(4, 3)
    z2 = torch.randn(4, 3)

    with pytest.raises(ValueError, match="temperature"):
        infonce_loss(z1, z2, temperature=0.0)

    with pytest.raises(ValueError, match="temperature"):
        infonce_loss(z1, z2, temperature=-1.0)


def test_make_two_views_uses_asymmetric_rates_and_seeded_views():
    data = load_graph("Synthetic", seed=0)

    x1, e1, x2, e2 = make_two_views(
        data.x,
        data.edge_index,
        edge_drop_1=0.0,
        edge_drop_2=1.0,
        feat_mask_1=0.0,
        feat_mask_2=1.0,
        seed=7,
    )
    x1_again, e1_again, x2_again, e2_again = make_two_views(
        data.x,
        data.edge_index,
        edge_drop_1=0.0,
        edge_drop_2=1.0,
        feat_mask_1=0.0,
        feat_mask_2=1.0,
        seed=7,
    )

    assert torch.equal(x1, data.x)
    assert torch.equal(e1, data.edge_index)
    assert e2.shape == (2, 0)
    assert torch.equal(x2, torch.zeros_like(data.x))
    assert torch.equal(x1, x1_again)
    assert torch.equal(e1, e1_again)
    assert torch.equal(x2, x2_again)
    assert torch.equal(e2, e2_again)


def test_variance_loss_is_positive_for_collapsed_embeddings():
    collapsed = torch.ones(12, 8)
    spread = torch.randn(12, 8) * 3.0

    assert variance_loss(collapsed).item() > 0.9
    assert variance_loss(spread).item() < variance_loss(collapsed).item()


def test_symmetric_cross_entropy_loss_is_finite_and_differentiable():
    logits = torch.tensor([[4.0, -1.0], [-0.5, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = symmetric_cross_entropy_loss(logits, labels, alpha=1.0, beta=0.1)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ball_majority_vote_corrects_minority_labels():
    assignment = torch.tensor([0, 0, 0, 1, 1])
    noisy_labels = torch.tensor([0, 0, 1, 2, 2])
    reliable = torch.tensor([True, True])

    corrected = ball_majority_vote(assignment, noisy_labels, reliable, num_classes=3)

    assert corrected[0].item() == 0
    assert corrected[1].item() == 0
    assert corrected[2].item() == 0
    assert corrected[3].item() == 2
    assert corrected[4].item() == 2


def test_ball_majority_vote_unreliable_ball_keeps_original():
    assignment = torch.tensor([0, 0, 1, 1])
    noisy_labels = torch.tensor([0, 1, 2, 3])
    reliable = torch.tensor([True, False])

    corrected = ball_majority_vote(assignment, noisy_labels, reliable, num_classes=4)

    assert corrected[2].item() == 2
    assert corrected[3].item() == 3


def test_ball_label_smoothing_shape_and_sum():
    assignment = torch.tensor([0, 0, 0, 1, 1])
    noisy_labels = torch.tensor([0, 0, 1, 2, 2])
    reliable = torch.tensor([True, True])

    soft = ball_label_smoothing(assignment, noisy_labels, reliable, num_classes=3, smoothing=0.2)

    assert soft.shape == (5, 3)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert soft[0, 0].item() > soft[0, 1].item()


def test_ball_label_smoothing_zero_smoothing_is_one_hot():
    assignment = torch.tensor([0, 0])
    noisy_labels = torch.tensor([1, 2])
    reliable = torch.tensor([True])

    soft = ball_label_smoothing(assignment, noisy_labels, reliable, num_classes=3, smoothing=0.0)

    expected = torch.zeros(2, 3)
    expected[0, 1] = 1.0
    expected[1, 2] = 1.0
    assert torch.allclose(soft, expected, atol=1e-6)


def test_inter_ball_separation_loss_finite_negative():
    centers = torch.randn(5, 8)
    reliable = torch.tensor([True, True, True, False, False])

    loss = inter_ball_separation_loss(centers, reliable)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() < 0


def test_inter_ball_separation_loss_single_ball_returns_zero():
    centers = torch.randn(3, 8)
    reliable = torch.tensor([True, False, False])

    loss = inter_ball_separation_loss(centers, reliable)

    assert loss.item() == 0.0


def test_ball_majority_vote_ignores_non_train_labels():
    """Val/test nodes are class 1 majority in ball 0, but train nodes are class 0.
    With train_mask, the vote must follow train nodes (class 0), not val/test."""
    # nodes 0,1 are train (label 0), nodes 2,3,4 are val/test (label 1)
    assignment = torch.tensor([0, 0, 0, 0, 0])
    noisy_labels = torch.tensor([0, 0, 1, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, False, False, False])

    corrected = ball_majority_vote(
        assignment, noisy_labels, reliable, num_classes=2, train_mask=train_mask,
    )
    # train nodes should get class 0 (train majority), not class 1 (overall majority)
    assert corrected[0].item() == 0
    assert corrected[1].item() == 0
    # all members get the train-majority label
    assert corrected[2].item() == 0


def test_ball_label_smoothing_ignores_non_train_labels():
    """Ball distribution should only reflect train members' labels."""
    assignment = torch.tensor([0, 0, 0, 0])
    noisy_labels = torch.tensor([0, 0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, False, False])

    soft = ball_label_smoothing(
        assignment, noisy_labels, reliable, num_classes=2, smoothing=1.0,
        train_mask=train_mask,
    )
    # With smoothing=1.0, soft target = ball distribution = 100% class 0 (only train labels)
    assert soft[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert soft[0, 1].item() == pytest.approx(0.0, abs=1e-5)


def test_gated_ball_soft_correction_blends_only_when_gate_passes():
    assignment = torch.tensor([0, 0, 0, 1, 1])
    noisy_labels = torch.tensor([0, 1, 1, 0, 1])
    reliable = torch.tensor([True, True])
    train_mask = torch.tensor([True, True, False, True, True])

    soft = gated_ball_soft_correction(
        assignment,
        noisy_labels,
        reliable,
        num_classes=2,
        train_mask=train_mask,
        tau_cov=2,
        tau_conf=0.5,
        alpha=0.3,
    )

    expected_node0 = torch.tensor([0.85, 0.15])
    expected_node2 = torch.tensor([0.15, 0.85])
    assert torch.allclose(soft[0], expected_node0, atol=1e-6)
    assert torch.allclose(soft[2], expected_node2, atol=1e-6)


def test_gated_ball_soft_correction_keeps_one_hot_when_coverage_fails():
    assignment = torch.tensor([0, 0, 0])
    noisy_labels = torch.tensor([0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, False, False])

    soft = gated_ball_soft_correction(
        assignment,
        noisy_labels,
        reliable,
        num_classes=2,
        train_mask=train_mask,
        tau_cov=2,
        tau_conf=0.0,
        alpha=0.3,
    )

    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert torch.allclose(soft, expected, atol=1e-6)


def test_gated_ball_soft_correction_keeps_one_hot_when_confidence_fails():
    assignment = torch.tensor([0, 0, 0])
    noisy_labels = torch.tensor([0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, False])

    soft = gated_ball_soft_correction(
        assignment,
        noisy_labels,
        reliable,
        num_classes=2,
        train_mask=train_mask,
        tau_cov=2,
        tau_conf=0.6,
        alpha=0.3,
    )

    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert torch.allclose(soft, expected, atol=1e-6)


def test_gated_ball_soft_correction_ignores_non_train_labels():
    assignment = torch.tensor([0, 0, 0, 0])
    noisy_labels = torch.tensor([0, 0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, False, False])

    soft = gated_ball_soft_correction(
        assignment,
        noisy_labels,
        reliable,
        num_classes=2,
        train_mask=train_mask,
        tau_cov=2,
        tau_conf=1.0,
        alpha=1.0,
    )

    assert soft[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert soft[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert soft[3, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert soft[3, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_ball_reliability_weights_only_uses_train_labels():
    assignment = torch.tensor([0, 0, 0, 0])
    noisy_labels = torch.tensor([0, 0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, False, False])

    weights = ball_reliability_weights(
        assignment,
        noisy_labels,
        reliable,
        train_mask=train_mask,
        weighted_k_ref=3,
        weighted_floor=0.25,
    )

    assert torch.allclose(weights, torch.full((4,), 2.0 / 3.0), atol=1e-6)


def test_ball_reliability_weights_respects_floor_for_low_coverage():
    assignment = torch.tensor([0, 0, 0])
    noisy_labels = torch.tensor([0, 1, 1])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, False, False])

    weights = ball_reliability_weights(
        assignment,
        noisy_labels,
        reliable,
        train_mask=train_mask,
        weighted_k_ref=10,
        weighted_floor=0.25,
    )

    assert torch.allclose(weights, torch.full((3,), 0.25), atol=1e-6)


def test_ball_reliability_weights_high_coverage_high_margin_is_one():
    assignment = torch.tensor([0, 0, 0, 0])
    noisy_labels = torch.tensor([1, 1, 1, 0])
    reliable = torch.tensor([True])
    train_mask = torch.tensor([True, True, True, False])

    weights = ball_reliability_weights(
        assignment,
        noisy_labels,
        reliable,
        train_mask=train_mask,
        weighted_k_ref=3,
        weighted_floor=0.25,
    )

    assert torch.allclose(weights, torch.ones(4), atol=1e-6)


def test_ball_reliability_weights_unreliable_ball_stays_one():
    assignment = torch.tensor([0, 0])
    noisy_labels = torch.tensor([0, 1])
    reliable = torch.tensor([False])
    train_mask = torch.tensor([True, True])

    weights = ball_reliability_weights(
        assignment,
        noisy_labels,
        reliable,
        train_mask=train_mask,
        weighted_k_ref=3,
        weighted_floor=0.25,
    )

    assert torch.allclose(weights, torch.ones(2), atol=1e-6)


def test_weighted_soft_target_loss_equals_mean_when_weights_are_one():
    logits = torch.tensor([[2.0, -1.0], [0.2, 1.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.ones(2)

    loss = weighted_soft_target_loss(logits, targets, weights)
    expected = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss.backward()

    assert torch.allclose(loss, expected, atol=1e-6)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_knn_majority_vote_only_uses_train_neighbors():
    """kNN should only look at train nodes as candidate neighbors."""
    # 5 nodes: 0,1 are train (class 0), 2,3,4 are test (class 1)
    # node 4 is closest to test nodes 2,3 (class 1), but only train neighbors allowed
    embeddings = torch.tensor([
        [1.0, 0.0],  # train, class 0
        [0.9, 0.1],  # train, class 0
        [0.0, 1.0],  # test, class 1
        [0.1, 0.9],  # test, class 1
        [0.05, 0.95],  # test, class 1 - nearest to nodes 2,3
    ])
    noisy_labels = torch.tensor([0, 0, 1, 1, 1])
    train_mask = torch.tensor([True, True, False, False, False])

    corrected = knn_majority_vote(
        embeddings, noisy_labels, num_classes=2, k=2, train_mask=train_mask,
    )
    # node 4's kNN from train pool are nodes 0,1 (class 0), so corrected to 0
    assert corrected[4].item() == 0


def test_knn_majority_vote_tiny_train_set_no_crash():
    """With only 1 train node, actual_k=0, should return original labels."""
    embeddings = torch.randn(5, 4)
    noisy_labels = torch.tensor([0, 1, 2, 0, 1])
    train_mask = torch.tensor([True, False, False, False, False])

    corrected = knn_majority_vote(
        embeddings, noisy_labels, num_classes=3, k=5, train_mask=train_mask,
    )
    assert torch.equal(corrected, noisy_labels)


def test_knn_label_smoothing_tiny_train_set_no_crash():
    """With only 1 train node, should return one-hot (no neighbors to smooth with)."""
    embeddings = torch.randn(5, 4)
    noisy_labels = torch.tensor([0, 1, 2, 0, 1])
    train_mask = torch.tensor([True, False, False, False, False])

    soft = knn_label_smoothing(
        embeddings, noisy_labels, num_classes=3, k=5, train_mask=train_mask,
    )
    assert soft.shape == (5, 3)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(5), atol=1e-5)
