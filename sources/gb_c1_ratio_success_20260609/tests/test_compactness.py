import torch

from gb_stage_b.balls import generate_angular_balls, reliable_ball_mask
from gb_stage_b.compactness import compactness_loss


def test_compactness_loss_is_finite_and_backprops_to_z_only():
    z = torch.randn(24, 8, requires_grad=True)
    balls = generate_angular_balls(z.detach(), min_size=3, quality_threshold=0.4)
    reliable = reliable_ball_mask(balls, min_size=3, max_size=100, ad_percentile=100.0)

    loss, stats = compactness_loss(z, balls.assignment, balls.centers, reliable)

    assert torch.isfinite(loss)
    assert stats["covered_nodes"] > 0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_compactness_loss_zero_when_no_reliable_balls():
    z = torch.randn(10, 4, requires_grad=True)
    balls = generate_angular_balls(z.detach(), min_size=2, quality_threshold=0.4)
    reliable = torch.zeros(balls.num_balls, dtype=torch.bool)

    loss, stats = compactness_loss(z, balls.assignment, balls.centers, reliable)

    assert loss.item() == 0.0
    assert stats["covered_nodes"] == 0


def test_compactness_loss_detaches_centers():
    z = torch.randn(12, 4, requires_grad=True)
    assignment = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    centers = torch.randn(3, 4, requires_grad=True)
    reliable = torch.tensor([True, False, True])

    loss, _ = compactness_loss(z, assignment, centers, reliable)
    loss.backward()

    assert z.grad is not None
    assert centers.grad is None


def test_compactness_loss_rejects_bad_shapes():
    z = torch.randn(5, 4)
    centers = torch.randn(2, 4)

    try:
        compactness_loss(z, torch.tensor([0, 0, 1]), centers, torch.ones(2, dtype=torch.bool))
    except ValueError as exc:
        assert "assignment" in str(exc)
    else:
        raise AssertionError("expected ValueError")
