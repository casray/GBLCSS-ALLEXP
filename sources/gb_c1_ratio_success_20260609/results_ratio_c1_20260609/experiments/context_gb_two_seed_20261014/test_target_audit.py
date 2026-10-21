import torch
from target_audit import loo_targets, clean_evidence


def test_self_vote_removed_and_ties_keep_distribution():
    # Removing class 0 leaves a 1/2 tie, while removing 1 leaves class 0 majority.
    labels = torch.tensor([0, 0, 1, 2])
    target, alpha = loo_targets(labels, [0]*4, [0, 1, 2, 3], 3)
    torch.testing.assert_close(alpha[:2], torch.zeros(2))
    torch.testing.assert_close(target[2], torch.tensor([5/12, .5, 1/12]))
    torch.testing.assert_close(alpha[2], torch.tensor(.5))


def test_heldout_labels_cannot_contribute_to_training_votes():
    first, first_alpha = loo_targets([0, 0, 1, 2], [0]*4, [0, 1, 2], 3)
    second, second_alpha = loo_targets([0, 0, 1, 1], [0]*4, [0, 1, 2], 3)
    torch.testing.assert_close(first[:3], second[:3])
    torch.testing.assert_close(first_alpha, second_alpha)
    torch.testing.assert_close(first[0], torch.tensor([.875, .125, 0.]))


def test_two_member_training_ball_falls_back_to_observed_label():
    target, alpha = loo_targets([1, 2, 0], [0]*3, [0, 1], 3)
    torch.testing.assert_close(target, torch.nn.functional.one_hot(torch.tensor([1, 2, 0]), 3).float())
    assert not alpha.any()


def test_clean_evidence_excludes_self_and_heldout_votes():
    support, margin = clean_evidence([0,0,1,1], [0]*4, [0,1,2], 2)
    torch.testing.assert_close(support, torch.tensor([2,2,2,0]))
    torch.testing.assert_close(margin, torch.tensor([0.,0.,-1.,0.]))
