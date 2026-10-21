import pytest
import torch

import run_recovery as run


def toy():
    train = torch.tensor([4, 1, 3, 0, 2])
    observed = torch.tensor([0, 1, 0, 1, 0, 1])
    probabilities = torch.tensor([[.9, .1], [.8, .2], [.9, .1], [.7, .3], [.6, .4], [.5, .5]])
    return train, observed, probabilities


def test_observed_probability_not_maximum_probability_controls_ranking():
    train, labels, probabilities = toy()
    selected = run.select_suspects(train, labels, probabilities)
    assert selected.tolist() == [1]


def test_stable_ties_and_ceiling_coverage():
    train, labels, probabilities = toy()
    probabilities[:] = .5
    assert run.select_suspects(train, labels, probabilities, .21).tolist() == [4, 1]


def test_correction_and_deletion_share_marked_nodes_and_preserve_order():
    train, labels, probabilities = toy()
    e, corrected, marked_e = run.realistic_intervention(train, labels, probabilities, 'E_correct')
    f, retained, marked_f = run.realistic_intervention(train, labels, probabilities, 'F_delete')
    assert torch.equal(marked_e, marked_f)
    assert torch.equal(e, train)
    assert f.tolist() == [4, 3, 0, 2]
    assert corrected[1] == 0
    assert corrected[0] == labels[0]
    assert retained[1] == 0 and retained[5] == 0
    assert corrected[5] == 0


def test_nontraining_labels_and_probabilities_cannot_change_selection():
    train, labels, probabilities = toy()
    first = run.realistic_intervention(train, labels, probabilities, 'E_correct')
    labels[5] = 12345
    probabilities[5] = torch.tensor([float('nan'), float('nan')])
    second = run.realistic_intervention(train, labels, probabilities, 'E_correct')
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_invalid_coverage_and_probability_are_rejected():
    train, labels, probabilities = toy()
    with pytest.raises(ValueError):
        run.select_suspects(train, labels, probabilities, 0.)
    probabilities[1] = float('nan')
    with pytest.raises(ValueError):
        run.select_suspects(train, labels, probabilities)


def test_true_labels_never_enter_realistic_intervention_and_fit_hides_test():
    train, labels, probabilities = toy()
    selected, permitted, _ = run.realistic_intervention(train, labels, probabilities, 'E_correct')
    true_labels = 1-labels
    visible = run.oracle.sanitized_targets(selected, permitted, torch.empty(0, dtype=torch.long), true_labels)
    assert torch.equal(visible[selected], permitted[selected])
    assert visible[5] == -1
    changed_truth = true_labels.clone()
    changed_truth[:] = 12345
    visible_again = run.oracle.sanitized_targets(selected, permitted, torch.empty(0, dtype=torch.long), changed_truth)
    assert torch.equal(visible, visible_again)
