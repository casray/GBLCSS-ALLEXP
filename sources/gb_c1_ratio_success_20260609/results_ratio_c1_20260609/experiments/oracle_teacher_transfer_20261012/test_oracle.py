from copy import deepcopy

import numpy as np
import pytest
import torch

import run_oracle as run


def toy():
    clean = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    noisy = clean.clone()
    noisy[[1, 4]] = 1-noisy[[1, 4]]
    train = torch.arange(6)
    return train, clean, noisy


def test_interventions_retain_only_original_train_and_match_classes():
    train, clean, noisy = toy()
    b, labels_b = run.intervention(train, clean, noisy, 2, 'B_all_clean')
    c, labels_c = run.intervention(train, clean, noisy, 2, 'C_oracle_filter')
    d, labels_d = run.intervention(train, clean, noisy, 2, 'D_matched_clean')
    assert torch.equal(b, train)
    assert torch.equal(labels_b[train], clean[train])
    assert torch.equal(c, torch.tensor([0, 2, 3, 5]))
    assert set(d.tolist()).issubset(set(train.tolist()))
    assert torch.equal(run.class_counts(labels_c,c,2),run.class_counts(labels_d,d,2))
    assert d.tolist() == sorted(d.tolist())
    assert torch.equal(run.intervention(train,clean,noisy,2,'D_matched_clean')[0],d)


def test_intervention_does_not_depend_on_nontraining_true_labels():
    train, clean, noisy = toy()
    changed = clean.clone()
    changed[6:] = 1234
    for arm in run.ARMS:
        first = run.intervention(train,clean,noisy,2,arm)
        second = run.intervention(train,changed,noisy,2,arm)
        assert all(torch.equal(a,b) for a,b in zip(first,second))


def test_sanitization_hides_test_and_discarded_labels():
    train, clean, noisy = toy()
    selected, labels = run.intervention(train,clean,noisy,2,'C_oracle_filter')
    visible = run.sanitized_targets(selected,labels,torch.tensor([6,7]),clean)
    assert torch.equal(visible[6:8],clean[6:8])
    assert (visible[torch.tensor([1,4,8,9])] == -1).all()
    assert labels[6:].sum() == 0


def test_all_clean_effective_inputs_ignore_original_noise_rate():
    train, clean, noisy = toy()
    different = clean.clone()
    different[0] = 1
    selected,a = run.intervention(train,clean,noisy,2,'B_all_clean')
    selected_b,b = run.intervention(train,clean,different,2,'B_all_clean')
    val = torch.tensor([6,7])
    h1 = run.effective_hash(selected,a,val,run.sanitized_targets(selected,a,val,clean))
    h2 = run.effective_hash(selected_b,b,val,run.sanitized_targets(selected_b,b,val,clean))
    assert h1 == h2


def test_missing_class_is_reported_and_not_silently_filled():
    with pytest.raises(ValueError,match='missing prototype'):
        run.intervention(torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([0,0]),2,'C_oracle_filter')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU ERASE preprocessing check')
def test_erase_preprocessing_uses_only_selected_annotations():
    train, clean, noisy = toy()
    selected, labels = run.intervention(train,clean,noisy,2,'C_oracle_filter')
    mask = torch.zeros(len(clean),dtype=torch.bool)
    mask[selected] = True
    edge = torch.tensor([[0,2,3,5,1,4,6,7],[2,0,5,3,4,1,7,6]])
    data = run.previous.GraphData('toy',torch.ones((len(clean),3)),edge,
              run.sanitized_targets(selected,labels,torch.tensor([6,7]),clean),mask,
              torch.zeros(len(clean),dtype=torch.bool),torch.zeros(len(clean),dtype=torch.bool),noisy_y=labels)
    data = run.previous.gpu_data(data)
    first, adjacency = run.previous.erase_preprocess(data,.6,2)
    altered = deepcopy(data)
    altered.noisy_y[~mask.cuda()] = 1
    second, _ = run.previous.erase_preprocess(altered,.6,2)
    assert torch.equal(first,second)
    assert first[~mask.cuda()].abs().sum() == 0
    assert adjacency.shape == (len(clean),len(clean))
