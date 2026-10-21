from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gblcss.data import inject_train_label_noise
from gblcss.experiment import make_smoke_graph

from gblcss import runner as cli


def test_named_groups_cover_supported_paper_methods():
    assert hasattr(cli, 'expand_methods'), 'Experiment groups must resolve to runnable methods'
    assert cli.expand_methods('structure') == ['gblcss', 'postel', 'als', 'sals', 'graphmlp', 'gsdn']
    assert cli.expand_methods('distillation') == ['gblcss', 'glnn_style', 'ffg2m', 'krd', 'vqgraph', 'gblcss_distill']
    assert cli.expand_methods('grouping') == ['gblcss', 'knn', 'kmeans']
    assert cli.expand_methods('matched,mlp_ce').count('mlp_ce') == 1
    assert cli.expand_methods('graphmlp') == ['graphmlp']


def test_validation_corruption_keeps_train_and_test_labels():
    graph, _ = make_smoke_graph(3)
    clean_validation = inject_train_label_noise(deepcopy(graph), 'uniform', 0.4, 3)
    noisy_validation = inject_train_label_noise(deepcopy(graph), 'uniform', 0.4, 3, noise_scope='train_val')
    assert torch.equal(clean_validation.noisy_y[graph.train_mask], noisy_validation.noisy_y[graph.train_mask])
    assert torch.equal(noisy_validation.noisy_y[graph.test_mask], graph.y[graph.test_mask])
    assert not torch.equal(noisy_validation.noisy_y[graph.val_mask], graph.y[graph.val_mask])
    repeated = inject_train_label_noise(deepcopy(graph), 'uniform', 0.4, 3, noise_scope='train_val')
    assert torch.equal(noisy_validation.noisy_y, repeated.noisy_y)


def test_nonstandard_protocol_config_is_rejected():
    config = cli.load_config(ROOT / 'configs/main/gblcss.yaml')
    cli.validate_main_config(config)
    config['split']['train'] = .3
    with pytest.raises(ValueError, match='split'):
        cli.validate_main_config(config)
    config = cli.load_config(ROOT / 'configs/main/gblcss.yaml')
    config['method']['leave_one_out'] = False
    with pytest.raises(ValueError, match='leave_one_out'):
        cli.validate_main_config(config)
