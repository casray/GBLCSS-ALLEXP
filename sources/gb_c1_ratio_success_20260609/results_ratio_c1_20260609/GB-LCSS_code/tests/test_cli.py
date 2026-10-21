import json

import pandas as pd

from gblcss.cli import save_tables
from gblcss.experiment import load_config, make_smoke_graph
from gblcss.runner import ROOT, run_method


def test_actual_training_rows_generate_tables(tmp_path):
    graph, representation = make_smoke_graph(0)
    config = load_config(ROOT / 'configs/main/gblcss.yaml')
    config['training']['epochs'] = 2
    rows = []
    for method in ['gblcss', 'mlp_ce']:
        result, _ = run_method(method, graph, representation, config,
                              seed=0, noise_rate=.4, device='cpu')
        rows.append(dict(suite='target_controls', task='train', dataset='synthetic',
                         method=method, seed=0, variant='default', noise_type='uniform',
                         noise_rate=.4, validation_protocol='clean',
                         test_accuracy=result.test_accuracy, best_epoch=result.best_epoch))
    save_tables(rows, tmp_path)
    table = pd.read_csv(tmp_path / 'tables/aggregate.csv')
    assert len(table) == 2
    assert table.seeds.eq(1).all()
    assert (tmp_path / 'tables/table5.csv').is_file()
    json.dumps(rows, allow_nan=False)
