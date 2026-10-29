import json
from pathlib import Path
import pytest
import torch
from contract import tasks,validate_target,factorial_effects

def test_cross_matrix_coverage():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    rows=tasks(cfg)
    assert len(rows)==32
    ids={(r['dataset'],r['seed'],r['features'],r['targets'],r['condition']) for r in rows}
    assert len(ids)==32
    assert all(r['features']!=r['targets'] for r in rows)
    assert sum(r['dataset']=='coauthorcs' and r['features']=='grace' for r in rows)==8
    assert sum(r['dataset']=='amazonpho' and r['features']=='graphecl' for r in rows)==8

def test_target_provenance_guard():
    t=torch.tensor([[.4,.6],[.7,.3]])
    validate_target(t,t.clone(),2,2)
    with pytest.raises(ValueError,match='source'):validate_target(t,t.flip(1),2,2)

@pytest.mark.parametrize('bad',[torch.tensor([[float('nan'),0.],[0.,1.]]),torch.tensor([[.4,.4],[0.,1.]]),torch.ones(3,2)/2])
def test_invalid_targets_rejected(bad):
    with pytest.raises(ValueError):validate_target(bad,bad,2,2)

def test_factorial_signs_and_interaction():
    # A-feature changes +3 with B targets; B-feature changes -4 with B targets.
    e=factorial_effects(80.,83.,85.,81.)
    assert e['target_B_minus_A_on_A']==3
    assert e['target_B_minus_A_on_B']==-4
    assert e['feature_B_minus_A_on_A_target']==5
    assert e['feature_B_minus_A_on_B_target']==-2
    assert e['interaction']==-7
