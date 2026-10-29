import json
from pathlib import Path
import pytest
from contract import tasks,effects,eligible

def test_new_matrix_contains_only_two_missing_cells():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8-sig'))
    rows=tasks(cfg)
    assert len(rows)==64 and len({tuple(r.values()) for r in rows})==64
    assert {(r['scale'],r['initialization']) for r in rows}=={('raw','unified'),('unit','legacy')}
    assert {(r['dataset'],r['seed']) for r in rows}=={(d,s) for d in cfg['datasets'] for s in [0,1]}

def test_factorial_additive_effects():
    r=effects(10,12,13,15)
    assert r==dict(normalization_legacy=3,normalization_unified=3,initialization_raw=2,initialization_unit=2,normalization_main=3,initialization_main=2,interaction=0,diagonal=5)

def test_factorial_interaction_and_exact_diagonal_decomposition():
    r=effects(10,11,12,17)
    assert r['interaction']==4 and r['normalization_legacy']==2 and r['normalization_unified']==6
    assert r['normalization_main']+r['initialization_main']==r['diagonal']==7

@pytest.mark.parametrize('noise,clean,want',[([1.,1.],[-.5,-.5],True),([.9,-.1],[0.,0.],False),([.2,.3],[0.,0.],False),([1.,1.],[-.7,-.7],False)])
def test_validation_gate_retains_clean_and_seed_requirements(noise,clean,want):
    assert eligible(noise,clean) is want
