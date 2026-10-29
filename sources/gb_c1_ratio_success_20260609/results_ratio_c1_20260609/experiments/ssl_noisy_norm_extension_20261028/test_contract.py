import json
from pathlib import Path
from contract import tasks,noisy_gate

def test_exact_missing_noisy_cells():
 c=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8-sig'));r=tasks(c)
 assert len(r)==36 and len({tuple(x.values()) for x in r})==36
 assert {x['dataset'] for x in r}=={'cora','citeseer','corafull'}
 assert all(x['condition']!='clean' and x['scale']=='unit' for x in r)

def test_noise_gate_positive_and_size():
 assert noisy_gate([.8,.6]) and not noisy_gate([.1,.2])

def test_noise_gate_both_seeds():
 assert not noisy_gate([2.,-.1])
