"""Independent post-grid audit; never used by training or validation selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(data)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(read(ROOT/'progress.json')['status']=='complete', 'Grid incomplete')
    config = read(ROOT/'config.json')
    with (ROOT/'results.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows)==192,'Expected 192 scored cells')
    hashes = read(ROOT/'frozen_inputs.json')['sha256']
    for path, expected in hashes.items():
        require(sha(path)==expected,'Changed frozen source: '+path)
    deltas, initial, files = [], {}, {}
    for name in config['datasets']:
        assets = torch.load(ROOT/'inputs'/name/'assets.pt', weights_only=True)
        reference = read(ROOT/'inputs'/name/'uniform0.4.json')
        stored = torch.load(reference['raw_source'],map_location='cpu',weights_only=False)[0]
        clean = (stored['y'] if isinstance(stored,dict) else stored.y).long().numpy()
        test_index, val_index = assets['test'].numpy(), assets['valid'].numpy()
        for row in [r for r in rows if r['dataset']==name]:
            folder = Path(row['folder'])
            result = read(folder/'result.json')
            probability_path = folder/'probabilities.pt'
            probability = torch.load(probability_path,weights_only=True).numpy()
            predicted = probability.argmax(1)
            accuracy = float(np.mean(predicted[test_index]==clean[test_index]))
            delta = abs(100*accuracy-float(row['test_percent']))
            require(delta<1e-10,'Independent accuracy differs')
            require(abs(100*np.mean(predicted[val_index]==clean[val_index])-float(row['validation_percent']))<1e-10,
                    'Independent validation differs')
            deltas.append(delta)
            trace = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            counts = [entry['valid_correct'] for entry in trace]
            require(int(np.argmax(counts))+1==result['best_epoch'],'Wrong selected epoch')
            if row['phase']!='probes':
                require(len(trace)==(100 if row['route']=='R2' else 200),'Wrong update budget')
                initial.setdefault(name,set()).add(result['initial_hash'])
                n = len(assets['train'])
                rate = config['conditions'][row['condition']][1]
                for entry in trace:
                    count = max(1,int((1-rate*min(1.,entry['epoch']/10))*n)) if row['rule'] in ('T2','T3') else n
                    require(entry['selected_count']==count,'Wrong sample retention')
                state = torch.load(folder/'student.pt',weights_only=True)
                require(set(state)=={'layers.0.weight','layers.0.bias','layers.1.weight','layers.1.bias',
                                    'batch_norms.0.weight','batch_norms.0.bias','batch_norms.0.running_mean',
                                    'batch_norms.0.running_var','batch_norms.0.num_batches_tracked'},'Deployment includes extra state')
                if row['route']=='R2':
                    pre = torch.load(Path(result['pretrained_source'])/'model.pt',weights_only=True)
                    for key,value in state.items():
                        if not key.startswith('layers.1.'):
                            require(torch.equal(value,pre['backbone.'+key]),'R2 frozen encoder/BN changed')
                files[str(folder/'student.pt')] = sha(folder/'student.pt')
            files[str(probability_path)] = sha(probability_path)
        require(len(initial[name])==1,'Initial deployed parameters differ across arms')
    references=[]
    for name in config['datasets']:
        old = WORKSPACE/'experiments/simmlp_matched_backbone_seed0_20261009/runs'/name/'simmlp_matched/protocol.json'
        a,b=read(old),read(ROOT/'inputs'/name/'uniform0.4.json')
        for field in ('dataset','seed','noise_type','noise_rate','split_hash','noise_hash','num_nodes'):
            require(a[field]==b[field],f'SimMLP protocol differs: {name}/{field}')
        require(a['data_file']==b['raw_source'],'SimMLP feature source differs')
        references.append(dict(dataset=name,split_and_noise_match=True,source=str(old),sha256=sha(old)))
    output=dict(status='passed',metrics_recomputed=len(rows),max_accuracy_difference_pp=max(deltas),
                first_validation_maximum_verified=True,selection_counts_verified=True,update_budgets_verified=True,
                r2_weights_and_bn_exactly_frozen=True,initial_states_paired=True,
                frozen_sources_verified=len(hashes),artifact_sha256=files,simmlp_references=references,
                scope='one protocol seed; no statistical significance or independent SSL repeat')
    (ROOT/'independent_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='artifact_sha256'},ensure_ascii=False))


if __name__=='__main__':
    main()
