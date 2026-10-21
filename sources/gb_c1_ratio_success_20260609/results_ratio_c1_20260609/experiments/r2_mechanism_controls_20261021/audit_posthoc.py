"""Independent post-score checks. No training changes or model selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'representation_supervision_20261021'
sys.path.insert(0, str(ROOT))
from kernels import loss_for


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    assert read(ROOT/'progress.json')['status']=='complete'
    with (ROOT/'results.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==156
    frozen=read(ROOT/'frozen_inputs.json')['sha256']
    for path, expected in frozen.items():
        assert sha(path)==expected, path
    max_difference=0.
    for name in ('cora','citeseer','corafull'):
        assets=torch.load(PREVIOUS/'inputs'/name/'assets.pt',weights_only=True)
        protocol=read(PREVIOUS/'inputs'/name/'clean.json')
        raw=torch.load(protocol['raw_source'],weights_only=False,map_location='cpu')[0]
        y=(raw['y'] if isinstance(raw,dict) else raw.y).numpy()
        ids=assets['test'].numpy()
        for row in [r for r in rows if r['dataset']==name]:
            p=torch.load(Path(row['folder'])/'probabilities.pt',weights_only=True).numpy()
            actual=100*int((p[ids].argmax(1)==y[ids]).sum())/len(ids)
            delta=abs(actual-float(row['test_percent']))
            assert delta<1e-10
            max_difference=max(delta,max_difference)
        groups=torch.load(ROOT/'inputs'/name/'groups.pt',weights_only=True)
        h=assets['h'][assets['train']].numpy().astype(np.float64)
        h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
        support=groups['knn_support'].numpy()
        knn=groups['knn'].numpy()
        sample=np.unique(np.linspace(0,len(h)-1,32).astype(int))
        with threadpool_limits(limits=4):
            scores=h[sample]@h.T
        for n,score in zip(sample,scores):
            m=int(support[n])
            if not m:
                continue
            score[n]=-np.inf
            chosen=knn[n,:m]
            excluded=np.ones(len(h),dtype=bool)
            excluded[chosen]=False
            assert float(np.max(score[excluded])-np.min(score[chosen]))<1e-5, 'Non-nearest neighbors'
            assert np.all(np.diff(score[chosen])<1e-5), 'Neighbor order not descending'
        shuffled=ROOT/'encoders'/name/'SHUFFLED_H'
        trace=[json.loads(line) for line in (shuffled/'epochs.jsonl').read_text().splitlines()]
        assert len(trace)==100
        selected=int(np.argmin([r['evaluation_alignment'] for r in trace]))+1
        assert selected==read(shuffled/'result.json')['selected_epoch']
        assert read(shuffled/'result.json')['labels_used'] is False
        assert read(ROOT/'encoders'/name/'RANDOM_ENCODER/result.json')['updates']==0
    logits=torch.tensor([[.2,-.3,1.],[-1.,2.,.3]],requires_grad=True)
    z=torch.tensor([0,2])
    target=F.one_hot(z,3).float()
    boot=loss_for(logits,z,target,'BOOT')
    bg,=torch.autograd.grad(boot,logits,retain_graph=True)
    cg,=torch.autograd.grad(F.cross_entropy(logits,z),logits)
    assert torch.allclose(bg,.8*cg,atol=1e-7)
    by={(r['dataset'],r['condition'],r['route'],r['rule']):float(r['test_percent']) for r in rows}
    paired=list(csv.DictReader((ROOT/'paired_deltas.csv').open(encoding='utf-8-sig')))
    for r in paired:
        assert abs(float(r['delta_pp'])-(float(r['a_test'])-float(r['b_test'])))<1e-10
    expected={}
    for rule in ('CE','GB','LS','GCE','BOOT','MASS','KNN','KM','RANDOM'):
        values=[by[n,c,'REAL',rule] for n in ('cora','citeseer','corafull')
                for c in ('uniform0.4','uniform0.6','pair0.4')]
        expected[rule]=float(np.mean(values))
    summary=list(csv.DictReader((ROOT/'summary.csv').open(encoding='utf-8-sig')))
    for r in summary:
        if r['scope']=='noisy' and r['route']=='REAL':
            assert abs(float(r['test_mean_percent'])-expected[r['rule']])<1e-10
    result=dict(status='passed',independent_accuracy_rows=156,max_accuracy_difference_pp=max_difference,
                paired_differences_checked=len(paired),independent_noisy_macro_means=expected,
                geometry_neighbor_queries_checked=96,neighbor_boundary_tolerance=1e-5,
                label_free_shuffled_checkpoint_minimum_verified=True,random_encoder_zero_updates=True,
                boot_gradient_equals_0p8_ce_verified=True,frozen_sources=len(frozen),
                source_sha256=sha(__file__),analytical_script_sha256=sha(ROOT/'analyze.py'),
                scope='posthoc validation only; seed0, no new fitting, no significance test')
    (ROOT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
