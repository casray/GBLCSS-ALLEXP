"""Independent read-only race, aggregate and CPU checkpoint audit."""
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = WORK / 'experiments/r2_integrated_20261026'
HISTORY = WORK / 'experiments/sc_support_channel_pilot_20261023/diagnostics/e1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def rank(r, joint=True):
    base = (-r['valid_correct']/r['valid_total'], r['valid_nll'],
            r['mean_lambda'], -r['support'], r['best_epoch'])
    a = r['arm']
    return base + ((0 if a=='CE' else 1 if '_L' in a else 2, a),) if joint else base


def main():
    assert read(ROOT/'complete.json')['audit']=='passed'
    assert read(ROOT/'audit.json')['status']=='passed'
    assert not (ROOT/'run.lock').exists()
    for manifest in ('sources.json', 'inputs.json'):
        for p, h in read(ROOT/manifest).items():
            assert sha(p)==h, p
    frozen = read(ROOT/'selections_frozen.json')
    assert len(frozen['choices'])==60
    assert sha(ROOT/'reuse_manifest.json')==frozen['reuse_manifest_sha256']
    for category in ('choices', 'races'):
        for p, h in frozen[category].items():
            assert sha(ROOT/p)==h, p
    assert read(ROOT/'testing_started.json')['started'] >= frozen['frozen']
    refs = read(ROOT/'reuse_manifest.json')
    logcache = {}
    prefix_checks = race_checks = selection_checks = 0
    for p in (ROOT/'races').rglob('*.json'):
        if p.stem not in ('GLOBAL','JOINT'):
            continue
        race = read(p)
        joint = race['kind']=='JOINT'
        cid = f"{race['dataset']}/seed{race['seed']}/R2/{race['condition']}"
        expected = None
        for step in race['rounds']:
            scores = step['scores']
            prefix = step['prefix']
            if expected is not None:
                assert [r['arm'] for r in scores]==expected+(['CE'] if prefix==200 else [])
            for r in scores:
                folder = refs[cid+'/'+r['arm']]['folder']
                if folder not in logcache:
                    logcache[folder] = [json.loads(x) for x in (Path(folder)/'epochs.jsonl').read_text().splitlines()]
                logs = logcache[folder][:prefix]
                assert [x['epoch'] for x in logs]==list(range(1,prefix+1))
                best = max(logs, key=lambda x:x['validation_correct'])
                assert (best['epoch'],best['validation_correct'],best['validation_nll']) == (r['best_epoch'],r['valid_correct'],r['valid_nll'])
                assert r['epoch']==prefix
                prefix_checks += 1
            ordered = sorted(scores, key=lambda r:rank(r,joint))
            if prefix<200:
                count = (14 if prefix==25 else 4) if joint else (6 if prefix==25 else 2)
                expected = [r['arm'] for r in ordered[:count]]
                if 'S32_L1.0' not in expected:
                    expected[-1]='S32_L1.0'
                assert len(set(expected))==count
            else:
                expected=[ordered[0]['arm']]
            assert step['survivors']==expected
        assert race['selected']==expected[0]
        race_checks += 1
    for p in frozen['choices']:
        choice = read(ROOT/p)
        fixed = choice['fixed_scores']
        cid = f"{choice['dataset']}/seed{choice['seed']}/R2/{choice['condition']}"
        glob = read(ROOT/'races'/cid/'GLOBAL.json')
        joint = read(ROOT/'races'/cid/'JOINT.json')
        fx = min(fixed,key=rank)
        ag = next(r for r in glob['rounds'][-1]['scores'] if r['arm']==glob['selected'])
        expected = dict(FIXED_GLOBAL=min([r for r in fixed if r['arm']=='CE' or '_L' in r['arm']],key=rank)['arm'],
                        FIXED_EXPANDED=fx['arm'], ADAPTIVE_GLOBAL=glob['selected'],
                        ADAPTIVE_EXPANDED=joint['selected'], SINGLE_MODULE_VAL=min([fx,ag],key=rank)['arm'])
        for method, arm in expected.items():
            assert choice['choices'][method]==cid+'/'+arm
            selection_checks += 1
    selected, paired = rows('selected_results.csv'), rows('paired_differences.csv')
    keys = ('dataset','seed','condition','method')
    lookup = {tuple(r[k] for k in keys):r for r in selected}
    assert len(lookup)==len(selected)==600
    metrics = ('test_accuracy','test_macro_f1','validation','valid_macro_f1')
    for r in paired:
        key = tuple(r[k] for k in keys[:-1])
        a,b = lookup[key+(r['method'],)],lookup[key+(r['reference'],)]
        for m in metrics:
            assert abs(float(r[m+'_diff_pp'])-100*(float(a[m])-float(b[m])))<1e-10
    summary_checks = 0
    for prefix,seeds in [('',(0,1,2)),('seeds12_',(1,2))]:
        for source,filename,identities,ms in [(selected,'summary.csv',('dataset','condition','method'),metrics),
                (paired,'paired_summary.csv',('dataset','condition','method','reference'),tuple(m+'_diff_pp' for m in metrics))]:
            for r in rows(prefix+filename):
                matched=[x for x in source if int(x['seed']) in seeds and all(
                    (k=='dataset' and r[k]=='ALL') or (k=='condition' and r[k]=='NOISY_MACRO' and x[k]!='clean') or x[k]==r[k] for k in identities)]
                assert {int(x['seed']) for x in matched}==set(seeds)
                for m in ms:
                    values=[np.mean([float(x[m]) for x in matched if int(x['seed'])==s]) for s in seeds]
                    assert abs(np.mean(values)-float(r[m+'_mean']))<1e-10
                    assert abs(np.std(values,ddof=1)-float(r[m+'_std_ddof1']))<1e-10
                summary_checks += 1
    project = WORK.parents[1]
    files = {'cora':project/'external/NoisyGL/data/Cora/processed/data.pt',
             'citeseer':project/'external/NoisyGL/data/CiteSeer/processed/data.pt',
             'corafull':project/'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
             'coauthorcs':project/'external/NoisyGL/data/CS/processed/data.pt',
             'amazonpho':project/'external/NoisyGL/data/Photo/processed/data.pt'}
    samples=[]
    for name,path in files.items():
        data=torch.load(path,map_location='cpu',weights_only=False)[0]
        x,y=(data['x'],data['y']) if isinstance(data,dict) else (data.x,data.y)
        x,y=x.float(),y.long()
        for seed in (0,1,2):
            geo=torch.load(HISTORY/name/f'seed{seed}/geometry.pt',map_location='cpu',weights_only=True)
            for condition in ('clean','uniform0.6'):
                r=lookup[(name,str(seed),condition,'ADAPTIVE_EXPANDED')]
                folder=Path(r['folder'])
                ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                p=torch.load(folder/'probabilities.pt',map_location='cpu',weights_only=True)
                assert sha(ck['encoder_path'])==ck['encoder_file_sha256']
                enc=torch.load(ck['encoder_path'],map_location='cpu',weights_only=True)
                h=F.relu(F.linear(x,enc['backbone.layers.0.weight'],enc['backbone.layers.0.bias']))
                b='backbone.batch_norms.0.'
                h=F.batch_norm(h,enc[b+'running_mean'],enc[b+'running_var'],enc[b+'weight'],enc[b+'bias'],training=False,eps=1e-5)
                q=F.linear(h,ck['model']['weight'],ck['model']['bias']).softmax(1)
                err=float((q-p).abs().max())
                te=geo['test'];pred=p[te].argmax(1)
                assert err<1e-5 and torch.equal(q[te].argmax(1),pred)
                assert abs(accuracy_score(y[te],pred)-float(r['test_accuracy']))<1e-12
                assert abs(f1_score(y[te],pred,labels=np.arange(p.shape[1]),average='macro',zero_division=0)-float(r['test_macro_f1']))<1e-12
                samples.append(dict(dataset=name,seed=seed,condition=condition,cpu_max_error=err))
    report=dict(status='passed',race_checks=race_checks,prefix_checks=prefix_checks,selection_checks=selection_checks,
                paired_checks=len(paired),summary_checks=summary_checks,reload_samples=samples,
                max_cpu_error=max(r['cpu_max_error'] for r in samples),audit_source_sha256=sha(__file__))
    (HERE/'PARENT_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='reload_samples'},indent=2))


if __name__=='__main__':
    main()
