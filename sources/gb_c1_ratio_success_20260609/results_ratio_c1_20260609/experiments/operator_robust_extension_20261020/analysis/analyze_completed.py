"""Descriptive completion audit; does not train or select new models."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np


def main(root,source,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(source)))
    from experiment import build_jobs, paired_summary, source_signature, atomic_json, file_hash
    config=json.loads((root/'config.json').read_text(encoding='utf8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    jobs=json.loads((root/'jobs.json').read_text(encoding='utf8'))
    assert jobs==build_jobs(config)
    assert file_hash(root/'config.json')==protocol['config_sha256']
    assert source_signature(source)[1]==protocol['source_sha256']
    rows=[json.loads(p.read_text(encoding='utf8')) for p in sorted((root/'results').glob('*.json'))]
    assert len(rows)==len(jobs)==192
    assert {r['job']['id'] for r in rows}=={j['id'] for j in jobs}
    audit=json.loads((root/'PREDICTION_AUDIT.json').read_text(encoding='utf8'))
    assert audit['audited']==192 and audit['status']=='passed'
    groups={}
    for r in rows:
        j=r['job'];groups.setdefault((j['dataset'],j['noise'],j['variant']),[]).append(r)
    with (root/'summary.csv').open(encoding='utf8',newline='') as f:summary=list(csv.DictReader(f))
    assert len(summary)==len(groups)==64
    for s in summary:
        rr=groups[s['dataset'],s['noise'],s['variant']]
        values=[100*r['test_accuracy'] for r in rr]
        assert len(rr)==int(s['n'])==3
        assert abs(statistics.mean(values)-float(s['accuracy_mean']))<1e-10
        assert abs(statistics.stdev(values)-float(s['accuracy_std']))<1e-10
        assert abs(statistics.mean(100*r['macro_f1'] for r in rr)-float(s['macro_f1_mean']))<1e-10
    pairs=paired_summary(rows)
    with (root/'paired.csv').open(encoding='utf8',newline='') as f:stored=list(csv.DictReader(f))
    assert len(pairs)==len(stored)==72
    for a,b in zip(pairs,stored):
        assert (a['dataset'],a['noise'],a['comparison'])==(b['dataset'],b['noise'],b['comparison'])
        assert a['delta_pp']==json.loads(b['delta_pp'])
    positive={('cora','u0.6'),('citeseer','u0.4'),('citeseer','u0.6'),('pubmed','u0.6')}
    def pair(ds,no,comparison):return next(r for r in pairs if (r['dataset'],r['noise'],r['comparison'])==(ds,no,comparison))
    def passes(r):return r['delta_pp_mean']>=1 and sum(x>0 for x in r['delta_pp'])>=2
    plugin=sum(passes(pair(ds,no,'tfr_S_shared-tfr_P_ind')) for ds,no in positive)
    projection=sum(passes(pair(ds,no,'tfr_S_shared-tfr_P_shared')) for ds,no in positive)
    dose=sum(passes(pair(ds,no,'ce_M_shared-ce_P_shared')) for ds,no in positive)
    repair=[]
    for no in ['clean','u0.4']:
        m=pair('corafull',no,'ce_M_shared-ce_P_shared')
        s=pair('corafull',no,'ce_M_shared-ce_S_shared')
        repair.append(dict(noise=no,m_minus_p=m['delta_pp_mean'],m_minus_s=s['delta_pp_mean'],
                           passes=m['delta_pp_mean']>=-1 and s['delta_pp_mean']>0))
    source_checks=[];endpoint=[];epochs=[]
    for ds in sorted({j['dataset'] for j in jobs}):
        folder=root/'cache'/ds
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf8'));old=Path(meta['old_directory'])
        assert file_hash(old/'metadata.json')==meta['original_metadata_sha256']
        for name,item in meta['original_cache_files'].items():
            assert file_hash(old/name)==item['sha256'];source_checks.append(dict(dataset=ds,file=name,sha256=item['sha256']))
        for name,digest in meta['new_files'].items():assert file_hash(folder/name)==digest
        assert file_hash(meta['original_metadata']['source'])==meta['original_metadata']['source_sha256']
        y=np.load(old/'y.npy')
        for r in rows:
            j=r['job']
            if j['dataset']!=ds:continue
            epochs.append(dict(**j,family=r['variant_spec']['family'],best_epoch=r['best_primary_epoch'],stop_epoch=r['epochs']))
            with np.load(root/'results'/r['prediction_file']) as saved:
                tr=saved['train_indices'];z=saved['train_noisy_labels'];p=saved['train_prediction'];flipped=z!=y[tr]
                endpoint.append(dict(**j,flipped_count=int(flipped.sum()),unflipped_count=int((~flipped).sum()),
                    fit_wrong_label=float(np.mean(p[flipped]==z[flipped])) if flipped.any() else None,
                    recover_true_label=float(np.mean(p[flipped]==y[tr][flipped])) if flipped.any() else None,
                    unflipped_accuracy=float(np.mean(p[~flipped]==y[tr][~flipped])) if (~flipped).any() else None))
    boundary=[r for r in epochs if r['family']=='ce' and (r['dataset'],r['noise']) in positive]
    decision=dict(plugin_positive_cells=plugin,projection_positive_cells=projection,plugin_gate=plugin>=2,
                  dose_positive_cells=dose,corafull_repair=repair,dose_gate=dose>=2 and all(x['passes'] for x in repair),
                  ce_best_epoch5=sum(x['best_epoch']==5 for x in boundary),ce_positive_total=len(boundary),
                  decision='Pause current strong-plugin/automatic-granularity expansion; dense-validation closure proposed, not executed')
    atomic_json(output/'FINAL_DIAGNOSTICS.json',dict(verification_status='ANALYZED; saved artifacts verified, no independent retraining',
        source_sha256=protocol['source_sha256'],config_sha256=protocol['config_sha256'],audited=192,
        verified_summary_groups=64,verified_pair_rows=72,decision=decision,source_checks=source_checks,
        epoch_distribution=epochs,endpoint_predictions=endpoint))
    for name,data in [('epoch_distribution.csv',epochs),('endpoint_predictions.csv',endpoint)]:
        with (output/name).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    for name in ['PREDICTION_AUDIT.json','summary.csv','paired.csv','status.json','protocol.json']:
        shutil.copy2(root/name,output/name)
    print(json.dumps(dict(audited=192,summary_groups=64,paired_rows=72,decision=decision)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(args.run_root,args.source,args.output)
