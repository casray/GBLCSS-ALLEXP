"""Paired X-only students with dataset-appropriate validation selection."""
import argparse
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import common as c
from metric_training import train_metric_mlp
from gblcss.data import set_seed


def expected_keys():
    selected=c.read(c.ROOT/'gate.json')['selected_datasets']
    return {(n,s,z,a) for n in c.CONFIG['datasets'] for s in c.SEEDS for z in c.conditions(n) for a in c.CONFIG['baseline_arms']+(c.CONFIG['candidate_arms'] if n in selected else [])}


def summarize():
    rows=[c.read(p) for p in sorted((c.ROOT/'students').glob('*/seed*/*/*/result.json'))]
    if not rows:return rows
    c.d.csv_write(c.ROOT/'student_results.csv',rows)
    frame=pd.DataFrame(rows);numeric=[k for k in frame.select_dtypes('number').columns if k!='seed']
    summary=frame.groupby(['dataset','noise','arm'])[numeric].agg(['mean','std','count']);summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(c.ROOT/'student_summary.csv',index=False)
    lookup={(r['dataset'],r['seed'],r['noise'],r['arm']):r for r in rows};pairs=[]
    for key,r in lookup.items():
        if r['arm'] not in ['R_TREE','X_TREE','BFS']:continue
        for ref in ['CE','OLD64','H_TREE','X_TREE','SHUFFLE_TREE']:
            other=lookup.get((*key[:3],ref))
            if other and ref!=r['arm']:
                pairs.append(dict(dataset=r['dataset'],seed=r['seed'],noise=r['noise'],arm=r['arm'],reference=ref,primary_metric=r['primary_metric'],delta_pp=100*(r['test_primary']-other['test_primary'])))
    c.d.csv_write(c.ROOT/'paired_student_deltas.csv',pairs)
    if pairs:
        pd.DataFrame(pairs).groupby(['dataset','noise','arm','reference','primary_metric']).delta_pp.agg(['mean','std','count']).reset_index().to_csv(c.ROOT/'paired_student_summary.csv',index=False)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');a=parser.parse_args()
    if c.read(c.ROOT/'audit/targets.json')['status']!='passed':raise ValueError('Target audit required')
    c.verify_manifest(c.ROOT/'source_manifest.json')
    c.verify_manifest(c.ROOT/'preparation_manifest.json')
    import metric_training
    files=[Path(__file__),Path(metric_training.__file__),c.ROOT/'common.py',c.ROOT/'PLAN.md',c.ROOT/'config.json',c.ROOT/'gate.json',c.ROOT/'source_manifest.json',c.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py',c.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py']
    c.freeze_manifest(c.ROOT/'student_source_manifest.json',files)
    keys=expected_keys();torch.set_num_threads(c.CONFIG['threads']);start=time.perf_counter()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'student_raw.log',encoding='utf-8')])
    c.write(c.ROOT/'student_status.json',dict(status='running',expected=len(keys)))
    try:
        with threadpool_limits(limits=c.CONFIG['threads']):
            for name in c.CONFIG['datasets']:
                primary=c.CONFIG['classification']['primary'].get(name,'accuracy')
                for seed in c.SEEDS:
                    for noise in c.conditions(name):
                        graph,protocol=c.load_case(name,seed,noise);tr=graph.train_index
                        parent=c.ROOT/'targets'/name/f'seed{seed}'/noise
                        if protocol!=c.read(parent/'protocol.json'):raise ValueError('Protocol changed')
                        c.verify_manifest(parent/'frozen_targets.json')
                        for arm in c.ARMS:
                            if (name,seed,noise,arm) not in keys:continue
                            folder=c.ROOT/'students'/name/f'seed{seed}'/noise/arm
                            path=parent/(arm+'.npz');digest=c.sha(path)
                            if (folder/'result.json').exists():
                                row=c.read(folder/'result.json')
                                if row['target_hash']!=digest or c.sha(folder/'logits.pt')!=row['logits_hash']:raise ValueError('Student cache changed')
                                continue
                            folder.mkdir(parents=True,exist_ok=True)
                            with np.load(path) as target:local=torch.from_numpy(target['target'].copy())
                            all_targets=torch.nn.functional.one_hot(graph.noisy_y,graph.num_classes).float();all_targets[tr]=local
                            config=dict(dataset=name,seed=seed,noise=noise,arm=arm,target_hash=digest,protocol=protocol,primary_metric=primary,training=c.CONFIG['training'],device=a.device,model_input='original X only',checkpoint='clean validation '+primary)
                            c.write(folder/'config.json',config);set_seed(seed);tick=time.perf_counter()
                            logging.info('START %s seed%d %s %s',name,seed,noise,arm)
                            if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                            result=train_metric_mlp(graph,all_targets,primary_metric=primary,device=a.device,**c.CONFIG['training'])
                            torch.save(result.logits,folder/'logits.pt');c.write(folder/'history.json',result.history)
                            row=dict(dataset=name,seed=seed,noise=noise,arm=arm,primary_metric=primary,test_primary=result.test_metrics[primary],validation_primary=result.val_metrics[primary],best_epoch=result.best_epoch,
                                elapsed_seconds=time.perf_counter()-tick,target_hash=digest,logits_hash=c.sha(folder/'logits.pt'),newly_trained=True,
                                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None)
                            for split,metrics in [('train',result.train_metrics),('val',result.val_metrics),('test',result.test_metrics)]:
                                row.update({split+'_'+k:v for k,v in metrics.items()})
                            c.write(folder/'result.json',row);summarize()
                            logging.info('DONE %s seed%d %s %s test_%s=%.6f seconds=%.2f',name,seed,noise,arm,primary,row['test_primary'],row['elapsed_seconds'])
                            del result,all_targets,local
        rows=summarize();actual={(r['dataset'],r['seed'],r['noise'],r['arm']) for r in rows}
        if actual!=keys or len(rows)!=len(keys):raise ValueError('Incomplete student keys')
        c.verify_manifest(c.ROOT/'student_source_manifest.json')
        c.verify_manifest(c.ROOT/'source_manifest.json')
        c.verify_manifest(c.ROOT/'preparation_manifest.json')
        c.write(c.ROOT/'student_status.json',dict(status='completed',student_runs=len(rows),expected=len(keys),elapsed_seconds=time.perf_counter()-start))
    except Exception as exc:
        c.write(c.ROOT/'student_status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
