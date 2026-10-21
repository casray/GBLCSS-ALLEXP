"""Matched seed0 pilot: change only supervision neighborhoods, reuse original MLP."""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_screen as screen
from run_students import TRAINING, full_targets, aggregate
from gblcss.data import inject_train_label_noise, set_seed
from gblcss.training import train_x_only_mlp

ROOT=screen.ROOT;d=screen.d
ARMS=['CE','OLD64','SQRT_SUPPORT','KM_MATCH','KNN_MATCH']


def summarize():
    rows=[json.loads(f.read_text(encoding='utf-8')) for f in (ROOT/'students').glob('*/seed*/*/*/result.json')]
    d.csv_write(ROOT/'student_results.csv',rows)
    if not rows:return
    d.csv_write(ROOT/'student_summary.csv',aggregate(rows))
    lookup={(r['dataset'],r['seed'],r['noise'],r['arm']):r for r in rows};pairs=[]
    for key,row in lookup.items():
        for base in ['CE','OLD64','KM_MATCH','KNN_MATCH']:
            if base==key[3]:continue
            reference=lookup.get((*key[:3],base))
            if reference:pairs.append(dict(dataset=key[0],seed=key[1],noise=key[2],arm=key[3],reference=base,
                accuracy_delta_pp=100*(row['test_accuracy']-reference['test_accuracy'])))
    d.csv_write(ROOT/'student_paired_deltas.csv',pairs)


def historical_student(name,noise,arm):
    if arm not in ['CE','OLD64']:return None
    source_arm='GB64' if arm=='OLD64' else 'CE'
    if name in ['texas','wisconsin','roman']:
        folder=screen.BASELINE/'students'/name/'seed0'/noise/source_arm
    else:folder=screen.MECHANISM/'students'/name/'grace'/'seed0'/noise/source_arm
    path=folder/'result.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda')
    parser.add_argument('--datasets',nargs='+',default=screen.NAMES)
    args=parser.parse_args();torch.set_num_threads(4)
    if json.loads((ROOT/'screen_status.json').read_text(encoding='utf-8'))['status']!='completed':
        raise ValueError('Mechanism screen must finish first')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'pilot_raw.log',encoding='utf-8')])
    inputs={str(Path(p).resolve()):d.sha(p) for p in [__file__,ROOT/'run_screen.py',ROOT/'source_manifest.json',
        screen.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py',screen.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py',
        screen.MECHANISM/'run_students.py']}
    manifest_path=ROOT/'pilot_source_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=inputs:
        raise ValueError('Pilot sources differ from prior run; use a new experiment version')
    d.write(manifest_path,inputs)
    for p,digest in json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8')).items():
        if d.sha(p)!=digest:raise ValueError('Screen source changed before pilot: '+p)
    status=dict(status='running',started=time.time(),seed=0,pilot=True)
    d.write(ROOT/'pilot_status.json',status)
    try:
        with threadpool_limits(limits=4):
            for name in args.datasets:
                graph=screen.load_case(name,0);tr=graph.train_index
                for noise,rate in d.CONFIG['noise_conditions'].items():
                    data=inject_train_label_noise(graph,'uniform' if noise=='clean' else noise,rate,0)
                    parent=ROOT/'runs'/name/'seed0'/noise
                    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
                    frozen=json.loads((parent/'frozen_targets.json').read_text(encoding='utf-8'))
                    if d.fingerprint(tr.numpy(),data.val_index.numpy(),data.test_index.numpy())!=protocol['split_hash']:
                        raise ValueError('Pilot split mismatch')
                    if d.fingerprint(data.noisy_y[tr].numpy())!=protocol['noisy_hash']:
                        raise ValueError('Pilot noise mismatch')
                    for arm in ARMS:
                        folder=ROOT/'students'/name/'seed0'/noise/arm
                        if (folder/'result.json').exists():continue
                        file=parent/(arm+'.npz')
                        if d.sha(file)!=frozen[arm]:raise ValueError('Frozen target changed')
                        with np.load(file) as a:targets=full_targets(data.noisy_y,tr,data.num_classes,a['target'])
                        d.write(folder/'config.json',dict(dataset=name,seed=0,noise=noise,arm=arm,
                            training=TRAINING,device=args.device,pilot=True,target_path=str(file),target_hash=frozen[arm],
                            split_hash=protocol['split_hash'],noise_hash=protocol['noisy_hash'],
                            sources='pilot_source_manifest.json plus source_manifest.json'))
                        set_seed(0);start=time.time();logging.info('START %s %s %s',name,noise,arm)
                        result=train_x_only_mlp(data,targets,device=args.device,**TRAINING)
                        torch.save(result.logits,folder/'logits.pt')
                        row=dict(dataset=name,representation='grace',seed=0,noise=noise,arm=arm,
                            train_accuracy=result.train_accuracy,validation_accuracy=result.validation_accuracy,
                            test_accuracy=result.test_accuracy,best_epoch=result.best_epoch,
                            elapsed_seconds=time.time()-start,target_hash=frozen[arm])
                        prior=historical_student(name,noise,arm)
                        if prior:
                            row['historical_accuracy_delta_pp']=100*(result.test_accuracy-prior['test_accuracy'])
                        d.write(folder/'result.json',row);summarize()
                        logging.info('DONE %s %s %s test=%.6f',name,noise,arm,result.test_accuracy)
                        del targets,result
        for p,digest in inputs.items():
            if d.sha(p)!=digest:raise ValueError('Pilot source changed during training')
        status.update(status='completed',elapsed_seconds=time.time()-status['started'],
            student_runs=len(list((ROOT/'students').glob('*/seed*/*/*/result.json'))))
    except Exception as exc:
        status.update(status='failed',error=repr(exc));d.write(ROOT/'pilot_status.json',status);raise
    d.write(ROOT/'pilot_status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
