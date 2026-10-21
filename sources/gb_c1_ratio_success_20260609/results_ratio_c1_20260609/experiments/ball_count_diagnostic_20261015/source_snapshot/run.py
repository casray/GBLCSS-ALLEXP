"""Nested ball-count intervention; fixed representations, support and student protocol."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import data_tools as dt
from data_tools import ROOT,WORKSPACE,CONFIG,DATA,H_PATHS,d
sys.path.insert(0,str(dt.GREET_ROOT))
from target_tools import ball_bundle
from gblcss.granular_ball import build_granular_balls
from gblcss.granular_ball_trajectory import build_ball_trajectory
from gblcss.representation import load_frozen_representation
from gblcss.data import set_seed
from gblcss.training import train_x_only_mlp


def target_key(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype=np.float32).tobytes()).hexdigest()


def arm_metadata(arm):
    if arm=='CE':return dict(space='COMMON',kind='CE',requested_balls=0)
    space,kind=arm.split('_',1)
    if space not in CONFIG['spaces']:raise ValueError('Unknown representation')
    if kind=='OLD64':return dict(space=space,kind='OLD64',requested_balls=64)
    if kind.startswith('B') and kind[1:].isdigit() and int(kind[1:])>0:
        return dict(space=space,kind='B',requested_balls=int(kind[1:]))
    raise ValueError('Unknown ball arm')


def status(stage,**extra):
    d.write(ROOT/'status.json',dict(status='running',stage=stage,
        updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))


def freeze_sources():
    files=[ROOT/n for n in ['config.json','PLAN.md','run.py','data_tools.py','audit_report.py','plot_curves.py',
                            'prepare_data.py','prepare_embeddings.py','test_data.py','test_execution.py','test_trajectory.py','test_audit.py',
                            'dataset_manifest.json','embedding_source_manifest.json']]
    files += [Path(d.__file__),dt.GREET_ROOT/'target_tools.py',dt.GREET_ROOT/'methods.py']
    files += [WORKSPACE/'GB-LCSS_code/src/gblcss'/n for n in
             ['granular_ball.py','granular_ball_adaptive.py','granular_ball_trajectory.py','representation.py','data.py','models.py','training.py']]
    files += [DATA[n] for n in CONFIG['datasets']]
    files += [H_PATHS[s][n] for n in CONFIG['datasets'] for s in CONFIG['spaces']]
    for n in ['roman','corafull','coauthorcs']:
        for seed in CONFIG['protocol_seeds']:
            files += [dt.PREVIOUS/'partitions'/n/f'seed{seed}'/s/'assignments.npz' for s in CONFIG['spaces']]
            files += [dt.PREVIOUS/'targets'/n/f'seed{seed}'/c/'protocol.json' for c in CONFIG['conditions']]
    manifest={str(p.resolve()):d.sha(p) for p in files}
    path=ROOT/'source_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:raise ValueError('Execution inputs changed')
    d.write(path,manifest)
    for p in files:
        if p.suffix not in ['.py','.md']:continue
        out=ROOT/'source_snapshot'/p.name;out.parent.mkdir(exist_ok=True)
        if out.exists() and d.sha(out)!=d.sha(p):raise ValueError('Source snapshot differs')
        if not out.exists():shutil.copyfile(p,out)


def verify_sources():
    for manifest in ['embedding_source_manifest.json','source_manifest.json']:
        for p,digest in json.loads((ROOT/manifest).read_text(encoding='utf-8')).items():
            if d.sha(p)!=digest:raise ValueError('Frozen source/input changed: '+p)


def geometry_stats(h,groups,train):
    h=np.asarray(h,dtype=np.float64)
    scale=np.max(np.abs(h),axis=1,keepdims=True)
    h=h/np.maximum(scale,np.finfo(np.float64).tiny)
    h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),np.finfo(np.float64).tiny)
    sizes=np.bincount(groups);support=np.bincount(groups[train],minlength=len(sizes))
    sse=0.
    for ball in range(len(sizes)):
        z=h[groups==ball].astype(np.float64);sse+=float(np.square(z-z.mean(0)).sum())
    return dict(actual_balls=len(sizes),num_balls=len(sizes),min_size=int(sizes.min()),median_size=float(np.median(sizes)),
        max_size=int(sizes.max()),min_train_support=int(support.min()),mean_other_votes=float((support[groups[train]]-1).mean()),
        train_support_quantiles=np.quantile(support,[0,.25,.5,.75,1]).tolist(),normalized_input_sse=sse,normalized_sse=sse/len(h))


def partitions(name,seed,space,graph):
    folder=ROOT/'partitions'/name/f'seed{seed}'/space
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True); train=graph.train_index.numpy()
    h=load_frozen_representation(H_PATHS[space][name],graph.num_nodes).numpy()
    if name in CONFIG['new_datasets']:
        old=build_granular_balls(h,max_balls=64,min_ball_size=32,split_improvement=.05,random_state=seed).assignment
    else:
        with np.load(dt.PREVIOUS/'partitions'/name/f'seed{seed}'/space/'assignments.npz') as a:old=a['OLD64'].copy()
    assignments={'OLD64':old};stats={'OLD64':dict(requested_target=64,stop_reason='historical_or_original_rule')}
    logging.info('TREE %s seed%d %s',name,seed,space)
    result=build_ball_trajectory(h,budgets=dt.budgets_for(graph.num_nodes),train_index=train,
        min_train_per_ball=CONFIG['trajectory']['min_train_per_ball'],
        min_leaf_size=CONFIG['trajectory']['min_leaf_size'],random_state=seed)
    for budget,groups in result.snapshots.items():
        key='B'+str(budget);assignments[key]=groups;stats[key]=result.summaries[budget]
    for key,groups in assignments.items():stats[key].update(geometry_stats(h,groups,train))
    np.savez_compressed(folder/'assignments.npz',**assignments)
    np.savez_compressed(folder/'split.npz',train=train,val=graph.val_index.numpy(),test=graph.test_index.numpy())
    d.write(folder/'stats.json',stats);d.write(folder/'trajectory_trace.json',result.split_trace)
    d.write(folder/'complete.json',dict(assignment_sha256=d.sha(folder/'assignments.npz'),h_sha256=d.sha(H_PATHS[space][name]),
            split_hash=d.fingerprint(train,graph.val_index.numpy(),graph.test_index.numpy()),single_split_trajectory=True))


def get_bundles(name,seed,observed,classes,train):
    onehot=np.eye(classes)[observed]
    bundles={'CE':dict(r=onehot,q=onehot,alpha=np.zeros(len(train)),support=np.zeros(len(train),dtype=np.int64),target=onehot)}
    for space in CONFIG['spaces']:
        with np.load(ROOT/'partitions'/name/f'seed{seed}'/space/'assignments.npz') as a:
            for key in a.files:bundles[space+'_'+key]=ball_bundle(a[key][train],observed,classes)
    return bundles


def summarize():
    for kind,pattern in [('target','targets/*/seed*/*/metrics.json'),('student','students/*/seed*/*/*/result.json')]:
        rows=[]
        for path in ROOT.glob(pattern):
            value=json.loads(path.read_text(encoding='utf-8'));rows.extend(value if isinstance(value,list) else [value])
        if not rows:continue
        frame=pd.DataFrame(rows);frame.to_csv(ROOT/(kind+'_results.csv'),index=False,encoding='utf-8-sig')
        metric='target_brier' if kind=='target' else 'test_accuracy'
        frame.groupby(['dataset','noise','space','kind','arm','requested_balls'])[metric].agg(['count','mean','std']).reset_index().to_csv(
            ROOT/(kind+'_summary.csv'),index=False,encoding='utf-8-sig')


def targets(name,seed):
    for condition in CONFIG['conditions']:
        folder=ROOT/'targets'/name/f'seed{seed}'/condition
        if (folder/'metrics.json').exists():continue
        graph,protocol=dt.load_case(name,seed,condition);train=graph.train_index.numpy();observed=graph.noisy_y[train].numpy()
        bundles=get_bundles(name,seed,observed,graph.num_classes,train);folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'observed_train.npy',observed);frozen={}
        for arm,bundle in bundles.items():
            np.savez_compressed(folder/(arm+'.npz'),**bundle);frozen[arm]=d.sha(folder/(arm+'.npz'))
        d.write(folder/'protocol.json',protocol);d.write(folder/'frozen_targets.json',frozen)
        clean=graph.y[train].numpy();clean_bundles=get_bundles(name,seed,clean,graph.num_classes,train);rows=[]
        for arm,bundle in bundles.items():
            meta=arm_metadata(arm);space=meta['space'];key=arm.split('_',1)[1] if arm!='CE' else ''
            stats={} if arm=='CE' else json.loads((ROOT/'partitions'/name/f'seed{seed}'/space/'stats.json').read_text(encoding='utf-8'))[key]
            metric=d.target_metrics(bundle['target'],bundle['q'],bundle['alpha'],clean,observed)
            r=clean_bundles[arm]['r'];other=r.copy();other[np.arange(len(train)),clean]=-np.inf
            margin=r[np.arange(len(train)),clean]-other.max(1);valid=bundle['support']>=2
            rows.append(dict(dataset=name,seed=seed,noise=condition,arm=arm,**meta,**metric,
                actual_balls=stats.get('actual_balls',0),mean_support=float(bundle['support'].mean()),
                min_support=int(bundle['support'].min()),support_coverage=float(valid.mean()),
                positive_clean_margin_fraction=float(np.mean(valid&(margin>0))) if arm!='CE' else None,
                mean_clean_margin=float(margin[valid].mean()) if valid.any() else None,
                normalized_sse=stats.get('normalized_sse')))
        d.write(folder/'metrics.json',rows);summarize()
        logging.info('TARGETS %s seed%d %s rows%d',name,seed,condition,len(rows))


def students(name,seed,device):
    for condition in CONFIG['conditions']:
        graph,protocol=dt.load_case(name,seed,condition);tr=graph.train_index
        parent=ROOT/'targets'/name/f'seed{seed}'/condition
        assert json.loads((parent/'protocol.json').read_text(encoding='utf-8'))==protocol
        frozen=json.loads((parent/'frozen_targets.json').read_text(encoding='utf-8'))
        metrics={x['arm']:x for x in json.loads((parent/'metrics.json').read_text(encoding='utf-8'))}
        cache={}
        for arm in dt.arms_for(graph.num_nodes):
            target=parent/(arm+'.npz')
            if d.sha(target)!=frozen[arm]:raise ValueError('Frozen target changed')
            with np.load(target) as a:local=a['target'].astype(np.float32)
            semantic=target_key(local);folder=ROOT/'students'/name/f'seed{seed}'/condition/arm
            if (folder/'result.json').exists():
                prior=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                if prior['target_hash']!=frozen[arm] or prior['target_semantic_hash']!=semantic:raise ValueError('Resumed target mismatch')
                cache.setdefault(semantic,prior);continue
            meta=arm_metadata(arm);actual=metrics[arm]['actual_balls']
            d.write(folder/'config.json',dict(dataset=name,seed=seed,noise=condition,arm=arm,protocol=protocol,
                training=CONFIG['training'],target_hash=frozen[arm],target_semantic_hash=semantic,target_path=str(target),device=device))
            status('students',dataset=name,seed=seed,noise=condition,arm=arm)
            if semantic in cache:
                row=dict(cache[semantic]);row.update(arm=arm,**meta,actual_balls=actual,target_hash=frozen[arm],
                    training_executed=False,elapsed_seconds=0.,alias_of=cache[semantic]['arm'])
                d.write(folder/'result.json',row)
                logging.info('ALIAS %s seed%d %s %s -> %s',name,seed,condition,arm,row['alias_of'])
            else:
                full=torch.nn.functional.one_hot(graph.noisy_y,graph.num_classes).float();full[tr]=torch.from_numpy(local)
                set_seed(seed);start=time.time()
                result=train_x_only_mlp(graph,full,device=device,**CONFIG['training'])
                logits=folder/'logits.pt';torch.save(result.logits,logits)
                macro=float(f1_score(graph.y[graph.test_index].numpy(),result.logits[graph.test_index].argmax(1).numpy(),
                    labels=list(range(graph.num_classes)),average='macro',zero_division=0))
                row=dict(dataset=name,seed=seed,noise=condition,arm=arm,**meta,actual_balls=actual,
                    train_accuracy=result.train_accuracy,validation_accuracy=result.validation_accuracy,test_accuracy=result.test_accuracy,
                    test_macro_f1=macro,best_epoch=result.best_epoch,elapsed_seconds=time.time()-start,
                    target_hash=frozen[arm],target_semantic_hash=semantic,logits_path=str(logits.resolve()),
                    training_executed=True,unique_model_id=f'{name}/seed{seed}/{condition}/{arm}')
                d.write(folder/'result.json',row);cache[semantic]=row
                logging.info('STUDENT %s seed%d %s %s test%.6f',name,seed,condition,arm,result.test_accuracy)
            summarize()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda')
    parser.add_argument('--wait-embeddings',action='store_true');args=parser.parse_args()
    torch.set_num_threads(CONFIG['threads'])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'raw.log',encoding='utf-8')])
    try:
        while True:
            p=ROOT/'embedding_status.json';embedding=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
            if embedding.get('status')=='completed':break
            if embedding.get('status')=='failed':raise ValueError('Embedding preparation failed: '+str(embedding))
            if not args.wait_embeddings:raise ValueError('Complete new embeddings first, or --wait-embeddings')
            status('waiting_for_embeddings',embedding=embedding);time.sleep(5)
        freeze_sources();records=[]
        with threadpool_limits(limits=CONFIG['threads']):
            for name in CONFIG['datasets']:
                graph,_=dt.load_case(name,0,'clean');records.append(dict(dataset=name,**dt.describe(graph)))
                for seed in CONFIG['protocol_seeds']:
                    graph,_=dt.load_case(name,seed,'clean')
                    for space in CONFIG['spaces']:
                        status('partitions',dataset=name,seed=seed,space=space);partitions(name,seed,space,graph)
                    status('targets',dataset=name,seed=seed);targets(name,seed)
            d.write(ROOT/'dataset_statistics.json',records);verify_sources()
            for name in CONFIG['datasets']:
                for seed in CONFIG['protocol_seeds']:students(name,seed,args.device)
        summarize();verify_sources()
        from audit_report import audit_and_report
        status('audit');audit_and_report(make_plots=False)
        from plot_curves import make_plots
        status('plotting');make_plots()
        d.write(ROOT/'status.json',dict(status='completed',stage='all',
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S')))
    except Exception as exc:
        d.write(ROOT/'status.json',dict(status='failed',error=repr(exc),updated_local=time.strftime('%Y-%m-%d %H:%M:%S')));raise


if __name__=='__main__':main()
