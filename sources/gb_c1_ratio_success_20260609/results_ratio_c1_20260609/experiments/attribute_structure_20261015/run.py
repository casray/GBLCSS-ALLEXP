"""Freeze label-free regions, then build and diagnose original LCSS targets."""
import argparse
import gc
import logging
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import common as c
from features import structural_views
from group_targets import group_vote
from gblcss.granular_ball import build_granular_balls
from gblcss.granular_ball_topology import build_topology_balls
from gblcss.granular_ball_trajectory import build_ball_trajectory
from gblcss.representation import load_frozen_representation


def code_files():
    return [c.ROOT/p for p in ['PLAN.md','config.json','common.py','features.py','group_targets.py','run.py']]+[
        c.WORKSPACE/'GB-LCSS_code/src/gblcss'/p for p in ['granular_ball.py','granular_ball_adaptive.py','granular_ball_trajectory.py','granular_ball_topology.py','targets.py','data.py','representation.py']]+[Path(c.d.__file__)]


def prepare_views(name,g):
    folder=c.ROOT/'views'/name
    if (folder/'manifest.json').exists():
        c.verify_manifest(folder/'manifest.json');return
    logging.info('VIEWS START %s',name)
    views=structural_views(g.x,g.edge_index,shuffle_seed=c.CONFIG['residual']['shuffle_seed'])
    folder.mkdir(parents=True,exist_ok=True)
    for key in ['X','R','SHUFFLE','permutation']:np.save(folder/(key+'.npy'),views[key])
    c.freeze_manifest(folder/'manifest.json',[folder/(key+'.npy') for key in ['X','R','SHUFFLE','permutation']]+[c.DATA[name]])
    c.write(folder/'input.json',dict(dataset=name,data_sha256=c.sha(c.DATA[name]),features_sha256=c.fingerprint(g.x.numpy()),edges_sha256=c.fingerprint(g.edge_index.numpy()),label_usage='none'))
    del views;gc.collect();logging.info('VIEWS DONE %s',name)


def partitions(name,seed):
    folder=c.ROOT/'partitions'/name/f'seed{seed}'
    if (folder/'manifest.json').exists():
        c.verify_manifest(folder/'manifest.json');return
    g=c.load_graph(name,seed);tr=g.train_index.numpy()
    prepare_views(name,g)
    budget=math.ceil(math.sqrt(g.num_nodes));folder.mkdir(parents=True,exist_ok=True)
    assignments={};stats={}
    h=load_frozen_representation(c.H[name],g.num_nodes).numpy()
    tick=time.perf_counter()
    with threadpool_limits(limits=1):
        result=build_granular_balls(h,max_balls=64,min_ball_size=32,split_improvement=.05,random_state=seed)
    assignments['OLD64']=result.assignment;stats['OLD64']={**result.summary,'seconds':time.perf_counter()-tick}
    bfs=build_topology_balls(g.edge_index,g.num_nodes)
    assignments['BFS']=bfs.assignment;stats['BFS']=bfs.summary
    for arm,space in [('H_TREE',None),('X_TREE','X'),('R_TREE','R'),('SHUFFLE_TREE','SHUFFLE')]:
        tick=time.perf_counter();logging.info('PARTITION %s seed%d %s',name,seed,arm)
        z=h if space is None else np.load(c.ROOT/'views'/name/(space+'.npy'),mmap_mode='r')
        with threadpool_limits(limits=1):
            result=build_ball_trajectory(z,budgets=[budget],train_index=tr,min_leaf_size=16,min_train_per_ball=16,random_state=seed)
        assignments[arm]=result.snapshots[budget]
        stats[arm]={**result.summaries[budget],'seconds':time.perf_counter()-tick}
        c.write(folder/(arm+'_trace.json'),result.split_trace)
        del z,result;gc.collect()
    np.savez_compressed(folder/'assignments.npz',**assignments)
    np.savez_compressed(folder/'split.npz',train=tr,val=g.val_index.numpy(),test=g.test_index.numpy())
    for arm,a in assignments.items():
        _,inverse=np.unique(a,return_inverse=True)
        size=np.bincount(inverse);train=np.bincount(inverse[tr],minlength=len(size))
        stats[arm].update(actual_balls=len(size),singletons=int((size==1).sum()),train_count_quantiles=np.quantile(train,[0,.25,.5,.75,1]).tolist(),size_quantiles=np.quantile(size,[0,.25,.5,.75,1]).tolist(),eligible_rate=float(np.mean(train[inverse[tr]]>=3)))
    c.write(folder/'stats.json',stats)
    c.freeze_manifest(folder/'manifest.json',[folder/'assignments.npz',folder/'split.npz',folder/'stats.json',c.DATA[name],c.H[name],c.ROOT/'views'/name/'manifest.json'])
    logging.info('PARTITIONS DONE %s seed%d %s',name,seed,{a:s['actual_balls'] for a,s in stats.items()})


def targets(name,seed):
    parent=c.ROOT/'targets'/name/f'seed{seed}'
    if (parent/'metrics.json').exists():
        for noise in c.conditions(name):c.verify_manifest(parent/noise/'frozen_targets.json')
        return
    with np.load(c.ROOT/'partitions'/name/f'seed{seed}'/'assignments.npz') as a:assignments={k:a[k].copy() for k in a.files}
    rows=[]
    for noise in c.conditions(name):
        graph,protocol=c.load_case(name,seed,noise);tr=graph.train_index.numpy();observed=graph.noisy_y[tr].numpy();classes=graph.num_classes
        folder=parent/noise;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'observed.npy',observed)
        c.write(folder/'protocol.json',protocol)
        for arm in c.ARMS:
            if arm=='CE':
                r=q=np.eye(classes)[observed];alpha=np.zeros(len(tr));support=np.zeros(len(tr),dtype=np.int64)
            else:r,q,alpha,support,_=group_vote(assignments[arm][tr],observed,classes)
            target=c.d.mix(q,alpha,observed).astype(np.float32)
            np.savez_compressed(folder/(arm+'.npz'),target=target,r=r.astype(np.float32),q=q.astype(np.float32),alpha=alpha.astype(np.float32),support=support)
        c.freeze_manifest(folder/'frozen_targets.json',[folder/(a+'.npz') for a in c.ARMS]+[folder/'observed.npy',folder/'protocol.json'])
        # Clean truth is consulted only after every target for this case is frozen.
        clean=graph.y[tr].numpy()
        for arm in c.ARMS:
            with np.load(folder/(arm+'.npz')) as v:
                t,q,alpha,support=(v[k] for k in ['target','q','alpha','support'])
                metrics=c.d.target_metrics(t,q,alpha,clean,observed)
                valid=support>=2
                margin_fraction=None;margin_mean=None
                if arm!='CE':
                    clean_r=group_vote(assignments[arm][tr],clean,classes)[0]
                    others=clean_r.copy();others[np.arange(len(tr)),clean]=-np.inf
                    margin=clean_r[np.arange(len(tr)),clean]-others.max(1)
                    margin_fraction=float(np.mean(valid&(margin>0)))
                    margin_mean=float(margin[valid].mean()) if valid.any() else None
                high=alpha>=.75
                rows.append(dict(dataset=name,seed=seed,noise=noise,arm=arm,**metrics,
                    eligible_rate=float(valid.mean()),eligible_q_accuracy=float(np.mean(q.argmax(1)[valid]==clean[valid])) if valid.any() else None,
                    positive_clean_margin_fraction=margin_fraction,mean_clean_margin=margin_mean,
                    mean_support=float(support.mean()),high_alpha_wrong_q_fraction=float(np.mean(high&(q.argmax(1)!=clean)))))
        logging.info('TARGET FROZEN %s seed%d %s',name,seed,noise)
    c.write(parent/'metrics.json',rows)


def summarize():
    rows=[r for p in sorted((c.ROOT/'targets').glob('*/seed*/metrics.json')) for r in c.read(p)]
    if not rows:return rows
    c.d.csv_write(c.ROOT/'target_metrics.csv',rows)
    f=pd.DataFrame(rows);nums=[x for x in f.select_dtypes('number').columns if x!='seed']
    stats=f.groupby(['dataset','noise','arm'])[nums].agg(['mean','std','count']);stats.columns=['_'.join(a) for a in stats.columns]
    stats.reset_index().to_csv(c.ROOT/'target_summary.csv',index=False)
    if len(rows)==420:
        gate=c.compute_gate(rows);c.write(c.ROOT/'gate.json',dict(datasets=gate,selected_datasets=[n for n,v in gate.items() if v['passed']],target_rows=len(rows),source='all frozen metrics.json records, not cached CSV'))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=c.CONFIG['datasets'],choices=c.CONFIG['datasets']);a=parser.parse_args()
    torch.set_num_threads(c.CONFIG['threads'])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'target_raw.log',encoding='utf-8')])
    c.freeze_manifest(c.ROOT/'source_manifest.json',code_files())
    c.write(c.ROOT/'target_status.json',dict(status='running',requested=a.datasets))
    try:
        with threadpool_limits(limits=c.CONFIG['threads']):
            for name in a.datasets:
                for seed in c.SEEDS:
                    partitions(name,seed);targets(name,seed);summarize()
        rows=summarize();c.verify_manifest(c.ROOT/'source_manifest.json')
        c.write(c.ROOT/'target_status.json',dict(status='completed' if len(rows)==420 else 'partial',target_rows=len(rows)))
    except Exception as exc:
        c.write(c.ROOT/'target_status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
