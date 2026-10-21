"""Freeze adaptive partitions and unchanged LCSS targets before quality evaluation."""
from pathlib import Path
import argparse
import json
import logging
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
MECHANISM=ROOT.parent/'heterophily_mechanism_20261014'
BASELINE=ROOT.parent/'heterophily_original_baseline_20261014'
sys.path.insert(0,str(MECHANISM))
import diagnostic as d
sys.path.insert(0,str(BASELINE))
import run_original_baseline as baseline
from gblcss.granular_ball import build_granular_balls, _measure_ball
from gblcss.data import inject_train_label_noise

DATA=dict(baseline.FILES,corafull=d.DATA['corafull'],coauthorcs=d.DATA['coauthorcs'])
H=dict(d.CACHED_H,texas=BASELINE/'representations/texas/grace/H.pt',
       wisconsin=BASELINE/'representations/wisconsin/grace/H.pt',
       roman=MECHANISM/'representations/Roman-empire/grace/H.pt')
NAMES=['texas','wisconsin','roman','corafull','coauthorcs']
ARMS=['CE','OLD64','OLD_SQRT','ADAPT64','SQRT_GEOM','SQRT_SUPPORT','KM_MATCH','KNN_MATCH']


def load_case(name,seed):
    return d.load_graph(name,seed) if name in ['corafull','coauthorcs'] else baseline.load_graph(name,seed)


def reference_folder(name,seed):
    return (BASELINE if name in ['texas','wisconsin'] else MECHANISM)/'runs'/name/'grace'/f'seed{seed}'


def geometry_stats(h,assignment,train):
    sizes=[];support=[];sse=0.
    for ball in np.unique(assignment):
        ids=np.flatnonzero(assignment==ball);z=h[ids].astype(np.float64)
        sizes.append(len(ids));support.append(int(np.count_nonzero(assignment[train]==ball)))
        sse+=float(np.square(z-z.mean(0)).sum())
    return dict(actual_balls=len(sizes),min_size=min(sizes),median_size=float(np.median(sizes)),
        max_size=max(sizes),max_node_fraction=max(sizes)/len(h),min_train_support=min(support),
        unsupported_balls=int(np.count_nonzero(np.array(support)<3)),common_input_sse=sse)


def old_terminal_probe(h,assignment,seed):
    records=[]
    count=len(np.unique(assignment))
    for ball in np.unique(assignment):
        ids=np.flatnonzero(assignment==ball)
        if len(ids)<64:
            records.append(dict(ball=int(ball),size=len(ids),reason='parent_below_2x32'));continue
        # For a completed64-budget partition no terminal rejection is inferred.
        if count>=64:
            records.append(dict(ball=int(ball),size=len(ids),reason='budget_reached'));continue
        labels=KMeans(n_clusters=2,n_init=10,random_state=seed+count-1+int(ball)).fit_predict(h[ids])
        children=[ids[labels==v] for v in [0,1]]
        reason='degenerate' if min(map(len,children))==0 else 'child_below32' if min(map(len,children))<32 else 'distance_gate'
        row=dict(ball=int(ball),size=len(ids),child_sizes=list(map(len,children)),reason=reason)
        if min(map(len,children))>0:
            old=_measure_ball(h,ids,0)['avg_distance']
            new=sum(_measure_ball(h,c,0)['avg_distance']*len(c) for c in children)/len(ids)
            row.update(relative_mean_distance_gain=(old-new)/max(old,1e-12))
        records.append(row)
    return records


def summarize():
    rows=[r for f in sorted((ROOT/'runs').glob('*/seed*/metrics.json'))
          for r in json.loads(f.read_text(encoding='utf-8'))]
    d.csv_write(ROOT/'target_metrics.csv',rows)
    if not rows:return
    frame=pd.DataFrame(rows)
    groups=frame.groupby(['dataset','noise','arm'])
    summary=groups[['target_brier','q_accuracy','clean_harm_rate','noise_recovery_rate','positive_clean_margin_fraction','mean_support']].agg(['mean','std'])
    summary.columns=['_'.join(v) for v in summary.columns]
    summary.reset_index().to_csv(ROOT/'target_summary.csv',index=False)
    lookup={(r['dataset'],r['seed'],r['noise'],r['arm']):r for r in rows};pairs=[]
    for key,row in lookup.items():
        for base in ['OLD64','OLD_SQRT','SQRT_GEOM','KM_MATCH','KNN_MATCH']:
            if key[3]==base:continue
            reference=lookup.get((*key[:3],base))
            if reference:pairs.append(dict(dataset=key[0],seed=key[1],noise=key[2],arm=key[3],reference=base,
                brier_delta=row['target_brier']-reference['target_brier'],
                clean_harm_delta=row['clean_harm_rate']-reference['clean_harm_rate']))
    d.csv_write(ROOT/'paired_deltas.csv',pairs)


def run_case(name,seed,device):
    from gblcss.granular_ball_adaptive import build_adaptive_granular_balls
    folder=ROOT/'runs'/name/f'seed{seed}'
    if (folder/'metrics.json').exists():return
    graph=load_case(name,seed);train=graph.train_index.numpy()
    h=d.load_frozen_representation(H[name],graph.num_nodes).numpy()
    target=math.ceil(math.sqrt(graph.num_nodes));old=reference_folder(name,seed)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'split.npz',train=train,val=graph.val_index.numpy(),test=graph.test_index.numpy())
    with np.load(old/'assignments.npz') as source:assignments={'OLD64':source['GB64'].copy()}
    stats={'OLD64':dict(requested_target=64,source=str(old/'assignments.npz'),cached=True)}
    logging.info('START %s seed%d target=%d',name,seed,target)
    start=time.time()
    result=build_granular_balls(h,max_balls=target,min_ball_size=32,split_improvement=.05,random_state=seed)
    assignments['OLD_SQRT']=result.assignment
    stats['OLD_SQRT']=dict(requested_target=target,**result.summary,seconds=time.time()-start)
    for arm,budget,support in [('ADAPT64',64,None),('SQRT_GEOM',target,None),('SQRT_SUPPORT',target,train)]:
        start=time.time()
        result=build_adaptive_granular_balls(h,target_balls=budget,train_index=support,
                                           min_train_per_ball=4,random_state=seed)
        assignments[arm]=result.assignment
        stats[arm]=dict(**result.summary,seconds=time.time()-start)
        d.write(folder/(arm+'_split_trace.json'),result.split_trace)
        d.write(folder/(arm+'_balls.json'),[dict(ball,
            train_count=int(np.count_nonzero(result.assignment[train]==ball['ball_id']))) for ball in result.balls])
    count=len(np.unique(assignments['SQRT_SUPPORT']))
    assignments['KM_MATCH']=KMeans(n_clusters=count,n_init=10,random_state=seed).fit_predict(h)
    np.savez_compressed(folder/'assignments.npz',**assignments)
    for arm,assignment in assignments.items():
        stats.setdefault(arm,{}).update(geometry_stats(h,assignment,train))
        if arm in ['OLD64','OLD_SQRT','KM_MATCH']:
            balls=[]
            for ball_id in np.unique(assignment):
                indices=np.flatnonzero(assignment==ball_id)
                stat=_measure_ball(h,indices,int(ball_id))
                z=h[indices].astype(np.float64)
                stat.update(sse=float(np.square(z-z.mean(0)).sum()),
                    train_count=int(np.count_nonzero(assignment[train]==ball_id)))
                balls.append(stat)
            d.write(folder/(arm+'_balls.json'),balls)
    d.write(folder/'partition_stats.json',stats)
    if seed==0:d.write(folder/'old_terminal_probe.json',old_terminal_probe(h,assignments['OLD64'],seed))
    relations={arm:d.group_weights(assignment[train]) for arm,assignment in assignments.items()}
    relations['KNN_MATCH']=d.knn_weights(h[train],np.diff(relations['SQRT_SUPPORT'].indptr),device)
    d.write(folder/'relations.json',{arm:dict(hash=d.fingerprint(w.indptr,w.indices,w.data),
        nonzeros=int(w.nnz),self_votes=int(np.count_nonzero(w.diagonal()))) for arm,w in relations.items()})
    rows=[]
    for noise,rate in d.CONFIG['noise_conditions'].items():
        data=inject_train_label_noise(graph,'uniform' if noise=='clean' else noise,rate,seed)
        observed=data.noisy_y[train].numpy();classes=data.num_classes
        prior_protocol=json.loads((old/noise/'protocol.json').read_text(encoding='utf-8'))
        prior_frozen=json.loads((old/noise/'frozen_targets.json').read_text(encoding='utf-8'))
        if prior_protocol['split_fingerprint']!=d.fingerprint(train,data.val_index.numpy(),data.test_index.numpy()):
            raise ValueError('Historical train/validation/test split mismatch')
        if prior_protocol['noisy_train_fingerprint']!=d.fingerprint(observed):
            raise ValueError('Historical noise mismatch')
        if d.sha(old/noise/'GB64.npz')!=prior_frozen['GB64']:
            raise ValueError('Historical frozen OLD64 target changed')
        condition=folder/noise;condition.mkdir(exist_ok=True)
        np.save(condition/'observed_train.npy',observed)
        distributions={arm:d.vote(w,observed,classes) for arm,w in relations.items()}
        onehot=np.eye(classes)[observed]
        distributions['CE']=(onehot,onehot,np.zeros(len(train)),np.zeros(len(train)),np.zeros(len(train)))
        frozen={}
        for arm,(r,q,alpha,count,effective) in distributions.items():
            target_values=d.mix(q,alpha,observed).astype(np.float32)
            np.savez_compressed(condition/(arm+'.npz'),r=r.astype(np.float32),q=q.astype(np.float32),
                alpha=alpha.astype(np.float32),target=target_values,support=count,effective=effective)
            frozen[arm]=d.sha(condition/(arm+'.npz'))
        d.write(condition/'frozen_targets.json',frozen)
        d.write(condition/'protocol.json',dict(dataset=name,seed=seed,noise=noise,rate=rate,
            split_hash=d.fingerprint(train,data.val_index.numpy(),data.test_index.numpy()),
            noisy_hash=d.fingerprint(observed),
            val_clean=bool(torch.equal(data.noisy_y[data.val_mask],data.y[data.val_mask])),
            test_clean=bool(torch.equal(data.noisy_y[data.test_mask],data.y[data.test_mask])),
            historical_protocol=str(old/noise/'protocol.json'),historical_target_hash=prior_frozen['GB64']))
        # Compare historical targets to establish a matched old baseline.
        with np.load(old/noise/'GB64.npz') as prior, np.load(condition/'OLD64.npz') as saved:
            if not np.array_equal(prior['target'],saved['target']):raise ValueError('Historical OLD64 targets differ')
        clean=data.y[train].numpy()
        for arm,(r,q,alpha,count,effective) in distributions.items():
            with np.load(condition/(arm+'.npz')) as artifact:target_values=artifact['target']
            metric=d.target_metrics(target_values,q,alpha,clean,observed)
            positive=margin_mean=None
            if arm!='CE':
                clean_r=d.vote(relations[arm],clean,classes)[0]
                other=clean_r.copy();other[np.arange(len(train)),clean]=-np.inf
                margin=clean_r[np.arange(len(train)),clean]-other.max(1)
                valid=count>=2;positive=float(np.mean(valid&(margin>0)))
                margin_mean=float(np.mean(margin[valid])) if valid.any() else None
            rows.append(dict(dataset=name,seed=seed,noise=noise,arm=arm,**metric,
                positive_clean_margin_fraction=positive,mean_clean_margin=margin_mean,
                mean_support=float(np.mean(count)),eligible_rate=float(np.mean(count>=2)),
                mean_effective=float(np.mean(effective))))
        logging.info('FROZEN %s seed%d %s arms=%d',name,seed,noise,len(frozen))
    d.write(folder/'metrics.json',rows);summarize()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=NAMES)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'screen_raw.log',encoding='utf-8')])
    sources=[__file__,ROOT/'PLAN.md',d.__file__,baseline.__file__,
        WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball.py',
        WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball_adaptive.py',
        WORKSPACE/'GB-LCSS_code/src/gblcss/representation.py',
        WORKSPACE/'GB-LCSS_code/src/gblcss/targets.py',WORKSPACE/'GB-LCSS_code/src/gblcss/data.py']
    sources.extend(DATA[name] for name in args.datasets);sources.extend(H[name] for name in args.datasets)
    for name in args.datasets:
        for seed in args.seeds:
            prior=reference_folder(name,seed)
            old_manifest=json.loads((prior.parent/'source_manifest.json').read_text(encoding='utf-8'))
            for p,digest in old_manifest.items():
                if d.sha(p)!=digest:raise ValueError('Historical source/input changed: '+p)
            sources.extend([prior/'assignments.npz',prior/'split.npz',prior.parent/'source_manifest.json'])
            for noise in d.CONFIG['noise_conditions']:
                sources.extend([prior/noise/'protocol.json',prior/noise/'frozen_targets.json',prior/noise/'GB64.npz'])
    manifest={str(Path(p).resolve()):d.sha(p) for p in sources}
    path=ROOT/'source_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Existing experiment source manifest differs; use a new version directory')
    d.write(path,manifest)
    d.write(ROOT/'config.json',dict(datasets=args.datasets,seeds=args.seeds,arms=ARMS,
        requested_scale='ceil(sqrt(full_node_count))',min_train_per_ball=4,geometry_size_divisor=4,
        noise=d.CONFIG['noise_conditions'],device=args.device,source_sha=d.sha(__file__),ssl_seeds=[0]))
    status=dict(status='running',datasets=args.datasets,seeds=args.seeds,started=time.time())
    d.write(ROOT/'screen_status.json',status)
    try:
        with threadpool_limits(limits=4):
            for name in args.datasets:
                for seed in args.seeds:run_case(name,seed,args.device)
        for p,digest in manifest.items():
            if d.sha(p)!=digest:raise ValueError('Source changed during screen: '+p)
        status.update(status='completed',elapsed_seconds=time.time()-status['started'],
                      protocol_runs=len(list((ROOT/'runs').glob('*/seed*/metrics.json'))))
    except Exception as exc:
        status.update(status='failed',error=repr(exc));d.write(ROOT/'screen_status.json',status);raise
    d.write(ROOT/'screen_status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
