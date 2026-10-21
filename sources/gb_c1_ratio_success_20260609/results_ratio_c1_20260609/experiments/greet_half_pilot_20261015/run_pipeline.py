"""Frozen-target diagnostics, cross-fitted HF proxy, and gated student queue."""
from pathlib import Path
import argparse
import json
import logging
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from threadpoolctl import threadpool_limits

from prepare_greet import ROOT,WORKSPACE,write,sha,prepare
from methods import rank_gate,match_budget
from hf_proxy import filtered_features,cross_fit_scores
from target_tools import ball_bundle,gated_bundle,mix
sys.path.insert(0,str(WORKSPACE/'experiments/heterophily_mechanism_20261014'))
import diagnostic_v2 as old
from gblcss.data import inject_train_label_noise,set_seed
from gblcss.granular_ball import build_granular_balls
from gblcss.training import train_x_only_mlp

CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
METHOD_FILES=['methods.py','prepare_greet.py','hf_proxy.py','target_tools.py','run_pipeline.py']


def freeze_sources():
    files={name:sha(ROOT/name) for name in METHOD_FILES}
    files['config.json']=sha(ROOT/'config.json')
    folder=ROOT/'source_snapshot'; folder.mkdir(exist_ok=True)
    for name,digest in files.items():
        target=folder/name
        if target.exists() and sha(target)!=digest:raise ValueError('Execution source changed: '+name)
        if not target.exists():shutil.copyfile(ROOT/name,target)
    write(ROOT/'execution_sources.json',files)
    return files


def status(stage,**extra):
    write(ROOT/'pipeline_status.json',dict(status='running',stage=stage,updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))


def graph_protocol(name,seed,condition):
    graph=old.load_graph(name,seed)
    graph=inject_train_label_noise(graph,'uniform' if condition=='clean' else condition,CONFIG['conditions'][condition],seed)
    train=graph.train_index.numpy(); observed=graph.noisy_y[graph.train_index].numpy()
    parent=old.ROOT/'runs'/name/'grace'/f'seed{seed}'
    historical=json.loads((parent/condition/'protocol.json').read_text(encoding='utf-8'))
    split_hash=old.fingerprint(train,graph.val_index.numpy(),graph.test_index.numpy())
    noise_hash=old.fingerprint(observed)
    if historical['split_fingerprint']!=split_hash or historical['noisy_train_fingerprint']!=noise_hash:
        raise ValueError('Historical split/noise mismatch')
    with np.load(parent/'assignments.npz') as a:assignment=a['GB64'].copy()
    with np.load(parent/condition/'GB64.npz') as a:base={k:a[k].copy() for k in ['q','alpha','target','support']}
    base['alpha']=base['alpha'].astype(np.float64)
    reconstructed=ball_bundle(assignment[train],observed,graph.num_classes)
    if not np.allclose(base['target'],reconstructed['target'],atol=2e-6,rtol=0):raise ValueError('Formal baseline mismatch')
    protocol=dict(dataset=name,seed=seed,noise=condition,split_fingerprint=split_hash,
                  noisy_train_fingerprint=noise_hash,data_sha256=old.sha(old.DATA[name]),
                  original_target_sha256=sha(parent/condition/'GB64.npz'),
                  validation_clean=bool(torch.equal(graph.noisy_y[graph.val_index],graph.y[graph.val_index])),
                  test_clean=bool(torch.equal(graph.noisy_y[graph.test_index],graph.y[graph.test_index])),
                  execution_config_sha256=sha(ROOT/'config.json'))
    return graph,assignment,base,protocol


def save_bundle(folder,arm,bundle):
    np.savez_compressed(folder/f'{arm}.npz',**{key:bundle[key] for key in ['q','alpha','target','support']})
    return sha(folder/f'{arm}.npz')


def bundle_metrics(bundle,truth,observed):
    metric=old.target_metrics(bundle['target'],bundle['q'],bundle['alpha'],truth,observed)
    valid=bundle['support']>=2
    metric.update(support_coverage=float(valid.mean()),
                  q_accuracy_supported=float((bundle['q'][valid].argmax(1)==truth[valid]).mean()) if valid.any() else None)
    return metric


def aggregate_targets():
    records=[]
    for path in (ROOT/'targets').glob('*/*/seed*/*/metrics.json'):
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    if records:
        frame=pd.DataFrame(records); frame.to_csv(ROOT/'target_results.csv',index=False,encoding='utf-8')
        frame.groupby(['dataset','stage','arm','noise'])['target_brier'].agg(['count','mean','std']).reset_index().to_csv(
            ROOT/'target_summary.csv',index=False,encoding='utf-8')
    return records


def edge_diagnostic(name,graph):
    folder=ROOT/'representations'/name
    if (folder/'edge_diagnostic.json').exists():return
    edges=torch.load(folder/'edges.pt',weights_only=True).numpy()
    weights=torch.load(folder/'edge_weights.pt',weights_only=True).numpy()
    matrix=sparse.csr_matrix((weights,edges),shape=(graph.num_nodes,graph.num_nodes))
    upper=sparse.triu((matrix+matrix.T)/2,k=1,format='coo')
    truth=(graph.y.numpy()[upper.row]==graph.y.numpy()[upper.col]).astype(int)
    order=np.argsort(-upper.data,kind='stable')
    result=dict(edge_count=len(truth),same_class_base_rate=float(truth.mean()),
                auroc=float(roc_auc_score(truth,upper.data)) if len(np.unique(truth))==2 else None,
                ap=float(average_precision_score(truth,upper.data)) if truth.any() else None,
                precision={str(rate):float(np.mean(truth[order[:max(1,int(rate*len(order)))]])) for rate in [.01,.05,.1,.2]},
                use='post-freeze diagnosis only; not checkpoint or edge threshold selection')
    write(folder/'edge_diagnostic.json',result)


def run_a(names,seeds,device):
    for name in names:
        hpath=ROOT/'representations'/name/'H.pt'
        summary=json.loads((hpath.parent/'summary.json').read_text(encoding='utf-8'))
        if sha(hpath)!=summary['H_sha256']:raise ValueError('GREET H changed')
        h=torch.load(hpath,map_location='cpu',weights_only=True)
        h=torch.nn.functional.normalize(h.float(),dim=1)
        for seed in seeds:
            common=ROOT/'partitions'/name/f'seed{seed}'; common.mkdir(parents=True,exist_ok=True)
            path=common/'assignments.npz'
            if path.exists():
                with np.load(path) as a:gb=a['GB'].copy(); km=a['KM'].copy()
            else:
                status('A_partitions',dataset=name,seed=seed)
                logging.info('A PARTITION %s seed%d',name,seed)
                balls=build_granular_balls(h,**CONFIG['balls'],random_state=seed); gb=balls.assignment
                km=KMeans(n_clusters=len(np.unique(gb)),n_init=10,random_state=seed).fit_predict(h.numpy())
                np.savez_compressed(path,GB=gb,KM=km)
                write(common/'config.json',dict(H_sha256=sha(hpath),assignment_sha256=sha(path),
                    actual_balls=len(np.unique(gb)),rules=CONFIG['balls'],seed=seed))
            graph,original,base,protocol=graph_protocol(name,seed,'clean')
            train=graph.train_index.numpy(); classes=graph.num_classes
            counts=np.bincount(gb[train])[gb[train]]-1
            rel_path=common/'knn.npz'
            if rel_path.exists():relation=sparse.load_npz(rel_path)
            else:
                relation=old.knn_weights(h[train].numpy(),counts,device=device)
                sparse.save_npz(rel_path,relation)
            edge_diagnostic(name,graph)
            for condition in CONFIG['conditions']:
                folder=ROOT/'targets'/name/'A'/f'seed{seed}'/condition; folder.mkdir(parents=True,exist_ok=True)
                if (folder/'metrics.json').exists():continue
                graph,original,base,protocol=graph_protocol(name,seed,condition)
                observed=graph.noisy_y[train].numpy()
                bundles={'GRACE_GB':base,'GREET_GB':ball_bundle(gb[train],observed,classes),
                         'GREET_KM':ball_bundle(km[train],observed,classes)}
                _,q,a,support,_=old.vote(relation,observed,classes)
                bundles['GREET_KNN']=dict(q=q,alpha=a,target=mix(q,a,observed),support=support)
                hashes={arm:save_bundle(folder,arm,bundle) for arm,bundle in bundles.items()}
                write(folder/'protocol.json',protocol); write(folder/'frozen_targets.json',hashes)
                # All methods frozen above; clean labels now enter retrospective metrics.
                truth=graph.y[train].numpy(); rows=[]
                for arm,bundle in bundles.items():
                    row=dict(dataset=name,stage='A',seed=seed,noise=condition,arm=arm,
                             actual_balls=int(len(np.unique(gb if arm.startswith('GREET') else original))),
                             target_sha256=hashes[arm],**bundle_metrics(bundle,truth,observed))
                    if arm in ['GRACE_GB','GREET_GB','GREET_KM']:
                        group=original if arm=='GRACE_GB' else gb if arm=='GREET_GB' else km
                        clean=ball_bundle(group[train],truth,classes)
                        rival=clean['r'].copy(); rival[np.arange(len(truth)),truth]=-1
                        margin=clean['r'][np.arange(len(truth)),truth]-rival.max(1)
                        row['positive_clean_margin_coverage']=float(((margin>0)&(bundle['support']>=2)).mean())
                    rows.append(row)
                write(folder/'metrics.json',rows); aggregate_targets()
                logging.info('A DONE %s seed%d %s GB Brier %.6f',name,seed,condition,rows[1]['target_brier'])
        del h,graph
        if torch.cuda.is_available():torch.cuda.empty_cache()


def run_b(names,seeds,device):
    for name in names:
        status('B_features',dataset=name)
        graph=old.load_graph(name,0)
        low,high,h=filtered_features(graph.x.numpy(),graph.edge_index.numpy())
        features=dict(attribute=graph.x.numpy(),low=low,high=high)
        for seed in seeds:
            for condition in CONFIG['conditions']:
                folder=ROOT/'targets'/name/'B'/f'seed{seed}'/condition; folder.mkdir(parents=True,exist_ok=True)
                if (folder/'metrics.json').exists():continue
                graph,assignment,base,protocol=graph_protocol(name,seed,condition)
                train=graph.train_index.numpy(); observed=graph.noisy_y[train].numpy(); h_local=h[train]
                aux_folder=ROOT/'auxiliary'/name/f'seed{seed}'/condition; aux_folder.mkdir(parents=True,exist_ok=True)
                status('B_crossfit',dataset=name,seed=seed,noise=condition)
                losses,folds=cross_fit_scores(features,observed,train,graph.num_classes,CONFIG['auxiliary'],seed,aux_folder,device)
                packed={f'{channel}_{epoch}':value for channel,epochs in losses.items() for epoch,value in epochs.items()}
                packed['h']=h_local
                np.savez_compressed(aux_folder/'scores.npz',**packed)
                write(aux_folder/'protocol.json',protocol)
                scores={'ATTR':losses['attribute'][30],'LOW':losses['low'][30],'HIGH':losses['high'][30],
                        'FIXED':(losses['low'][30]+losses['high'][30])/2,
                        'ADAPTIVE':h_local*losses['low'][30]+(1-h_local)*losses['high'][30]}
                bundles={'GRACE_GB':base}
                for key,score in scores.items():bundles['HF_'+key]=gated_bundle(base,observed,score)
                desired=bundles['HF_ADAPTIVE']['alpha'].sum()
                constant=desired/base['alpha'].sum() if base['alpha'].sum() else 0
                shuffled=np.random.RandomState(seed+251015).permutation(rank_gate(scores['ADAPTIVE']))
                for key,a in [('HF_CONST',base['alpha']*constant),('HF_SHUFFLE',match_budget(base['alpha'],shuffled,desired))]:
                    bundles[key]=dict(q=base['q'],alpha=a,target=mix(base['q'],a,observed),support=base['support'])
                if abs(bundles['HF_SHUFFLE']['alpha'].sum()-desired)>1e-8:raise ValueError('Budget mismatch')
                np.save(folder/'adaptive_gate.npy',rank_gate(scores['ADAPTIVE']))
                hashes={arm:save_bundle(folder,arm,bundle) for arm,bundle in bundles.items()}
                write(folder/'protocol.json',protocol); write(folder/'frozen_targets.json',hashes)
                truth=graph.y[train].numpy(); corruption=(observed!=truth).astype(int); rows=[]
                detection={key:dict(auroc=float(roc_auc_score(corruption,score)),ap=float(average_precision_score(corruption,score)))
                           if len(np.unique(corruption))==2 else dict(auroc=None,ap=None) for key,score in scores.items()}
                write(aux_folder/'detection.json',dict(noise_base_rate=float(corruption.mean()),scores=detection))
                for arm,bundle in bundles.items():
                    rows.append(dict(dataset=name,stage='B',seed=seed,noise=condition,arm=arm,
                                     target_sha256=hashes[arm],**bundle_metrics(bundle,truth,observed)))
                write(folder/'metrics.json',rows); aggregate_targets()
                logging.info('B DONE %s seed%d %s adaptive Brier %.6f',name,seed,condition,
                             next(row['target_brier'] for row in rows if row['arm']=='HF_ADAPTIVE'))
        del features,low,high,h,graph
        if torch.cuda.is_available():torch.cuda.empty_cache()


def decision():
    frame=pd.DataFrame(aggregate_targets()); table=frame.pivot(index=['dataset','seed','noise','stage'],columns='arm',values='target_brier')
    records=[]
    for name in CONFIG['datasets']:
        a=table.xs((name,'A'),level=('dataset','stage')); a=a[a.index.get_level_values('noise')!='clean']
        b=table.xs((name,'B'),level=('dataset','stage')); b=b[b.index.get_level_values('noise')!='clean']
        delta_a=a['GREET_GB']-a['GRACE_GB']
        comparisons={arm:float((b['HF_ADAPTIVE']-b[arm]).mean()) for arm in
                     ['GRACE_GB','HF_CONST','HF_SHUFFLE','HF_ATTR','HF_LOW','HF_HIGH','HF_FIXED']}
        a_pass=bool(delta_a.mean()<=-CONFIG['gates']['brier_improvement'] and
                    (delta_a.groupby(level='noise').mean()<0).all())
        b_pass=bool(comparisons['GRACE_GB']<=-CONFIG['gates']['brier_improvement'] and
                    comparisons['HF_CONST']<=-CONFIG['gates']['brier_improvement'] and
                    all(value<0 for arm,value in comparisons.items() if arm!='GRACE_GB') and
                    ((b['HF_ADAPTIVE']-b['GRACE_GB']).groupby(level='noise').mean()<0).all())
        records.append(dict(dataset=name,greet_minus_grace_brier=float(delta_a.mean()),greet_pass=a_pass,
                            hf_differences=comparisons,hf_pass=b_pass,
                            greet_minus_km=float((a['GREET_GB']-a['GREET_KM']).mean()),
                            greet_minus_knn=float((a['GREET_GB']-a['GREET_KNN']).mean())))
    result=dict(status='screened',datasets=records,A_pass=sum(r['greet_pass'] for r in records)>=CONFIG['gates']['minimum_datasets'],
                B_pass=sum(r['hf_pass'] for r in records)>=CONFIG['gates']['minimum_datasets'],
                criterion='Exploratory engineering gates, not statistical significance',student_runs_completed=0)
    write(ROOT/'decision.json',result); return result


def run_students(result,names,seeds,device):
    if not (result['A_pass'] or result['B_pass']):return
    arms=[('A','GRACE_GB')]
    if result['A_pass']:arms += [('A',arm) for arm in ['GREET_GB','GREET_KM','GREET_KNN']]
    if result['B_pass']:arms += [('B',arm) for arm in ['HF_ADAPTIVE','HF_ATTR','HF_LOW','HF_HIGH','HF_FIXED','HF_CONST','HF_SHUFFLE']]
    for name in names:
        for seed in seeds:
            for condition in CONFIG['conditions']:
                graph,_,_,protocol=graph_protocol(name,seed,condition)
                candidates=[('CE',None)]
                for stage,arm in arms:
                    path=ROOT/'targets'/name/stage/f'seed{seed}'/condition/f'{arm}.npz'
                    with np.load(path) as a:local=a['target'].copy()
                    candidates.append((arm,local))
                if result['A_pass'] and result['B_pass']:
                    path=ROOT/'targets'/name/'A'/f'seed{seed}'/condition/'GREET_GB.npz'
                    with np.load(path) as a:q=a['q'].copy();alpha=a['alpha'].copy()
                    gate=np.load(ROOT/'targets'/name/'B'/f'seed{seed}'/condition/'adaptive_gate.npy')
                    observed=graph.noisy_y[graph.train_index].numpy()
                    candidates.append(('FUSION',mix(q,alpha*gate,observed)))
                for arm,local in candidates:
                    folder=ROOT/'students'/name/f'seed{seed}'/condition/arm; folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'result.json').exists():continue
                    targets=torch.nn.functional.one_hot(graph.noisy_y,graph.num_classes).float()
                    if local is not None:targets[graph.train_index]=torch.as_tensor(local,dtype=torch.float32)
                    torch.save(targets,folder/'targets.pt')
                    write(folder/'config.json',dict(protocol=protocol,training=CONFIG['student'],arm=arm,
                                                  targets_sha256=sha(folder/'targets.pt')))
                    status('student',dataset=name,seed=seed,noise=condition,arm=arm)
                    set_seed(seed); start=time.time()
                    trained=train_x_only_mlp(graph,targets,device=device,**CONFIG['student'])
                    torch.save(trained.logits,folder/'logits.pt')
                    accuracy=float((trained.logits[graph.test_index].argmax(1)==graph.y[graph.test_index]).float().mean())
                    if accuracy!=trained.test_accuracy:raise ValueError('Saved logits inconsistent')
                    write(folder/'result.json',dict(dataset=name,seed=seed,noise=condition,arm=arm,
                        test_accuracy=accuracy,validation_accuracy=trained.validation_accuracy,best_epoch=trained.best_epoch,
                        elapsed_seconds=time.time()-start,targets_sha256=sha(folder/'targets.pt'),logits_sha256=sha(folder/'logits.pt')))
                    logging.info('STUDENT %s seed%d %s %s %.4f',name,seed,condition,arm,accuracy)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['A','B','finish','all'],default='all')
    parser.add_argument('--datasets',nargs='+',choices=CONFIG['datasets'])
    parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--wait-pretrained',action='store_true')
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(ROOT/'pipeline_raw.log',encoding='utf-8')])
    names=args.datasets or CONFIG['datasets']; device=args.device or CONFIG['device']
    if args.datasets and args.stage in ['all','finish']:parser.error('Partial datasets only supported in A/B stages')
    torch.set_num_threads(CONFIG['threads']); freeze_sources()
    try:
        if args.wait_pretrained:
            deadline=time.time()+3*3600
            while not all((ROOT/'representations'/name/'summary.json').exists() for name in names):
                if time.time()>deadline:raise TimeoutError('Waiting for pretraining exceeded three hours')
                status('waiting_for_pretrained',datasets=names)
                time.sleep(20)
        with threadpool_limits(limits=CONFIG['threads']):
            if args.stage=='all':
                for name in names:prepare(name,CONFIG['greet'],device)
            if args.stage in ['A','all']:run_a(names,CONFIG['protocol_seeds'],device)
            if args.stage in ['B','all']:run_b(names,CONFIG['protocol_seeds'],device)
            if args.stage in ['finish','all']:
                result=decision(); run_students(result,names,CONFIG['protocol_seeds'],device)
                import analyze_audit
                analyze_audit.main()
        write(ROOT/'pipeline_status.json',dict(status='completed',stage=args.stage,datasets=names,updated_local=time.strftime('%Y-%m-%d %H:%M:%S')))
    except Exception as exc:
        write(ROOT/'pipeline_status.json',dict(status='failed',stage=args.stage,error=str(exc)))
        logging.exception('Pipeline failed'); raise


if __name__=='__main__':main()
