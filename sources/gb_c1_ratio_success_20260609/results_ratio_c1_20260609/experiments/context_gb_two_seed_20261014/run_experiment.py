"""Resumable two-seed, label-free pretraining and frozen GB-LCSS evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import dropout_edge
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE/'GB-LCSS_code/src'))
from gblcss.data import GraphData, percent_split_indices, inject_train_label_noise, set_seed
from gblcss.grace import _row_normalize_features, _drop_features
from gblcss.granular_ball import build_granular_balls
from gblcss.targets import build_leave_one_out_targets, TargetResult
from gblcss.diagnostics import target_metrics
from gblcss.training import train_x_only_mlp, TrainResult, accuracy, soft_cross_entropy
from gblcss.models import GCN
import core
from pingda import build_generators, sample_augmentation

DATA = {
    'cora': PROJECT/'external/NoisyGL/data/Cora/processed/data.pt',
    'citeseer': PROJECT/'external/NoisyGL/data/CiteSeer/processed/data.pt',
    'corafull': PROJECT/'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
}
ARMS = ['grace', 'pingda', 'pingda_context', 'pingda_hreg']
SPACES = ['grace_h','pingda_h','pingda_mean_r','context_h','context_r','hreg_h']
TRAIN = dict(hidden=64, dropout=.5, learning_rate=.01, weight_decay=5e-4, epochs=200)
NOISE = {'clean':0., 'uniform':.4, 'pair':.4}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        if torch.is_tensor(a): a = a.detach().cpu().numpy()
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def csv_write(path, rows):
    if not rows:return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def load_graph(name, seed):
    raw = torch.load(DATA[name], map_location='cpu', weights_only=False)[0]
    if isinstance(raw,dict): x,edges,y=raw['x'],raw['edge_index'],raw['y']
    else:x,edges,y=raw.x,raw.edge_index,raw.y
    expected={'cora':(2708,1433,7),'citeseer':(3327,3703,6),'corafull':(19793,8710,70)}
    if (len(y),x.shape[1],int(y.max())+1)!=expected[name]:
        raise ValueError(f'dataset identity mismatch for {name}')
    tr,va,te=percent_split_indices(len(y),seed)
    def mask(idx):
        result=torch.zeros(len(y),dtype=torch.bool);result[torch.as_tensor(idx)]=True;return result
    return GraphData(name,x.float(),edges.long(),y.long(),mask(tr),mask(va),mask(te),
                     train_index=torch.as_tensor(tr),val_index=torch.as_tensor(va),test_index=torch.as_tensor(te))


def pretrain(features, edges, config, arm, seed, device, folder, epochs, chunk):
    """Only X/A are accepted; no label or split argument enters SSL."""
    set_seed(seed)
    model = core.WeightedGRACE(features.shape[1], config['hidden_dim'], config['projection_dim'],
                               config['temperature'], config['activation']).to(device)
    x=_row_normalize_features(features.float()).to(device);edges=edges.to(device)
    head=None; edge_gen=None;feat_gen=None
    if arm!='grace':
        edge_gen,feat_gen=build_generators(x.shape[1],device)
    if arm in ['pingda_context','pingda_hreg']:
        # Isolate head initialization so node/generator RNG matches the no-head arm.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
            torch.manual_seed(seed+17001)
            head=nn.Sequential(nn.Linear(config['hidden_dim'],128),nn.ReLU(),nn.Linear(128,128)).to(device)
    parameters=list(model.parameters())+(list(head.parameters()) if head is not None else [])
    optimizer=torch.optim.Adam(parameters,lr=config['learning_rate'],weight_decay=1e-5)
    edge_opt=feat_opt=None
    if edge_gen is not None:
        edge_opt=torch.optim.Adam(edge_gen.parameters(),lr=.0001,weight_decay=.0001)
        feat_opt=torch.optim.Adam(feat_gen.parameters(),lr=.001,weight_decay=.0001)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    records=[];start=time.time()
    for epoch in range(epochs):
        model.train();optimizer.zero_grad(set_to_none=True)
        if edge_opt is not None:
            edge_opt.zero_grad(set_to_none=True);feat_opt.zero_grad(set_to_none=True)
            x1,_=sample_augmentation(x,edges,edge_gen,feat_gen,learnable_edge=False)
            x2,w2=sample_augmentation(x,edges,edge_gen,feat_gen)
            e1=e2=edges
            h1=model(x1,e1);h2=model(x2,e2,w2)
        else:
            e1,_=dropout_edge(edges,p=config['edge_drop_rates'][0],training=True)
            e2,_=dropout_edge(edges,p=config['edge_drop_rates'][1],training=True)
            h1=model(_drop_features(x,config['feature_mask_rates'][0]),e1)
            h2=model(_drop_features(x,config['feature_mask_rates'][1]),e2)
        node=core.contrastive_loss(model.project(h1),model.project(h2),config['temperature'],chunk)
        ctx_loss=node.new_zeros(());stats={'valid_roots':0,'invariance':0.,'variance':0.,'covariance':0.}
        extra_grads=None
        if head is not None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
                torch.manual_seed(seed*100000+epoch+27001)
                if arm=='pingda_context':
                    s1=core.sample_context_edges(e1,len(x),16);s2=core.sample_context_edges(e2,len(x),16)
                    m1,v1=core.neighbor_mean(h1,s1);m2,v2=core.neighbor_mean(h2,s2)
                    valid=v1&v2
                else:m1,m2=h1,h2;valid=torch.ones(len(x),dtype=torch.bool,device=device)
                ids=torch.nonzero(valid).flatten()
                if len(ids)>1024:ids=ids[torch.randperm(len(ids),device=device)[:1024]]
            c1,c2=head(m1[ids]),head(m2[ids])
            ctx_loss,stats=core.vicreg(c1,c2,torch.ones(len(ids),dtype=torch.bool,device=device))
            weighted=.1*ctx_loss/25
            extra_grads=torch.autograd.grad(weighted,parameters,retain_graph=True,allow_unused=True)
        if not torch.isfinite(node+ctx_loss):raise FloatingPointError(f'{arm} nonfinite loss at {epoch}')
        node.backward()
        if extra_grads is not None:
            for p,g in zip(parameters,extra_grads):
                if g is not None:p.grad=g if p.grad is None else p.grad+g
        optimizer.step()
        if edge_opt is not None:edge_opt.step();feat_opt.step()
        records.append(dict(epoch=epoch+1,node_loss=float(node.detach()),context_loss=float(ctx_loss.detach()),
                            elapsed_seconds=time.time()-start,**stats))
        if (epoch+1)%20==0 or epoch==0:
            csv_write(folder/'loss.csv',records)
            logging.info('SSL %s seed%d epoch%d/%d node=%.5f ctx=%.5f seconds=%.1f',
                          arm,seed,epoch+1,epochs,float(node.detach()),float(ctx_loss.detach()),time.time()-start)
    model.eval()
    if head is not None:head.eval()
    with torch.no_grad():
        h=model(x,edges);m,valid=core.neighbor_mean(h,edges)
        c=head(m) if head is not None and arm=='pingda_context' else m
        out=dict(h=h.cpu(),c=c.cpu(),structural_valid=valid.cpu())
    torch.save({'model':model.state_dict(),'head':head.state_dict() if head is not None else None,
                'edge_generator':edge_gen.state_dict() if edge_gen is not None else None,
                'feature_generator':feat_gen.state_dict() if feat_gen is not None else None},folder/'checkpoint.pt')
    csv_write(folder/'loss.csv',records)
    del model,head,edge_gen,feat_gen,x,edges
    if device=='cuda':torch.cuda.empty_cache()
    return out,dict(elapsed_seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0,
                    h_geometry=core.geometry(out['h']),c_geometry=core.geometry(out['c']))


def train_gcn(data,targets,device,epochs):
    set_seed(int(data._experiment_seed))
    model=GCN(data.x.shape[1],64,data.num_classes,2,.5).to(device)
    x=data.x.to(device);edges=data.edge_index.to(device)
    adjacency=torch.sparse_coo_tensor(edges,torch.ones(edges.shape[1],device=device),
        (data.num_nodes,data.num_nodes)).coalesce()
    tr=data.train_index.to(device);va=data.val_index.to(device);y=data.y.to(device)
    targets=targets.to(device);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=5e-4)
    best=-1.;best_epoch=0;state=None
    for epoch in range(epochs):
        model.train();opt.zero_grad();logits=model(x,adjacency)
        loss=soft_cross_entropy(logits[tr],targets[tr]);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():val=accuracy(model(x,adjacency)[va],y[va])
        if val>best:best=val;best_epoch=epoch+1;state=deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    with torch.no_grad():logits=model(x,adjacency).cpu()
    return TrainResult(accuracy(logits[data.train_mask],data.noisy_y[data.train_mask]),best,
        accuracy(logits[data.test_mask],data.y[data.test_mask]),best_epoch,logits)


def make_spaces(outputs):
    spaces={}
    for arm,prefix in [('grace','grace_h'),('pingda','pingda_h'),('pingda_context','context_h'),('pingda_hreg','hreg_h')]:
        spaces[prefix]=(outputs[arm]['h'],None)
    for arm,prefix in [('pingda','pingda_mean_r'),('pingda_context','context_r')]:
        r,valid=core.combine_space(outputs[arm]['h'],outputs[arm]['c'],.5,outputs[arm]['structural_valid'])
        spaces[prefix]=(r,valid)
    return spaces


def targets_for_space(data,assignment,valid,base_assignment):
    tr=data.train_index
    if valid is None:
        return build_leave_one_out_targets(data.noisy_y,assignment,tr,data.num_classes),assignment
    # Eligible-space LOO votes use only valid members. Missing contexts fall back to
    # the full H partition's original targets, never mixed padded coordinates.
    indices=torch.nonzero(valid).flatten()
    local_train=torch.nonzero(data.train_mask[indices]).flatten()
    local=build_leave_one_out_targets(data.noisy_y[indices],assignment,local_train,data.num_classes)
    base=build_leave_one_out_targets(data.noisy_y,base_assignment,tr,data.num_classes)
    target=base.soft_target.clone();alpha=base.alpha.clone();pseudo=base.pseudo_label.clone()
    target[indices]=local.soft_target;alpha[indices]=local.alpha;pseudo[indices]=local.pseudo_label
    full=base_assignment.copy()+int(assignment.max()+1)
    full[indices.numpy()]=assignment
    return TargetResult(target,alpha,pseudo,{'fallback_nodes':int((~valid).sum())}),full


def summarize(output):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((output/'students').glob('*/seed*/*/*/*/result.json'))]
    csv_write(output/'results.csv',rows)
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ['dataset','noise','head','space','budget'])
        groups.setdefault(key,[]).append(row['test_accuracy']*100)
    summary=[dict(zip(['dataset','noise','head','space','budget'],key),seed_count=len(values),
                  mean_accuracy_pct=float(np.mean(values)),sample_std_pp=float(np.std(values,ddof=1)) if len(values)>1 else None)
             for key,values in sorted(groups.items())]
    csv_write(output/'summary.csv',summary)
    lookup={tuple(r[k] for k in ['dataset','seed','noise','head','space','budget']):r for r in rows}
    paired=[]
    for key,row in lookup.items():
        for reference in ['CE','grace_h','pingda_h','pingda_mean_r','hreg_h']:
            if row['space']==reference:continue
            ref=lookup.get((*key[:4],reference,0 if reference=='CE' else key[5]))
            if ref:paired.append(dict(dataset=key[0],seed=key[1],noise=key[2],head=key[3],space=key[4],
                budget=key[5],reference=reference,accuracy_delta_pp=100*(row['test_accuracy']-ref['test_accuracy'])))
    csv_write(output/'paired_deltas.csv',paired)
    return len(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=list(DATA));parser.add_argument('--seeds',nargs='+',type=int,default=[0,1])
    parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--student-epochs',type=int,default=200);parser.add_argument('--chunk-size',type=int,default=1024)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--stage',choices=['all','pretrain','downstream'],default='all')
    args=parser.parse_args();torch.set_num_threads(4)
    output=ROOT/('smoke' if args.smoke else 'runs');output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(output/'raw.log',encoding='utf-8')])
    cfg=yaml.safe_load((WORKSPACE/'GB-LCSS_code/configs/representations/grace.yaml').read_text(encoding='utf-8'))
    sources=[Path(__file__),ROOT/'core.py',ROOT/'pingda.py',ROOT/'PLAN.md',
             WORKSPACE/'GB-LCSS_code/configs/representations/grace.yaml']
    sources += [WORKSPACE/'GB-LCSS_code/src/gblcss'/n for n in ['data.py','targets.py','granular_ball.py','training.py','models.py','diagnostics.py','grace.py']]
    sources += sorted((WORKSPACE/'vendor/PiNGDA').glob('*.py'))
    sources += [WORKSPACE/'vendor/PiNGDA/provenance.json']
    manifest={str(p.resolve()):sha(p) for p in sources}
    manifest_path=output/'source_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('source changed since prior run; create a new version')
    write(manifest_path,manifest)
    config=dict(datasets=args.datasets,seeds=args.seeds,arms=ARMS,spaces=SPACES,budgets=[64,128,256],
        noise=NOISE,pretrain_epochs=args.epochs,student_epochs=args.student_epochs,device=args.device,
        context_lambda=.1,context_mix=.5,vote_mix=.5,context_neighbors=16,context_dim=128,
        node_negatives='all exact block recomputation',chunk_size=args.chunk_size,
        scope='two complete seeds exploratory; no KNN/KMeans comparator',
        data={name:{'path':str(DATA[name]),'sha256':sha(DATA[name])} for name in args.datasets},
        environment={'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
                     'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None})
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        # Environment platform is stable; stage alone intentionally doesn't change config.
        raise ValueError('config differs from previous run')
    write(config_path,config)
    status=dict(status='running',stage=args.stage,started=time.time(),
                completed_pretrains=len(list((output/'pretrain').glob('*/seed*/*/complete.json'))),
                completed_students=len(list((output/'students').glob('*/seed*/*/*/*/result.json'))))
    write(output/'status.json',status)
    try:
        with threadpool_limits(limits=4):
            for name in args.datasets:
                for seed in args.seeds:
                    graph=load_graph(name,seed)
                    protocol=dict(split_hash=fingerprint(graph.train_index,graph.val_index,graph.test_index),
                        nodes=graph.num_nodes,classes=graph.num_classes,edges=graph.edge_index.shape[1])
                    write(output/'protocol'/name/f'seed{seed}.json',protocol)
                    outputs={}
                    for arm in ARMS:
                        folder=output/'pretrain'/name/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
                        if (folder/'complete.json').exists():
                            outputs[arm]=torch.load(folder/'representation.pt',weights_only=True)
                            if sha(folder/'representation.pt')!=json.loads((folder/'complete.json').read_text(encoding='utf-8'))['sha256']:
                                raise ValueError('representation changed')
                            continue
                        if args.stage=='downstream':raise ValueError('missing pretraining '+str(folder))
                        logging.info('START SSL %s seed%d %s',name,seed,arm)
                        write(folder/'config.json',dict(arm=arm,seed=seed,dataset=name,epochs=args.epochs,
                            base=cfg['datasets'][name],split_not_used=True,labels_not_used=True))
                        result,info=pretrain(graph.x,graph.edge_index,cfg['datasets'][name],arm,seed,args.device,folder,args.epochs,args.chunk_size)
                        torch.save(result,folder/'representation.pt')
                        write(folder/'complete.json',dict(sha256=sha(folder/'representation.pt'),**info))
                        outputs[arm]=result;status['completed_pretrains']+=1;write(output/'status.json',status)
                    if args.stage=='pretrain':continue
                    spaces=make_spaces(outputs);partitions={}
                    for space,(h,valid) in spaces.items():
                        for budget in [64,128,256]:
                            folder=output/'partitions'/name/f'seed{seed}'/space/f'B{budget}'
                            if (folder/'assignment.npy').exists():
                                partitions[(space,budget)]=np.load(folder/'assignment.npy');continue
                            logging.info('BALLS %s seed%d %s B%d',name,seed,space,budget)
                            subset=h[valid] if valid is not None else h
                            if len(subset)==0:raise ValueError('all contexts invalid')
                            balls=build_granular_balls(subset,max_balls=budget,min_ball_size=32,split_improvement=.05,random_state=seed)
                            folder.mkdir(parents=True,exist_ok=True);np.save(folder/'assignment.npy',balls.assignment)
                            write(folder/'summary.json',dict(summary=balls.summary,balls=balls.balls,
                                valid_nodes=len(subset),total_nodes=len(h),validity_hash=fingerprint(valid) if valid is not None else None))
                            partitions[(space,budget)]=balls.assignment
                    for noise,rate in NOISE.items():
                        data=inject_train_label_noise(graph,'uniform' if noise=='clean' else noise,rate,seed)
                        data._experiment_seed=seed
                        noise_hash=fingerprint(data.noisy_y[data.train_index])
                        onehot=F.one_hot(data.noisy_y,data.num_classes).float()
                        jobs=[('CE',0,onehot,None)]
                        for space,(h,valid) in spaces.items():
                            for budget in [64,128,256]:
                                assignment=partitions[(space,budget)]
                                base_space='context_h' if space=='context_r' else 'pingda_h'
                                result,full_assignment=targets_for_space(data,assignment,valid,
                                    partitions[(base_space,budget)] if valid is not None else assignment)
                                metrics=target_metrics(data,full_assignment,result)
                                metrics['fallback_nodes']=int((~valid).sum()) if valid is not None else 0
                                metrics['clean_margin_scope']='hybrid assignment diagnostic; fallback full-H votes may differ' if valid is not None else 'exact'
                                folder=output/'targets'/name/f'seed{seed}'/noise/space/f'B{budget}'
                                folder.mkdir(parents=True,exist_ok=True)
                                torch.save({'targets':result.soft_target,'alpha':result.alpha,'assignment':torch.as_tensor(full_assignment)},folder/'targets.pt')
                                write(folder/'metrics.json',dict(dataset=name,seed=seed,noise=noise,space=space,budget=budget,
                                    split_hash=protocol['split_hash'],noise_hash=noise_hash,**metrics))
                                jobs.append((space,budget,result.soft_target,metrics))
                        for space,budget,targets,metrics in jobs:
                            for head_name in ['mlp','gcn']:
                                folder=output/'students'/name/f'seed{seed}'/noise/head_name/f'{space}_B{budget}'
                                if (folder/'result.json').exists():continue
                                folder.mkdir(parents=True,exist_ok=True)
                                target_hash=fingerprint(targets[data.train_index])
                                write(folder/'config.json',dict(dataset=name,seed=seed,noise=noise,head=head_name,space=space,budget=budget,
                                    split_hash=protocol['split_hash'],noise_hash=noise_hash,target_hash=target_hash,
                                    training=dict(TRAIN,epochs=args.student_epochs)))
                                set_seed(seed);start=time.time()
                                logging.info('START CLS %s seed%d %s %s %s B%d',name,seed,noise,head_name,space,budget)
                                result=train_x_only_mlp(data,targets,device=args.device,**dict(TRAIN,epochs=args.student_epochs)) if head_name=='mlp' else train_gcn(data,targets,args.device,args.student_epochs)
                                torch.save(result.logits,folder/'logits.pt')
                                row=dict(dataset=name,seed=seed,noise=noise,head=head_name,space=space,budget=budget,
                                    test_accuracy=result.test_accuracy,validation_accuracy=result.validation_accuracy,
                                    train_accuracy=result.train_accuracy,best_epoch=result.best_epoch,elapsed_seconds=time.time()-start,
                                    split_hash=protocol['split_hash'],noise_hash=noise_hash,target_hash=target_hash,
                                    num_balls=metrics['num_balls'] if metrics else 0,
                                    soft_target_brier=metrics['soft_target_brier'] if metrics else None)
                                write(folder/'result.json',row);status['completed_students']+=1
                                write(output/'status.json',status)
                        summarize(output)
            for p,digest in manifest.items():
                if sha(p)!=digest:raise ValueError('source changed during experiment: '+p)
        status.update(status='completed',elapsed_seconds=time.time()-status['started'],student_files=summarize(output))
    except Exception as exc:
        status.update(status='failed',error=repr(exc));write(output/'status.json',status);raise
    write(output/'status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
