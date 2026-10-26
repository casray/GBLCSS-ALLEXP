"""Frozen support/neighbor ablation; fitting never scores test labels."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'neighbor_preserving_ball_20261022'
FUSION = ROOT.parent/'gce_lcss_fusion_20261022'
OLD = ROOT.parent/'representation_supervision_20261021'
sys.path.insert(0, str(OLD))
from core import AlignedMLP, make_targets, state_hash
from protocol import load_dataset, raw_data, read, write, sha, fingerprint
from gblcss.data import set_seed, percent_split_indices
from gblcss.models import MLP
sys.path.insert(0, str(BASE))
import generator as g
sys.path.insert(0, str(ROOT))
from kernels import support_costs, count_matched_cut, cost_components

CONFIG = read(ROOT/'config.json')


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def initialize():
    torch.set_num_threads(CONFIG['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('Verified GPU required')


def verify():
    for file, digest in read(ROOT/'frozen_inputs.json')['sha256'].items():
        if sha(file) != digest:
            raise RuntimeError('Frozen source/input modified: '+file)


def freeze_sources():
    """Finish source freezing after parallel analysis implementation, before fits."""
    if list((ROOT/'runs').glob('**/result.json')) or list((ROOT/'smoke').glob('**/result.json')):
        raise RuntimeError('Source freeze must precede all fitting')
    required = ['run.py','analyze.py','kernels.py','test_kernels.py','PLAN.md','config.json']
    for name in required:
        if not (ROOT/name).is_file():
            raise RuntimeError('Missing required source: '+name)
    manifest = read(ROOT/'frozen_inputs.json')
    if manifest['config']!=CONFIG:
        raise RuntimeError('Prepared configuration differs; cached inputs cannot be reused')
    for file,digest in manifest['sha256'].items():
        local_code=Path(file).parent==ROOT and Path(file).suffix=='.py'
        if not local_code and sha(file)!=digest:
            raise RuntimeError('Input/historical source changed before final freeze: '+file)
    shutil.copy2(ROOT/'frozen_inputs.json',ROOT/'preparation_manifest.json')
    for file in [*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'config.json',ROOT/'environment.json',ROOT/'geometry.csv']:
        manifest['sha256'][str(file)]=sha(file)
    manifest['stage']='all sources frozen before smoke and full fits'
    write(ROOT/'frozen_inputs.json',manifest)
    verify()
    print('All required sources frozen before fitting',flush=True)


def input_dir(name, seed):
    return ROOT/'inputs'/name/f'seed{seed}'


def encoder_path(name, seed):
    return (OLD/'pretraining'/name/'model.pt' if seed == 0 else
            FUSION/'encoders'/name/f'seed{seed}'/'model.pt')


def reference_run(name, seed, condition, route, arm):
    if arm in ('CE', 'OLD64'):
        return FUSION/'runs'/name/f'seed{seed}'/condition/(route+'_'+('LCSS' if arm=='OLD64' else 'CE'))
    if seed == 0 and arm in ('NP', 'KMNP', 'KNNMATCH'):
        return BASE/'runs'/name/condition/(route+'_'+('KM' if arm=='KMNP' else arm))
    return None


def counts_for_partition(z, assignment, classes):
    counts = np.zeros((len(z), classes), np.float32)
    for ball in np.unique(assignment):
        rows = np.flatnonzero(assignment == ball)
        counts[rows] = np.bincount(z[rows], minlength=classes)
    counts[np.arange(len(z)), z] -= 1
    return counts


def reference_targets(z, counts):
    classes = counts.shape[1]
    out = np.eye(classes)[z].copy()
    for row, count in enumerate(counts):
        if count.sum() < 2:
            continue
        vote = count.astype(float)/count.sum()
        weight = (classes*vote.max()-1)/(classes-1)
        winners = np.flatnonzero(count == count.max())
        q = vote if len(winners)>1 else .5*vote+.5*np.eye(classes)[winners[0]]
        out[row] = (1-weight)*out[row]+weight*q
    return out


def prepare():
    initialize()
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Already frozen; use run')
    sources = dict(read(BASE/'frozen_inputs.json')['sha256'])
    sources.update(read(FUSION/'frozen_inputs.json')['sha256'])
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError('Historical source changed: '+path)
    geometry, checks = [], []
    for name in CONFIG['datasets']:
        data = load_dataset(name, 'cpu')
        h = g.normalize(data['h'].numpy())
        for seed in CONFIG['seeds']:
            source = FUSION/'inputs'/name/f'seed{seed}'
            assets = torch.load(source/'assets.pt', weights_only=True)
            train = assets['train'].numpy()
            expected = percent_split_indices(len(h), seed)
            for key, value in zip(('train','valid','test'), expected):
                if not np.array_equal(assets[key].numpy(), value):
                    raise RuntimeError('Split order mismatch')
            folder = input_dir(name, seed)
            folder.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            if seed == 0:
                stored_tree = torch.load(BASE/'inputs'/name/'tree.pt', weights_only=True)
                nodes = [{**n,'members':n['members'].numpy()} for n in stored_tree['nodes']]
                near = torch.load(BASE/'inputs'/name/'geometry.pt', weights_only=True)['near'].numpy()
                tree_source = 'reused audited seed0 tree and neighbors'
            else:
                tree = g.build_tree(h, train, CONFIG['max_leaves'], 64,
                                    CONFIG['min_nodes'], CONFIG['min_train'], seed=seed)
                nodes = tree['nodes']
                near = g.neighbors(h[train], train, len(train)-1)
                tree_source = 'new same-specification tree and neighbors'
            tree_seconds = time.perf_counter()-t0
            k = min(CONFIG['reference_k'], len(train)-1)
            full = g.node_costs(nodes, train, near[:, :k])
            size_cost, identity_cost = cost_components(nodes, train, near[:, :k])
            if not np.allclose(full, size_cost+identity_cost, rtol=0, atol=1e-10):
                raise RuntimeError('TV decomposition failed')
            np.testing.assert_allclose(size_cost, support_costs(nodes, k), rtol=0, atol=1e-10)
            selected, _ = g.select_cut(nodes, full)
            size_selected, _ = g.select_cut(nodes, size_cost)
            count_selected = count_matched_cut(nodes, len(selected))
            parts = dict(OLD64=assets['assignment'].numpy(),
                NP=g.assignment(nodes, selected, len(h)),
                SUPPORT=g.assignment(nodes, size_selected, len(h)),
                COUNT=g.assignment(nodes, count_selected, len(h)))
            for arm, count in [('KMNP',len(selected)), ('KMOLD',len(np.unique(parts['OLD64'])))]:
                with threadpool_limits(limits=1):
                    parts[arm] = KMeans(count, init='k-means++', n_init=10, max_iter=300,
                        algorithm='lloyd', random_state=seed).fit_predict(h)
            if seed == 0:
                previous = torch.load(BASE/'inputs'/name/'geometry.pt', weights_only=True)
                for current, old in [('NP','NP'),('KMNP','KM')]:
                    if not np.array_equal(parts[current], previous['partitions'][old].numpy()):
                        raise RuntimeError('Seed0 partition bridge failed: '+current)
            np_support = np.bincount(parts['NP'][train])[parts['NP'][train]]-1
            for arm, part in parts.items():
                sizes = np.bincount(part[train], minlength=len(np.unique(part)))
                s = sizes[part[train]]-1
                tv, size_sum = 0., 0.
                for b in np.unique(part[train]):
                    rows = np.flatnonzero(part[train]==b)
                    donors = len(rows)-1
                    overlap = (part[train][near[rows,:k]]==b).sum(1)
                    tv += float((1-overlap/max(k,donors)).sum())
                    size_sum += len(rows)*abs(donors-k)/max(k,donors)
                geometry.append(dict(dataset=name,seed=seed,arm=arm,regions=len(np.unique(part)),
                    train_nodes=len(train),support_mean=float(s.mean()),support_min=int(s.min()),
                    support_max=int(s.max()),tv_per_train=tv/len(train),
                    size_cost_per_train=size_sum/len(train),
                    identity_cost_per_train=(tv-size_sum)/len(train),tree_source=tree_source,
                    tree_neighbor_seconds=tree_seconds))
            torch.save(dict(nodes=[{**n,'members':torch.tensor(n['members'])} for n in nodes],
                full_cost=torch.tensor(full),size_cost=torch.tensor(size_cost),
                identity_cost=torch.tensor(identity_cost),np_leaves=selected,
                support_leaves=size_selected,count_leaves=count_selected), folder/'tree.pt')
            torch.save(dict(**assets,partitions={a:torch.tensor(p).long() for a,p in parts.items()},
                near=torch.tensor(near),support=torch.tensor(np_support),k=k),folder/'geometry.pt')
            for condition in CONFIG['conditions']:
                cache = torch.load(source/(condition+'.pt'), weights_only=True)
                z, classes = cache['observed'], data['classes']
                targets, alphas, supports = {}, {}, {}
                for arm in CONFIG['arms']:
                    if arm == 'CE':
                        counts = np.zeros((len(z),classes),np.float32)
                        target, alpha = F.one_hot(z, classes).float(), torch.zeros(len(z))
                    else:
                        counts = (g.neighbor_counts(z.numpy(),near,np_support,classes) if arm=='KNNMATCH'
                                  else counts_for_partition(z.numpy(),parts[arm][train],classes))
                        target, alpha = g.targets_from_counts(z,torch.tensor(counts))
                    err = float(np.abs(reference_targets(z.numpy(),counts)-target.numpy()).max())
                    if err > 3e-7:
                        raise RuntimeError('Independent target reference failed')
                    if arm=='OLD64' and not torch.equal(target,cache['target']):
                        raise RuntimeError('Historical target changed')
                    targets[arm],alphas[arm],supports[arm] = target,alpha,torch.tensor(counts.sum(1))
                    checks.append(dict(dataset=name,seed=seed,condition=condition,arm=arm,error=err))
                    for route in CONFIG['routes']:
                        anchor = reference_run(name,seed,condition,route,arm)
                        if anchor:
                            record = read(anchor/'result.json')
                            # Historical CE metadata hashed the loaded LCSS payload;
                            # its objective replaced that payload with observed one-hot labels.
                            recorded_target = cache['target'] if arm=='CE' else target
                            if record['status']!='ok' or record['target_hash']!=state_hash(dict(target=recorded_target)):
                                raise RuntimeError('Invalid reusable anchor')
                            if record['epochs']!=CONFIG['epochs' if route=='R0' else 'head_epochs']:
                                raise RuntimeError('Anchor duration differs')
                            if arm in ('CE','OLD64') and record['input_sha256']!=sha(source/(condition+'.pt')):
                                raise RuntimeError('Anchor input changed')
                            for path in anchor.glob('*'):
                                sources[str(path)]=sha(path)
                torch.save(dict(observed=z,valid_y=cache['valid_y'],targets=targets,
                    alphas=alphas,supports=supports), folder/(condition+'.pt'))
                sources[str(source/(condition+'.pt'))]=sha(source/(condition+'.pt'))
            for path in [source/'assets.pt',encoder_path(name,seed),*folder.glob('*')]:
                sources[str(path)]=sha(path)
            print(f'PREPARED {name} seed{seed}: NP={len(selected)}, SUPPORT={len(size_selected)}, COUNT={len(count_selected)}',flush=True)
    csv_write(ROOT/'geometry.csv',geometry)
    write(ROOT/'preflight.json',dict(target_checks=checks,expected_cells=576,new=360,reused=216))
    write(ROOT/'environment.json',dict(executable=sys.executable,torch=torch.__version__,
        hip=torch.version.hip,gpu=torch.cuda.get_device_name(0),platform=platform.platform()))
    for path in [*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'config.json',ROOT/'environment.json',ROOT/'geometry.csv']:
        sources[str(path)]=sha(path)
    write(ROOT/'frozen_inputs.json',dict(config=CONFIG,sha256=sources))
    print('FROZEN full experiment before outcomes',flush=True)


def fit(name, seed, condition, route, arm, smoke=False):
    folder = ROOT/('smoke' if smoke else 'runs')/name/f'seed{seed}'/condition/(route+'_'+arm)
    if (folder/'result.json').exists():
        record = read(folder/'result.json')
        if record['status']!='ok':
            raise RuntimeError('Review unfinished run: '+str(folder))
        return record
    folder.mkdir(parents=True,exist_ok=True)
    anchor = reference_run(name,seed,condition,route,arm)
    if anchor and not smoke:
        result = dict(**{k:v for k,v in read(anchor/'result.json').items() if k not in ('arm','reused','source_folder')},
                      arm=arm,reused=True,source_folder=str(anchor))
        cache = torch.load(input_dir(name,seed)/(condition+'.pt'),weights_only=True)
        result['historical_target_hash']=result['target_hash']
        result['target_hash']=state_hash(dict(target=cache['targets'][arm]))
        result['historical_input_sha256']=result['input_sha256']
        result['input_sha256']=sha(input_dir(name,seed)/(condition+'.pt'))
        write(folder/'result.json',result)
        return result
    data = load_dataset(name)
    geo = torch.load(input_dir(name,seed)/'geometry.pt',weights_only=True)
    cache = torch.load(input_dir(name,seed)/(condition+'.pt'),weights_only=True)
    train,valid = geo['train'].cuda(),geo['valid'].cuda()
    target,valid_y = cache['targets'][arm].cuda(),cache['valid_y'].cuda()
    set_seed(seed)
    model = AlignedMLP(data['x'].shape[1],data['classes'],data['h'].shape[1],CONFIG['hidden'],CONFIG['dropout']).cuda()
    enc = encoder_path(name,seed) if route=='R2' else None
    if enc:
        model.load_state_dict(torch.load(enc,map_location='cuda',weights_only=True))
        model.freeze_encoder()
        frozen = state_hash(model.encoder_state())
        with torch.no_grad():
            hidden = model.encode(data['x']).detach()
        params = model.classifier_parameters()
    else:
        frozen,hidden,params = None,None,model.backbone.parameters()
    optimizer = torch.optim.Adam(params,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    epochs = 3 if smoke else CONFIG['epochs' if route=='R0' else 'head_epochs']
    meta = dict(dataset=name,seed=seed,condition=condition,route=route,arm=arm,epochs=epochs,
        initial_hash=state_hash(model.deployment_state()),
        initial_rng_hash=state_hash(dict(cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state())),
        target_hash=state_hash(dict(target=target)),
        input_sha256=sha(input_dir(name,seed)/(condition+'.pt')),encoder_source=str(enc) if enc else None,
        smoke=smoke,reused=False)
    reference = read(reference_run(name,seed,condition,route,'CE')/'result.json')
    for key in ('initial_hash','initial_rng_hash'):
        if meta[key]!=reference[key]:
            raise RuntimeError('Initialization bridge failed: '+key)
    write(folder/'config.json',meta)
    write(folder/'result.json',dict(**meta,status='running'))
    best,selected,state = -1,0,None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as handle:
        for epoch in range(1,epochs+1):
            if time.perf_counter()-started>CONFIG['timeout_seconds_per_fit']:
                raise TimeoutError('Per-fit timeout')
            if route=='R2':
                model.train_head()
                logits=model.backbone.layers[-1](hidden)
            else:
                model.train()
                logits=model(data['x'])[0]
            optimizer.zero_grad(set_to_none=True)
            weights=torch.ones(len(train),device=logits.device,dtype=logits.dtype)
            loss=(weights*(-target*logits[train].log_softmax(1)).sum(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluated=model.backbone.layers[-1](hidden) if route=='R2' else model(data['x'])[0]
                correct=int((evaluated[valid].argmax(1)==valid_y).sum())
            if correct>best:
                best,selected,state=correct,epoch,deepcopy(model.deployment_state())
            handle.write(json.dumps(dict(epoch=epoch,loss=float(loss.detach()),validation_correct=correct))+'\n')
            handle.flush()
    if frozen and state_hash(model.encoder_state())!=frozen:
        raise RuntimeError('Frozen encoder/BN changed')
    model.backbone.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probabilities=model(data['x'])[0].softmax(1).cpu()
    deployed=MLP(data['x'].shape[1],CONFIG['hidden'],data['classes'],2,CONFIG['dropout']).cuda().eval()
    deployed.load_state_dict(state)
    with torch.no_grad():
        diff=float((deployed(data['x']).softmax(1).cpu()-probabilities).abs().max())
    if diff!=0:
        raise RuntimeError('X-only deployment failed')
    torch.save(state,folder/'student.pt')
    torch.save(probabilities,folder/'probabilities.pt')
    torch.cuda.synchronize()
    result=dict(**meta,status='ok',best_epoch=selected,validation=best/len(valid),valid_correct=best,
        valid_total=len(valid),seconds=time.perf_counter()-started,frozen_encoder_unchanged=frozen is not None,
        deployment_difference=diff,checkpoint_sha256=sha(folder/'student.pt'),
        probabilities_sha256=sha(folder/'probabilities.pt'),folder=str(folder))
    write(folder/'result.json',result)
    return result


def run(smoke=False):
    initialize()
    verify()
    if read(ROOT/'frozen_inputs.json').get('stage')!='all sources frozen before smoke and full fits':
        raise RuntimeError('Run freeze before fitting')
    if not smoke:
        audit=read(ROOT/'smoke_audit.json')
        if not audit.get('passed') or audit.get('frozen_manifest_sha256')!=sha(ROOT/'frozen_inputs.json'):
            raise RuntimeError('Current frozen experiment requires a successful smoke audit')
    if smoke:
        jobs=[('cora',seed,'uniform0.4',route,arm) for seed in CONFIG['seeds']
              for route in CONFIG['routes'] for arm in ('CE','OLD64','SUPPORT','COUNT')]
    else:
        jobs=[(name,seed,cond,route,arm) for name in CONFIG['datasets'] for seed in CONFIG['seeds']
              for cond in CONFIG['conditions'] for route in CONFIG['routes'] for arm in CONFIG['arms']]
    start=time.perf_counter()
    for number,job in enumerate(jobs,1):
        result=fit(*job,smoke=smoke)
        write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(
            status='completed' if number==len(jobs) else 'running',completed=number,expected=len(jobs),
            last_job=job,seconds=time.perf_counter()-start))
        print(f'{number}/{len(jobs)} {job}: val={100*result["validation"]:.3f}, epoch={result["best_epoch"]}, reused={result["reused"]}',flush=True)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','freeze','smoke','run'])
    phase=parser.parse_args().phase
    try:
        if phase=='prepare':
            prepare()
        elif phase=='freeze':
            freeze_sources()
        else:
            run(phase=='smoke')
    except Exception:
        write(ROOT/f'failure_{phase}.json',dict(traceback=traceback.format_exc()))
        raise
