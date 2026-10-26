"""Paired GPU fits. Training has no access to clean test labels."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
EXP = ROOT.parent
sys.path.insert(0, str(ROOT))
from run_diagnostics import save_json, save_csv, sha, CONFIG

spec = importlib.util.spec_from_file_location('historical_r2', EXP/'r2_support_5dataset_20261022/run.py')
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
from core import AlignedMLP, state_hash, alignment_loss
from gblcss.models import MLP, GCN
from gblcss.data import set_seed


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(path):
    record_input(path)
    return torch.load(path, map_location='cpu', weights_only=True)


def record_input(path):
    path = Path(path).resolve()
    registry = ROOT/'fit_inputs.json'
    items = read(registry) if registry.exists() else {}
    key = str(path)
    digest = sha(path)
    if key in items and items[key] != digest:
        raise RuntimeError('Input changed: '+key)
    if key not in items:
        items[key] = digest
        save_json(registry, items)
    return digest


def initialize():
    if datetime.now().astimezone() < datetime.fromisoformat(CONFIG['not_before']):
        raise RuntimeError('not_before has not elapsed')
    torch.set_num_threads(CONFIG['training']['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU fallback')
    manifest = ROOT/'fit_sources.json'
    files = [*ROOT.glob('*.py'), ROOT/'PLAN.md', ROOT/'config.json',
             Path(r2.__file__), Path(sys.modules['core'].__file__),
             Path(sys.modules['gblcss.models'].__file__), Path(sys.modules['gblcss.data'].__file__),
             Path(r2.generator.__file__), Path(sys.modules['common'].__file__)]
    current = {str(p.resolve()): sha(p) for p in files}
    if manifest.exists():
        for p, digest in read(manifest).items():
            if sha(p) != digest:
                raise RuntimeError('Frozen fit source changed: '+p)
    else:
        save_json(manifest, current)
    free, total = torch.cuda.mem_get_info()
    save_json(ROOT/'fit_environment.json', dict(torch=torch.__version__, device=torch.cuda.get_device_name(0),
        free_bytes=free, total_bytes=total, pid=os.getpid(), started=datetime.now().astimezone().isoformat()))
    if free < 4*1024**3:
        raise RuntimeError('GPU memory occupied; leave queued, do not kill other jobs')


_data = {}


def data(name):
    if name not in _data:
        raw = r2.raw_data(name)
        h, hp, mp = r2.load_h(name, raw)
        for path in [r2.DATA_FILES[name], hp, mp]:
            record_input(path)
        # Only class cardinality is retained; fitting uses validation labels from the locked input.
        _data.clear()
        _data[name] = dict(x=raw.x.float().cuda(), h=h.cuda(), h_cpu=h,
                          classes=int(raw.y.max())+1, edge_index=raw.edge_index.long())
    return _data[name]


def cell(name, seed, condition):
    base = ROOT/'diagnostics/e1'/name/f'seed{seed}'
    geo = load(base/'geometry.pt')
    pack = load(base/f'{condition}.pt')
    return geo, pack


def historical_encoder(name, seed):
    if name in ('cora','citeseer','corafull'):
        path, _ = r2.encoder_paths(name, seed)
    else:
        path = EXP/'r2_support_5dataset_20261022/pretraining'/name/f'seed{seed}'/'model.pt'
    return path


def partition_target(z, part, classes):
    from run_diagnostics import import_file
    kernel = import_file('sc_counts_fit', ROOT/'kernels.py')
    counts = kernel.loo_counts(z.numpy(), part.numpy(), classes)
    return r2.generator.targets_from_counts(z, torch.as_tensor(counts).float(), **CONFIG['target'])[0]


def historical_target(name, seed, condition):
    geo, pack = cell(name, seed, condition)
    target = partition_target(pack['observed'], geo['legacy32'][geo['train']], data(name)['classes'])
    old_root = EXP/('neighbor_support_ablation_20261022' if name in ('cora','citeseer','corafull')
                   else 'r2_support_5dataset_20261022')
    old_path = old_root/'inputs'/name/f'seed{seed}'/f'{condition}.pt'
    if old_path.exists():
        old = load(old_path)
        previous = old['targets']['SUPPORT'] if 'targets' in old else old['target']
        if not torch.allclose(previous, target, atol=3e-7, rtol=0):
            raise RuntimeError('Historical SUPPORT target mismatch')
        target = previous
    return target


def control_targets(name, seed, condition):
    folder = ROOT/'controls'/name/f'seed{seed}'
    out = folder/f'{condition}.pt'
    if out.exists():
        return load(out)
    d = data(name)
    geo, pack = cell(name, seed, condition)
    train, z = geo['train'], pack['observed']
    s = pack['selected_support']
    assignment = geo['partitions'][s]
    groups = len(torch.unique(assignment))
    folder.mkdir(parents=True, exist_ok=True)
    km_path = folder/f'km_{groups}.pt'
    if km_path.exists():
        km = load(km_path)
    else:
        with threadpool_limits(limits=1):
            km = torch.from_numpy(KMeans(groups, init='k-means++', n_init=10, max_iter=300,
                algorithm='lloyd', random_state=seed).fit_predict(d['h_cpu'].numpy()))
        torch.save(km, km_path)
    near_path = folder/'neighbors.pt'
    if near_path.exists():
        near = load(near_path).numpy()
    else:
        near = r2.generator.neighbors(d['h_cpu'][train].numpy(), train.numpy(), len(train)-1)
        torch.save(torch.from_numpy(near), near_path)
    support = torch.bincount(assignment[train])[assignment[train]]-1
    counts = r2.generator.neighbor_counts(z.numpy(), near, support.numpy(), d['classes'])
    old_path = EXP/'main_protocol_replay_20260922/matrix'/name/'default'/f'seed{seed}.npz'
    record_input(old_path)
    with np.load(old_path) as old:
        old_assignment = torch.from_numpy(old['assignment']).long()
    values = dict(CE=F.one_hot(z,d['classes']).float(),
        OLD64=partition_target(z,old_assignment[train],d['classes']),
        LEGACY32=historical_target(name,seed,condition),
        KM=partition_target(z,km[train],d['classes']),
        KNN=r2.generator.targets_from_counts(z,torch.from_numpy(counts),**CONFIG['target'])[0])
    torch.save(values,out)
    return values


def copy_encoder(model, state):
    current = model.state_dict()
    for key in current:
        if key.startswith(('backbone.layers.0.', 'backbone.batch_norms.', 'projection.')):
            current[key] = state[key]
    model.load_state_dict(current)


def loss_value(logits, target, observed=None, gce=False):
    logp = logits.log_softmax(1)
    if gce:
        return ((1-torch.exp(.7*logp.gather(1,observed[:,None]).squeeze(1)))/.7).mean()
    return -(target*logp).sum(1).mean()


def fit(folder, name, seed, condition, arm, target, epochs, encoder=None, frozen=False,
        all_nodes=False, gce=False, graph=False):
    folder = Path(folder)
    target_digest = state_hash(dict(target=target))
    if (folder/'result.json').exists():
        done = read(folder/'result.json')
        if done.get('status') == 'complete':
            for k, val in dict(epochs=epochs, target_hash=target_digest, frozen=frozen, all_nodes=all_nodes).items():
                if done[k] != val:
                    raise RuntimeError('Completed fit configuration mismatch '+str(folder))
            for filename, field in [('student.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
                if sha(folder/filename) != done[field]:
                    raise RuntimeError('Completed artifact changed')
            return done
        raise RuntimeError('Incomplete fit must be archived as a failure before retry: '+str(folder))
    d = data(name)
    geo, pack = cell(name,seed,condition)
    x, classes = d['x'],d['classes']
    train, valid = geo['train'].cuda(),geo['valid'].cuda()
    valid_y, observed = pack['valid_y'].cuda(),pack['observed'].cuda()
    t = target.cuda()
    set_seed(seed)
    adjacency = None
    if graph:
        MLP(x.shape[1],64,classes,2,.5).cuda()
        model = GCN(x.shape[1],64,classes,2,.5).cuda()
        edges = d['edge_index'].cuda()
        adjacency = torch.sparse_coo_tensor(edges,torch.ones(edges.shape[1],device='cuda'),
                                            (len(x),len(x))).coalesce()
        params = model.parameters()
        get_state = lambda: {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        model = AlignedMLP(x.shape[1],classes,d['h'].shape[1],64,.5).cuda()
        if encoder is not None:
            copy_encoder(model,load(encoder))
        if frozen:
            model.freeze_encoder()
            with torch.no_grad():
                hidden = model.encode(x).detach()
            params = model.classifier_parameters()
        else:
            params = model.backbone.parameters()
        get_state = model.deployment_state
    initial = get_state()
    initial_encoder = state_hash(model.encoder_state()) if frozen else None
    meta = dict(dataset=name,seed=seed,condition=condition,arm=arm,epochs=epochs,
        frozen=frozen,all_nodes=all_nodes,gce=gce,graph=graph,target_hash=target_digest,
        initial_hash=state_hash(initial),initial_rng_hash=state_hash(dict(cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state())),
        encoder=str(encoder) if encoder else None,encoder_sha256=sha(encoder) if encoder else None,
        folder=str(folder),source_manifest_sha256=sha(ROOT/'fit_sources.json'))
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(target.cpu(),folder/'target.pt')
    save_json(folder/'config.json',meta)
    save_json(folder/'result.json',dict(**meta,status='running'))
    optimizer = torch.optim.Adam(params,lr=.01,weight_decay=.0005)
    best, selected, state = -1,0,None
    torch.cuda.synchronize()
    start = time.perf_counter()
    def forward():
        if graph:
            return model(x,adjacency)
        return model.backbone.layers[-1](hidden) if frozen else model(x)[0]
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as handle:
        for epoch in range(1,epochs+1):
            if frozen:
                model.train_head()
            else:
                model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = forward()
            loss = loss_value(logits if all_nodes else logits[train],t,observed,gce)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss '+str(folder))
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                correct = int((forward()[valid].argmax(1)==valid_y).sum())
            if correct > best:
                best,selected,state = correct,epoch,deepcopy(get_state())
            handle.write(json.dumps(dict(epoch=epoch,loss=float(loss.detach()),validation_correct=correct))+'\n')
            handle.flush()
    if frozen and state_hash(model.encoder_state()) != initial_encoder:
        raise RuntimeError('Frozen encoder or BN changed')
    (model if graph else model.backbone).load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probabilities = forward().softmax(1).cpu()
    difference = 0.
    if not graph:
        deployed = MLP(x.shape[1],64,classes,2,.5).cuda().eval()
        deployed.load_state_dict(state)
        with torch.no_grad():
            difference = float((deployed(x).softmax(1).cpu()-probabilities).abs().max())
        if difference != 0:
            raise RuntimeError('X-only deployment failed')
    torch.save(state,folder/'student.pt')
    torch.save(probabilities,folder/'probabilities.pt')
    torch.cuda.synchronize()
    result = dict(**meta,status='complete',best_epoch=selected,validation=best/len(valid),
        valid_correct=best,valid_total=len(valid),seconds=time.perf_counter()-start,
        frozen_encoder_unchanged=frozen,deployment_difference=difference,
        checkpoint_sha256=sha(folder/'student.pt'),probabilities_sha256=sha(folder/'probabilities.pt'))
    save_json(folder/'result.json',result)
    print(f'DONE {folder.relative_to(ROOT)} val={100*result["validation"]:.3f} epoch={selected} seconds={result["seconds"]:.1f}',flush=True)
    return result


def alignment(name,seed,budget=600):
    folder = ROOT/'alignment'/name/f'seed{seed}'
    if (folder/'complete.json').exists():
        done = read(folder/'complete.json')
        if done['budget'] < budget:
            raise RuntimeError('Alignment continuation requires its saved optimizer; do not overwrite')
        return folder
    if (folder/'epochs.jsonl').exists():
        raise RuntimeError('Incomplete alignment must be archived before restart')
    d = data(name)
    set_seed(seed)
    model = AlignedMLP(d['x'].shape[1],d['classes'],d['h'].shape[1],64,.5).cuda()
    optimizer = torch.optim.Adam(model.encoder_parameters()+list(model.projection.parameters()),lr=.01,weight_decay=.0005)
    folder.mkdir(parents=True,exist_ok=True)
    best,selected,state = float('inf'),0,None
    start = time.perf_counter()
    snapshots = []
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,budget+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = alignment_loss(model.projection(model.encode(d['x'])),d['h'])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluation = float(alignment_loss(model.projection(model.encode(d['x'])),d['h']))
            if not np.isfinite(evaluation):
                raise FloatingPointError('Nonfinite alignment')
            if evaluation < best:
                best,selected = evaluation,epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            log.write(json.dumps(dict(epoch=epoch,training=float(loss.detach()),evaluation=evaluation))+'\n')
            if epoch in (100,300,600):
                torch.save(state,folder/f'prefix{epoch}.pt')
                snapshots.append(dict(prefix=epoch,best_epoch=selected,loss=best,seconds=time.perf_counter()-start))
                print(f'ALIGN {name} seed{seed} prefix{epoch} bestloss={best:.6f}',flush=True)
    historical = load(historical_encoder(name,seed))
    prefix = load(folder/'prefix100.pt')
    maxdiff = max(float((prefix[k]-historical[k]).abs().max()) for k in prefix)
    save_json(folder/'bridge.json',dict(historical=str(historical_encoder(name,seed)),max_absolute_difference=maxdiff,
        exact=state_hash(historical)==state_hash(prefix),comparison_policy='paired new prefixes; historical100 kept as reference'))
    save_json(folder/'complete.json',dict(budget=budget,snapshots=snapshots,seconds=time.perf_counter()-start))
    return folder


def teacher(name,seed,condition):
    if condition != 'clean':
        folder = EXP/'teacher_support_lcss_20261022/runs'/name/f'seed{seed}'/condition/'SUPPORT_LCSS'
        source = EXP/'teacher_support_lcss_20261022/inputs'/name/f'seed{seed}'/f'{condition}.pt'
        pack = load(source)
        geo, current = cell(name,seed,condition)
        for k,old in [('train','train_index'),('valid','valid_index'),('test','test_index')]:
            if not torch.equal(geo[k],pack[old]):
                raise RuntimeError('Teacher split differs')
        if not torch.equal(current['observed'],pack['observed_train']):
            raise RuntimeError('Teacher noise differs')
        if not torch.allclose(historical_target(name,seed,condition),pack['targets']['SUPPORT_LCSS'],atol=3e-7,rtol=0):
            raise RuntimeError('Teacher target differs')
    else:
        folder = ROOT/'clean_teachers'/name/f'seed{seed}'
        fit(folder,name,seed,condition,'SUPPORT_TEACHER',historical_target(name,seed,condition),200,graph=True)
    prob = load(folder/'probabilities.pt')
    d = data(name)
    if prob.shape != (len(d['x']),d['classes']) or not torch.isfinite(prob).all() or not torch.allclose(prob.sum(1),torch.ones(len(prob)),atol=1e-6):
        raise RuntimeError('Invalid teacher probabilities')
    return prob,folder
