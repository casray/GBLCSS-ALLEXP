from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'representation_supervision_20261021'
FUSION = ROOT.parent/'gce_lcss_fusion_20261022'
sys.path.insert(0, str(OLD))
from core import AlignedMLP, make_targets, state_hash
from protocol import load_dataset, raw_data, read, write, sha, fingerprint
from gblcss.data import set_seed
from gblcss.models import MLP
sys.path.insert(0, str(ROOT))
import generator as g

CONFIG = read(ROOT/'config.json')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
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
            raise RuntimeError('Frozen input/source modified: '+file)


def counts_for_partition(z, assignment, classes):
    counts = np.zeros((len(z), classes), np.float32)
    for b in np.unique(assignment):
        rows = np.flatnonzero(assignment == b)
        counts[rows] = np.bincount(z[rows], minlength=classes)
    counts[np.arange(len(z)), z] -= 1
    return counts


def reference_targets(z, counts):
    classes = counts.shape[1]
    target = np.eye(classes)[z].copy()
    for row, count in enumerate(counts):
        if count.sum() < 2:
            continue
        vote = count.astype(float)/count.sum()
        alpha = (classes*vote.max()-1)/(classes-1)
        winners = np.flatnonzero(count == count.max())
        q = vote if len(winners) > 1 else .5*vote+.5*np.eye(classes)[winners[0]]
        target[row] = (1-alpha)*target[row]+alpha*q
    return target


def prepare():
    initialize()
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Already frozen')
    sources = dict(read(FUSION/'frozen_inputs.json')['sha256'])
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError('Historical frozen source modified: '+path)
    stats, checks = [], []
    for name in CONFIG['datasets']:
        data = load_dataset(name, 'cpu')
        source = FUSION/'inputs'/name/'seed0'
        assets = torch.load(source/'assets.pt', weights_only=True)
        for key in ('train', 'valid', 'test', 'assignment'):
            if not torch.equal(assets[key], data[key]):
                raise RuntimeError('Historical protocol mismatch')
        train = assets['train'].numpy()
        h = g.normalize(data['h'].numpy())
        folder = ROOT/'inputs'/name
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tree = g.build_tree(h, train, CONFIG['max_leaves'], CONFIG['snapshot_leaves'],
                            CONFIG['min_nodes'], CONFIG['min_train'], seed=0)
        tree_seconds = time.perf_counter()-t0
        print(f'TREE {name}: {len(tree["fine_leaves"])} leaves, {tree_seconds:.2f}s', flush=True)
        t0 = time.perf_counter()
        near = g.neighbors(h[train], train, len(train)-1)
        knn_seconds = time.perf_counter()-t0
        k = min(CONFIG['reference_k'], len(train)-1)
        t0 = time.perf_counter()
        costs = g.node_costs(tree['nodes'], train, near[:, :k])
        leaves, cost = g.select_cut(tree['nodes'], costs)
        np_assignment = g.assignment(tree['nodes'], leaves, len(h))
        selection_seconds = time.perf_counter()-t0
        t0 = time.perf_counter()
        with threadpool_limits(limits=1):
            km = KMeans(len(leaves), init='k-means++', n_init=10, max_iter=300,
                        algorithm='lloyd', random_state=0).fit_predict(h)
        km_seconds = time.perf_counter()-t0
        partitions = dict(OLD64=assets['assignment'].numpy(), TREE64=tree['snapshot_assignment'],
                          FINE=tree['fine_assignment'], NP=np_assignment, KM=km)
        support = np.bincount(np_assignment[train])[np_assignment[train]]-1
        if support.min() < 2:
            raise RuntimeError('NP violates support limit')
        # Persist tensor-only data so future loading does not require arbitrary pickle execution.
        saved_nodes = [{**node, 'members': torch.tensor(node['members'])} for node in tree['nodes']]
        torch.save(dict(nodes=saved_nodes, fine_leaves=tree['fine_leaves'],
                        snapshot_leaves=tree['snapshot_leaves'], np_leaves=leaves,
                        costs=torch.tensor(costs), cost=cost), folder/'tree.pt')
        torch.save(dict(**assets, partitions={a:torch.tensor(v).long() for a,v in partitions.items()},
                        near=torch.tensor(near), support=torch.tensor(support), k=k), folder/'geometry.pt')
        for arm, part in partitions.items():
            counts = np.bincount(part[train], minlength=len(np.unique(part)))
            per_node = counts[part[train]]-1
            value = 0.
            for b in np.unique(part[train]):
                rows = np.flatnonzero(part[train] == b)
                overlap = (part[train][near[rows, :k]] == b).sum(1)
                value += g.tv_cost(overlap, k, len(rows)-1).sum() if len(rows)>1 else len(rows)
            stats.append(dict(dataset=name, arm=arm, regions=len(np.unique(part)), train_nodes=len(train),
                              support_mean=float(per_node.mean()), support_min=int(per_node.min()),
                              support_max=int(per_node.max()), tv_per_train=value/len(train),
                              tree_seconds=tree_seconds, full_knn_seconds=knn_seconds,
                              selection_seconds=selection_seconds, km_seconds=km_seconds,
                              full_neighbor_bytes=int(near.nbytes), tree_max_depth=max(n['depth'] for n in tree['nodes'])))
        for condition in CONFIG['conditions']:
            cache = torch.load(source/(condition+'.pt'), weights_only=True)
            z, classes = cache['observed'], data['classes']
            targets, alphas, supports = {}, {}, {}
            for arm in CONFIG['arms']:
                if arm == 'CE':
                    target, alpha = F.one_hot(z, classes).float(), torch.zeros(len(z))
                    counts = np.zeros((len(z), classes), np.float32)
                elif arm in partitions:
                    counts = counts_for_partition(z.numpy(), partitions[arm][train], classes)
                    target, alpha = g.targets_from_counts(z, torch.tensor(counts))
                    old_target, old_alpha = make_targets(z, torch.tensor(partitions[arm][train]), classes)
                    if not torch.equal(target, old_target) or not torch.equal(alpha, old_alpha):
                        raise RuntimeError('Old target implementation not exactly reproduced')
                else:
                    counts = g.neighbor_counts(z.numpy(), near, np.full(len(train), k) if arm == 'KNN32' else support, classes)
                    target, alpha = g.targets_from_counts(z, torch.tensor(counts))
                err = float(np.abs(reference_targets(z.numpy(), counts)-target.numpy()).max())
                if err > 3e-7:
                    raise RuntimeError('Independent target reference mismatch')
                if arm == 'OLD64' and not torch.equal(target, cache['target']):
                    raise RuntimeError('Historical OLD64 target mismatch')
                targets[arm], alphas[arm], supports[arm] = target, alpha, torch.tensor(counts.sum(1))
                checks.append(dict(dataset=name, condition=condition, arm=arm, target_reference_error=err,
                                   target_hash=state_hash(dict(target=target))))
            torch.save(dict(observed=z, valid_y=cache['valid_y'], targets=targets, alphas=alphas, supports=supports),
                       folder/(condition+'.pt'))
            for route in CONFIG['routes']:
                for arm, rule in [('CE','CE'), ('OLD64','LCSS')]:
                    anchor = FUSION/'runs'/name/'seed0'/condition/(route+'_'+rule)
                    result = read(anchor/'result.json')
                    if result['status'] != 'ok' or result['input_sha256'] != sha(source/(condition+'.pt')):
                        raise RuntimeError('Invalid anchor input/status')
                    if result['epochs'] != CONFIG['epochs' if route == 'R0' else 'head_epochs']:
                        raise RuntimeError('Anchor training duration differs')
                    for file in anchor.glob('*'):
                        sources[str(file)] = sha(file)
            sources[str(source/(condition+'.pt'))] = sha(source/(condition+'.pt'))
        sources[str(source/'assets.pt')] = sha(source/'assets.pt')
        for path in folder.glob('*'):
            sources[str(path)] = sha(path)
        print(f'PREPARED {name}: NP={len(leaves)}, k={k}, no labels in partition selection', flush=True)
    csv_write(ROOT/'geometry.csv', stats)
    write(ROOT/'preflight.json', dict(target_checks=checks, expected_cells=192, expected_new=144, expected_anchors=48))
    write(ROOT/'environment.json', dict(executable=sys.executable, torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), platform=platform.platform(), threads=torch.get_num_threads()))
    for path in list(ROOT.glob('*.py'))+[ROOT/'PLAN.md', ROOT/'config.json', ROOT/'environment.json']:
        sources[str(path)] = sha(path)
    write(ROOT/'frozen_inputs.json', dict(config=CONFIG, sha256=sources))
    print('FROZEN: source and input manifest complete', flush=True)


def new_model(data):
    set_seed(0)
    return AlignedMLP(data['x'].shape[1], data['classes'], data['h'].shape[1], CONFIG['hidden'], CONFIG['dropout']).cuda()


def fit(name, condition, route, arm, smoke=False):
    folder = ROOT/('smoke' if smoke else 'runs')/name/condition/(route+'_'+arm)
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        if result['status'] != 'ok':
            raise RuntimeError('Review failed run before resuming: '+str(folder))
        return result
    folder.mkdir(parents=True, exist_ok=True)
    if not smoke and arm in ('CE','OLD64'):
        source = FUSION/'runs'/name/'seed0'/condition/(route+'_'+('LCSS' if arm == 'OLD64' else 'CE'))
        result = {**read(source/'result.json'), 'arm':arm, 'reused':True, 'source_folder':str(source)}
        write(folder/'result.json', result)
        return result
    data = load_dataset(name)
    geo = torch.load(ROOT/'inputs'/name/'geometry.pt', weights_only=True)
    cache = torch.load(ROOT/'inputs'/name/(condition+'.pt'), weights_only=True)
    train, valid = geo['train'].cuda(), geo['valid'].cuda()
    target, valid_y = cache['targets'][arm].cuda(), cache['valid_y'].cuda()
    model = new_model(data)
    encoder_path = OLD/'pretraining'/name/'model.pt' if route == 'R2' else None
    if encoder_path:
        model.load_state_dict(torch.load(encoder_path, map_location='cuda', weights_only=True))
        model.freeze_encoder()
        frozen_hash = state_hash(model.encoder_state())
        with torch.no_grad():
            hidden = model.encode(data['x']).detach()
        params = model.classifier_parameters()
    else:
        frozen_hash, hidden, params = None, None, model.backbone.parameters()
    optimizer = torch.optim.Adam(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    epochs = 3 if smoke else CONFIG['head_epochs'] if route == 'R2' else CONFIG['epochs']
    meta = dict(dataset=name, seed=0, condition=condition, route=route, arm=arm, epochs=epochs,
                initial_hash=state_hash(model.deployment_state()),
                initial_rng_hash=state_hash(dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state())),
                input_sha256=sha(ROOT/'inputs'/name/(condition+'.pt')),
                encoder_source=str(encoder_path) if encoder_path else None,
                target_hash=state_hash(dict(target=target)), smoke=smoke, reused=False)
    reference = read(FUSION/'runs'/name/'seed0'/condition/(route+'_LCSS')/'result.json')
    for key in ('initial_hash','initial_rng_hash'):
        if meta[key] != reference[key]:
            raise RuntimeError('Initialization bridge mismatch: '+key)
    write(folder/'config.json', meta)
    write(folder/'result.json', dict(**meta, status='running'))
    best, epoch_best, checkpoint = -1, 0, None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as f:
        for epoch in range(1, epochs+1):
            if time.perf_counter()-started > CONFIG['timeout_seconds_per_fit']:
                raise TimeoutError('Per-fit timeout')
            if route == 'R2':
                model.train_head()
                logits = model.backbone.layers[-1](hidden)
            else:
                model.train()
                logits = model(data['x'])[0]
            optimizer.zero_grad(set_to_none=True)
            # Identical reduction to the previous CE/LCSS objective, including unit weights.
            logp = logits[train].log_softmax(1)
            weights = torch.ones(len(train), device=logp.device, dtype=logp.dtype)
            loss = (weights*(-target*logp).sum(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluated = model.backbone.layers[-1](hidden) if route == 'R2' else model(data['x'])[0]
                correct = int((evaluated[valid].argmax(1) == valid_y).sum())
            if correct > best:
                best, epoch_best, checkpoint = correct, epoch, deepcopy(model.deployment_state())
            f.write(json.dumps(dict(epoch=epoch, loss=float(loss.detach()), validation_correct=correct))+'\n')
            f.flush()
    if frozen_hash and state_hash(model.encoder_state()) != frozen_hash:
        raise RuntimeError('Frozen encoder/BN changed')
    model.backbone.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        probabilities = model(data['x'])[0].softmax(1).cpu()
    deployed = MLP(data['x'].shape[1], CONFIG['hidden'], data['classes'], 2, CONFIG['dropout']).cuda().eval()
    deployed.load_state_dict(checkpoint)
    with torch.no_grad():
        diff = float((deployed(data['x']).softmax(1).cpu()-probabilities).abs().max())
    if diff != 0:
        raise RuntimeError('X-only deployment mismatch')
    torch.save(checkpoint, folder/'student.pt')
    torch.save(probabilities, folder/'probabilities.pt')
    torch.cuda.synchronize()
    result = dict(**meta, status='ok', best_epoch=epoch_best, validation=best/len(valid), valid_correct=best,
                  valid_total=len(valid), seconds=time.perf_counter()-started,
                  frozen_encoder_unchanged=frozen_hash is not None, deployment_difference=diff,
                  checkpoint_sha256=sha(folder/'student.pt'), probabilities_sha256=sha(folder/'probabilities.pt'),
                  folder=str(folder))
    write(folder/'result.json', result)
    return result


def run(smoke=False):
    initialize()
    verify()
    started = time.perf_counter()
    if smoke:
        jobs = [('cora','uniform0.4',route,arm) for route in CONFIG['routes'] for arm in CONFIG['arms']]
    else:
        jobs = [(name,cond,route,arm) for name in CONFIG['datasets'] for cond in CONFIG['conditions']
                for route in CONFIG['routes'] for arm in CONFIG['arms']]
    for number, job in enumerate(jobs, 1):
        result = fit(*job, smoke=smoke)
        write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),
              dict(status='running' if number < len(jobs) else 'completed', completed=number, expected=len(jobs),
                   last_job=job, seconds=time.perf_counter()-started))
        print(f'{number}/{len(jobs)} {job}: val={100*result["validation"]:.3f}, epoch={result["best_epoch"]}, reused={result["reused"]}', flush=True)
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare','smoke','run'])
    args = parser.parse_args()
    try:
        prepare() if args.phase == 'prepare' else run(args.phase == 'smoke')
    except Exception:
        write(ROOT/f'failure_{args.phase}.json', dict(traceback=traceback.format_exc()))
        raise
