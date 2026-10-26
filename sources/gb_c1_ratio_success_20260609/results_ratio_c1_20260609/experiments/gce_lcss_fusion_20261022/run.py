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

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'representation_supervision_20261021'
sys.path.insert(0, str(OLD))
from core import AlignedMLP, alignment_loss, make_targets, state_hash
from protocol import load_dataset, raw_data, read, write, sha, fingerprint, MATRIX, WORKSPACE
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices, set_seed
from gblcss.models import MLP
sys.path.insert(0, str(ROOT))
from kernels import objective, mass_target, target_diagnostic

CONFIG = read(ROOT/'config.json')


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
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
            raise RuntimeError('Frozen input/source modified: '+file)


def numpy_targets(z, assignment, classes):
    result = np.zeros((len(z), classes), np.float64)
    for group in np.unique(assignment):
        ids = np.flatnonzero(assignment == group)
        counts = np.bincount(z[ids], minlength=classes)
        for row in ids:
            own = np.eye(classes)[z[row]]
            votes = counts-own
            if votes.sum() < 2:
                result[row] = own
                continue
            r = votes/votes.sum()
            alpha = (classes*r.max()-1)/(classes-1)
            winners = np.flatnonzero(votes == votes.max())
            consensus = r if len(winners) > 1 else .5*r+.5*np.eye(classes)[winners[0]]
            result[row] = (1-alpha)*own+alpha*consensus
    return result


def prepare():
    initialize()
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Already frozen; use run, not prepare')
    sources = dict(read(OLD/'frozen_inputs.json')['sha256'])
    for p, h in sources.items():
        if sha(p) != h:
            raise RuntimeError('Historical source/input modified: '+p)
    records, diagnostics = [], []
    for name in CONFIG['datasets']:
        raw = raw_data(name)
        data0 = load_dataset(name, 'cpu')
        for seed in CONFIG['seeds']:
            meta_path = MATRIX/name/'default'/f'seed{seed}.json'
            meta = read(meta_path)
            with np.load(meta_path.with_suffix('.npz')) as archive:
                assignment = torch.tensor(archive['assignment'].copy(), dtype=torch.long)
                historical = {k: archive[k].copy() for k in ('targets', 'train_index', 'noisy', 'clean')}
            indices = tuple(torch.tensor(a) for a in percent_split_indices(len(raw.x), seed))
            if not np.array_equal(indices[0].numpy(), historical['train_index']):
                raise RuntimeError('Historical training order mismatch')
            if fingerprint(*indices) != meta['split_hash']:
                raise RuntimeError('Historical split fingerprint mismatch')
            source = Path(meta['ball_embedding_path'])
            old_source = read(OLD/'inputs'/name/'uniform0.4.json')['representation_source']
            if source.resolve() != Path(old_source).resolve():
                raise RuntimeError('Unexpected distinct H source; review before continuing')
            folder = ROOT/'inputs'/name/f'seed{seed}'
            assets = dict(train=indices[0], valid=indices[1], test=indices[2], assignment=assignment)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(assets, folder/'assets.pt')
            masks = []
            for ids in indices:
                mask = torch.zeros(len(raw.x), dtype=torch.bool)
                mask[ids] = True
                masks.append(mask)
            for condition, (kind, rate) in CONFIG['conditions'].items():
                graph = GraphData(name, raw.x.float(), raw.edge_index, raw.y.long(), *masks,
                                  train_index=indices[0], val_index=indices[1], test_index=indices[2])
                inject_train_label_noise(graph, kind, rate, seed)
                if not torch.equal(graph.y[~masks[0]], graph.noisy_y[~masks[0]]):
                    raise RuntimeError('Noise escaped training set')
                z = graph.noisy_y[indices[0]]
                y = graph.y[indices[0]]
                target, alpha = make_targets(z, assignment[indices[0]], data0['classes'])
                reference = numpy_targets(z.numpy(), assignment[indices[0]].numpy(), data0['classes'])
                difference = float(np.max(np.abs(reference-target.numpy())))
                if difference > 3e-7:
                    raise RuntimeError('Independent NumPy LOO mismatch')
                record = dict(dataset=name, seed=seed, condition=condition, split_hash=fingerprint(*indices),
                              noise_hash=fingerprint(graph.y, graph.noisy_y), assignment_hash=state_hash(dict(a=assignment)),
                              changed_train_labels=int((z != y).sum()), modified_val=0, modified_test=0,
                              classes=data0['classes'], target_reference_difference=difference,
                              representation_source=str(source), historical_partition=str(meta_path.with_suffix('.npz')),
                              independent_grace_repeats=False)
                if condition == 'uniform0.4':
                    hd = float(np.max(np.abs(target.numpy()-historical['targets'])))
                    if record['noise_hash'] != meta['noise_hash'] or hd > 3e-7:
                        raise RuntimeError('Historical noisy labels or targets mismatch')
                    record['historical_target_difference'] = hd
                if seed == 0:
                    anchor = torch.load(OLD/'inputs'/name/(condition+'.pt'), weights_only=True)
                    if not torch.equal(z, anchor['observed']) or not torch.equal(target, anchor['target']):
                        raise RuntimeError('Seed0 historical target exact-match failed')
                torch.save(dict(observed=z, target=target, alpha=alpha, valid_y=graph.y[indices[1]]),
                           folder/(condition+'.pt'))
                write(folder/(condition+'.json'), record)
                records.append(record)
                for stratum, selected in [('all', torch.ones(len(y), dtype=torch.bool)), ('corrupted', z != y), ('clean', z == y)]:
                    diagnostics.append(dict(dataset=name, seed=seed, condition=condition, stratum=stratum,
                                            **target_diagnostic(target[selected], z[selected], y[selected])))
            for p in (meta_path, meta_path.with_suffix('.npz'), source):
                sources[str(p)] = sha(p)
            for p in folder.glob('*'):
                sources[str(p)] = sha(p)
        print('PREPARED '+name, flush=True)
    csv_write(ROOT/'offline_diagnostics.csv', diagnostics)
    write(ROOT/'preflight.json', dict(protocols=records, protocol_count=len(records),
          target_shared_between_R0_R2=True, expected_fits=CONFIG['expected_fits']))
    write(ROOT/'environment.json', dict(executable=sys.executable, torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), platform=platform.platform(), threads=torch.get_num_threads()))
    for p in list(ROOT.glob('*.py'))+[ROOT/'PLAN.md', ROOT/'config.json', ROOT/'environment.json']:
        sources[str(p)] = sha(p)
    for name in CONFIG['datasets']:
        for p in (OLD/'pretraining'/name).glob('*'):
            sources[str(p)] = sha(p)
    write(ROOT/'frozen_inputs.json', dict(config=CONFIG, sha256=sources))
    print('FROZEN 36 protocols; 504 fits; no oracle arm selection', flush=True)


def new_model(data, seed):
    set_seed(seed)
    return AlignedMLP(data['x'].shape[1], data['classes'], data['h'].shape[1], CONFIG['hidden'], CONFIG['dropout']).cuda()


def log(handle, row):
    handle.write(json.dumps(row, allow_nan=False)+'\n')
    handle.flush()


def encoder(data, name, seed):
    if seed == 0:
        return OLD/'pretraining'/name/'model.pt'
    folder = ROOT/'encoders'/name/f'seed{seed}'
    if (folder/'result.json').exists():
        if read(folder/'result.json')['status'] != 'ok':
            raise RuntimeError('Review failed encoder before resuming')
        return folder/'model.pt'
    folder.mkdir(parents=True, exist_ok=True)
    model = new_model(data, seed)
    optimizer = torch.optim.Adam(model.encoder_parameters()+list(model.projection.parameters()),
                                 lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    best, selected, checkpoint = float('inf'), 0, None
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as f:
        for epoch in range(1, CONFIG['alignment_epochs']+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = alignment_loss(model.projection(model.encode(data['x'])), data['h'])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(alignment_loss(model.projection(model.encode(data['x'])), data['h']))
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite alignment')
            if value < best:
                best, selected, checkpoint = value, epoch, deepcopy(model.state_dict())
            log(f, dict(epoch=epoch, alignment=float(loss.detach()), evaluation_alignment=value))
    torch.save({k:v.cpu() for k,v in checkpoint.items()}, folder/'model.pt')
    write(folder/'result.json', dict(status='ok', labels_used=False, selected_epoch=selected,
          selection='minimum label-free alignment', value=best, sha256=sha(folder/'model.pt'), seconds=time.perf_counter()-started))
    return folder/'model.pt'


def fit(name, seed, condition, route, rule, smoke=False):
    folder = ROOT/('smoke' if smoke else 'runs')/name/f'seed{seed}'/condition/(route+'_'+rule)
    if (folder/'result.json').exists():
        record = read(folder/'result.json')
        if record['status'] != 'ok':
            raise RuntimeError('Review failed run before resuming: '+str(folder))
        return record
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(name)
    assets = torch.load(ROOT/'inputs'/name/f'seed{seed}'/'assets.pt', weights_only=True)
    cache = torch.load(ROOT/'inputs'/name/f'seed{seed}'/(condition+'.pt'), weights_only=True)
    train, valid = assets['train'].cuda(), assets['valid'].cuda()
    observed, target, valid_y = cache['observed'].cuda(), cache['target'].cuda(), cache['valid_y'].cuda()
    encoder_path = encoder(data, name, seed) if route == 'R2' else None
    model = new_model(data, seed)
    if encoder_path:
        model.load_state_dict(torch.load(encoder_path, map_location='cuda', weights_only=True))
        model.freeze_encoder()
        frozen_hash = state_hash(model.encoder_state())
        with torch.no_grad():
            hidden = model.encode(data['x']).detach()
        params = model.classifier_parameters()
    else:
        frozen_hash, hidden = None, None
        params = model.backbone.parameters()
    optimizer = torch.optim.Adam(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    epochs = 3 if smoke else CONFIG['head_epochs'] if route == 'R2' else CONFIG['epochs']
    meta = dict(dataset=name, seed=seed, condition=condition, route=route, rule=rule, epochs=epochs,
                initial_hash=state_hash(model.deployment_state()),
                initial_rng_hash=state_hash(dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state())),
                input_sha256=sha(ROOT/'inputs'/name/f'seed{seed}'/(condition+'.pt')),
                encoder_source=str(encoder_path) if encoder_path else None,
                target_hash=state_hash(dict(target=target)), smoke=smoke)
    write(folder/'config.json', meta)
    write(folder/'result.json', dict(**meta, status='running'))
    best, epoch_best, checkpoint = -1, 0, None
    means = []
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
            loss, weights = objective(logits[train], observed, target, rule, CONFIG['gce_q'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                evaluated = model.backbone.layers[-1](hidden) if route == 'R2' else model(data['x'])[0]
                correct = int((evaluated[valid].argmax(1) == valid_y).sum())
            if correct > best:
                best, epoch_best, checkpoint = correct, epoch, deepcopy(model.deployment_state())
            avg = float(weights.mean())
            means.append(avg)
            log(f, dict(epoch=epoch, loss=float(loss.detach()), validation_correct=correct,
                        mean_weight=avg, min_weight=float(weights.min()), max_weight=float(weights.max()),
                        weight_ess=float(weights.sum().square()/weights.square().sum().clamp_min(1e-12))))
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
                  valid_total=len(valid), mean_weight=float(np.mean(means)), seconds=time.perf_counter()-started,
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
        jobs = [('cora', 0, 'uniform0.4', route, rule) for route in CONFIG['routes'] for rule in CONFIG['rules']]
    else:
        jobs = [(name, seed, cond, route, rule) for name in CONFIG['datasets'] for seed in CONFIG['seeds']
                for cond in CONFIG['conditions'] for route in CONFIG['routes'] for rule in CONFIG['rules']]
    for number, job in enumerate(jobs, 1):
        result = fit(*job, smoke=smoke)
        write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),
              dict(status='running' if number < len(jobs) else 'completed', completed=number, expected=len(jobs),
                   last_job=job, seconds=time.perf_counter()-started))
        print(f'{number}/{len(jobs)} {job}: val={100*result["validation"]:.3f} epoch={result["best_epoch"]}', flush=True)
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    try:
        if args.phase == 'prepare':
            prepare()
        else:
            run(args.phase == 'smoke')
    except Exception:
        write(ROOT/f'failure_{args.phase}.json', dict(traceback=traceback.format_exc()))
        raise
