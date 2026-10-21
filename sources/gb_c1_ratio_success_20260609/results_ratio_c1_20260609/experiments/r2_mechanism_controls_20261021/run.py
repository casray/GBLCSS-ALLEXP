from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'representation_supervision_20261021'
sys.path.insert(0, str(PREVIOUS))
from core import AlignedMLP, alignment_loss, make_targets, state_hash
from protocol import load_dataset, load_condition, read, write, sha
from gblcss.data import set_seed
from gblcss.models import MLP
from kernels import (mass_matched, loss_for, targets_from_counts, neighbor_indices,
                     support_counts, random_partition, shuffle_rows)

CONFIG = read(ROOT/'config.json')


def initialize():
    torch.set_num_threads(CONFIG['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('Verified ROCm GPU required')


def new_model(data):
    set_seed(CONFIG['seed'])
    return AlignedMLP(data['x'].shape[1], data['classes'], data['h'].shape[1]).cuda()


def log_line(handle, value):
    handle.write(json.dumps(value, allow_nan=False)+'\n')
    handle.flush()


def verify_inputs():
    for source, expected in read(ROOT/'frozen_inputs.json')['sha256'].items():
        if sha(source) != expected:
            raise RuntimeError('Frozen source changed: '+source)


def control_encoder(name, route):
    folder = ROOT/'encoders'/name/route
    if (folder/'result.json').exists():
        if read(folder/'result.json')['status'] != 'ok':
            raise RuntimeError('Control encoder failed; explicit review required')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(name)
    model = new_model(data)
    initial = state_hash(model.deployment_state())
    started = time.perf_counter()
    selected, best = 0, None
    write(folder/'result.json', dict(status='running', route=route, dataset=name))
    if route == 'SHUFFLED_H':
        target, order = shuffle_rows(data['h'], CONFIG['shuffle_seed'])
        torch.save(order.cpu(), folder/'permutation.pt')
        optimizer = torch.optim.Adam(model.encoder_parameters()+list(model.projection.parameters()),
                                     lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        value_best = float('inf')
        with (folder/'epochs.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(1, CONFIG['alignment_epochs']+1):
                if time.perf_counter()-started > CONFIG['timeout_seconds_per_fit']:
                    raise TimeoutError('Control pretraining timeout')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = alignment_loss(model.projection(model.encode(data['x'])), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite shuffled alignment')
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    value = float(alignment_loss(model.projection(model.encode(data['x'])), target))
                if value < value_best:
                    value_best, selected, best = value, epoch, deepcopy(model.state_dict())
                log_line(log, dict(epoch=epoch, train_alignment=float(loss.detach()), evaluation_alignment=value))
        model.load_state_dict(best)
    else:
        value_best = None
    torch.save({k:v.cpu() for k,v in model.state_dict().items()}, folder/'model.pt')
    torch.cuda.synchronize()
    write(folder/'result.json', dict(status='ok', dataset=name, route=route, labels_used=False,
          updates=CONFIG['alignment_epochs'] if route=='SHUFFLED_H' else 0, selected_epoch=selected,
          initial_hash=initial, encoder_hash=state_hash(model.encoder_state()), best_alignment=value_best,
          seconds=time.perf_counter()-started, checkpoint_sha256=sha(folder/'model.pt')))
    print(f'ENCODER {name}/{route} selected={selected}', flush=True)
    return folder


def prepare():
    initialize()
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Preparation already frozen; do not overwrite')
    sources = dict(read(PREVIOUS/'frozen_inputs.json')['sha256'])
    for path, expected in sources.items():
        if sha(path) != expected:
            raise RuntimeError('Previous input/source changed: '+path)
    for path in list(ROOT.glob('*.py'))+[ROOT/'PLAN.md', ROOT/'config.json']:
        sources[str(path)] = sha(path)
    write(ROOT/'starting_manifest.json', dict(sha256=sources))
    records = []
    for name in CONFIG['datasets']:
        started = time.perf_counter()
        data = load_dataset(name)
        folder = ROOT/'inputs'/name
        folder.mkdir(parents=True, exist_ok=True)
        train = data['train']
        assignment = data['assignment']
        support = support_counts(assignment[train])
        neighbors = neighbor_indices(data['h'][train], support)
        random = random_partition(assignment, train, CONFIG['random_partition_seed'])
        with threadpool_limits(limits=CONFIG['threads']):
            km = KMeans(n_clusters=len(assignment.unique()), n_init=CONFIG['kmeans_n_init'],
                        max_iter=CONFIG['kmeans_max_iter'], tol=CONFIG['kmeans_tol'],
                        random_state=CONFIG['seed']).fit(data['h'].cpu().numpy())
        groups = dict(GB=assignment, KM=torch.tensor(km.labels_, device='cuda'), RANDOM=random)
        torch.save(dict(assignment={k:v.cpu() for k,v in groups.items()}, knn=neighbors.cpu(),
                        knn_support=support.cpu()), folder/'groups.pt')
        max_votes = neighbors.shape[1]
        valid_neighbors = torch.arange(max_votes, device='cuda')[None, :] < support[:, None]
        for condition in CONFIG['conditions']:
            cache = load_condition(name, condition)
            z = cache['observed']
            targets = dict(CE=F.one_hot(z, data['classes']).float(), GB=cache['target'],
                           MASS=mass_matched(cache['target'], z))
            for group in ('KM', 'RANDOM'):
                targets[group], _ = make_targets(z, groups[group][train], data['classes'])
            votes = torch.zeros((len(train), data['classes']), device='cuda')
            neighbor_labels = z[neighbors]
            votes.scatter_add_(1, neighbor_labels, valid_neighbors.float())
            targets['KNN'], _ = targets_from_counts(z, votes)
            # Validate the independent count-to-target path on the historical GB assignment.
            one_hot = F.one_hot(z, data['classes']).float()
            _, inv = torch.unique(assignment[train], sorted=True, return_inverse=True)
            counts = torch.zeros((int(inv.max())+1, data['classes']), device='cuda')
            counts.index_add_(0, inv, one_hot)
            check, _ = targets_from_counts(z, counts[inv]-one_hot)
            if not torch.equal(check, cache['target']):
                if float((check-cache['target']).abs().max()) > 1e-7:
                    raise RuntimeError('New vote kernel differs from historical target')
            torch.save({k:v.cpu() for k,v in targets.items()}, folder/(condition+'.pt'))
        for route in CONFIG['control_routes']:
            control_encoder(name, route)
        for path in (PREVIOUS/'pretraining'/name).glob('*'):
            sources[str(path)] = sha(path)
        for condition in CONFIG['conditions']:
            for rule in ('T0', 'T1'):
                anchor = PREVIOUS/'main'/name/condition/('R2_'+rule)
                for leaf in ('result.json', 'probabilities.pt', 'student.pt'):
                    sources[str(anchor/leaf)] = sha(anchor/leaf)
        records.append(dict(dataset=name, actual_regions=len(assignment.unique()),
                            training_nodes=len(train), gb_support_mean=float(support.float().mean()),
                            kmeans_regions=len(np.unique(km.labels_)), kmeans_inertia=float(km.inertia_),
                            kmeans_iterations=int(km.n_iter_), preprocessing_seconds=time.perf_counter()-started))
        if time.perf_counter()-started > CONFIG['preprocessing_timeout_seconds']:
            raise TimeoutError('Preprocessing exceeded graph budget')
        print(f'PREPARED {name}: groups={len(assignment.unique())}, support={float(support.float().mean()):.2f}', flush=True)
        load_dataset.cache_clear()
        torch.cuda.empty_cache()
    for subtree in ('inputs', 'encoders'):
        for path in (ROOT/subtree).rglob('*'):
            if path.is_file():
                sources[str(path)] = sha(path)
    for path, expected in read(ROOT/'starting_manifest.json')['sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Source changed during preparation')
    write(ROOT/'environment.json', dict(executable=sys.executable, python=sys.version, torch=torch.__version__,
          hip=torch.version.hip, gpu=torch.cuda.get_device_name(0), platform=platform.platform(), threads=CONFIG['threads']))
    sources[str(ROOT/'environment.json')] = sha(ROOT/'environment.json')
    write(ROOT/'preflight.json', dict(records=records, test_metrics_computed=False))
    write(ROOT/'frozen_inputs.json', dict(config=CONFIG, sha256=sources, created=time.time()))


def encoder_path(name, route):
    return (PREVIOUS/'pretraining'/name/'model.pt' if route=='REAL' else ROOT/'encoders'/name/route/'model.pt')


def fit(name, condition, route, rule, engineering=False):
    folder = ROOT/('engineering' if engineering else 'runs')/name/condition/(route+'_'+rule)
    if (folder/'result.json').exists():
        saved = read(folder/'result.json')
        if saved['status'] != 'ok':
            raise RuntimeError('Failed fit needs explicit review: '+str(folder))
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    data, cache = load_dataset(name), load_condition(name, condition)
    targets = torch.load(ROOT/'inputs'/name/(condition+'.pt'), weights_only=True)
    target = targets.get(rule, targets['CE']).cuda()
    model = new_model(data)
    initial_hash = state_hash(model.deployment_state())
    model.load_state_dict(torch.load(encoder_path(name, route), map_location='cuda', weights_only=True))
    classifier_hash = state_hash(dict(model.backbone.layers[-1].state_dict()))
    model.freeze_encoder()
    encoder_before = state_hash(model.encoder_state())
    with torch.no_grad():
        hidden = model.encode(data['x']).detach()
    optimizer = torch.optim.Adam(model.classifier_parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    epochs = 3 if engineering else CONFIG['head_epochs']
    meta = dict(dataset=name, condition=condition, route=route, rule=rule, seed=CONFIG['seed'],
                initial_hash=initial_hash, classifier_hash=classifier_hash, encoder_hash=encoder_before,
                encoder_source=str(encoder_path(name, route)), target_hash=state_hash(dict(target=target)),
                updates=epochs, engineering=engineering,
                protocol=read(PREVIOUS/'inputs'/name/(condition+'.json')))
    write(folder/'config.json', meta)
    write(folder/'result.json', dict(**meta, status='running'))
    best, selected, best_state = -1, 0, None
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with (folder/'epochs.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(1, epochs+1):
                if time.perf_counter()-started > CONFIG['timeout_seconds_per_fit']:
                    raise TimeoutError('Head fit timeout')
                model.train_head()
                optimizer.zero_grad(set_to_none=True)
                logits = model.backbone.layers[-1](hidden)
                loss = loss_for(logits[data['train']], cache['observed'], target, rule,
                                epsilon=CONFIG['ls_epsilon'], q=CONFIG['gce_q'],
                                label_weight=CONFIG['bootstrap_label_weight'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite head loss')
                loss.backward()
                grad_norm = float(torch.stack([p.grad.square().sum() for p in model.classifier_parameters()]).sum().sqrt())
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_logits = model.backbone.layers[-1](hidden)[data['valid']]
                    correct = int((validation_logits.argmax(1)==cache['valid_y']).sum())
                if correct > best:
                    best, selected, best_state = correct, epoch, deepcopy(model.state_dict())
                log_line(log, dict(epoch=epoch, loss=float(loss.detach()), gradient_norm=grad_norm, valid_correct=correct))
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probability = model(data['x'])[0].softmax(1).cpu()
        if state_hash(model.encoder_state()) != encoder_before:
            raise RuntimeError('Frozen encoder/BN changed')
        deployed = MLP(data['x'].shape[1], 64, data['classes'], 2, .5).cuda().eval()
        state = model.deployment_state()
        deployed.load_state_dict(state)
        with torch.no_grad():
            diff = float((deployed(data['x']).softmax(1).cpu()-probability).abs().max())
        if diff != 0:
            raise RuntimeError('X-only export mismatch')
        anchor_diff = None
        if not engineering and route=='REAL' and rule in ('CE','GB'):
            old = PREVIOUS/'main'/name/condition/('R2_T0' if rule=='CE' else 'R2_T1')
            anchor_diff = float((torch.load(old/'probabilities.pt', weights_only=True)-probability).abs().max())
            if anchor_diff != 0 or read(old/'result.json')['best_epoch'] != selected:
                raise RuntimeError('Previous R2 anchor not exactly reproduced')
        torch.save(state, folder/'student.pt')
        torch.save(probability, folder/'probabilities.pt')
        torch.cuda.synchronize()
        result = dict(**meta, status='ok', best_epoch=selected, valid_correct=best, valid_total=len(data['valid']),
                      validation=best/len(data['valid']), final_validation=correct/len(data['valid']),
                      seconds=time.perf_counter()-started, deployment_difference=diff, anchor_difference=anchor_diff,
                      student_sha256=sha(folder/'student.pt'), probability_sha256=sha(folder/'probabilities.pt'), folder=str(folder))
        write(folder/'result.json', result)
        return result
    except Exception:
        write(folder/'result.json', dict(**meta, status='failed', traceback=traceback.format_exc()))
        raise


def smoke():
    initialize()
    verify_inputs()
    rows = []
    for route, rule in (('REAL','LS'), ('REAL','GCE'), ('REAL','BOOT'), ('REAL','MASS'),
                        ('REAL','KNN'), ('REAL','KM'), ('REAL','RANDOM'), ('RANDOM_ENCODER','GB'), ('SHUFFLED_H','CE')):
        row = fit('cora', 'uniform0.4', route, rule, True)
        rows.append({k:row[k] for k in ('route','rule','validation','deployment_difference')})
    write(ROOT/'smoke.json', dict(status='passed', rows=rows))
    print('SMOKE passed: nine three-update fits', flush=True)


def grid():
    initialize()
    verify_inputs()
    if read(ROOT/'smoke.json')['status'] != 'passed':
        raise RuntimeError('Smoke not passed')
    started = time.time()
    completed = 0
    for name in CONFIG['datasets']:
        for condition in CONFIG['conditions']:
            cells = [('REAL', r) for r in CONFIG['real_rules']]
            cells += [(route, r) for route in CONFIG['control_routes'] for r in CONFIG['control_rules']]
            for route, rule in cells:
                row = fit(name, condition, route, rule)
                completed += 1
                write(ROOT/'progress.json', dict(status='running', completed=completed, total=CONFIG['expected_fits'],
                      last=f'{name}/{condition}/{route}/{rule}', seconds=time.time()-started))
                print(f'DONE {completed}/{CONFIG["expected_fits"]} {name} {condition} {route}/{rule} val={row["validation"]:.4f}', flush=True)
        load_dataset.cache_clear()
        torch.cuda.empty_cache()
    verify_inputs()
    write(ROOT/'progress.json', dict(status='complete', completed=completed, total=CONFIG['expected_fits'], seconds=time.time()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','smoke','grid'])
    command = parser.parse_args().command
    write(ROOT/('execution_'+command+'.json'), dict(command=sys.argv, executable=sys.executable, time=time.time()))
    try:
        globals()[command]()
    except Exception:
        write(ROOT/('failure_'+command+'.json'), dict(traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__=='__main__':
    class Journal:
        def __init__(self, console, file):
            self.console, self.file = console, file

        def write(self, value):
            self.console.write(value)
            self.file.write(value)

        def flush(self):
            self.console.flush()
            self.file.flush()

    with (ROOT/'raw_execution.log').open('a', encoding='utf-8') as file:
        sys.stdout = Journal(sys.stdout, file)
        try:
            main()
        finally:
            sys.stdout.flush()
            sys.stdout = sys.stdout.console
