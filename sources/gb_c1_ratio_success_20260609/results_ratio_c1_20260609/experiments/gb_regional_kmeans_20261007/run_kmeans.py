from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'gb_regional_stability_20261007'))
import run_stability as stable

old, previous = stable.old, stable.previous
read, write, sha = stable.read, stable.write, stable.sha
DATASETS = stable.DATASETS
KM = dict(init='k-means++', n_init=10, random_state=0, max_iter=300,
          tol=1e-4, algorithm='lloyd')
SETTINGS = {**stable.SETTINGS, 'student_seeds': [0, 1, 2], 'grouping': 'kmeans',
            'kmeans': KM, 'clustering_threads': 4}


def folder_for(dataset, rate, seed):
    return ROOT / 'runs' / dataset / f'uniform{rate}' / f'seed{seed}' / 'KM'


def prepare():
    old.require_gpu()
    stable.verify()
    if (ROOT / 'frozen_inputs.json').exists():
        raise RuntimeError('experiment already frozen')
    sys.path.insert(0, str(previous.NOISYGL))
    from scripts.linear_probe_embeddings import embedding_path_for
    from utils.frozen_grace_targets import load_frozen_embeddings
    from utils.ordinary_gb import _l2_normalize, _to_numpy_embeddings

    paths = [ROOT / name for name in ('PLAN.md', 'run_kmeans.py', 'analyze.py', 'test_kmeans.py')]
    paths += [stable.ROOT / 'run_stability.py', stable.ROOT / 'frozen_v2_inputs.json']
    paths += [previous.NOISYGL / name for name in ('scripts/linear_probe_embeddings.py',
              'utils/frozen_grace_targets.py', 'utils/ordinary_gb.py')]
    for dataset in DATASETS:
        configs = ROOT.parent / 'main_protocol_replay_20260922' / 'matrix' / dataset / 'default'
        source_config = configs / 'seed0.config.json'
        args = read(source_config)['args']
        source = embedding_path_for(args['ball_embedding_root'] or args['embedding_root'], dataset, 'grace')
        cache = torch.load(old.cache_folder(dataset, .4) / 'regional.pt', weights_only=True)
        assignment = cache['assignments']['R1'].numpy()
        alternate = torch.load(old.cache_folder(dataset, .6) / 'regional.pt', weights_only=True)
        assert np.array_equal(assignment, alternate['assignments']['R1'].numpy())
        original = np.load(configs / 'seed0.npz')['assignment']
        assert np.array_equal(assignment, original)
        features = _l2_normalize(_to_numpy_embeddings(load_frozen_embeddings(source, len(assignment))))
        assert np.isfinite(features).all()
        count = len(np.unique(assignment))
        started = time.perf_counter()
        with threadpool_limits(limits=4):
            model = KMeans(n_clusters=count, **KM).fit(features)
        groups = model.labels_.astype(np.int64)
        sizes = np.bincount(groups, minlength=count)
        assert sizes.min() > 0 and len(groups) == len(assignment)
        destination = ROOT / 'partitions' / dataset
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / 'assignment.npy', groups)
        write(destination / 'config.json', dict(embedding_source=str(source), embedding_sha256=sha(source),
              shape=list(features.shape), normalized_feature_hash=old.state_hash({'x': torch.from_numpy(features)}),
              count=count, sizes=sizes.tolist(), gb_sizes=np.bincount(assignment).tolist(),
              kmeans=KM, sklearn=sklearn.__version__, inertia=float(model.inertia_), iterations=int(model.n_iter_),
              seconds=time.perf_counter()-started, assignment_hash=old.assignment_hash(groups),
              gb_assignment_hash=old.assignment_hash(assignment), ari_with_gb=adjusted_rand_score(assignment, groups)))
        paths += [source_config, source, configs / 'seed0.npz', destination / 'assignment.npy', destination / 'config.json']
        for rate in (.4, .6):
            paths += [old.cache_folder(dataset, rate) / name for name in ('regional.pt', 'config.json')]
            for seed in (0, 1, 2):
                for arm in ('R0', 'R1'):
                    paths += [stable.folder_for(dataset, rate, seed, arm) / name for name in
                              ('config.json', 'result.json', 'student_probabilities.pt', 'epochs.jsonl')]
        print(f'PARTITION {dataset}: {count} groups, size {sizes.min()}..{sizes.max()}', flush=True)
    write(ROOT / 'environment.json', dict(python=sys.version, executable=sys.executable,
          torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
          sklearn=sklearn.__version__, settings=SETTINGS))
    paths.append(ROOT / 'environment.json')
    write(ROOT / 'frozen_inputs.json', dict(hashes={str(p): sha(p) for p in paths}, settings=SETTINGS))


def verify():
    stable.verify()
    for path, expected in read(ROOT / 'frozen_inputs.json')['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen source/input changed: {path}')


def fit(dataset, rate, seed, destination=None, replay=False):
    folder = destination or folder_for(dataset, rate, seed)
    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
        raise RuntimeError('refusing to overwrite completed fit')
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=dataset, noise_rate=rate, student_seed=seed, arm='GB_replay' if replay else 'KM')
    write(folder / 'result.json', dict(**meta, status='running'))
    try:
        data, protocol = previous.load_data(dataset, rate)
        cached_config = read(old.cache_folder(dataset, rate) / 'config.json')
        assert protocol == cached_config['protocol']
        data = previous.gpu_data(data)
        cache = torch.load(old.cache_folder(dataset, rate) / 'regional.pt', map_location='cpu', weights_only=True)
        teacher = cache['teacher'].cuda().detach()
        assignment = cache['assignments']['R1'] if replay else torch.from_numpy(np.load(ROOT / 'partitions' / dataset / 'assignment.npy'))
        region = stable.RegionalKL(teacher, assignment.cuda())
        previous.set_seed(seed)
        model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
        initial_hash = old.state_hash(model.state_dict())
        rng_hash = old.state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()})
        for arm in ('R0', 'R1'):
            reference = read(stable.folder_for(dataset, rate, seed, arm) / 'config.json')
            assert initial_hash == reference['initial_state_hash'] and rng_hash == reference['initial_rng_hash']
            assert protocol == reference['protocol']
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
        write(folder / 'config.json', dict(**meta, settings=SETTINGS, protocol=protocol,
              initial_state_hash=initial_hash, initial_rng_hash=rng_hash,
              input_cache=str(old.cache_folder(dataset, rate) / 'regional.pt'),
              assignment_hash=old.assignment_hash(assignment.numpy()), teacher_source=cached_config['teacher_source']))
        best_val, best_epoch, best_state = -1., -1, None
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as log, (folder / 'trajectory.jsonl').open('w', encoding='utf-8') as trace:
            for epoch in range(1, 201):
                observe = epoch == 1 or epoch % 10 == 0
                model.train()
                optimizer.zero_grad()
                logits = model(data.x)
                node_loss = previous.soft_cross_entropy(logits, teacher)
                region_loss = region(logits.softmax(1))
                loss = node_loss + region_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite loss at epoch {epoch}')
                grad = {}
                if observe:
                    rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                    grad = stable.gradient_observation(node_loss, region_loss, model.parameters())
                    assert all(p.grad is None for p in model.parameters())
                    assert torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_gpu, torch.cuda.get_rng_state())
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model(data.x)
                    valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
                    probabilities = logits.softmax(1) if observe or epoch in stable.SNAPSHOTS else None
                    if observe:
                        diagnostics = stable.trace_metrics(probabilities, teacher, region, data)
                if valid > best_val:
                    best_val, best_epoch, best_state = valid, epoch, deepcopy(model.state_dict())
                log.write(json.dumps(dict(epoch=epoch, loss=loss.item(), node_loss=node_loss.item(), region_loss=region_loss.item(), valid=valid)) + '\n')
                log.flush()
                if observe:
                    trace.write(json.dumps(dict(epoch=epoch, **grad, **diagnostics)) + '\n')
                    trace.flush()
                if epoch in stable.SNAPSHOTS:
                    torch.save(model.state_dict(), folder / f'epoch{epoch}.pt')
                    torch.save(probabilities.cpu(), folder / f'epoch{epoch}_probabilities.pt')
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                probabilities = model(data.x).softmax(1)
                selected = stable.trace_metrics(probabilities, teacher, region, data)
        torch.cuda.synchronize()
        result = dict(**meta, status='ok', best_epoch=best_epoch, initial_state_hash=initial_hash,
                      initial_rng_hash=rng_hash, seconds=time.perf_counter()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      selected_diagnostics=selected, **old.exact_metrics(probabilities, data))
        torch.save(model.state_dict(), folder / 'student.pt')
        torch.save(probabilities.cpu(), folder / 'student_probabilities.pt')
        write(folder / 'result.json', result)
        return result
    except Exception:
        write(folder / 'result.json', dict(**meta, status='failed', traceback=traceback.format_exc()))
        raise


def replay_check():
    destination = ROOT / 'engineering' / 'GB_replay'
    result = fit('cora', .4, 0, destination, replay=True)
    reference = stable.folder_for('cora', .4, 0, 'R1')
    left = torch.load(reference / 'student_probabilities.pt', weights_only=True)
    right = torch.load(destination / 'student_probabilities.pt', weights_only=True)
    a = [json.loads(x)['valid'] for x in (reference / 'epochs.jsonl').read_text().splitlines()]
    b = [json.loads(x)['valid'] for x in (destination / 'epochs.jsonl').read_text().splitlines()]
    check = dict(bit_exact=torch.equal(left, right), max_abs_difference=float((left-right).abs().max()),
                 validation_trajectory_identical=a == b, best_epoch_identical=result['best_epoch'] == read(reference / 'result.json')['best_epoch'])
    check['passed'] = check['bit_exact'] and check['validation_trajectory_identical'] and check['best_epoch_identical']
    write(ROOT / 'replay_check.json', check)
    assert check['passed'], check
    print(json.dumps(check), flush=True)


def grid():
    assert read(ROOT / 'replay_check.json')['passed']
    for dataset in DATASETS:
        for rate in (.4, .6):
            for seed in (0, 1, 2):
                folder = folder_for(dataset, rate, seed)
                if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                print(f'START {dataset} u{rate} seed{seed}', flush=True)
                with (folder / 'console.log').open('w', encoding='utf-8') as handle, redirect_stdout(handle):
                    result = fit(dataset, rate, seed)
                print(f'DONE {dataset} u{rate} seed{seed}: {100*result["student_test"]:.4f}, epoch={result["best_epoch"]}, seconds={result["seconds"]:.1f}', flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    write(ROOT / 'execution.json', dict(status='ok', fits=30, student_seeds=[0, 1, 2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'replay', 'grid'])
    mode = parser.parse_args().mode
    old.require_gpu()
    if mode == 'prepare':
        prepare()
    else:
        verify()
        replay_check() if mode == 'replay' else grid()
        verify()
