from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PRIOR = ROOT.parent / 'robust_graph_seed0_20261006'
GUIDED = ROOT.parent / 'gb_guided_distillation_seed0_20261006'
sys.path.insert(0, str(PRIOR))
import run_screen as previous
from regional_loss import RegionalKL

DATASETS = previous.DATASETS
ARMS = ['R0', 'R1', 'R2', 'R3', 'R4']
SETTINGS = dict(strength=1., shuffle_seed=137, hidden=64, layers=2,
                batch_norm=True, dropout=.5, lr=.01, weight_decay=.0005,
                epochs=200, temperature=1, seed=0, scope='all nodes',
                selection='first strict improvement of clean validation accuracy')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f'{key}:{array.dtype}:{array.shape}\n'.encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU training fallback is forbidden')
    torch.set_num_threads(4)


def prior_folder(dataset, rate):
    return PRIOR / f'runs/{dataset}/uniform{rate}/erase'


def cache_folder(dataset, rate):
    return ROOT / f'cache/{dataset}/uniform{rate}'


def run_folder(dataset, rate, arm):
    return ROOT / f'runs/{dataset}/uniform{rate}/{arm}'


def assignment_hash(assignment):
    array = np.ascontiguousarray(assignment, dtype='<i8')
    header = f'dtype={array.dtype.str};shape={array.shape[0]};order=C\n'.encode('ascii')
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def verify_original_code():
    for name, expected in read(PRIOR / 'environment.json')['gblcss_source_hashes'].items():
        if sha(WORKSPACE / name) != expected:
            raise RuntimeError(f'original source changed: {name}')


def prepare():
    require_gpu()
    verify_original_code()
    if (ROOT / 'frozen_inputs.json').exists():
        raise RuntimeError('experiment already frozen; refusing to overwrite')
    started = time.perf_counter()
    sources = {}
    for path in [ROOT / 'PLAN.md', ROOT / 'regional_loss.py', ROOT / 'test_regional_loss.py',
                 ROOT / 'run_regional.py', ROOT / 'analyze.py',
                 PRIOR / 'run_screen.py', PRIOR / 'adapters.py', PRIOR / 'environment.json']:
        sources[str(path)] = sha(path)
    for name in read(PRIOR / 'environment.json')['gblcss_source_hashes']:
        sources[str(WORKSPACE / name)] = sha(WORKSPACE / name)
    for dataset in DATASETS:
        sources[str(previous.DATA_FILES[dataset])] = sha(previous.DATA_FILES[dataset])
        for rate in (.4, .6):
            data, protocol = previous.load_data(dataset, rate)
            prior = prior_folder(dataset, rate)
            reference = read(prior / 'result.json')
            configuration = read(prior / 'config.json')
            assert reference['status'] == 'ok'
            assert reference['split_hash'] == protocol['split_hash']
            assert reference['noise_hash'] == protocol['noise_hash']
            for key in ['hidden', 'layers', 'dropout', 'lr', 'weight_decay', 'epochs', 'temperature']:
                assert configuration['student'][key] == SETTINGS[key], key
            assert configuration['student']['scope'] == 'all nodes'
            archive_path = PRIOR / f'protocol_cache/{dataset}_u{rate}.pt'
            archive = torch.load(archive_path, map_location='cpu', weights_only=True)
            for key, value in dict(train_index=data.train_index, val_index=data.val_index,
                                   test_index=data.test_index, clean=data.y, noisy=data.noisy_y).items():
                assert torch.equal(archive[key], value), f'protocol archive mismatch: {key}'
            ball_path = WORKSPACE / f'experiments/main_protocol_replay_20260922/matrix/{dataset}/default/seed0.npz'
            with np.load(ball_path) as stored:
                assignment = stored['assignment'].copy()
            ball_reference = read(ball_path.with_suffix('.json'))
            assert assignment_hash(assignment) == ball_reference['ball_assignment_hash']
            guided_path = GUIDED / f'targets/{dataset}/uniform{rate}/targets.pt'
            guided = torch.load(guided_path, map_location='cpu', weights_only=False)
            assert np.array_equal(assignment, guided['assignment'])
            teacher_path = prior / 'teacher_probabilities.pt'
            teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
            assert torch.equal(teacher, guided['targets']['D0'])
            assert teacher.shape == (data.num_nodes, data.num_classes)
            random_assignment = np.random.default_rng(137).permutation(assignment)
            assert np.array_equal(np.unique(assignment, return_counts=True)[1],
                                  np.unique(random_assignment, return_counts=True)[1])
            assignments = dict(R1=torch.as_tensor(assignment.copy(), dtype=torch.long),
                               R2=torch.as_tensor(random_assignment.copy(), dtype=torch.long),
                               R3=torch.zeros(data.num_nodes, dtype=torch.long))
            teacher_means = {}
            for arm, ids in assignments.items():
                loss = RegionalKL(teacher.cuda(), ids.cuda())
                teacher_means[arm] = loss.teacher_mean.cpu()
            folder = cache_folder(dataset, rate)
            folder.mkdir(parents=True, exist_ok=False)
            torch.save(dict(teacher=teacher, assignments=assignments, teacher_means=teacher_means),
                       folder / 'regional.pt')
            write(folder / 'config.json', dict(dataset=dataset, noise_rate=rate,
                  protocol=protocol, settings=SETTINGS, teacher_source=str(prior),
                  ball_assignment_hash=assignment_hash(assignment),
                  random_assignment_hash=assignment_hash(random_assignment),
                  ball_count=int(np.unique(assignment).size),
                  group_sizes=np.unique(assignment, return_counts=True)[1].tolist(),
                  teacher_settings=configuration['teacher']))
            for path in [archive_path, ball_path, ball_path.with_suffix('.json'), guided_path,
                         teacher_path, prior / 'teacher.pt', prior / 'student.pt',
                         prior / 'student_probabilities.pt', prior / 'result.json', prior / 'config.json',
                         folder / 'regional.pt', folder / 'config.json']:
                sources[str(path)] = sha(path)
            print(f'FROZEN {dataset} u{rate}: N={data.num_nodes}, balls={np.unique(assignment).size}', flush=True)
    torch.cuda.synchronize()
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          torch=torch.__version__, hip=torch.version.hip, os=platform.platform(),
          gpu=torch.cuda.get_device_name(0), threads=4, settings=SETTINGS))
    sources[str(ROOT / 'environment.json')] = sha(ROOT / 'environment.json')
    write(ROOT / 'frozen_inputs.json', dict(date='2026-10-07', hashes=sources,
          preparation_seconds=time.perf_counter()-started,
          labels='only validation labels used for checkpoint selection; clean labels for post-fit diagnostics'))


def verify_frozen():
    hashes = read(ROOT / 'frozen_inputs.json')['hashes'].copy()
    amendment_path = ROOT / 'guard_amendment.json'
    if amendment_path.exists():
        for item in read(amendment_path)['sources']:
            path = item['path']
            if Path(path).resolve() not in {(ROOT / 'run_regional.py').resolve(), (ROOT / 'analyze.py').resolve()}:
                raise RuntimeError(f'unapproved source amendment: {path}')
            if hashes[path] != item['original_sha256'] or sha(item['snapshot']) != hashes[path]:
                raise RuntimeError(f'original source snapshot changed: {path}')
            hashes[path] = item['revised_sha256']
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen input changed: {path}')


def exact_metrics(probabilities, data):
    predicted = probabilities.argmax(1)
    values = {}
    for split, labels in [('train', data.noisy_y), ('valid', data.y), ('test', data.y)]:
        index = getattr(data, f'{"val" if split == "valid" else split}_index')
        correct = int((predicted[index] == labels[index]).sum())
        values[f'student_{split}'] = correct / index.numel()
        values[f'{split}_correct'] = correct
        values[f'{split}_total'] = index.numel()
    return values


def initial_hash(data):
    previous.set_seed(0)
    model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    result = state_hash(model.state_dict())
    del model
    return result


def train_modified(data, teacher, region, scale, handle):
    previous.set_seed(0)
    model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    start_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best_val, best_epoch, best_state = -1., -1, None
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x)
        node_loss = previous.soft_cross_entropy(logits, teacher)
        region_loss = region(logits.softmax(1)) if region is not None else logits.new_zeros(())
        loss = scale * node_loss + SETTINGS['strength'] * region_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite objective at epoch {epoch}')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(data.x)
            valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
        if valid > best_val:
            best_val, best_epoch, best_state = valid, epoch, deepcopy(model.state_dict())
        handle.write(json.dumps(dict(phase='student', epoch=epoch, loss=float(loss),
                     node_loss=float(node_loss), region_loss=float(region_loss), valid=valid)) + '\n')
        handle.flush()
        if epoch == 1 or epoch % 25 == 0:
            print(f'student epoch={epoch} loss={float(loss):.5f} region={float(region_loss):.5f} val={valid:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(data.x).softmax(1)
    return model, probabilities, best_epoch, start_hash


def fit(dataset, rate, arm, verified=False):
    require_gpu()
    if not verified:
        verify_frozen()
    folder = run_folder(dataset, rate, arm)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
        raise RuntimeError('completed run exists; refusing to overwrite')
    write(folder / 'result.json', dict(dataset=dataset, noise_rate=rate, arm=arm, status='running'))
    try:
        frozen = read(cache_folder(dataset, rate) / 'config.json')
        data, protocol = previous.load_data(dataset, rate)
        assert protocol == frozen['protocol']
        data = previous.gpu_data(data)
        cache = torch.load(cache_folder(dataset, rate) / 'regional.pt', map_location='cpu', weights_only=True)
        teacher = cache['teacher'].cuda().detach()
        region = None
        if arm in ['R1', 'R2', 'R3']:
            region = RegionalKL(teacher, cache['assignments'][arm].cuda())
            torch.testing.assert_close(region.teacher_mean.cpu(), cache['teacher_means'][arm], rtol=1e-5, atol=1e-6)
        scale = 2. if arm == 'R4' else 1.
        expected_initial_hash = initial_hash(data)
        write(folder / 'config.json', dict(**frozen, arm=arm, node_scale=scale,
              region_strength=1. if region is not None else 0.,
              initial_state_hash=expected_initial_hash, objective='soft CE plus size-weighted regional KL'))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as handle:
            if arm == 'R0':
                model, probabilities, best_epoch = previous.train_student(data, teacher, 200, handle)
                actual_initial_hash = expected_initial_hash
            else:
                model, probabilities, best_epoch, actual_initial_hash = train_modified(data, teacher, region, scale, handle)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert actual_initial_hash == expected_initial_hash
        assert torch.isfinite(probabilities).all() and (probabilities.sum(1)-1).abs().max() < 1e-5
        result = dict(dataset=dataset, noise_rate=rate, arm=arm, seed=0, status='ok',
                      student_seconds=elapsed, best_epoch=best_epoch,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      initial_state_hash=actual_initial_hash, **exact_metrics(probabilities, data))
        teacher_test = int((teacher[data.test_index].argmax(1) == data.y[data.test_index]).sum())
        result['teacher_test'] = teacher_test / data.test_index.numel()
        if arm == 'R0':
            old = torch.load(prior_folder(dataset, rate) / 'student_probabilities.pt', map_location='cpu', weights_only=True)
            reference = read(prior_folder(dataset, rate) / 'result.json')
            result['replay_max_abs_difference'] = float((probabilities.cpu()-old).abs().max())
            result['replay_bit_exact'] = bool(torch.equal(probabilities.cpu(), old))
            previous_metrics = exact_metrics(old.cuda(), data)
            result['replay_passed'] = (result['replay_max_abs_difference'] <= 1e-6
                 and best_epoch == reference['student_best_epoch']
                 and result['valid_correct'] == previous_metrics['valid_correct']
                 and result['test_correct'] == previous_metrics['test_correct'])
            if not result['replay_passed']:
                raise RuntimeError(f'baseline replay mismatch: {result}')
        torch.save(model.state_dict(), folder / 'student.pt')
        torch.save(probabilities.cpu(), folder / 'student_probabilities.pt')
        write(folder / 'result.json', result)
        return result
    except Exception:
        failure = traceback.format_exc()
        write(folder / 'result.json', dict(dataset=dataset, noise_rate=rate, arm=arm,
                                          status='failed', traceback=failure))
        raise


def save_results():
    rows = [read(path) for path in sorted((ROOT / 'runs').glob('*/*/*/result.json'))]
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (ROOT / 'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def grid():
    require_gpu()
    verify_frozen()
    replay = read(run_folder('cora', .4, 'R0') / 'result.json')
    assert replay['status'] == 'ok' and replay['replay_passed']
    started = time.perf_counter()
    for dataset in DATASETS:
        for rate in (.4, .6):
            for arm in ARMS:
                folder = run_folder(dataset, rate, arm)
                if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                print(f'START {dataset} u{rate} {arm}', flush=True)
                with (folder / 'console.log').open('w', encoding='utf-8') as handle:
                    with redirect_stdout(handle):
                        result = fit(dataset, rate, arm, verified=True)
                print(f'DONE {dataset} u{rate} {arm}: test={100*result["student_test"]:.4f}, '
                      f'epoch={result["best_epoch"]}, seconds={result["student_seconds"]:.2f}', flush=True)
                save_results()
                gc.collect()
                torch.cuda.empty_cache()
    verify_frozen()
    write(ROOT / 'execution.json', dict(status='ok', grid_seconds=time.perf_counter()-started,
                                        completed_runs=50, datasets=DATASETS, arms=ARMS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--rate', type=float, choices=[.4, .6], default=.4)
    parser.add_argument('--arm', choices=ARMS, default='R0')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.grid:
        grid()
    elif args.fit:
        print(json.dumps(fit(args.dataset, args.rate, args.arm)), flush=True)
    else:
        parser.error('select --prepare, --grid or --fit')
