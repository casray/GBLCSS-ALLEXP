from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PRIOR = ROOT.parent / 'robust_graph_seed0_20261006'
sys.path.insert(0, str(PRIOR))
import run_screen as previous
from guided_targets import eligible_mask, shuffled_assignment, build_arms

DATASETS = previous.DATASETS
ARMS = ['D0', 'D1', 'D2', 'D3', 'D4']
SETTINGS = dict(strength=.5, beta=.5, min_votes=2, shuffle_seed=137,
                hidden=64, layers=2, batch_norm=True, dropout=.5, lr=.01,
                weight_decay=.0005, epochs=200, temperature=1, seed=0,
                objective='all-node soft cross entropy', selection='strict clean validation improvement')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, content):
    previous.write_json(path, content)


def cache_folder(dataset, rate):
    return ROOT / 'targets' / dataset / f'uniform{rate}'


def prior_folder(dataset, rate):
    return PRIOR / 'runs' / dataset / f'uniform{rate}' / 'erase'


def assignment_hash(assignment):
    canonical = np.ascontiguousarray(assignment, dtype='<i8')
    header = f'dtype={canonical.dtype.str};shape={canonical.shape[0]};order=C\n'.encode('ascii')
    return hashlib.sha256(header + canonical.tobytes()).hexdigest()


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU training fallback forbidden')
    torch.set_num_threads(4)


def verify_sources():
    expected = read(PRIOR / 'environment.json')['gblcss_source_hashes']
    for name, fingerprint in expected.items():
        assert sha(WORKSPACE / name) == fingerprint, f'source changed: {name}'


def prepare():
    require_gpu()
    verify_sources()
    if (ROOT / 'frozen_inputs.json').exists():
        raise RuntimeError('targets already frozen; do not overwrite an active experiment')
    started = time.perf_counter()
    sources = {}
    for path in [ROOT / 'run_guided.py', ROOT / 'guided_targets.py', ROOT / 'test_guided_targets.py',
                 PRIOR / 'run_screen.py', PRIOR / 'adapters.py', PRIOR / 'environment.json',
                 ROOT.parent / 'gb_guided_distillation_proposal_20261006/PLAN.md']:
        sources[str(path)] = sha(path)
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          torch=torch.__version__, hip=torch.version.hip, os=platform.platform(),
          gpu=torch.cuda.get_device_name(0), threads=4, settings=SETTINGS))
    for dataset in DATASETS:
        for rate in (.4, .6):
            begin = time.perf_counter()
            data, protocol = previous.load_data(dataset, rate)
            prior = prior_folder(dataset, rate)
            reference = read(prior / 'result.json')
            assert reference['status'] == 'ok'
            assert reference['split_hash'] == protocol['split_hash']
            assert reference['noise_hash'] == protocol['noise_hash']
            ball_path = WORKSPACE / f'experiments/main_protocol_replay_20260922/matrix/{dataset}/default/seed0.npz'
            assignment = np.load(ball_path)['assignment']
            ball_ref = read(ball_path.with_suffix('.json'))
            assert assignment_hash(assignment) == ball_ref['ball_assignment_hash']
            train_mask = data.train_mask.numpy()
            eligible = eligible_mask(assignment, train_mask, SETTINGS['min_votes'])
            random_assignment = shuffled_assignment(assignment, eligible, SETTINGS['shuffle_seed'])
            assert np.array_equal(random_assignment[~eligible], assignment[~eligible])
            for mask in (np.ones_like(train_mask), train_mask):
                assert np.array_equal(np.unique(assignment[mask], return_counts=True),
                                      np.unique(random_assignment[mask], return_counts=True))
            assert np.array_equal(eligible, eligible_mask(random_assignment, train_mask, 2))
            data = previous.gpu_data(data)
            teacher = torch.load(prior / 'teacher_probabilities.pt', weights_only=True).cuda()
            gb = previous.build_leave_one_out_targets(data.noisy_y, assignment, data.train_index,
                            data.num_classes, min_train_per_ball=2, beta=.5)
            random_gb = previous.build_leave_one_out_targets(data.noisy_y, random_assignment, data.train_index,
                            data.num_classes, min_train_per_ball=2, beta=.5)
            targets, weights = build_arms(teacher, gb.soft_target, random_gb.soft_target,
                                         data.noisy_y, gb.alpha, torch.tensor(eligible, device='cuda'))
            assert torch.equal(targets['D0'], teacher)
            folder = cache_folder(dataset, rate)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(dict(targets={k: v.cpu() for k, v in targets.items()},
                            weights={k: v.cpu() for k, v in weights.items()},
                            gb=gb.soft_target.cpu(), random_gb=random_gb.soft_target.cpu(), alpha=gb.alpha.cpu(),
                            eligible=torch.tensor(eligible), assignment=assignment,
                            random_assignment=random_assignment), folder / 'targets.pt')
            torch.cuda.synchronize()
            frozen = dict(dataset=dataset, noise_rate=rate, protocol=protocol, settings=SETTINGS,
                          prior_result=reference, ball_assignment_hash=assignment_hash(assignment),
                          random_assignment_hash=assignment_hash(random_assignment),
                          eligible_nodes=int(eligible.sum()), train_nodes=int(train_mask.sum()),
                          preparation_seconds=time.perf_counter() - begin,
                          gb_diagnostics=gb.diagnostics, random_gb_diagnostics=random_gb.diagnostics)
            write(folder / 'config.json', frozen)
            for path in [folder / 'targets.pt', folder / 'config.json', ball_path, ball_path.with_suffix('.json'),
                         prior / 'teacher_probabilities.pt', prior / 'student_probabilities.pt',
                         prior / 'teacher.pt', prior / 'student.pt', prior / 'result.json', prior / 'config.json',
                         PRIOR / f'protocol_cache/{dataset}_u{rate}.pt']:
                sources[str(path)] = sha(path)
            print(f'FROZEN {dataset} u{rate}: eligible={int(eligible.sum())}, seconds={time.perf_counter()-begin:.2f}', flush=True)
    write(ROOT / 'frozen_inputs.json', dict(hashes=sources, preparation_seconds=time.perf_counter()-started,
          diagnostic_labels='not consulted for target design; used only in post-freeze analysis'))


def verify_frozen():
    for path, expected in read(ROOT / 'frozen_inputs.json')['hashes'].items():
        assert sha(path) == expected, f'frozen input changed: {path}'


def fit(dataset, rate, arm, replay=False):
    require_gpu()
    verify_frozen()
    folder = ROOT / ('replay' if replay else 'runs') / dataset / f'uniform{rate}' / arm
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frozen = read(cache_folder(dataset, rate) / 'config.json')
    data, protocol = previous.load_data(dataset, rate)
    assert protocol == frozen['protocol']
    data = previous.gpu_data(data)
    cache = torch.load(cache_folder(dataset, rate) / 'targets.pt', weights_only=False)
    target = cache['targets'][arm].cuda()
    write(folder / 'config.json', dict(**frozen, arm=arm, reused=False))
    torch.cuda.reset_peak_memory_stats()
    with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as handle:
        model, probabilities, best_epoch = previous.train_student(data, target, 200, handle)
    torch.cuda.synchronize()
    result = dict(dataset=dataset, noise_rate=rate, arm=arm, seed=0, status='ok', reused=False,
                  best_epoch=best_epoch, seconds=time.perf_counter()-started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  **{f'student_{k}': v for k, v in previous.metrics(probabilities, data).items()},
                  teacher_test=frozen['prior_result']['teacher_test'])
    torch.save(model.state_dict(), folder / 'student.pt')
    torch.save(probabilities.cpu(), folder / 'student_probabilities.pt')
    if replay:
        old = torch.load(prior_folder(dataset, rate) / 'student_probabilities.pt', weights_only=True)
        result['replay_max_abs_difference'] = float((probabilities.cpu()-old).abs().max())
        result['replay_bit_exact'] = bool(torch.equal(probabilities.cpu(), old))
        result['replay_passed'] = (result['replay_max_abs_difference'] <= 1e-6
                                   and result['best_epoch'] == frozen['prior_result']['student_best_epoch']
                                   and result['student_test'] == frozen['prior_result']['student_test']
                                   and result['student_valid'] == frozen['prior_result']['student_valid'])
    write(folder / 'result.json', result)
    print(json.dumps(result), flush=True)
    if replay and not result['replay_passed']:
        raise RuntimeError('D0 replay failed; do not launch grid')


def grid():
    require_gpu()
    verify_frozen()
    assert read(ROOT / 'replay/cora/uniform0.4/D0/result.json')['replay_passed']
    for dataset in DATASETS:
        for rate in (.4, .6):
            frozen = read(cache_folder(dataset, rate) / 'config.json')
            prior = frozen['prior_result']
            folder = ROOT / 'runs' / dataset / f'uniform{rate}' / 'D0'
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / 'result.json', dict(dataset=dataset, noise_rate=rate, arm='D0', seed=0,
                  status='ok', reused=True, source=str(prior_folder(dataset, rate)),
                  best_epoch=prior['student_best_epoch'], seconds=prior['student_seconds'],
                  student_train=prior['student_train'], student_valid=prior['student_valid'],
                  student_test=prior['student_test'], teacher_test=prior['teacher_test']))
            for arm in ARMS[1:]:
                folder = ROOT / 'runs' / dataset / f'uniform{rate}' / arm
                if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                print(f'START {dataset} u{rate} {arm}', flush=True)
                with (folder / 'console.log').open('w', encoding='utf-8') as handle:
                    process = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                               '--fit', '--dataset', dataset, '--rate', str(rate), '--arm', arm],
                               stdout=handle, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f'{dataset} u{rate} {arm} failed; see {folder / "console.log"}')
                result = read(folder / 'result.json')
                print(f'DONE {dataset} u{rate} {arm}: test={100*result["student_test"]:.4f}, epoch={result["best_epoch"]}', flush=True)
                rows = [read(p) for p in sorted((ROOT / 'runs').glob('*/*/*/result.json'))]
                columns = sorted(set().union(*(row.keys() for row in rows)))
                with (ROOT / 'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--rate', type=float, choices=[.4, .6], default=.4)
    parser.add_argument('--arm', choices=ARMS, default='D3')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.replay:
        fit('cora', .4, 'D0', replay=True)
    elif args.grid:
        grid()
    elif args.fit:
        fit(args.dataset, args.rate, args.arm)
    else:
        parser.error('select --prepare, --replay, --grid or --fit')
