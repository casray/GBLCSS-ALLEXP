from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'gb_strong_kd_screen_20261007'))
import run_challenge as screen

stable, old, previous = screen.stable, screen.old, screen.previous
read, write, sha = screen.read, screen.write, screen.sha
DATASETS = screen.DATASETS
MATRIX = ROOT.parent / 'main_protocol_replay_20260922/matrix'


def cells():
    for dataset in DATASETS:
        for rate in (.4, .6):
            for seed in (1, 2):
                yield dataset, rate, seed


def load_data(dataset, rate, seed):
    stored = torch.load(previous.DATA_FILES[dataset], map_location='cpu', weights_only=False)
    raw = previous.Data.from_dict(stored[0]) if isinstance(stored[0], dict) else stored[0]
    indices = previous.percent_split_indices(raw.x.shape[0], seed)
    masks = []
    for index in indices:
        mask = torch.zeros(raw.x.shape[0], dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    data = previous.GraphData(dataset, raw.x.float(), raw.edge_index.long(), raw.y.long(),
                              *masks, train_index=torch.tensor(indices[0]),
                              val_index=torch.tensor(indices[1]), test_index=torch.tensor(indices[2]))
    previous.inject_train_label_noise(data, 'uniform', rate, seed)
    split_hash = previous.fingerprint(*indices)
    noise_hash = previous.fingerprint(data.y, data.noisy_y)
    reference_path = MATRIX / dataset / 'default' / f'seed{seed}.json'
    reference = read(reference_path)
    assert split_hash == reference['split_hash']
    if rate == .4:
        assert noise_hash == reference['noise_hash']
    else:
        rows = []
        for path in (previous.NOISYGL / 'runs').glob('mlp*/teacher_u0p6_seeds0_4.csv'):
            with path.open(encoding='utf-8-sig') as handle:
                rows.extend(r for r in csv.DictReader(handle) if r['dataset'] == dataset and r['seed'] == str(seed))
        if not rows:
            path = previous.NOISYGL / 'runs/mlp57_xteacher_loo_cleanup_20260813/xteacher_uniform0p6_seeds0_4.csv'
            with path.open(encoding='utf-8-sig') as handle:
                rows.extend(r for r in csv.DictReader(handle) if r.get('dataset') == dataset
                            and r.get('seed') == str(seed) and r.get('noise_scope') == 'train')
        assert rows and all(r['split_hash'] == split_hash and r['noise_hash'] == noise_hash for r in rows)
    with np.load(reference_path.with_suffix('.npz')) as cache:
        assignment = cache['assignment'].copy()
    assert old.assignment_hash(assignment) == reference['ball_assignment_hash']
    assert assignment.shape == (data.num_nodes,)
    assert torch.equal(data.noisy_y[~data.train_mask], data.y[~data.train_mask])
    meta = dict(dataset=dataset, rate=rate, seed=seed, split_hash=split_hash,
                noise_hash=noise_hash, assignment_hash=old.assignment_hash(assignment),
                partition_source=str(reference_path.with_suffix('.npz')),
                representation_source=reference['ball_embedding_path'],
                modified_train_count=int((data.noisy_y != data.y).sum()))
    return data, meta, torch.as_tensor(assignment, dtype=torch.long)


def folder_for(dataset, rate, seed):
    return ROOT / 'runs' / dataset / f'uniform{rate}' / f'seed{seed}'


def prepare():
    assert not (ROOT / 'frozen_inputs.json').exists()
    screen.verify()
    paths = list(ROOT.glob('*.py')) + [ROOT / 'PLAN.md', screen.ROOT / 'frozen_inputs.json']
    paths += [previous.ROOT / 'run_screen.py', previous.ROOT / 'adapters.py']
    paths += list((previous.ROOT / 'vendor/erase/scripts').glob('*.py'))
    for dataset, rate, seed in cells():
        data, meta, assignment = load_data(dataset, rate, seed)
        folder = folder_for(dataset, rate, seed)
        folder.mkdir(parents=True, exist_ok=True)
        write(folder / 'protocol.json', meta)
        torch.save(dict(train=data.train_index, val=data.val_index, test=data.test_index,
                        clean=data.y, noisy=data.noisy_y, assignment=assignment), folder / 'protocol.pt')
        paths += [folder / 'protocol.json', folder / 'protocol.pt', previous.DATA_FILES[dataset],
                  Path(meta['partition_source']), Path(meta['partition_source']).with_suffix('.json'),
                  Path(meta['representation_source'])]
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          platform=platform.platform(), torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), seeds=[1, 2], DIST_temperature=1, region_weight=1))
    paths.append(ROOT / 'environment.json')
    write(ROOT / 'frozen_inputs.json', dict(hashes={str(p): sha(p) for p in set(paths)}))
    print('Frozen: 20 protocols, 20 teachers, 40 students', flush=True)


def verify():
    screen.verify()
    for path, expected in read(ROOT / 'frozen_inputs.json')['hashes'].items():
        assert sha(path) == expected, path


def teacher_fit(data, folder, seed):
    out = folder / 'teacher'
    out.mkdir(exist_ok=True)
    if (out / 'result.json').exists() and read(out / 'result.json')['status'] == 'ok':
        return torch.load(out / 'probabilities.pt', weights_only=True).cuda()
    previous.set_seed(seed)
    start = time.perf_counter()
    with (out / 'epochs.jsonl').open('w', encoding='utf-8') as log:
        model, probs, epoch, settings = previous.train_erase(data, 400, log)
    torch.cuda.synchronize()
    assert torch.isfinite(probs).all() and (probs.sum(1)-1).abs().max() < 1e-5
    torch.save(dict(state=model.state_dict(), settings=settings), out / 'teacher.pt')
    torch.save(probs.cpu(), out / 'probabilities.pt')
    write(out / 'config.json', dict(seed=seed, settings={k: v for k, v in settings.items() if k != 'probe_state'}))
    write(out / 'result.json', dict(status='ok', seconds=time.perf_counter()-start,
          best_epoch=epoch, valid_correct=int((probs[data.val_index].argmax(1) == data.y[data.val_index]).sum())))
    return probs.detach()


def student_fit(data, teacher, assignment, seed, use_region, folder):
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'result.json').exists(), 'completed student must not be overwritten'
    region = stable.RegionalKL(teacher, assignment.cuda())
    objective = screen.KDLoss('DIST', teacher, 1.)
    previous.set_seed(seed)
    model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    config = dict(seed=seed, DIST_temperature=1., region_weight=int(use_region), epochs=200,
                  hidden=64, layers=2, dropout=.5, lr=.01, weight_decay=.0005,
                  initial_state_hash=old.state_hash(model.state_dict()),
                  initial_rng_hash=old.state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()}),
                  teacher_hash=old.state_hash({'probabilities': teacher.cpu()}),
                  assignment_hash=old.assignment_hash(assignment.cpu().numpy()))
    write(folder / 'config.json', config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best_val, best_epoch, best_state = -1, -1, None
    start = time.perf_counter()
    with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x)
            node_loss = objective(logits)
            region_loss = region(logits.softmax(1)) if use_region else logits.new_zeros(())
            loss = node_loss + region_loss if use_region else node_loss
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(data.x)
                correct = int((logits[data.val_index].argmax(1) == data.y[data.val_index]).sum())
            if correct > best_val:
                best_val, best_epoch, best_state = correct, epoch, deepcopy(model.state_dict())
            log.write(json.dumps(dict(epoch=epoch, loss=float(loss), node_loss=float(node_loss),
                                     region_loss=float(region_loss), valid_correct=correct)) + '\n')
            log.flush()
        torch.save(logits.softmax(1).cpu(), folder / 'epoch200_probabilities.pt')
        torch.save(model.state_dict(), folder / 'epoch200.pt')
        model.load_state_dict(best_state)
        with torch.no_grad():
            probs = model(data.x).softmax(1)
    torch.save(model.state_dict(), folder / 'student.pt')
    torch.save(probs.cpu(), folder / 'probabilities.pt')
    torch.cuda.synchronize()
    write(folder / 'result.json', dict(status='ok', best_epoch=best_epoch, valid_correct=best_val,
                                       seconds=time.perf_counter()-start))


def replay():
    data, _, _ = load_data('cora', .4, 0)
    data = previous.gpu_data(data)
    cache = torch.load(old.cache_folder('cora', .4) / 'regional.pt', weights_only=True)
    checks = []
    for region in (False, True):
        folder = ROOT / 'engineering' / f'GB{int(region)}'
        student_fit(data, cache['teacher'].cuda(), cache['assignments']['R1'], 0, region, folder)
        reference = screen.folder_for('cora', .4, 'DIST', 1., region)
        a = torch.load(folder / 'probabilities.pt', weights_only=True)
        b = torch.load(reference / 'student_probabilities.pt', weights_only=True)
        trajectory = [json.loads(s)['valid_correct'] for s in (folder/'epochs.jsonl').read_text().splitlines()]
        original = [json.loads(s)['valid_correct'] for s in (reference/'epochs.jsonl').read_text().splitlines()]
        checks.append(dict(region=region, exact=torch.equal(a,b), trajectory_equal=trajectory == original,
                           max_difference=float((a-b).abs().max())))
    write(ROOT/'replay_check.json', checks)
    assert all(c['exact'] and c['trajectory_equal'] for c in checks), checks
    print(checks, flush=True)


def grid():
    assert all(c['exact'] and c['trajectory_equal'] for c in read(ROOT/'replay_check.json'))
    for number, (dataset, rate, seed) in enumerate(cells(), 1):
        folder = folder_for(dataset, rate, seed)
        print(f'START {number}/20 {dataset} {rate} seed{seed}', flush=True)
        try:
            data, meta, assignment = load_data(dataset, rate, seed)
            assert meta == read(folder / 'protocol.json')
            data = previous.gpu_data(data)
            with (folder/'console.log').open('a', encoding='utf-8') as log, redirect_stdout(log):
                teacher = teacher_fit(data, folder, seed)
                for region in (False, True):
                    target = folder / f'GB{int(region)}'
                    if not (target / 'result.json').exists():
                        student_fit(data, teacher, assignment, seed, region, target)
                a, b = [read(folder/f'GB{i}/config.json') for i in (0,1)]
                for key in ('initial_state_hash','initial_rng_hash','teacher_hash','assignment_hash'):
                    assert a[key] == b[key], key
            print(f'DONE {number}/20 {dataset} {rate} seed{seed}', flush=True)
            del data, teacher
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            write(folder/'failure.json', dict(traceback=traceback.format_exc()))
            raise
    verify()
    write(ROOT/'execution.json', dict(status='ok', teachers=20, students=40))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','replay','grid'])
    args = parser.parse_args()
    old.require_gpu()
    if args.mode == 'prepare':
        prepare()
    else:
        verify()
        replay() if args.mode == 'replay' else grid()
