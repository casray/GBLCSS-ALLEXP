from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'gb_regional_stability_20261007'))
import run_stability as stable
from losses import KDLoss

old, previous = stable.old, stable.previous
read, write, sha = stable.read, stable.write, stable.sha
DATASETS = stable.DATASETS
CANDIDATES = {'KD': (1., 4.), 'DIST': (1., 4.), 'DGKD-T': (.7, 1.)}
SETTINGS = dict(student_seed=0, protocol_seed=0, hidden=64, layers=2,
                dropout=.5, batch_norm=True, lr=.01, weight_decay=.0005,
                epochs=200, region_strength=1., candidates=CANDIDATES,
                scope='all nodes', dgkd_alpha=15., dgkd_beta=3.,
                dgkd_target='frozen teacher argmax on ALL nodes',
                selection='first strict maximum clean validation accuracy')
warnings.filterwarnings('ignore', message='size_average and reduce args will be deprecated')


def folder_for(dataset, rate, family, temperature, region, replay=False):
    if family == 'KD' and temperature == 1 and not replay:
        return stable.folder_for(dataset, rate, 0, 'R1' if region else 'R0')
    return ROOT / ('engineering' if replay else 'runs') / dataset / f'uniform{rate}' / f'{family}_T{temperature:g}_GB{int(region)}'


def all_cells():
    for dataset in DATASETS:
        for rate in (.4, .6):
            for family, candidates in CANDIDATES.items():
                for temperature in candidates:
                    for region in (False, True):
                        yield dataset, rate, family, temperature, region


def prepare():
    old.require_gpu()
    stable.verify()
    if (ROOT / 'frozen_inputs.json').exists():
        raise RuntimeError('experiment already frozen')
    paths = [ROOT / name for name in ('PLAN.md', 'losses.py', 'run_challenge.py', 'analyze.py', 'test_losses.py', 'test_selection.py', 'selection.py')]
    paths += list((ROOT / 'vendor').glob('*.py'))
    paths += [stable.ROOT / 'run_stability.py', stable.ROOT / 'frozen_v2_inputs.json']
    checks = {}
    for dataset in DATASETS:
        for rate in (.4, .6):
            cache_path = old.cache_folder(dataset, rate) / 'regional.pt'
            teacher = torch.load(cache_path, weights_only=True)['teacher']
            checks[f'{dataset}/{rate}'] = dict(exact_zeros=int((teacher == 0).sum()),
                                              minimum=float(teacher.min()), shape=list(teacher.shape))
            paths += [cache_path, old.cache_folder(dataset, rate) / 'config.json']
            for region in (False, True):
                ref = folder_for(dataset, rate, 'KD', 1, region)
                paths += [ref / name for name in ('config.json', 'result.json', 'student.pt', 'student_probabilities.pt', 'epochs.jsonl')]
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          platform=platform.platform(), torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), teacher_checks=checks, settings=SETTINGS,
          source_commits=dict(DGKD='ae8a495a054578d4b5e36aa5e7d3a3f4872037f6',
                              DIST='6f7a70a1dd57fd88413d83909d3d53243068f701')))
    paths.append(ROOT / 'environment.json')
    write(ROOT / 'frozen_inputs.json', dict(hashes={str(p): sha(p) for p in paths}, settings=SETTINGS))
    print('FROZEN: 100 new fits + 20 references, seed 0; labels excluded from losses', flush=True)


def verify():
    stable.verify()
    for path, expected in read(ROOT / 'frozen_inputs.json')['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen input changed: {path}')


def fit(dataset, rate, family, temperature, use_region, replay=False):
    folder = folder_for(dataset, rate, family, temperature, use_region, replay)
    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
        raise RuntimeError('refusing to overwrite a completed fit')
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=dataset, noise_rate=rate, student_seed=0, family=family,
                temperature=temperature, region=use_region, replay=replay)
    write(folder / 'result.json', dict(**meta, status='running'))
    try:
        data, protocol = previous.load_data(dataset, rate)
        cache_config = read(old.cache_folder(dataset, rate) / 'config.json')
        assert protocol == cache_config['protocol']
        data = previous.gpu_data(data)
        cache = torch.load(old.cache_folder(dataset, rate) / 'regional.pt', weights_only=True)
        teacher = cache['teacher'].cuda().detach()
        region = stable.RegionalKL(teacher, cache['assignments']['R1'].cuda())
        objective = KDLoss(family, teacher, temperature)
        previous.set_seed(0)
        model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
        initial = old.state_hash(model.state_dict())
        rng = old.state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()})
        for arm in ('R0', 'R1'):
            reference = read(stable.folder_for(dataset, rate, 0, arm) / 'config.json')
            assert initial == reference['initial_state_hash'] and rng == reference['initial_rng_hash']
            assert protocol == reference['protocol']
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
        write(folder / 'config.json', dict(**meta, settings=SETTINGS, protocol=protocol,
              initial_state_hash=initial, initial_rng_hash=rng,
              teacher_source=cache_config['teacher_source'], input_cache=str(old.cache_folder(dataset, rate) / 'regional.pt'),
              assignment_hash=old.assignment_hash(cache['assignments']['R1'].numpy())))
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
                node_loss = objective(logits)
                region_loss = region(logits.softmax(1)) if use_region or observe else logits.new_zeros(())
                loss = node_loss + region_loss if use_region else node_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite loss at epoch {epoch}')
                if observe:
                    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                    grad = stable.gradient_observation(node_loss, region_loss, model.parameters())
                    assert all(p.grad is None for p in model.parameters())
                    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model(data.x)
                    correct = int((logits[data.val_index].argmax(1) == data.y[data.val_index]).sum())
                    valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
                    probabilities = logits.softmax(1) if observe or epoch in stable.SNAPSHOTS else None
                    if observe:
                        diagnostics = stable.trace_metrics(probabilities, teacher, region, data)
                if valid > best_val:
                    best_val, best_epoch, best_state, best_correct = valid, epoch, deepcopy(model.state_dict()), correct
                log.write(json.dumps(dict(epoch=epoch, loss=loss.item(), node_loss=node_loss.item(),
                                         region_loss=region_loss.item(), valid=valid, valid_correct=correct)) + '\n')
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
        result = dict(**meta, status='ok', best_epoch=best_epoch, initial_state_hash=initial,
                      initial_rng_hash=rng, seconds=time.perf_counter()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      valid_correct=best_correct, valid_total=data.val_index.numel(),
                      student_valid=best_correct/data.val_index.numel(), selected_diagnostics=selected)
        torch.save(model.state_dict(), folder / 'student.pt')
        torch.save(probabilities.cpu(), folder / 'student_probabilities.pt')
        write(folder / 'result.json', result)
        return result
    except Exception:
        write(folder / 'result.json', dict(**meta, status='failed', traceback=traceback.format_exc()))
        raise


def replay():
    checks = []
    for use_region in (False, True):
        args = ('cora', .4, 'KD', 1., use_region)
        folder = folder_for(*args, replay=True)
        if not (folder / 'result.json').exists() or read(folder / 'result.json')['status'] != 'ok':
            fit(*args, replay=True)
        reference = folder_for(*args)
        a = torch.load(folder / 'student_probabilities.pt', weights_only=True)
        b = torch.load(reference / 'student_probabilities.pt', weights_only=True)
        left = [json.loads(x)['valid'] for x in (folder / 'epochs.jsonl').read_text().splitlines()]
        right = [json.loads(x)['valid'] for x in (reference / 'epochs.jsonl').read_text().splitlines()]
        checks.append(dict(region=use_region, bit_exact=torch.equal(a, b),
                           max_abs_difference=float((a-b).abs().max()), validation_identical=left == right,
                           epoch_identical=read(folder/'result.json')['best_epoch'] == read(reference/'result.json')['best_epoch']))
    passed = all(x['bit_exact'] and x['validation_identical'] and x['epoch_identical'] for x in checks)
    write(ROOT / 'replay_check.json', dict(passed=passed, checks=checks))
    assert passed, checks
    print(json.dumps(checks), flush=True)


def grid():
    assert read(ROOT / 'replay_check.json')['passed']
    completed, started = 0, time.perf_counter()
    for args in all_cells():
        if args[2] == 'KD' and args[3] == 1:
            continue
        folder = folder_for(*args)
        if (folder / 'result.json').exists() and read(folder/'result.json')['status'] == 'ok':
            completed += 1
            continue
        folder.mkdir(parents=True, exist_ok=True)
        print(f'START {args}', flush=True)
        with (folder / 'console.log').open('w', encoding='utf-8') as handle, redirect_stdout(handle):
            result = fit(*args)
        completed += 1
        print(f'DONE {completed}/100 {args}: val={result["student_valid"]:.6f}, epoch={result["best_epoch"]}, sec={result["seconds"]:.1f}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    verify()
    write(ROOT / 'execution.json', dict(status='ok', new_fits=completed, reused_fits=20,
                                      seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'replay', 'grid'])
    args = parser.parse_args()
    old.require_gpu()
    if args.mode == 'prepare':
        prepare()
    else:
        verify()
        replay() if args.mode == 'replay' else grid()
