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

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'gb_regional_distillation_seed0_20261007'
sys.path.insert(0, str(BASE))
import run_regional as old
from regional_loss import RegionalKL as OriginalRegionalKL, residual_components

previous = old.previous
DATASETS = old.DATASETS
SEEDS = (1, 2)
ARMS = ('R0', 'R1')
SNAPSHOTS = (20, 50, 100, 200)
SETTINGS = {**old.SETTINGS, 'seed': None, 'student_seeds': list(SEEDS),
            'teacher_split_noise_seed': 0, 'strength': 1.,
            'trace_every': 10, 'snapshot_epochs': list(SNAPSHOTS),
            'regional_reduction': 'fixed dense membership matrix multiplication'}
read, write, sha = old.read, old.write, old.sha


def folder_for(dataset, rate, seed, arm):
    category = 'reference_seed0' if seed == 0 else 'runs'
    return ROOT / category / dataset / f'uniform{rate}' / f'seed{seed}' / arm


class RegionalKL(OriginalRegionalKL):
    def __init__(self, teacher, assignment):
        super().__init__(teacher, assignment)
        self.membership = teacher.new_zeros((self.sizes.numel(), teacher.shape[0]))
        self.membership[self.groups, torch.arange(teacher.shape[0], device=teacher.device)] = 1/self.sizes[self.groups]
        self.teacher_mean = (self.membership @ teacher.detach()).detach()

    def __call__(self, student):
        if student.shape != self.shape or student.device != self.groups.device:
            raise ValueError('student and teacher must have matching shape and device')
        means = self.membership @ student
        divergence = F.kl_div(means.clamp_min(torch.finfo(means.dtype).tiny).log(),
                              self.teacher_mean, reduction='none').sum(1)
        return (self.weights * divergence).sum()


def gradient_observation(node_loss, region_loss, parameters):
    parameters = tuple(parameters)
    node = torch.autograd.grad(node_loss, parameters, retain_graph=True)
    region = torch.autograd.grad(region_loss, parameters, retain_graph=True)
    a = sum(value.detach().double().square().sum() for value in node)
    b = sum(value.detach().double().square().sum() for value in region)
    dot = sum((x.detach().double() * y.detach().double()).sum() for x, y in zip(node, region))
    return dict(node_gradient_norm=float(a.sqrt()), region_gradient_norm=float(b.sqrt()),
                region_to_node_gradient_ratio=float((b / a.clamp_min(1e-30)).sqrt()),
                gradient_cosine=float(dot / (a*b).sqrt().clamp_min(1e-30)))


def trace_metrics(probabilities, teacher, region, data):
    index = data.val_index
    pred, teacher_pred = probabilities[index].argmax(1), teacher[index].argmax(1)
    labels = data.y[index]
    one_hot = F.one_hot(labels, data.num_classes).float()
    result = dict(pointwise_kl=float(F.kl_div(probabilities.clamp_min(1e-30).log(),
                                            teacher, reduction='batchmean')),
                  real_region_kl=float(region(probabilities)),
                  validation_brier=float((probabilities[index]-one_hot).square().sum(1).mean()),
                  validation_correct=int((pred == labels).sum()),
                  validation_total=index.numel(),
                  teacher_wrong_student_right=int(((teacher_pred != labels) & (pred == labels)).sum()),
                  teacher_right_student_wrong=int(((teacher_pred == labels) & (pred != labels)).sum()))
    result.update({f'residual_{key}': value for key, value in
                   residual_components(probabilities, teacher, region.groups).items()})
    return result


def prepare():
    old.require_gpu()
    old.verify_frozen()
    if (ROOT / 'frozen_v2_inputs.json').exists():
        raise RuntimeError('experiment already frozen')
    paths = [ROOT / name for name in ['PLAN.md', 'run_stability.py', 'analyze.py', 'test_observation.py',
                                      'numerical_amendment.md', 'replay_diagnosis.json']]
    paths += list((ROOT / 'revision_before_deterministic').glob('*'))
    paths += [BASE / name for name in ['frozen_inputs.json', 'guard_amendment.json',
                                      'run_regional.py', 'regional_loss.py']]
    for dataset in DATASETS:
        for rate in (.4, .6):
            paths += [old.cache_folder(dataset, rate) / name for name in ['regional.pt', 'config.json']]
            for arm in ARMS:
                paths += [old.run_folder(dataset, rate, arm) / name for name in
                          ['result.json', 'student.pt', 'student_probabilities.pt', 'epochs.jsonl']]
    write(ROOT / 'environment_v2.json', dict(python=sys.version, executable=sys.executable,
          torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
          platform=platform.platform(), settings=SETTINGS))
    paths.append(ROOT / 'environment_v2.json')
    write(ROOT / 'frozen_v2_inputs.json', dict(date='2026-10-07',
          hashes={str(path): sha(path) for path in paths}, settings=SETTINGS))
    print('FROZEN: 40 new fits; fixed teacher, split, noise, partitions; lambda=1', flush=True)


def verify():
    old.verify_frozen()
    for path, expected in read(ROOT / 'frozen_v2_inputs.json')['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen source/input changed: {path}')


def fit(dataset, rate, seed, arm, observations=True, destination=None):
    folder = destination or folder_for(dataset, rate, seed, arm)
    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
        raise RuntimeError('refusing to overwrite completed fit')
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=dataset, noise_rate=rate, student_seed=seed, arm=arm,
                observations=observations)
    write(folder / 'result.json', dict(**meta, status='running'))
    try:
        data, protocol = previous.load_data(dataset, rate)
        cached_config = read(old.cache_folder(dataset, rate) / 'config.json')
        assert protocol == cached_config['protocol']
        data = previous.gpu_data(data)
        cache = torch.load(old.cache_folder(dataset, rate) / 'regional.pt',
                           map_location='cpu', weights_only=True)
        teacher = cache['teacher'].cuda().detach()
        region = RegionalKL(teacher, cache['assignments']['R1'].cuda())
        torch.testing.assert_close(region.teacher_mean.cpu(), cache['teacher_means']['R1'],
                                   rtol=1e-5, atol=1e-6)
        previous.set_seed(seed)
        model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
        initial_hash = old.state_hash(model.state_dict())
        rng_hash = old.state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()})
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
        write(folder / 'config.json', dict(**meta, settings=SETTINGS, protocol=protocol,
              initial_state_hash=initial_hash, initial_rng_hash=rng_hash,
              input_cache=str(old.cache_folder(dataset, rate) / 'regional.pt'),
              teacher_source=cached_config['teacher_source'],
              ball_assignment_hash=cached_config['ball_assignment_hash']))
        best_val, best_epoch, best_state = -1., -1, None
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as log, \
             (folder / 'trajectory.jsonl').open('w', encoding='utf-8') as trace:
            for epoch in range(1, 201):
                observe = observations and (epoch == 1 or epoch % 10 == 0)
                model.train()
                optimizer.zero_grad()
                logits = model(data.x)
                node_loss = previous.soft_cross_entropy(logits, teacher)
                region_loss = region(logits.softmax(1)) if arm == 'R1' or observe else logits.new_zeros(())
                loss = node_loss + region_loss if arm == 'R1' else node_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite loss at epoch {epoch}')
                grad = {}
                if observe:
                    rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                    grad = gradient_observation(node_loss, region_loss, model.parameters())
                    assert all(parameter.grad is None for parameter in model.parameters())
                    assert torch.equal(rng_cpu, torch.get_rng_state())
                    assert torch.equal(rng_gpu, torch.cuda.get_rng_state())
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model(data.x)
                    valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
                    probabilities = logits.softmax(1) if observe or epoch in SNAPSHOTS else None
                    if observe:
                        diagnostics = trace_metrics(probabilities, teacher, region, data)
                if valid > best_val:
                    best_val, best_epoch, best_state = valid, epoch, deepcopy(model.state_dict())
                log.write(json.dumps(dict(epoch=epoch, loss=loss.item(), node_loss=node_loss.item(),
                          region_loss=region_loss.item() if arm == 'R1' or observe else None,
                          valid=valid)) + '\n')
                log.flush()
                if observe:
                    trace.write(json.dumps(dict(epoch=epoch, gradient_stage='pre_update_train_forward',
                                metrics_stage='post_update_eval', **grad, **diagnostics)) + '\n')
                    trace.flush()
                if epoch in SNAPSHOTS:
                    torch.save(model.state_dict(), folder / f'epoch{epoch}.pt')
                    torch.save(probabilities.cpu(), folder / f'epoch{epoch}_probabilities.pt')
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                probabilities = model(data.x).softmax(1)
                selected = trace_metrics(probabilities, teacher, region, data)
        torch.cuda.synchronize()
        result = dict(**meta, status='ok', best_epoch=best_epoch, initial_state_hash=initial_hash,
                      initial_rng_hash=rng_hash, seconds=time.perf_counter()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      selected_diagnostics=selected, **old.exact_metrics(probabilities, data))
        if seed == 0:
            original = old.run_folder(dataset, rate, arm)
            reference = read(original / 'result.json')
            reference_prob = torch.load(original / 'student_probabilities.pt',
                                        map_location='cpu', weights_only=True)
            original_log = [json.loads(line) for line in (original / 'epochs.jsonl').read_text().splitlines()]
            current_log = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
            result['replay_max_abs_difference'] = float((reference_prob-probabilities.cpu()).abs().max())
            result['replay_bit_exact'] = torch.equal(reference_prob, probabilities.cpu())
            result['validation_trajectory_identical'] = all(a['valid'] == b['valid'] for a,b in zip(original_log,current_log))
            if arm == 'R0':
                assert result['replay_max_abs_difference'] <= 1e-6
                assert result['validation_trajectory_identical']
                for key in ['best_epoch', 'valid_correct', 'test_correct']:
                    assert result[key] == reference[key], (arm, key)
        torch.save(model.state_dict(), folder / 'student.pt')
        torch.save(probabilities.cpu(), folder / 'student_probabilities.pt')
        write(folder / 'result.json', result)
        return result
    except Exception:
        write(folder / 'result.json', dict(**meta, status='failed', traceback=traceback.format_exc()))
        raise


def launch(dataset, rate, seed, arm):
    folder = folder_for(dataset, rate, seed, arm)
    if (folder / 'result.json').exists() and read(folder / 'result.json')['status'] == 'ok':
        return
    folder.mkdir(parents=True, exist_ok=True)
    print(f'START {dataset} u{rate} student_seed={seed} {arm}', flush=True)
    with (folder / 'console.log').open('w', encoding='utf-8') as handle:
        with redirect_stdout(handle):
            result = fit(dataset, rate, seed, arm)
    print(f'DONE {dataset} u{rate} seed={seed} {arm}: test={100*result["student_test"]:.4f}, '
          f'epoch={result["best_epoch"]}, seconds={result["seconds"]:.2f}', flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def grid():
    old.require_gpu()
    verify()
    assert read(ROOT / 'observation_replay.json')['passed']
    started = time.perf_counter()
    for dataset in DATASETS:
        for rate in (.4, .6):
            for seed in (0,)+SEEDS:
                for arm in ARMS:
                    launch(dataset, rate, seed, arm)
    verify()
    write(ROOT / 'execution.json', dict(status='ok', new_runs=40, numerical_reference_runs=20,
                                        student_seeds=list(SEEDS), seconds=time.perf_counter()-started))


def check_observation():
    for arm in ARMS:
        launch('cora', .4, 0, arm)
    control = ROOT / 'engineering' / 'unobserved_R1'
    if not (control/'result.json').exists() or read(control/'result.json')['status'] != 'ok':
        fit('cora', .4, 0, 'R1', observations=False, destination=control)
    observed = folder_for('cora', .4, 0, 'R1')
    left = torch.load(observed/'student_probabilities.pt',map_location='cpu',weights_only=True)
    right = torch.load(control/'student_probabilities.pt',map_location='cpu',weights_only=True)
    a = [json.loads(v) for v in (observed/'epochs.jsonl').read_text().splitlines()]
    b = [json.loads(v) for v in (control/'epochs.jsonl').read_text().splitlines()]
    result = dict(max_abs_difference=float((left-right).abs().max()),
                  bit_exact=torch.equal(left,right),
                  validation_trajectory_identical=all(x['valid']==y['valid'] for x,y in zip(a,b)),
                  best_epoch_identical=read(observed/'result.json')['best_epoch']==read(control/'result.json')['best_epoch'])
    result['passed'] = result['max_abs_difference']<=1e-6 and result['validation_trajectory_identical'] and result['best_epoch_identical']
    write(ROOT/'observation_replay.json',result)
    assert result['passed'],result
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'replay', 'grid'])
    args = parser.parse_args()
    old.require_gpu()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'replay':
        verify()
        check_observation()
        verify()
    else:
        grid()
