from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch

import weighted_dist as wd
import diagnose as knowledge

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SOURCE = knowledge.SOURCE
PRIOR = ROOT.parent / 'gb_signal_increment_diagnostic_20261012'
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_GitHub/src'))
sys.path.insert(0, str(ROOT.parent / 'gb_strong_kd_screen_20261007'))
from gblcss.models import MLP
from vendor.dist_official import DIST

SEEDS = list(range(1000, 1020))
BUDGETS = [.1, .25, .5]
ARMS = ('Uniform', 'GB_size', 'GB_adaptive', 'KM_adaptive', 'BKM_adaptive', 'Random_adaptive')


def folder(dataset, rate, seed):
    return SOURCE / 'runs' / dataset / f'uniform{rate}' / f'seed{seed}'


def partition_paths(dataset, k):
    return (knowledge.ROOT / 'partitions' / f'{dataset}_K{k}' / 'assignment.npy',
            PRIOR / 'partitions' / f'{dataset}_K{k}' / 'BKM.npy',
            PRIOR / 'partitions' / f'{dataset}_K{k}' / 'config.json')


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(f'{key}:{array.dtype}:{array.shape}\n'.encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def export(name, rows):
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze():
    assert not (ROOT/'frozen_inputs.json').exists()
    knowledge.verify()
    preliminary = knowledge.read(knowledge.ROOT/'summary.json')
    assert preliminary['proceed_generic_dist']
    paths = [ROOT/name for name in ('PLAN.md', 'weighted_dist.py', 'test_weighted_dist.py',
                                  'diagnose.py', 'analyze.py')]
    paths += [knowledge.ROOT/name for name in ('diagnose.py', 'frozen_inputs.json', 'summary.json')]
    paths += [WORKSPACE/'GB-LCSS_GitHub/src/gblcss/models.py',
              ROOT.parent/'gb_strong_kd_screen_20261007/vendor/dist_official.py',
              PRIOR/'input_manifest.json', PRIOR/'output_manifest.json']
    prior_hashes = knowledge.read(PRIOR/'output_manifest.json')
    for dataset in knowledge.DATASETS:
        paths.append(knowledge.DATA_FILES[dataset])
        for rate in (.4, .6):
            for seed in (1, 2):
                base = folder(dataset, rate, seed)
                meta = knowledge.read(base/'protocol.json')
                a = torch.load(base/'protocol.pt', map_location='cpu', weights_only=True)['assignment'].numpy()
                k = len(np.unique(a))
                km, bkm, config = partition_paths(dataset, k)
                info = knowledge.read(config)
                assert info['embedding_sha256'] == knowledge.sha(meta['representation_source'])
                assert info['region_count'] == k
                assert knowledge.sha(bkm) == prior_hashes[str(bkm.relative_to(PRIOR))]
                assert knowledge.assignment_hash(np.load(bkm)) == info['assignment_hashes']['BKM']
                paths += [km, bkm, config, Path(meta['representation_source'])]
                paths += [base/name for name in ('protocol.json', 'protocol.pt',
                                                'teacher/probabilities.pt', 'GB0/config.json',
                                                'GB0/epoch200.pt', 'GB0/epoch200_probabilities.pt')]
    hashes = {str(path.resolve()): knowledge.sha(path) for path in sorted(set(paths))}
    original = knowledge.read(SOURCE/'frozen_inputs.json')['hashes']
    for path, value in hashes.items():
        if path in original:
            assert original[path] == value
    knowledge.write(ROOT/'frozen_inputs.json', dict(hashes=hashes, budgets=BUDGETS,
                    sampling_seeds=SEEDS, arms=ARMS, states=['initial', 'epoch200'],
                    true_labels_used=False, compute='GPU DIST and classification-head gradient'))
    knowledge.write(ROOT/'environment.json', dict(python=sys.version, executable=sys.executable,
                    torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                    cuda_available=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    print(f'FROZEN {len(hashes)} files')


def verify():
    for path, expected in knowledge.read(ROOT/'frozen_inputs.json')['hashes'].items():
        assert knowledge.sha(path) == expected, f'changed input: {path}'


def head_gradient(logits, teacher, weights, hidden, population):
    z = logits.detach().requires_grad_(True)
    loss = wd.weighted_dist(z, teacher, weights, population)
    gradient = torch.autograd.grad(loss, z)[0]
    weight_gradient = gradient.T @ hidden
    result = torch.cat((weight_gradient.flatten(), gradient.sum(0)))
    if not torch.isfinite(result).all() or not torch.isfinite(loss):
        raise RuntimeError('nonfinite loss/gradient')
    return loss.detach(), result.detach()


def run():
    assert not (ROOT/'execution.json').exists()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU fallback')
    verify()
    torch.set_num_threads(4)
    started = time.perf_counter()
    metrics, aggregates, allocations, references = [], [], [], []
    for dataset in knowledge.DATASETS:
        stored = torch.load(knowledge.DATA_FILES[dataset], map_location='cpu', weights_only=False)
        raw = stored[0]
        x = (raw['x'] if isinstance(raw, dict) else raw.x).float().cuda()
        del stored, raw
        for rate in (.4, .6):
            for seed in (1, 2):
                base = folder(dataset, rate, seed)
                meta = knowledge.read(base/'protocol.json')
                gb = torch.load(base/'protocol.pt', map_location='cpu', weights_only=True)['assignment'].numpy()
                assert knowledge.assignment_hash(gb) == meta['assignment_hash']
                n, k = len(gb), len(np.unique(gb))
                km_path, bkm_path, _ = partition_paths(dataset, k)
                groups = {'GB': gb, 'KM': np.load(km_path), 'BKM': np.load(bkm_path),
                          'Random': knowledge.random_groups(gb, 100)}
                assert len(x) == n
                for a in groups.values():
                    assert a.shape == (n,) and np.array_equal(np.unique(a), np.arange(k))
                t_raw = torch.load(base/'teacher/probabilities.pt', map_location='cpu', weights_only=True)
                assert torch.isfinite(t_raw).all() and (t_raw >= 0).all()
                torch.testing.assert_close(t_raw.sum(1), torch.ones(n), atol=1e-5, rtol=1e-5)
                teacher = t_raw.cuda().clamp_min(torch.finfo(t_raw.dtype).tiny).log().softmax(1)
                teacher_np = teacher.cpu().double().numpy()
                sampling = {}
                for fraction in BUDGETS:
                    budget = int(n*fraction)
                    for arm in ARMS:
                        if arm == 'Uniform':
                            a, method = gb, 'uniform'
                        else:
                            name, method = arm.split('_')
                            a = groups[name]
                        plan = wd.prepare(a, teacher_np, budget, method)
                        for j, count in enumerate(plan['allocation']):
                            allocations.append(dict(dataset=dataset, rate=rate, seed=seed,
                                fraction=fraction, arm=arm, group=j,
                                nodes=len(plan['groups'][j]), allocated=int(count)))
                        for repeat in SEEDS:
                            ids, w = wd.take(plan, np.random.default_rng(repeat).random(n))
                            assert len(ids) == budget and len(np.unique(ids)) == budget
                            assert np.isclose(w.sum(), 1) and (w > 0).all()
                            sampling[fraction, arm, repeat] = (torch.as_tensor(ids, device='cuda'),
                                                              torch.as_tensor(w, dtype=x.dtype, device='cuda'))
                config = knowledge.read(base/'GB0/config.json')
                for state in ('initial', 'epoch200'):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    model = MLP(x.shape[1], 64, teacher.shape[1], 2, .5).cuda()
                    assert state_hash(model.state_dict()) == config['initial_state_hash']
                    if state == 'epoch200':
                        model.load_state_dict(torch.load(base/'GB0/epoch200.pt', map_location='cuda', weights_only=True))
                    model.eval()
                    captured = []
                    hook = model.layers[-1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
                    with torch.no_grad():
                        logits = model(x)
                    hook.remove()
                    hidden = captured[0]
                    if state == 'epoch200':
                        expected = torch.load(base/'GB0/epoch200_probabilities.pt', weights_only=True, map_location='cuda')
                        torch.testing.assert_close(logits.softmax(1), expected, rtol=1e-5, atol=1e-6)
                        assert torch.equal(logits.argmax(1), expected.argmax(1))
                    weights = torch.full((n,), 1/n, device='cuda')
                    full_loss, full_grad = head_gradient(logits, teacher, weights, hidden, n)
                    reference = DIST()(logits, teacher.clamp_min(torch.finfo(teacher.dtype).tiny).log())
                    torch.testing.assert_close(full_loss, reference, rtol=1e-5, atol=2e-6)
                    norm2 = full_grad.square().sum()
                    if float(norm2) <= 1e-20:
                        raise RuntimeError('zero full gradient; relative metrics undefined')
                    key = dict(dataset=dataset, rate=rate, seed=seed, state=state)
                    references.append(dict(**key, full_loss=float(full_loss), gradient_norm=float(norm2.sqrt()),
                                           state_hash=state_hash(model.state_dict())))
                    for fraction in BUDGETS:
                        for arm in ARMS:
                            grad_sum = torch.zeros_like(full_grad)
                            for repeat in SEEDS:
                                ids, w = sampling[fraction, arm, repeat]
                                loss, grad = head_gradient(logits[ids], teacher[ids], w, hidden[ids], n)
                                grad_sum += grad
                                error = (grad-full_grad).square().sum()/norm2
                                cosine = torch.nn.functional.cosine_similarity(grad, full_grad, dim=0)
                                metrics.append(dict(**key, fraction=fraction, arm=arm, repeat=repeat,
                                    nodes_sampled=len(ids), loss=float(loss), full_loss=float(full_loss),
                                    loss_error=float(loss-full_loss), grad_relative_mse=float(error),
                                    gradient_cosine=float(cosine)))
                            mean_grad = grad_sum/len(SEEDS)
                            bias = (mean_grad-full_grad).norm()/norm2.sqrt()
                            aggregates.append(dict(**key, fraction=fraction, arm=arm,
                                                   mean_gradient_relative_bias=float(bias)))
                    del model, logits, hidden, full_grad
                    print(f'DONE {dataset} u{rate} seed{seed} {state}', flush=True)
                del sampling, teacher
        del x
        torch.cuda.empty_cache()
    export('replicates.csv', metrics)
    export('aggregates.csv', aggregates)
    export('allocations.csv', allocations)
    export('references.csv', references)
    verify()
    knowledge.write(ROOT/'execution.json', dict(status='ok', cases=20, states=40,
                    sampling_repeats=len(SEEDS), metric_rows=len(metrics),
                    seconds=time.perf_counter()-started, input_hash_check='passed',
                    true_label_metrics=0, training_runs=0, device=torch.cuda.get_device_name(0)))
    print('COMPLETE: 40 states, unchanged input hashes, no student training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    output = sys.stdout
    with (ROOT/f'{args.mode}.log').open('a', encoding='utf-8') as handle:
        class Tee:
            def write(self, value):
                output.write(value)
                handle.write(value)
                handle.flush()
            def flush(self):
                output.flush()
                handle.flush()
        with redirect_stdout(Tee()), redirect_stderr(Tee()):
            try:
                freeze() if args.mode == 'freeze' else run()
            except Exception:
                traceback.print_exc()
                raise
