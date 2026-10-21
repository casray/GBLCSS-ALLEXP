from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import torch

from protocol import ROOT, WORKSPACE, DATASETS, load_data, final_metrics, write_json
import sim_probe
import tined_formal


def sources():
    files = [ROOT / n for n in ['protocol.py', 'sim_probe.py', 'tined_formal.py', 'run_grid.py']]
    files += [WORKSPACE / 'GB-LCSS_GitHub/src/gblcss' / n for n in ['data.py', 'models.py']]
    files += [ROOT.parent / 'new_graph_mlp_seed0_20261009' / n for n in ['common.py', 'tined_adapter.py']]
    files += [ROOT.parent / 'simmlp_matched_backbone_seed0_20261009/matched_adapter.py',
              ROOT.parent / 'simmlp_tined_audit_seed0_20261009/simmlp_adapter.py']
    files += sorted((ROOT.parent / 'simmlp_tined_audit_seed0_20261009/vendor/simmlp').glob('*.py'))
    files += sorted((ROOT.parent / 'new_graph_mlp_seed0_20261009/vendor/tined').glob('*.py'))
    files += [ROOT.parent / 'new_graph_mlp_seed0_20261009/vendor/tined/best_records/best_records.json']
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def location(method, dataset, rate, seed, smoke=False):
    return ROOT / ('smoke' if smoke else 'runs') / method / dataset / f'uniform_{rate:g}' / f'seed{seed}'


def fit(method, dataset, rate, seed, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU fallback')
    if method == 'simmlp' and rate != .4:
        raise ValueError('SimMLP uniform .6 was not requested')
    torch.set_num_threads(4)
    folder = location(method, dataset, rate, seed, smoke)
    folder.mkdir(parents=True, exist_ok=True)
    data, protocol = load_data(dataset, seed, rate)
    hashes = sources()
    if (folder / 'result.json').exists():
        previous = json.loads((folder / 'result.json').read_text())
        if (previous['status'] != 'ok' or previous['smoke'] != smoke or
            previous['protocol'] != protocol or previous['source_hashes'] != hashes):
            raise RuntimeError('Completed artifact identity mismatch')
        for name in ['probabilities.pt', 'checkpoint.pt', 'epochs.jsonl']:
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != previous['artifact_hashes'][name]:
                raise RuntimeError('Completed artifact hash mismatch: ' + name)
        print('Verified completed, skipping:', folder, flush=True)
        return
    if (folder / 'epochs.jsonl').exists():
        raise RuntimeError('Partial artifacts preserved; do not overwrite or silently retry')
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'source_hashes.json', hashes)
    write_json(folder / 'environment.json', dict(torch=str(torch.__version__),
               gpu=torch.cuda.get_device_name(0), python=sys.executable, platform=platform.platform(),
               threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        if method == 'simmlp':
            _, _, config = sim_probe.build_probe(data, seed)
            if seed == 0 and not smoke:
                original = sim_probe.BASE / 'runs' / dataset / 'simmlp_matched'
                record = json.loads((original / 'result.json').read_text())
                if record['status'] != 'ok' or record['smoke'] or record['split_hash'] != protocol['split_hash'] or record['noise_hash'] != protocol['noise_hash']:
                    raise RuntimeError('Seed0 reuse identity mismatch')
                for name in ['probabilities.pt', 'checkpoint.pt', 'epochs.jsonl']:
                    shutil.copy2(original / name, folder / name)
                result = {key: record[key] for key in ['best_val', 'best_epoch', 'parameters', 'frozen_encoder_hash']}
                result.update(config=config, reused_run=str(original))
            else:
                result = sim_probe.run(data, seed, folder, smoke)
        else:
            config = tined_formal.effective_config(dataset, smoke, seed)
            write_json(folder / 'config.json', config)
            result = tined_formal.run(data, config, folder)
        torch.cuda.synchronize()
        probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        metrics = final_metrics(dataset, seed, probabilities)
        if abs(metrics['valid'] - result['best_val']) > 1e-9:
            raise RuntimeError('Saved validation mismatch')
        if method == 'tined':
            teacher_metrics = final_metrics(dataset, seed, torch.load(folder / 'teacher_probabilities.pt', map_location='cpu', weights_only=True))
            if abs(teacher_metrics['valid'] - result['teacher_best_val']) > 1e-9:
                raise RuntimeError('Teacher validation mismatch')
            result['teacher_test'] = teacher_metrics['test']
        result.update(dataset=dataset, method=method, noise_type='uniform', noise_rate=rate, seed=seed,
                      status='ok', smoke=smoke, test=metrics['test'], valid=metrics['valid'],
                      wall_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      protocol=protocol, source_hashes=hashes,
                      artifact_hashes={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                      for name in ['probabilities.pt', 'checkpoint.pt', 'epochs.jsonl']})
        write_json(folder / 'config.json', result['config'])
        write_json(folder / 'result.json', result)
        print(json.dumps({key: result[key] for key in ['method', 'dataset', 'noise_rate', 'seed', 'test', 'valid', 'best_epoch', 'wall_seconds']}), flush=True)
    except Exception:
        write_json(folder / 'failure.json', dict(error=traceback.format_exc()))
        raise


def grid():
    for method, rates in [('simmlp', [.4]), ('tined', [.4, .6])]:
        for rate in rates:
            for dataset in DATASETS:
                for seed in range(5):
                    folder = location(method, dataset, rate, seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'run_grid.py'),
                               '--method', method, '--dataset', dataset, '--rate', str(rate), '--seed', str(seed)]
                    with (folder / 'console.log').open('a', encoding='utf-8') as log:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                 env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                    print(method, dataset, rate, seed, 'exit', process.returncode, flush=True)
                    if process.returncode:
                        raise SystemExit(process.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--method', choices=['simmlp', 'tined'], default='simmlp')
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--rate', type=float, choices=[.4, .6], default=.4)
    parser.add_argument('--seed', type=int, choices=range(5), default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.grid:
        grid()
    else:
        fit(args.method, args.dataset, args.rate, args.seed, args.smoke)
