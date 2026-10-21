from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'gssc_matched_seed0_20261009'
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('gssc_seed0_runner', BASE / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
adapter = previous.adapter
DATASETS = previous.DATASETS
load_data, final_metrics, write_json, set_seed = (
    previous.load_data, previous.final_metrics, previous.write_json, previous.set_seed)


def sources():
    return {**previous.sources(), str(ROOT / 'run_grid.py'):
            hashlib.sha256((ROOT / 'run_grid.py').read_bytes()).hexdigest()}


def location(dataset, seed, smoke=False):
    if dataset not in DATASETS or isinstance(seed, bool) or seed not in range(5):
        raise ValueError('Only approved datasets and seeds0..4')
    return ROOT / ('smoke' if smoke else 'runs') / dataset / 'uniform_0.4' / f'seed{seed}'


def verify_sources(record):
    expected = previous.sources() if record.get('reused_run') else sources()
    if record['source_hashes'] != expected:
        raise RuntimeError('Actual training-source identity mismatch')
    if record['orchestration_source_hashes'] != sources():
        raise RuntimeError('Orchestration source changed')


def audit(folder, data, record, protocol):
    verify_sources(record)
    if record['config'] != adapter.effective_config(data.name, record['smoke'], record['seed']):
        raise RuntimeError('Fixed adaptation configuration mismatch')
    if record['seed'] != protocol['seed'] or record['dataset'] != protocol['dataset']:
        raise RuntimeError('Seed/dataset identity mismatch')
    # Historical audit has a fixed source-table interface; actual sources verified above.
    projected = {**record, 'source_hashes': previous.sources()}
    return previous.audit(folder, data, projected, protocol)


def summarize(records):
    if len(records) != 5 or sorted(r['seed'] for r in records) != list(range(5)):
        raise ValueError('Exactly one completed result per seed0..4 is required')
    values = [100*float(r['test']) for r in records]
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
        raise ValueError('Invalid accuracy')
    return dict(mean_percent=statistics.mean(values), std_percent=statistics.stdev(values),
                seeds=5, std_ddof=1)


def fit(dataset, seed, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU fallback')
    torch.set_num_threads(4)
    folder = location(dataset, seed, smoke)
    folder.mkdir(parents=True, exist_ok=True)
    data, protocol = load_data(dataset, seed, .4)
    config = adapter.effective_config(dataset, smoke, seed)
    frozen = sources()
    if (folder / 'result.json').exists():
        record = json.loads((folder / 'result.json').read_text())
        if record['status'] != 'ok' or record['config'] != config:
            raise RuntimeError('Completed result identity mismatch')
        audit(folder, data, record, protocol)
        print('Verified completed:', dataset, seed, flush=True)
        return
    if any((folder / name).exists() for name in ('epochs.jsonl','checkpoint.pt','probabilities.pt','failure.json')):
        raise RuntimeError('Partial/failed artifacts preserved; refusing overwrite')
    write_json(folder / 'orchestration_source_hashes.json', frozen)
    orchestration_start = time.perf_counter()
    try:
        if seed == 0 and not smoke:
            origin = BASE / 'runs' / dataset / 'uniform_0.4' / 'seed0'
            record = json.loads((origin / 'result.json').read_text())
            if record['status'] != 'ok' or record['smoke'] or record['config'] != config:
                raise RuntimeError('Seed0 reuse configuration mismatch')
            previous.audit(origin, data, record, protocol)
            original_record_hash = hashlib.sha256((origin / 'result.json').read_bytes()).hexdigest()
            for name in ('config.json','protocol.json','source_hashes.json','environment.json',
                         'epochs.jsonl','checkpoint.pt','probabilities.pt','adapted_training_core.py'):
                shutil.copy2(origin / name, folder / name)
            if (origin / 'console.log').exists():
                shutil.copy2(origin / 'console.log', folder / 'original_console.log')
            record.update(reused_run=str(origin), original_result_sha256=original_record_hash,
                          orchestration_source_hashes=frozen,
                          probabilities_file=str(folder / 'probabilities.pt'))
        else:
            write_json(folder / 'config.json', config)
            write_json(folder / 'protocol.json', protocol)
            write_json(folder / 'source_hashes.json', frozen)
            write_json(folder / 'environment.json', dict(python=sys.executable, torch=str(torch.__version__),
                       gpu=torch.cuda.get_device_name(0), platform=platform.platform(), threads=4))
            (folder / 'adapted_training_core.py').write_text(adapter.TRAINING_SOURCE+'\n', encoding='utf-8')
            torch.cuda.reset_peak_memory_stats()
            set_seed(seed)
            start = time.perf_counter()
            record = adapter.run(data, config, folder)
            torch.cuda.synchronize()
            saved = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
            metrics = final_metrics(dataset, seed, saved)
            if abs(metrics['valid'] - record['best_val']) > 1e-9:
                raise RuntimeError('Saved validation accuracy mismatch')
            record.update(dataset=dataset, method='gssc_matched', seed=seed, noise_type='uniform',
                          noise_rate=.4, smoke=smoke, status='ok', test=metrics['test'],
                          valid=metrics['valid'], protocol=protocol,
                          wall_seconds=time.perf_counter()-start,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          source_hashes=frozen, orchestration_source_hashes=frozen,
                          artifact_hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                              for name in ('checkpoint.pt','probabilities.pt','epochs.jsonl','adapted_training_core.py')})
        if sources() != frozen:
            raise RuntimeError('Source changed during execution')
        record['audit'] = audit(folder, data, record, protocol)
        record['orchestration_wall_seconds'] = time.perf_counter()-orchestration_start
        write_json(folder / 'result.json', record)
        print(json.dumps({k:record[k] for k in ('dataset','seed','test','valid','best_epoch','wall_seconds')}), flush=True)
    except Exception:
        write_json(folder / 'failure.json', dict(error=traceback.format_exc()))
        raise


def grid():
    for dataset in DATASETS:
        for seed in range(5):
            folder = location(dataset, seed)
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable,'-B','-X','utf8','-u',str(ROOT/'run_grid.py'),
                       '--dataset',dataset,'--seed',str(seed)]
            with (folder/'console.log').open('a', encoding='utf-8') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            print(dataset,seed,'exit',code,flush=True)
            if code:
                raise SystemExit(code)
            record = json.loads((folder/'result.json').read_text())
            print(json.dumps({k:record[k] for k in ('dataset','seed','test','valid','best_epoch')}
                             | {'reused':bool(record.get('reused_run'))}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grid', action='store_true')
    p.add_argument('--dataset', choices=DATASETS, default='cora')
    p.add_argument('--seed', type=int, choices=range(5), default=1)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.grid:
        if args.smoke:
            raise ValueError('Smoke is separate from the full grid')
        grid()
    else:
        fit(args.dataset,args.seed,args.smoke)
