from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
from common import DATASETS, final_metrics, load_data, write_json
from gcn_adapter import effective_config, run


def fit(dataset, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    folder = ROOT / ('smoke' if smoke else 'runs') / dataset / 'tined_gcn'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        print('Completed, not rerunning:', folder, flush=True)
        return
    if (folder / 'epochs.jsonl').exists():
        raise FileExistsError('Preserve partial run before retrying')
    torch.set_num_threads(4)
    data, protocol = load_data(dataset)
    config = effective_config(dataset, smoke)
    write_json(folder / 'config.json', config)
    write_json(folder / 'protocol.json', protocol)
    previous = ROOT.parent / 'new_graph_mlp_seed0_20261009'
    vendor = previous / 'vendor/tined'
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(folder / 'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                                                python=sys.executable, platform=platform.platform(), author_commit=commit))
    sources = [ROOT / 'gcn_adapter.py', ROOT / 'run_gcn.py', previous / 'tined_adapter.py',
               previous / 'common.py', ROOT.parents[1] / 'GB-LCSS_GitHub/src/gblcss/models.py',
               ROOT.parents[1] / 'GB-LCSS_GitHub/src/gblcss/data.py']
    sources += [vendor / name for name in ['models.py', 'utils_gnn_property.py', 'train_and_eval.py',
                                          'train_student.py', 'tran.conf.yaml', 'best_records/best_records.json']]
    write_json(folder / 'source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        result = run(data, config, folder)
        torch.cuda.synchronize()
        probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        metrics = final_metrics(dataset, probabilities)
        teacher_metrics = final_metrics(dataset, torch.load(folder / 'teacher_probabilities.pt',
                                                            map_location='cpu', weights_only=True))
        if abs(metrics['valid'] - result['best_val']) > 1e-9:
            raise ValueError('Saved student validation mismatch')
        if abs(teacher_metrics['valid'] - result['teacher_best_val']) > 1e-9:
            raise ValueError('Saved teacher validation mismatch')
        result.update(protocol)
        result.update(status='ok', method='tined_gcn_adapted', smoke=smoke,
                      test=metrics['test'], valid=metrics['valid'], teacher_test=teacher_metrics['test'],
                      wall_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(folder / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception:
        write_json(folder / 'failure.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()
    if args.grid:
        for dataset in DATASETS:
            folder = ROOT / 'runs' / dataset / 'tined_gcn'
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                completed = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(ROOT / 'run_gcn.py'),
                                            '--dataset', dataset], stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            print(dataset, 'exit', completed.returncode, flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)
    else:
        fit(args.dataset, args.smoke)
