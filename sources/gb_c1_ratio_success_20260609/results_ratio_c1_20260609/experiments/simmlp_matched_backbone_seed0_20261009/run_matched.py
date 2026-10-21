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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
from common import DATASETS, final_metrics, load_data, set_seed, write_json
from matched_adapter import effective_config, run
import torch


def fit(dataset, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    outdir = ROOT / ('smoke' if smoke else 'runs') / dataset / 'simmlp_matched'
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / 'result.json').exists():
        print('Completed, not rerunning:', outdir, flush=True)
        return
    if (outdir / 'epochs.jsonl').exists():
        raise RuntimeError('Preserve partial run before retrying')
    torch.set_num_threads(4)
    cfg = effective_config(dataset, smoke)
    data, audit = load_data(dataset)
    write_json(outdir / 'protocol.json', audit)
    write_json(outdir / 'config.json', cfg)
    vendor = ROOT.parent / 'simmlp_tined_audit_seed0_20261009/vendor/simmlp'
    commit = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(outdir / 'environment.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                                                python=sys.executable, platform=platform.platform(), author_commit=commit))
    sources = [ROOT / 'matched_adapter.py', ROOT / 'run_matched.py',
               ROOT.parent / 'simmlp_tined_audit_seed0_20261009/simmlp_adapter.py',
               ROOT.parent / 'new_graph_mlp_seed0_20261009/common.py',
               ROOT.parents[1] / 'GB-LCSS_GitHub/src/gblcss/models.py',
               ROOT.parents[1] / 'GB-LCSS_GitHub/src/gblcss/data.py']
    sources += [vendor / name for name in ['model.py', 'loss.py', 'augment.py', 'eval.py', 'main.py']]
    sources += sorted((vendor / 'param/node').glob('*.json'))
    write_json(outdir / 'source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        set_seed(0)
        result = run(data, cfg, outdir)
        torch.cuda.synchronize()
        probs = torch.load(outdir / 'probabilities.pt', map_location='cpu', weights_only=True)
        metrics = final_metrics(dataset, probs)
        if abs(metrics['valid'] - result['best_val']) > 1e-8:
            raise ValueError('Saved prediction validation mismatch')
        result.update(audit)
        result.update(method='simmlp_matched_backbone', status='ok', smoke=smoke,
                      test=metrics['test'], valid=metrics['valid'],
                      wall_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(outdir / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception:
        write_json(outdir / 'failure.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='cora')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()
    if args.grid:
        for dataset in DATASETS:
            folder = ROOT / 'runs' / dataset / 'simmlp_matched'
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                process = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(ROOT / 'run_matched.py'),
                                          '--dataset', dataset], stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            print(dataset, 'exit', process.returncode, flush=True)
            if process.returncode:
                raise SystemExit(process.returncode)
    else:
        fit(args.dataset, args.smoke)
