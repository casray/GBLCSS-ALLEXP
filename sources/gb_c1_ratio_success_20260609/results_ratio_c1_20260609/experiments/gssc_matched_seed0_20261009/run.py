from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'baseline_main5seed_20261009'))
from protocol import DATASETS, WORKSPACE, load_data, final_metrics, write_json, set_seed, PROJECT
import adapter


def sources():
    files = [ROOT / n for n in ('adapter.py', 'run.py')]
    files += [ROOT.parent / 'baseline_main5seed_20261009/protocol.py',
              adapter.OLD / 'common.py', adapter.OLD / 'gssc_adapter.py',
              WORKSPACE / 'GB-LCSS_GitHub/src/gblcss/models.py',
              WORKSPACE / 'GB-LCSS_GitHub/src/gblcss/data.py',
              PROJECT / 'external/NoisyGL/utils/labelnoise.py']
    files += [adapter.core.VENDOR / n for n in
              ('code/main.py', 'code/model.py', 'param/best_parameters.json')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def audit(folder, data, record, expected_protocol=None):
    if record['source_hashes'] != sources():
        raise RuntimeError('Source changed since training')
    if expected_protocol is not None and record['protocol'] != expected_protocol:
        raise RuntimeError('Protocol identity mismatch')
    for name, digest in record['artifact_hashes'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Artifact identity mismatch: ' + name)
    rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    eligible = [r for r in rows if r['checkpoint_eligible']]
    best = max(eligible, key=lambda r: r['val'])
    cfg = record['config']
    if len(eligible) != cfg['supervised_epochs_total']:
        raise RuntimeError('Supervised epoch budget mismatch')
    if len(rows) != sum(cfg['phase_budgets'].values()):
        raise RuntimeError('Total phase budget mismatch')
    if best['epoch'] != record['best_epoch'] or best['val'] != record['best_val']:
        raise RuntimeError('First validation maximum mismatch')
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if checkpoint['config'] != cfg or checkpoint['best_epoch'] != best['epoch']:
        raise RuntimeError('Checkpoint metadata mismatch')
    model = adapter.MatchedContrast(data.x.shape[1], 64, data.num_classes, cfg).to(data.x.device)
    model.load_state_dict(checkpoint['state']['model'])
    model.eval()
    with torch.no_grad():
        replay = model.backbone(data.x).softmax(-1).cpu()
    saved = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(replay, saved, rtol=0, atol=0)
    if record['parameters'] != model.inference_parameters():
        raise RuntimeError('Inference parameter count mismatch')
    return dict(checkpoint_replay_max_abs_diff=float((replay-saved).abs().max()),
                first_validation_maximum=True, epochs=len(eligible),
                deployed_fc_layers=2, inference_parameters=model.inference_parameters())


def fit(dataset, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    torch.set_num_threads(4)
    folder = ROOT / ('smoke' if smoke else 'runs') / dataset / 'uniform_0.4' / 'seed0'
    folder.mkdir(parents=True, exist_ok=True)
    data, protocol = load_data(dataset, 0, .4)
    cfg = adapter.effective_config(dataset, smoke, 0)
    if (folder / 'result.json').exists():
        record = json.loads((folder / 'result.json').read_text())
        if record['status'] != 'ok' or record['config'] != cfg:
            raise RuntimeError('Completed result identity mismatch')
        audit(folder, data, record, protocol)
        print('Verified completed:', dataset, flush=True)
        return
    if any((folder / n).exists() for n in ('epochs.jsonl', 'checkpoint.pt', 'probabilities.pt', 'failure.json')):
        raise RuntimeError('Partial/failed run preserved; refusing overwrite')
    write_json(folder / 'config.json', cfg)
    write_json(folder / 'protocol.json', protocol)
    frozen = sources()
    write_json(folder / 'source_hashes.json', frozen)
    write_json(folder / 'environment.json', dict(python=sys.executable, torch=str(torch.__version__),
               gpu=torch.cuda.get_device_name(0), platform=platform.platform(), threads=4))
    (folder / 'adapted_training_core.py').write_text(adapter.TRAINING_SOURCE + '\n', encoding='utf-8')
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        set_seed(0)
        result = adapter.run(data, cfg, folder)
        torch.cuda.synchronize()
        if sources() != frozen:
            raise RuntimeError('Source changed during training')
        probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        metrics = final_metrics(dataset, 0, probabilities)
        if abs(metrics['valid'] - result['best_val']) > 1e-9:
            raise RuntimeError('Saved validation accuracy mismatch')
        result.update(dataset=dataset, method='gssc_matched', seed=0, noise_rate=.4,
                      noise_type='uniform', status='ok', smoke=smoke,
                      test=metrics['test'], valid=metrics['valid'], protocol=protocol,
                      wall_seconds=time.perf_counter()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      source_hashes=frozen,
                      artifact_hashes={n: hashlib.sha256((folder/n).read_bytes()).hexdigest()
                                      for n in ('checkpoint.pt', 'probabilities.pt', 'epochs.jsonl', 'adapted_training_core.py')})
        result['audit'] = audit(folder, data, result, protocol)
        write_json(folder / 'result.json', result)
        print(json.dumps({k: result[k] for k in ('dataset','test','valid','best_epoch','wall_seconds','parameters')}), flush=True)
    except Exception:
        write_json(folder / 'failure.json', dict(error=traceback.format_exc()))
        raise


def grid():
    for dataset in DATASETS:
        folder = ROOT / 'runs' / dataset / 'uniform_0.4' / 'seed0'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'console.log').open('a', encoding='utf-8') as log:
            code = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'run.py'),
                                   '--dataset', dataset], stdout=log, stderr=subprocess.STDOUT).returncode
        print(dataset, 'exit', code, flush=True)
        if code:
            raise SystemExit(code)
        result = json.loads((folder / 'result.json').read_text())
        print(json.dumps({k: result[k] for k in ('dataset','test','valid','best_epoch','wall_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=DATASETS, default='cora')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--grid', action='store_true')
    args = p.parse_args()
    if args.grid:
        if args.smoke:
            raise ValueError('Smoke is a separate short Cora integration check')
        grid()
    else:
        fit(args.dataset, args.smoke)
