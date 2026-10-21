"""Run paired original MLPs only for datasets passing the frozen target gate."""
import json
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import run_screen as screen
from gblcss.data import set_seed, inject_train_label_noise
from gblcss.training import train_x_only_mlp

ROOT = screen.ROOT
d = screen.d
TRAINING = dict(hidden=64, layers=2, dropout=.5, learning_rate=.01,
                weight_decay=.0005, epochs=200)


def summarize():
    rows = [screen.read(file) for file in sorted((ROOT/'students').glob('*/seed*/*/*/result.json'))]
    d.csv_write(ROOT/'student_results.csv', rows)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    groups = frame.groupby(['dataset', 'noise', 'arm']).test_accuracy.agg(['count', 'mean', 'std']).reset_index()
    groups.to_csv(ROOT/'student_summary.csv', index=False)
    lookup = {(r['dataset'], r['seed'], r['noise'], r['arm']): r for r in rows}
    paired = []
    for row in rows:
        if row['arm'] != 'BFS':
            continue
        for reference in ['CE', 'OLD64', 'H_GEOM', 'X_GEOM', 'RANDOM', 'H_KNN']:
            other = lookup.get((row['dataset'], row['seed'], row['noise'], reference))
            if other:
                paired.append(dict(dataset=row['dataset'], seed=row['seed'], noise=row['noise'],
                    reference=reference, accuracy_delta_pp=100*(row['test_accuracy']-other['test_accuracy'])))
    d.csv_write(ROOT/'paired_student_deltas.csv', paired)


def historical_result(name, seed, noise, arm):
    if arm not in ['CE', 'OLD64']:
        return None
    old_arm = 'GB64' if arm == 'OLD64' else 'CE'
    base = ROOT.parent/'heterophily_original_baseline_20261014/students'/name/f'seed{seed}'/noise/old_arm
    other = screen.MECHANISM/'students'/name/'grace'/f'seed{seed}'/noise/old_arm
    for folder in [base, other]:
        if (folder/'result.json').exists():
            return screen.read(folder/'result.json')
    return None


def main():
    gate = screen.read(ROOT/'gate.json')
    assert screen.read(ROOT/'audit/report.json')['status'] == 'passed'
    names = gate['selected_datasets']
    status = dict(status='running', started=time.time(), selected_datasets=names,
                  reused_student_runs=0, training=TRAINING)
    sources = [Path(__file__), ROOT/'PLAN.md', ROOT/'gate.json', ROOT/'run_screen.py',
               ROOT/'source_manifest.json', screen.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py',
               screen.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py']
    manifest = {str(p.resolve()): d.sha(p) for p in sources}
    path = ROOT/'student_source_manifest.json'
    if path.exists() and screen.read(path) != manifest:
        raise ValueError('Student sources changed; use a new version')
    d.write(path, manifest)
    d.write(ROOT/'student_status.json', status)
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/'student_raw.log', encoding='utf-8')])
    try:
        with threadpool_limits(limits=4):
            for name in names:
                for seed in screen.SEEDS:
                    graph = d.load_graph(name, seed)
                    tr = graph.train_index
                    for noise, rate in screen.NOISE.items():
                        data = inject_train_label_noise(graph, 'uniform' if noise == 'clean' else noise, rate, seed)
                        parent = ROOT/'runs'/name/f'seed{seed}'/noise
                        protocol = screen.read(parent/'protocol.json')
                        assert d.fingerprint(tr.numpy(), graph.val_index.numpy(), graph.test_index.numpy()) == protocol['split_hash']
                        assert d.fingerprint(data.noisy_y[tr].numpy()) == protocol['noise_hash']
                        frozen = screen.read(parent/'frozen_targets.json')
                        for arm in screen.ARMS:
                            folder = ROOT/'students'/name/f'seed{seed}'/noise/arm
                            if (folder/'result.json').exists():
                                continue
                            folder.mkdir(parents=True, exist_ok=True)
                            target_path = parent/(arm+'.npz')
                            assert d.sha(target_path) == frozen[arm]
                            with np.load(target_path) as artifact:
                                local = torch.as_tensor(artifact['target'].copy())
                            targets = torch.nn.functional.one_hot(data.noisy_y, data.num_classes).float()
                            targets[tr] = local
                            config = dict(dataset=name, seed=seed, noise=noise, arm=arm, training=TRAINING,
                                device='cuda', target_path=str(target_path), target_hash=frozen[arm],
                                split_hash=protocol['split_hash'], noise_hash=protocol['noise_hash'],
                                checkpoint_selection='clean validation only', student_seed=seed,
                                trainer_hash=d.sha(screen.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py'),
                                model_hash=d.sha(screen.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py'))
                            d.write(folder/'config.json', config)
                            set_seed(seed)
                            started = time.perf_counter()
                            logging.info('START %s seed%d %s %s', name, seed, noise, arm)
                            torch.cuda.reset_peak_memory_stats()
                            result = train_x_only_mlp(data, targets, device='cuda', **TRAINING)
                            elapsed = time.perf_counter()-started
                            torch.save(result.logits, folder/'logits.pt')
                            row = dict(dataset=name, seed=seed, noise=noise, arm=arm,
                                test_accuracy=result.test_accuracy, validation_accuracy=result.validation_accuracy,
                                train_accuracy=result.train_accuracy, best_epoch=result.best_epoch,
                                elapsed_seconds=elapsed, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                target_hash=frozen[arm], newly_trained=True)
                            prior = historical_result(name, seed, noise, arm)
                            if prior:
                                row['historical_accuracy_delta_pp'] = 100*(result.test_accuracy-prior['test_accuracy'])
                            d.write(folder/'result.json', row)
                            summarize()
                            logging.info('DONE %s seed%d %s %s test=%.6f seconds=%.2f',
                                         name, seed, noise, arm, result.test_accuracy, elapsed)
                            del result, targets, local
        for p, digest in manifest.items():
            assert d.sha(p) == digest, p
        expected = len(names)*5*3*7
        actual = len(list((ROOT/'students').glob('*/seed*/*/*/result.json')))
        assert actual == expected
        status.update(status='completed' if names else 'skipped_by_gate',
                      elapsed_seconds=time.time()-status['started'], student_runs=actual)
    except Exception as exc:
        status.update(status='failed', error=repr(exc))
        d.write(ROOT/'student_status.json', status)
        raise
    d.write(ROOT/'student_status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
