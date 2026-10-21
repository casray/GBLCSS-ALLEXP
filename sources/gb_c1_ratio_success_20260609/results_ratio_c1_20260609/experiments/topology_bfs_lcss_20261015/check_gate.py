"""Bind gate decisions and derived CSVs to independently audited raw case JSON."""
import itertools
import json
import numpy as np
import pandas as pd
import run_screen as screen

ROOT = screen.ROOT
d = screen.d


def main():
    raw = [r for p in sorted((ROOT/'runs').glob('*/seed*/metrics.json')) for r in screen.read(p)]
    keys = {(r['dataset'], r['seed'], r['noise'], r['arm']) for r in raw}
    expected = set(itertools.product(screen.NAMES, screen.SEEDS, screen.NOISE, screen.ARMS))
    assert len(raw) == len(keys) == 315 and keys == expected
    derived = pd.read_csv(ROOT/'target_metrics.csv')
    lookup = {(r['dataset'], r['seed'], r['noise'], r['arm']): r for r in raw}
    assert len(derived) == 315
    for row in derived.to_dict('records'):
        original = lookup[row['dataset'], row['seed'], row['noise'], row['arm']]
        assert abs(row['target_brier']-original['target_brier']) < 1e-12
    paired = pd.read_csv(ROOT/'paired_target_deltas.csv')
    assert len(paired) == 270
    for row in paired.to_dict('records'):
        key = (row['dataset'], row['seed'], row['noise'])
        expected_delta = lookup[*key, 'BFS']['target_brier']-lookup[*key, row['reference']]['target_brier']
        assert abs(row['brier_delta']-expected_delta) < 1e-12
    gate = screen.read(ROOT/'gate.json')
    for dataset in gate['datasets']:
        passes = []
        for check in dataset['checks']:
            values = np.array([lookup[dataset['dataset'], seed, check['noise'], 'BFS']['target_brier']
                              -lookup[dataset['dataset'], seed, check['noise'], check['reference']]['target_brier']
                              for seed in screen.SEEDS])
            assert abs(values.mean()-check['mean_brier_delta']) < 1e-12
            assert int((values < 0).sum()) == check['improved_seeds']
            passed = bool(values.mean() <= -.005 and (values < 0).sum() >= 4)
            assert passed == check['passed']
            passes.append(passed)
        assert all(passes) == dataset['proceed_to_students']
    conditional = []
    for name, seed, noise, arm in sorted(keys):
        if arm == 'CE':
            continue
        graph = None
        # Saved local clean labels come from the same loader; no training or updates.
        if 'case' not in locals() or case != (name, seed):
            case = (name, seed)
            graph = d.load_graph(name, seed)
            clean = graph.y[graph.train_index].numpy()
        path = ROOT/'runs'/name/f'seed{seed}'/noise/(arm+'.npz')
        with np.load(path) as artifact:
            support = artifact['support']
            q = artifact['q']
            alpha = artifact['alpha']
        valid = support >= 2
        conditional.append(dict(dataset=name, seed=seed, noise=noise, arm=arm,
            eligible_nodes=int(valid.sum()), q_accuracy_eligible=float((q.argmax(1)[valid] == clean[valid]).mean()) if valid.any() else None,
            mean_alpha=float(alpha.mean()), mean_alpha_eligible=float(alpha[valid].mean()) if valid.any() else None))
    d.csv_write(ROOT/'conditional_q_metrics.csv', conditional)
    summary = pd.DataFrame(conditional).groupby(['dataset', 'noise', 'arm'])[
        ['q_accuracy_eligible', 'mean_alpha', 'mean_alpha_eligible']].agg(['mean', 'std'])
    summary.columns = ['_'.join(k) for k in summary.columns]
    summary.reset_index().to_csv(ROOT/'conditional_q_summary.csv', index=False)
    d.write(ROOT/'gate_integrity.json', dict(status='passed', target_rows=315, paired_rows=270,
        check_source_hash=d.sha(__file__), gate_hash=d.sha(ROOT/'gate.json'),
        target_csv_hash=d.sha(ROOT/'target_metrics.csv'), paired_csv_hash=d.sha(ROOT/'paired_target_deltas.csv'),
        raw_metrics_hashes={str(p.resolve()): d.sha(p) for p in sorted((ROOT/'runs').glob('*/seed*/metrics.json'))}))
    print('PASS complete keys315 paired270 raw-JSON gate and conditional q metrics')


if __name__ == '__main__':
    main()
