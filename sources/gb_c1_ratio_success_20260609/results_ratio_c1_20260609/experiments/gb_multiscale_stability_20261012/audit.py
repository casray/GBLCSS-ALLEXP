from __future__ import annotations

import csv

import numpy as np

import screen as s


def audit():
    s.verify()
    manifest = s.read(s.ROOT / 'score_manifest.json')
    assert len(manifest) == 192
    assert all(s.sha(s.ROOT / p) == digest for p, digest in manifest.items())
    with (s.ROOT / 'coverage_curves.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1728
    maximum_error = 0.
    for row in rows:
        name, kind, seed = row['dataset'], row['kind'], int(row['seed'])
        scores = s.ROOT / 'scores' / name / f'{kind}_seed{seed}' / f"{row['method']}_{row['variant']}.npz"
        labels = s.ROOT / 'protocols' / name / f'{kind}_seed{seed}' / 'labels.npz'
        with np.load(scores) as out, np.load(labels) as truth:
            np.testing.assert_array_equal(out['train'], truth['train'])
            np.testing.assert_array_equal(out['observed'], truth['observed'])
            candidate = out['candidate']
            assert np.all(out['prediction'][candidate] != out['observed'][candidate])
            assert np.all(out['unique_support'][candidate] >= 2)
            assert np.all((out['stability'] >= 0) & (out['stability'] <= 1))
            assert np.isfinite(out['probability']).all()
            np.testing.assert_allclose(out['probability'].sum(1), 1, atol=1e-12)
            ranking = out[row['ranking']]
            if row['coverage_type'] == 'candidate_fraction':
                count = int(np.ceil(candidate.sum() * float(row['fraction'])))
            else:
                count = int(np.ceil(len(candidate) * float(row['fraction'])))
            enough = count > 0 and candidate.sum() >= count
            assert enough == (row['sufficient'] == 'True')
            weights = np.zeros(len(candidate))
            if enough:
                cutoff = np.sort(ranking[candidate])[::-1][count-1]
                above = candidate & (ranking > cutoff)
                tied = candidate & (ranking == cutoff)
                weights[above] = 1.
                weights[tied] = (count - above.sum()) / tied.sum()
            clean, obs, pred = truth['clean'], truth['observed'], out['prediction']
            fixed = (weights * ((clean != obs) & (pred == clean))).sum()
            harm = (weights * ((clean == obs) & (pred != clean))).sum()
            metrics = dict(selected=weights.sum(),
                           precision=(weights * (pred == clean)).sum() / weights.sum() if enough else None,
                           recall=fixed/max(int((clean != obs).sum()),1),
                           clean_corruption=harm/max(int((clean == obs).sum()),1),
                           actual_training_coverage=weights.sum()/len(clean),
                           net_accuracy_change=(fixed-harm)/len(clean), fixed=fixed, harmed=harm)
            for field, expected in metrics.items():
                if expected is None:
                    assert row[field] == ''
                else:
                    error = abs(float(row[field])-float(expected))
                    maximum_error = max(error, maximum_error)
                    assert error < 1e-12, (row, field, expected)
    with (s.ROOT / 'mechanism_deltas.csv').open(encoding='utf-8-sig', newline='') as handle:
        units = list(csv.DictReader(handle))
    assert len(units) == 4
    decision = s.read(s.ROOT / 'decision.json')
    valid = all(row['valid'] == 'True' for row in units)
    g1 = valid and np.mean([float(r['precision']) for r in units]) >= .03
    g1 = bool(g1 and sum(float(r['precision']) > 0 for r in units) >= 3
              and np.mean([float(r['clean_corruption']) for r in units]) <= 0)
    assert g1 == decision['G1']
    s.write(s.ROOT / 'audit.json', dict(status='passed', input_hashes_unchanged=True,
           score_hashes_unchanged=True, evaluated_rows=len(rows), score_files=len(manifest),
           independent_metrics_max_abs_difference=maximum_error, fixed_candidates_across_rankings=True,
           primary_gate_recomputed=True, model_training_runs=0))
    print(s.read(s.ROOT / 'audit.json'))


if __name__ == '__main__':
    audit()
