from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import torch

import run_diagnostic as d


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def audit():
    d.verify()
    root = d.ROOT
    with (root / 'per_cell_metrics.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 200
    assert len({(r['dataset'], r['rate'], r['seed'], r['arm']) for r in rows}) == 200
    folds = d.read(root / 'folds.json')
    assert len(folds) == 5
    for fold in folds:
        assert fold['heldout'] not in fold['training_datasets']
        assert len(fold['training_datasets']) == 4
        assert fold['training_cells'] == 16 and fold['evaluation_cells'] == 4
    cell_audits = []
    feature_count = 0
    for dataset in d.DATASETS:
        for rate in d.RATES:
            for seed in d.SEEDS:
                source = d.cell_folder(dataset, rate, seed)
                proto = torch.load(source / 'protocol.pt', weights_only=True)
                meta = d.read(source / 'protocol.json')
                assert fingerprint(*(proto[k].numpy() for k in ('train', 'val', 'test'))) == meta['split_hash']
                assert fingerprint(proto['clean'].numpy(), proto['noisy'].numpy()) == meta['noise_hash']
                assert d.assignment_hash(proto['assignment'].numpy()) == meta['assignment_hash']
                train, val, test = (proto[k].numpy() for k in ('train', 'val', 'test'))
                assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
                outside = np.ones(len(proto['clean']), dtype=bool)
                outside[train] = False
                np.testing.assert_array_equal(proto['clean'].numpy()[outside], proto['noisy'].numpy()[outside])
                with np.load(root / 'features' / dataset / f'seed{seed}' / f'uniform{rate}.npz') as features:
                    np.testing.assert_array_equal(features['train'], train)
                    assert set(features.files) == {'train', 'base', 'knn', 'GB', 'KM', 'BKM', 'Random'}
                    for key in features.files:
                        assert np.isfinite(features[key]).all()
                    feature_count += len(train)
                cell_audits.append(dict(dataset=dataset, rate=rate, seed=seed,
                                       split_noise_assignment_fingerprints=True,
                                       nontraining_labels_unchanged=True, diagnostic_rows=len(train)))
    maximum_difference = 0.
    for row in rows:
        dataset, rate, seed, arm = row['dataset'], float(row['rate']), int(row['seed']), row['arm']
        source = d.cell_folder(dataset, rate, seed)
        proto = torch.load(source / 'protocol.pt', weights_only=True)
        target = (proto['clean'][proto['train']] != proto['noisy'][proto['train']]).numpy().astype(int)
        with np.load(root / 'evaluation' / dataset / arm.replace('+', '_') / f'u{rate}_seed{seed}.npz') as result:
            np.testing.assert_array_equal(result['train'], proto['train'].numpy())
            np.testing.assert_array_equal(result['target'], target)
            scores = result['score']
            independent = dict(auc=roc_auc_score(target, scores), ap=average_precision_score(target, scores),
                               logloss=log_loss(target, scores, labels=[0, 1]), brier=brier_score_loss(target, scores))
            order = np.argsort(-scores, kind='stable')
            count = int(np.ceil(.2 * len(target)))
            selected = scores[order[count-1]]
            certain, ties = scores > selected, scores == selected
            true_selected = target[certain].sum() + (count-certain.sum()) * target[ties].mean()
            independent.update(precision20=true_selected/count,
                               recall20=true_selected/target.sum(),
                               clean_flag_rate20=(count-true_selected)/(target == 0).sum())
            for name, value in independent.items():
                difference = abs(value - float(row[name]))
                maximum_difference = max(maximum_difference, difference)
                assert difference < 1e-12, (row, name, value)
    expected = d.read(root / 'output_manifest.json')
    assert all(d.sha(root / path) == value for path, value in expected.items())
    decision = d.read(root / 'decision.json')
    macro = {(r['first'], r['second']): r for r in csv.DictReader(
        (root / 'macro_deltas.csv').open(encoding='utf-8-sig', newline=''))}
    primary = macro['Base+kNN+GB', 'Base+kNN']
    assert decision['G1'] == (float(primary['auc']) >= .02 and int(primary['positive_datasets']) >= 3)
    assert decision['G2'] == all(float(macro['Base+kNN+GB', other]['auc']) >= .02
                                and int(macro['Base+kNN+GB', other]['positive_datasets']) >= 3
                                for other in ('Base+kNN+KM', 'Base+kNN+BKM'))
    assert decision['G3'] == (float(primary['precision20']) >= .02 and float(primary['clean_flag_rate20']) <= 0)
    result = dict(status='passed', original_input_hash_check=True, output_hash_check=True,
                  protocol_cells=20, metric_rows=200, outer_dataset_folds=5,
                  diagnostic_response_records=feature_count, independent_metric_max_difference=maximum_difference,
                  fold_overlap=False, protocol_checks=cell_audits, gates_recomputed=True,
                  audit_source_sha256=d.sha(Path(__file__)))
    d.write(root / 'audit.json', result)
    print({k: v for k, v in result.items() if k != 'protocol_checks'})


if __name__ == '__main__':
    audit()
