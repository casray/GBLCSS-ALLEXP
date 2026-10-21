"""Adversarial unit checks for the independent artifact auditor."""
import unittest
import csv
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit import independent_targets, compare_arrays, check_history, independent_metrics, validate_target_keys, gate_from_metrics, compare_records, verify_csv, independent_pairs, verify_paired_csvs


class AuditTests(unittest.TestCase):
    def test_leave_one_out_ties_and_insufficient_votes(self):
        values = independent_targets(np.array([0, 0, 0, 1]), np.array([0, 1, 2, 1]), 3)
        np.testing.assert_allclose(values['r'][0], [0, .5, .5])
        np.testing.assert_allclose(values['q'][0], [0, .5, .5])
        self.assertAlmostEqual(float(values['alpha'][0]), .25)
        np.testing.assert_allclose(values['target'][0], [.75, .125, .125])
        self.assertEqual(values['support'][3], 0)
        np.testing.assert_array_equal(values['target'][3], [0, 1, 0])

    def test_corrupt_target_fails(self):
        values = independent_targets(np.array([0, 0, 0]), np.array([1, 1, 0]), 2)
        corrupt = values['target'].copy()
        corrupt[0, 0] += .001
        with self.assertRaises(ValueError):
            compare_arrays(corrupt, values['target'], 'changed target')
        with self.assertRaises(ValueError):
            compare_arrays(np.array([np.nan]), np.array([np.nan]), 'NaN')

    def test_history_earliest_max_and_saved_validation(self):
        history = [dict(epoch=i+1, validation_score=v, loss=1.) for i, v in enumerate([.5, .8, .8])]
        check_history(history, 2, .8, 3)
        for epoch, validation in [(3, .8), (2, .7)]:
            with self.assertRaises(ValueError):
                check_history(history, epoch, validation, 3)

    def test_metrics_use_softmax_probabilities_and_float32_accuracy(self):
        logits = torch.tensor([[100., 101.], [-100., -98.], [4., 2.], [3., 2.]])
        labels = np.array([0, 1, 0, 1])
        result = independent_metrics(logits, labels, binary=True)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['roc_auc'], .75)
        self.assertAlmostEqual(result['average_precision'], 5/6)

    def test_full_matrix_requires_all_420_unique_keys(self):
        import common as c
        rows = [dict(dataset=n, seed=s, noise=z, arm=a) for n in c.CONFIG['datasets']
                for s in c.SEEDS for z in c.conditions(n) for a in c.ARMS]
        self.assertEqual(validate_target_keys(rows, c.CONFIG['datasets']), 420)
        for corrupt in [rows[:-1], rows+[rows[0]], rows[:-1]+[dict(dataset='unknown', seed=0, noise='clean', arm='CE')]]:
            with self.assertRaises(ValueError):
                validate_target_keys(corrupt, c.CONFIG['datasets'])

    def test_gate_recomputed_and_changed_decision_fails(self):
        rows = [dict(dataset='roman', seed=s, noise=z, arm=a, target_brier=value)
                for s in range(5) for z in ['uniform', 'pair']
                for a, value in [('R_TREE', .1), ('X_TREE', .2), ('SHUFFLE_TREE', .2)]]
        gate = gate_from_metrics(rows, ['roman'])
        self.assertTrue(gate['roman']['passed'])
        corrupt = {'roman': dict(gate['roman'], passed=False)}
        with self.assertRaises(ValueError):
            compare_records(corrupt, gate, 'changed gate')
        rows[0]['target_brier'] = 1.
        self.assertFalse(gate_from_metrics(rows, ['roman'])['roman']['passed'])

    def test_stale_csv_value_fails(self):
        row = dict(dataset='roman', seed=0, noise='clean', arm='CE', target_brier=.5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'metrics.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            verify_csv(path, [row], ['roman'])
            with self.assertRaises(ValueError):
                verify_csv(path, [dict(row, target_brier=.4)], ['roman'])

    def test_history_accuracy_allows_one_ulp_but_preserves_exact_selection(self):
        history = [dict(epoch=i+1, validation_score=v, loss=1.) for i, v in enumerate([.5, .8, .8])]
        check_history(history, 2, .8+5.960464477539063e-8, 3, primary_metric='accuracy')
        with self.assertRaises(ValueError):
            check_history(history, 2, .8+1.21e-7, 3, primary_metric='accuracy')
        with self.assertRaises(ValueError):
            check_history(history, 2, .8+2e-12, 3, primary_metric='roc_auc')
        near_tie = [dict(epoch=1, validation_score=.8, loss=1.),
                    dict(epoch=2, validation_score=.8+5.960464477539063e-8, loss=1.)]
        with self.assertRaises(ValueError):
            check_history(near_tie, 1, .8, 2, primary_metric='accuracy')

    def test_stale_paired_values_keys_and_summary_fail(self):
        rows = [dict(dataset='roman', seed=s, noise='uniform', arm=a, primary_metric='accuracy', test_primary=v)
                for s in [0, 1] for a, v in [('CE', .3), ('OLD64', .4), ('H_TREE', .45),
                                           ('X_TREE', .5), ('BFS', .6), ('R_TREE', .7), ('SHUFFLE_TREE', .55)]]
        pairs = independent_pairs(rows)
        self.assertEqual(len(pairs), 28)
        self.assertAlmostEqual(next(p['delta_pp'] for p in pairs if p['arm'] == 'R_TREE' and p['reference'] == 'X_TREE'), 20.)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            def write_pairs(values):
                pd.DataFrame(values).to_csv(folder/'paired_student_deltas.csv', index=False)
            summary = pd.DataFrame(pairs).groupby(['dataset', 'noise', 'arm', 'reference', 'primary_metric']).delta_pp.agg(['mean', 'std', 'count']).reset_index()
            def write_summary(values):
                values.to_csv(folder/'paired_student_summary.csv', index=False)
            write_pairs(pairs); write_summary(summary)
            verify_paired_csvs(folder, rows, ['roman'])
            write_pairs([dict(p, delta_pp=p['delta_pp']+1) if i == 0 else p for i, p in enumerate(pairs)])
            with self.assertRaises(ValueError):
                verify_paired_csvs(folder, rows, ['roman'])
            write_pairs(pairs[:-1]+[pairs[0]])
            with self.assertRaises(ValueError):
                verify_paired_csvs(folder, rows, ['roman'])
            write_pairs(pairs)
            for column in ['mean', 'std', 'count']:
                corrupt = summary.copy()
                corrupt.loc[0, column] += 1
                write_summary(corrupt)
                with self.assertRaises(ValueError):
                    verify_paired_csvs(folder, rows, ['roman'])


if __name__ == '__main__':
    unittest.main()
