from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

import run_recovery as run


def read_csv(name):
    with (run.ROOT/name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def audit():
    assert run.read(run.ROOT/'execution.json')['status'] == 'ok'
    registration = run.read(run.ROOT/'audit_registration.json')
    assert run.sha(__file__) == registration['source_sha256']
    run.verify()
    rows = read_csv('results.csv')
    assert len(rows) == 90
    by_cell = {}
    score_error = 0.
    checked_protocols = 0
    supplemental = []
    for row in rows:
        dataset, rate, seed, arm = row['dataset'], float(row['rate']), int(row['seed']), row['arm']
        original = run.load_protocol(dataset, rate, seed)
        train, val, test, clean, noisy = [original[key].numpy() for key in ('train', 'val', 'test', 'clean', 'noisy')]
        folder = run.base_folder(dataset, rate, seed)
        stored = torch.load(folder/f'{arm}.pt', weights_only=True)
        selected = stored['selected'].numpy()
        expected_labels = np.zeros(len(clean), dtype=np.int64)
        if arm == 'A_noisy':
            expected_selected = train
            expected_labels[train] = noisy[train]
        elif arm == 'B_all_clean':
            expected_selected = train
            expected_labels[train] = clean[train]
        elif arm == 'C_oracle_filter':
            expected_selected = train[clean[train] == noisy[train]]
            expected_labels[expected_selected] = clean[expected_selected]
        else:
            p = torch.load(run.pair_location(dataset, rate, seed, 'A_noisy')/'teacher/probabilities.pt', weights_only=True).numpy()
            score = np.float32(1)-p[train, noisy[train]]
            marked = train[np.argsort(-score, kind='stable')[:math.ceil(.2*len(train))]]
            assert np.array_equal(marked, stored['marked'].numpy())
            expected_selected = train if arm == 'E_correct' else train[~np.isin(train, marked)]
            expected_labels[expected_selected] = noisy[expected_selected]
            if arm == 'E_correct':
                expected_labels[marked] = p[marked].argmax(1)
                originally_wrong = int((clean[train] != noisy[train]).sum())
                marked_wrong = int((clean[marked] != noisy[marked]).sum())
                repaired = int(((clean[marked] != noisy[marked]) & (p[marked].argmax(1) == clean[marked])).sum())
                damaged = int(((clean[marked] == noisy[marked]) & (p[marked].argmax(1) != clean[marked])).sum())
                residual = int((expected_labels[train] != clean[train]).sum())
                assert residual == originally_wrong-repaired+damaged
                class_diagnostics = []
                for cls in range(p.shape[1]):
                    members = train[clean[train] == cls]
                    changed = marked[clean[marked] == cls]
                    class_diagnostics.append(dict(true_class=cls, original_nodes=len(members),
                        initially_clean=int((noisy[members] == clean[members]).sum()),
                        corrected_clean=int((expected_labels[members] == clean[members]).sum()),
                        removed_nodes=len(changed)))
                supplemental.append(dict(dataset=dataset, rate=rate, seed=seed,
                    original_wrong=originally_wrong, marked_wrong=marked_wrong, repaired=repaired,
                    damaged=damaged, E_wrong=residual, E_noise_rate=residual/len(train),
                    F_wrong=originally_wrong-marked_wrong,
                    F_noise_rate=(originally_wrong-marked_wrong)/(len(train)-len(marked)),
                    class_diagnostics=class_diagnostics))
        assert np.array_equal(selected, expected_selected)
        assert np.array_equal(stored['noisy_y'].numpy(), expected_labels)
        visible = np.full(len(clean), -1, dtype=np.int64)
        visible[selected] = expected_labels[selected]
        visible[val] = clean[val]
        assert np.array_equal(visible, stored['visible_y'].numpy())
        assert (visible[test] == -1).all()
        assert not np.intersect1d(selected, val).size and not np.intersect1d(selected, test).size
        mask = np.zeros(len(clean), dtype=bool)
        mask[selected] = True
        assert np.array_equal(mask, stored['train_mask'].numpy())
        checked_protocols += 1
        result = {}
        for role in ('teacher', 'student'):
            p = torch.load(Path(row[role+'_folder'])/'probabilities.pt', weights_only=True).numpy()[test]
            correct = p.argmax(1) == clean[test]
            accuracy = int(correct.sum())*100/len(test)
            balanced = statistics.mean(int(correct[clean[test] == cls].sum())*100/int((clean[test] == cls).sum())
                                       for cls in np.unique(clean[test]))
            for key, value in [('accuracy', accuracy), ('balanced_accuracy', balanced)]:
                difference = abs(value-float(row[role+'_'+key]))
                score_error = max(score_error, difference)
                assert difference < 1e-10
            result[role] = accuracy
        by_cell.setdefault((dataset, rate, seed), {})[arm] = result
    paired = read_csv('paired_deltas.csv')
    for row in paired:
        cell = by_cell[(row['dataset'], float(row['rate']), int(row['seed']))]
        for a, b, name in [('E_correct', 'F_delete', 'E_minus_F'), ('E_correct', 'A_noisy', 'E_minus_A'),
                            ('F_delete', 'A_noisy', 'F_minus_A'), ('B_all_clean', 'A_noisy', 'B_minus_A'),
                            ('C_oracle_filter', 'A_noisy', 'C_minus_A'), ('B_all_clean', 'C_oracle_filter', 'B_minus_C'),
                            ('B_all_clean', 'E_correct', 'B_minus_E')]:
            for role in ('teacher', 'student'):
                assert abs(cell[a][role]-cell[b][role]-float(row[name+'_'+role+'_pp'])) < 1e-10
    decision = run.read(run.ROOT/'decision.json')
    for column, mean in decision['mean_deltas'].items():
        assert abs(statistics.mean(float(row[column]) for row in paired)-mean) < 1e-10
    run.write(run.ROOT/'independent_audit.json', dict(passed=True, logical_rows=len(rows),
              independently_verified_interventions=checked_protocols, max_metric_difference=score_error,
              test_labels_hidden=True, E_F_truth_independent_recipe_verified=True,
              all_paired_and_macro_effects_verified=True, source_sha256=run.sha(__file__)))
    run.write(run.ROOT/'supplemental_label_diagnostics.json', supplemental)
    print('Independent audit passed:', len(rows), 'rows; maximum accuracy difference', score_error, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    audit()
