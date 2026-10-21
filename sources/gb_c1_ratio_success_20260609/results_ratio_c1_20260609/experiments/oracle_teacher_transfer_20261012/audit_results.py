from __future__ import annotations

import csv

import numpy as np
import torch

import run_oracle as run


def main():
    assert run.read(run.ROOT / 'execution.json')['status'] == 'ok'
    run.verify()
    rows = list(csv.DictReader((run.ROOT / 'results.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 24
    checks = []
    history = []
    maximum_metric_difference = 0.
    for dataset, rate in run.cells():
        protocol = torch.load(run.source_folder(dataset, rate) / 'protocol.pt', weights_only=True)
        original_train = protocol['train']
        clean, noisy = protocol['clean'], protocol['noisy']
        test = protocol['test']
        mask_c = original_train[clean[original_train] == noisy[original_train]]
        class_count = torch.load(run.source_folder(dataset, rate) / 'teacher/probabilities.pt', weights_only=True).shape[1]
        teacher_init, student_init = [], []
        selected_sets = {}
        for arm in run.ARMS:
            intervention = torch.load(run.protocol_path(dataset, rate, arm), weights_only=True)
            selected = intervention['selected']
            selected_sets[arm] = selected
            assert set(selected.tolist()).issubset(set(original_train.tolist()))
            assert (intervention['visible_y'][test] == -1).all()
            assert (intervention['noisy_y'][~intervention['train_mask']] == 0).all()
            assert torch.equal(torch.where(intervention['train_mask'])[0], selected.sort().values)
            expected_labels = noisy[selected] if arm == 'A_noisy' else clean[selected]
            assert torch.equal(intervention['noisy_y'][selected], expected_labels)
            if arm == 'C_oracle_filter':
                assert torch.equal(selected, mask_c)
            folder = run.run_folder(dataset, rate, arm)
            tc = run.read(folder / 'teacher/config.json')
            sc = run.read(folder / 'student/config.json')
            teacher_init.append((tc['initial_state_hash'], tc['initial_rng_hash']))
            student_init.append((sc['initial_state_hash'], sc['initial_rng_hash']))
            teacher = torch.load(folder / 'teacher/probabilities.pt', weights_only=True)
            student = torch.load(folder / 'student/probabilities.pt', weights_only=True)
            assert sc['teacher_hash'] == run.state_hash({'probabilities': teacher})
            assert teacher.shape == student.shape == (len(clean), class_count)
            assert torch.isfinite(teacher).all() and torch.isfinite(student).all()
            row = next(r for r in rows if r['dataset'] == dataset and float(r['rate']) == rate and r['arm'] == arm)
            truth = clean[test]
            for name, probabilities in [('teacher', teacher[test]), ('student', student[test])]:
                independent = float((probabilities.argmax(1) == truth).double().mean() * 100)
                difference = abs(independent - float(row[name + '_accuracy']))
                maximum_metric_difference = max(maximum_metric_difference, difference)
                assert difference < 1e-12
            checks.append(dict(dataset=dataset, rate=rate, arm=arm, selected_labels_match=True,
                               test_labels_hidden=True, supervised=len(selected)))
        counts_c = torch.bincount(clean[selected_sets['C_oracle_filter']], minlength=class_count)
        counts_d = torch.bincount(clean[selected_sets['D_matched_clean']], minlength=class_count)
        assert torch.equal(counts_c, counts_d)
        assert len(set(teacher_init)) == len(set(student_init)) == 1
        for role, branch in [('teacher', 'teacher'), ('student', 'GB0')]:
            old = torch.load(run.source_folder(dataset, rate) / branch / 'probabilities.pt', weights_only=True)
            new = torch.load(run.run_folder(dataset, rate, 'A_noisy') / role / 'probabilities.pt', weights_only=True)
            old_accuracy = float((old[test].argmax(1) == clean[test]).double().mean() * 100)
            new_accuracy = float((new[test].argmax(1) == clean[test]).double().mean() * 100)
            history.append(dict(dataset=dataset, rate=rate, role=role, historical_accuracy=old_accuracy,
                                contemporary_accuracy=new_accuracy, delta_pp=new_accuracy - old_accuracy,
                                max_probability_difference=float((new-old).abs().max()),
                                argmax_changed=int((new.argmax(1) != old.argmax(1)).sum())))
    with (run.ROOT / 'historical_baseline_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for dataset in run.DATASETS:
        a = run.read(run.protocol_path(dataset, .4, 'B_all_clean').with_suffix('.json'))
        b = run.read(run.protocol_path(dataset, .6, 'B_all_clean').with_suffix('.json'))
        assert a['effective_input_hash'] == b['effective_input_hash']
    original = run.read(run.ROOT / 'engineering/frozen_inputs_before_retraining_check.json')['hashes']
    revised = run.read(run.ROOT / 'frozen_inputs.json')['hashes']
    changes = [path for path in original if original[path] != revised[path]]
    assert all(path.startswith(str(run.ROOT)) for path in changes)
    run.verify()
    run.write(run.ROOT / 'independent_audit.json', dict(passed=True, intervention_checks=checks,
          current_and_original_nonlocal_source_hashes_equal=True, changed_local_source_count=len(changes),
          paired_initializations_identical=True, test_accuracy_recalculation_max_difference=maximum_metric_difference,
          class_matched_subsets=True, B_noise_rate_input_equivalence=True,
          historical_baselines_are_secondary=True, audit_script_sha256=run.sha(__file__)))
    print(dict(passed=True,logical_pairs=24,intervention_checks=len(checks),
               metric_max_difference=maximum_metric_difference,original_inputs_unchanged=True),flush=True)


if __name__ == '__main__':
    main()
