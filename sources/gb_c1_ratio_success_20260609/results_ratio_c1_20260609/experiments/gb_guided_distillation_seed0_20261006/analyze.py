from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_guided as run

ROOT = run.ROOT


def save_csv(name, rows):
    fields = list(rows[0])
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def independent_gb(labels, assignment, train, classes):
    onehot = F.one_hot(labels, classes).float()
    target = onehot.clone()
    alpha = torch.zeros(len(labels))
    for ball in np.unique(assignment):
        members = torch.tensor(np.flatnonzero((assignment == ball) & train.numpy()))
        if len(members) < 3:
            continue
        hist = torch.bincount(labels[members], minlength=classes)
        votes = hist[None, :]-onehot[members]
        distribution = votes/(len(members)-1)
        peak, leading = distribution.max(1)
        concentration = ((classes*peak-1)/(classes-1)).clamp(0, 1)
        unique = (votes == votes.max(1, keepdim=True).values).sum(1) == 1
        consensus = distribution.clone()
        consensus[unique] = .5*distribution[unique]+.5*F.one_hot(leading[unique], classes).float()
        target[members] = ((1-concentration[:, None])*onehot[members]
                           + concentration[:, None]*consensus)
        alpha[members] = concentration
    return target, alpha


def analyze():
    torch.set_num_threads(4)
    run.verify_frozen()
    # Also verify our reusable trainer against the preceding experiment's manifest.
    with (run.PRIOR / 'sha256_manifest.csv').open(encoding='utf-8-sig') as handle:
        manifest = {row['path'].replace('\\', '/'): row['sha256'] for row in csv.DictReader(handle)}
    for name in ['run_screen.py', 'adapters.py', 'environment.json']:
        assert run.sha(run.PRIOR / name) == manifest[name]
        assert run.sha(ROOT / 'provenance' / name) == manifest[name]
    replay = run.read(ROOT / 'replay/cora/uniform0.4/D0/result.json')
    assert replay['replay_passed']
    paired, diagnostics, all_rows = [], [], []
    audit = dict(frozen_inputs_unchanged=True, prior_sources_unchanged=True,
                 replay=replay, complete_runs=0, target_checks=0,
                 metric_checks=0, checkpoint_selection_checks=0, independent_gb_checks=0)
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            old_settings = run.read(run.prior_folder(dataset, rate) / 'config.json')['student']
            for key in ['hidden', 'layers', 'dropout', 'lr', 'weight_decay', 'epochs', 'temperature']:
                assert old_settings[key] == run.SETTINGS[key]
            assert old_settings['initialization_seed'] == 0 and old_settings['rng_reset']
            assert old_settings['scope'] == 'all nodes'
            data, protocol = run.previous.load_data(dataset, rate)
            cache = torch.load(run.cache_folder(dataset, rate) / 'targets.pt', weights_only=False)
            teacher = torch.load(run.prior_folder(dataset, rate) / 'teacher_probabilities.pt', weights_only=True)
            train = data.train_mask
            test = data.test_mask
            eligible = cache['eligible']
            for name, assignment in [('gb', cache['assignment']), ('random_gb', cache['random_assignment'])]:
                expected_target, expected_alpha = independent_gb(data.noisy_y, assignment, train, data.num_classes)
                torch.testing.assert_close(cache[name], expected_target, rtol=1e-5, atol=1e-6)
                if name == 'gb':
                    torch.testing.assert_close(cache['alpha'], expected_alpha, rtol=1e-5, atol=1e-6)
                audit['independent_gb_checks'] += 1
            expected_weights = .5*cache['alpha']*(1-teacher.max(1).values)
            expected_weights[~eligible] = 0
            torch.testing.assert_close(cache['weights']['D3'], expected_weights, rtol=0, atol=0)
            true_train = data.y[train]
            onehot_train = F.one_hot(true_train, data.num_classes).float()
            teacher_class = teacher.argmax(1)
            gb_class = cache['gb'].argmax(1)
            disagree = train & (teacher_class != gb_class)
            teacher_test_count = int((teacher_class[test] == data.y[test]).sum())
            row = dict(dataset=dataset, noise_rate=rate, teacher=100*teacher_test_count/int(test.sum()))
            for arm in run.ARMS:
                folder = ROOT / 'runs' / dataset / f'uniform{rate}' / arm
                result = run.read(folder / 'result.json')
                assert result['status'] == 'ok'
                if arm == 'D0':
                    probs = torch.load(run.prior_folder(dataset, rate) / 'student_probabilities.pt', weights_only=True)
                else:
                    probs = torch.load(folder / 'student_probabilities.pt', weights_only=True)
                    epochs = [run.json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
                    assert len(epochs) == 200
                    assert all(np.isfinite(e['loss']) for e in epochs)
                    chosen = max(epochs, key=lambda e: e['valid'])
                    assert chosen['epoch'] == result['best_epoch']
                    assert chosen['valid'] == result['student_valid']
                    audit['checkpoint_selection_checks'] += 1
                assert probs.shape == teacher.shape
                assert torch.isfinite(probs).all() and (probs >= 0).all()
                assert (probs.sum(1)-1).abs().max() < 1e-5
                prediction = probs.argmax(1)
                masks = dict(train=data.train_mask, valid=data.val_mask, test=data.test_mask)
                correct_counts = {}
                for key, mask in masks.items():
                    labels = data.noisy_y if key == 'train' else data.y
                    correct_counts[key] = int((prediction[mask] == labels[mask]).sum())
                    total = int(mask.sum())
                    # GPU and CPU float32 means can differ by one ULP. Verify integer counts.
                    assert correct_counts[key] == round(result['student_' + key]*total)
                    assert abs(correct_counts[key]/total-result['student_' + key]) < 1e-7
                    audit['metric_checks'] += 1
                target, weight = cache['targets'][arm], cache['weights'][arm]
                assert torch.equal(target[~eligible], teacher[~eligible])
                assert torch.equal(target[weight == 0], teacher[weight == 0])
                assert torch.equal(target[~train], teacher[~train])
                assert (target >= 0).all() and torch.isfinite(target).all()
                assert (target.sum(1)-1).abs().max() < 1e-5
                audit['target_checks'] += 1
                audit['complete_runs'] += 1
                row[arm] = 100*correct_counts['test']/int(test.sum())
                all_rows.append(result)
                target_class = target.argmax(1)
                prediction = probs.argmax(1)
                change = (target-teacher).abs().sum(1)
                values = weight[eligible]
                diag = dict(dataset=dataset, noise_rate=rate, arm=arm,
                            train_nodes=int(train.sum()), eligible_nodes=int(eligible.sum()),
                            eligible_percent_train=100*float(eligible[train].float().mean()),
                            weighted_percent_train=100*float((weight[train] > 0).float().mean()),
                            changed_percent_train=100*float((change[train] > 0).float().mean()),
                            mean_weight_eligible=float(values.mean()), weight_sum=float(weight.sum()),
                            weight_q0=float(values.min()), weight_q50=float(values.median()),
                            weight_q90=float(torch.quantile(values, .9)), weight_q100=float(values.max()),
                            mean_l1_train=float(change[train].mean()),
                            target_brier_train=float(((target[train]-onehot_train)**2).sum(1).mean()),
                            teacher_brier_train=float(((teacher[train]-onehot_train)**2).sum(1).mean()),
                            gb_brier_train=float(((cache['gb'][train]-onehot_train)**2).sum(1).mean()),
                            target_argmax_change_percent_train=100*float((target_class[train] != teacher_class[train]).float().mean()),
                            teacher_wrong_gb_right=int((disagree & (teacher_class != data.y) & (gb_class == data.y)).sum()),
                            teacher_right_gb_wrong=int((disagree & (teacher_class == data.y) & (gb_class != data.y)).sum()),
                            target_corrected_train=int((train & (teacher_class != data.y) & (target_class == data.y)).sum()),
                            target_damaged_train=int((train & (teacher_class == data.y) & (target_class != data.y)).sum()),
                            student_corrected_test=int((test & (teacher_class != data.y) & (prediction == data.y)).sum()),
                            student_damaged_test=int((test & (teacher_class == data.y) & (prediction != data.y)).sum()),
                            teacher_to_student_pp=row[arm]-row['teacher'],
                            test_nodes=int(test.sum()))
                net = 100*(diag['student_corrected_test']-diag['student_damaged_test'])/diag['test_nodes']
                assert abs(net-diag['teacher_to_student_pp']) < 1e-4
                diagnostics.append(diag)
            assert torch.equal(cache['weights']['D1'], cache['weights']['D3'])
            assert torch.equal(cache['weights']['D4'], cache['weights']['D3'])
            torch.testing.assert_close(cache['weights']['D2'].sum(), cache['weights']['D3'].sum(), rtol=1e-5, atol=1e-6)
            for arm in ['D0', 'D1', 'D2', 'D4']:
                row['D3_minus_' + arm] = row['D3']-row[arm]
            row['D3_minus_teacher'] = row['D3']-row['teacher']
            paired.append(row)
    assert audit['complete_runs'] == 50
    means = {arm: float(np.mean([r[arm] for r in paired])) for arm in ['teacher'] + run.ARMS}
    comparison = {}
    for arm in ['D0', 'D1', 'D2', 'D4', 'teacher']:
        delta = np.array([r['D3_minus_'+arm] for r in paired])
        comparison[arm] = dict(mean_pp=float(delta.mean()), wins=int((delta > 0).sum()),
                               ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                               min_pp=float(delta.min()), max_pp=float(delta.max()))
    gates = dict(extra_gain=(comparison['D0']['wins'] >= 6 and comparison['D0']['mean_pp'] >= .5
                            and comparison['D0']['min_pp'] >= -1),
                 local_membership=(comparison['D4']['wins'] >= 7 and comparison['D4']['mean_pp'] >= .3),
                 better_than_labels=comparison['D1']['mean_pp'] > 0,
                 better_than_constant=comparison['D2']['mean_pp'] > 0,
                 retain_teacher=comparison['teacher']['min_pp'] >= -2)
    summary = dict(mean_accuracy_percent=means, candidate_comparisons=comparison, gates=gates,
                   advance_to_five_seeds=all(gates.values()), seed=0,
                   std='not estimable from one seed', inference='raw attributes only',
                   decision='continue only after user approval' if all(gates.values()) else 'stop this fixed candidate')
    run.write(ROOT / 'output_audit.json', audit)
    run.write(ROOT / 'summary.json', summary)
    save_csv('paired_results.csv', paired)
    save_csv('diagnostics.csv', diagnostics)
    lines = ['# Fixed ERASE teacher: GB-guided distillation screening', '',
             'Seed 0 only; uniform training-label noise 0.4 and 0.6. No tuning or teacher retraining.', '',
             'D0: ordinary distillation; D1: noisy-label anchor; D2: constant-weight GB target; '
             'D3: adaptive GB target; D4: shuffled size-matched GB target.', '',
             '| Dataset | Noise | Teacher | D0 | D1 | D2 | D3 | D4 | D3-D0 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in paired:
        lines.append('| '+row['dataset']+' | '+str(row['noise_rate'])+' | '+
                     ' | '.join(f'{row[k]:.3f}' for k in ['teacher']+run.ARMS+['D3_minus_D0'])+' |')
    lines.extend(['', '## Candidate paired comparisons', '', '| Reference | Mean delta (pp) | Wins/ties/losses | Worst delta |',
                  '|---|---:|---|---:|'])
    for arm, values in comparison.items():
        lines.append(f'| {arm} | {values["mean_pp"]:.4f} | {values["wins"]}/{values["ties"]}/{values["losses"]} | {values["min_pp"]:.4f} |')
    lines.extend(['', '## Predefined gates', ''] + [f'- {k}: {v}' for k, v in gates.items()])
    lines.extend(['', f'Decision: {summary["decision"]}.', '',
                  'All targets were fixed before the new student fits. Clean training labels enter only the post-freeze diagnostics.',
                  'D1 and D4 reuse the real-GB weights; they are mechanism controls, not complete no-ball methods.',
                  'D2 preserves total weight mass, not identical per-node probability displacement.',
                  'No standard deviations, significance tests, equivalence conclusions or SOTA claims follow from this one-seed screening.', '',
                  'Audit: 50 outputs, 150 recomputed metrics, 40 strict-validation checkpoint selections, '
                  'all frozen input hashes unchanged. See output_audit.json.', '',
                  'See diagnostics.csv for target coverage, weights, L1 movement, Brier scores and correction/damage counts.'])
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest_rows = []
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'sha256_manifest.csv':
            manifest_rows.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=run.sha(path)))
    save_csv('sha256_manifest.csv', manifest_rows)
    print(run.json.dumps(summary, indent=2))


if __name__ == '__main__':
    analyze()
