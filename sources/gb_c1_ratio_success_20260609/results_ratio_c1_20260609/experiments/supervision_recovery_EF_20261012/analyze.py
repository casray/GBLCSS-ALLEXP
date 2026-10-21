from __future__ import annotations

import csv
import json
import statistics as stats

import torch

import run_recovery as run

def export(name, rows):
    with (run.ROOT/name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(probabilities, labels):
    correct = probabilities.argmax(1) == labels
    recall = [correct[labels == c].double().mean().item() for c in torch.unique(labels)]
    onehot = torch.nn.functional.one_hot(labels, probabilities.shape[1]).float()
    nll = -probabilities[torch.arange(len(labels)), labels].clamp_min(1e-12).log().mean()
    return dict(accuracy=float(correct.double().mean()*100), balanced_accuracy=stats.mean(recall)*100,
                nll=float(nll), brier=float((probabilities-onehot).square().sum(1).mean()))


def reload_pair(data, folder):
    teacher = torch.load(folder/'teacher/probabilities.pt', weights_only=True)
    student = torch.load(folder/'student/probabilities.pt', weights_only=True)
    checkpoint = torch.load(folder/'teacher/teacher.pt', weights_only=True)
    conf = checkpoint['settings']
    Model, _ = run.previous.erase_classes()
    model = Model.GAT.get_model(data.x.shape[1], conf['hidden'], conf['embedding'], conf['layers'],
                    conf['heads'], conf['dropout'], 'cuda', False, False, False, False).cuda().eval()
    model.load_state_dict(checkpoint['state'])
    with torch.no_grad():
        observed = run.previous.probe_probabilities(model(data.x, data.edge_index),
                   tuple(item.cuda() for item in conf['probe_state']), data.num_classes).cpu()
    t_diff = float((observed-teacher).abs().max())
    assert t_diff <= 1e-6 and torch.equal(observed.argmax(1), teacher.argmax(1))
    student_model = run.previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda().eval()
    student_model.load_state_dict(torch.load(folder/'student/student.pt', weights_only=True))
    with torch.no_grad():
        observed_s = student_model(data.x).softmax(1).cpu()
    s_diff = float((observed_s-student).abs().max())
    assert s_diff <= 1e-6 and torch.equal(observed_s.argmax(1), student.argmax(1))
    for role, key in [('teacher', 'valid'), ('student', 'valid_correct')]:
        result = run.read(folder/role/'result.json')
        epochs = [json.loads(line) for line in (folder/role/'epochs.jsonl').read_text().splitlines()]
        best = max(epochs, key=lambda row: row[key])
        assert best['epoch'] == result['best_epoch']
        probabilities = teacher if role == 'teacher' else student
        correct = int((probabilities[data.val_index.cpu()].argmax(1) == data.y[data.val_index].cpu()).sum())
        assert correct == result['valid_correct']
    return teacher, student, dict(teacher_reload_difference=t_diff,
             student_reload_difference=s_diff, student_exact=torch.equal(student, observed_s),
             argmax_identical=True, first_maximum_validation_checked=True)


def diagnosis(protocol, probabilities, marked, corrected):
    train, clean, noisy = [protocol[key] for key in ('train', 'clean', 'noisy')]
    bad = clean[train] != noisy[train]
    marked_bad = clean[marked] != noisy[marked]
    pred = probabilities.argmax(1)
    fixed = marked_bad & (pred[marked] == clean[marked])
    harmed = ~marked_bad & (pred[marked] != clean[marked])
    still_bad = corrected[train] != clean[train]
    classes = probabilities.shape[1]
    clean_counts = torch.bincount(clean[train[~bad]], minlength=classes)
    return dict(train_nodes=len(train), originally_corrupted=int(bad.sum()), marked_nodes=len(marked),
          detection_precision=float(marked_bad.double().mean()),
          detection_recall=int(marked_bad.sum())/int(bad.sum()),
          all_corrupted_pseudo_accuracy=float((pred[train[bad]] == clean[train[bad]]).double().mean()),
          marked_corrupted_pseudo_accuracy=int(fixed.sum())/int(marked_bad.sum()),
          changed_labels=int((pred[marked] != noisy[marked]).sum()), fixed_labels=int(fixed.sum()),
          harmed_clean_labels=int(harmed.sum()), residual_noise_rate=float(still_bad.double().mean()),
          clean_labels_per_class_mean=float(clean_counts.double().mean()),
          clean_labels_per_class_min=int(clean_counts.min()),
          clean_labels_per_class_counts=json.dumps(clean_counts.tolist()))


COMPARISONS = [('E_correct', 'F_delete', 'E_minus_F'), ('E_correct', 'A_noisy', 'E_minus_A'),
               ('F_delete', 'A_noisy', 'F_minus_A'), ('B_all_clean', 'A_noisy', 'B_minus_A'),
               ('C_oracle_filter', 'A_noisy', 'C_minus_A'), ('B_all_clean', 'C_oracle_filter', 'B_minus_C'),
               ('B_all_clean', 'E_correct', 'B_minus_E')]


def main():
    assert run.read(run.ROOT/'execution.json')['status'] == 'ok'
    run.verify()
    rows, pairs, diagnostics, audits = [], [], [], []
    for dataset, rate, seed in run.cells():
        protocol = run.load_protocol(dataset, rate, seed)
        test = protocol['test']
        truth = protocol['clean'][test]
        data, _, _ = run.make_data(dataset, rate, seed, 'A_noisy')
        gpu = run.previous.gpu_data(data)
        initial = {role: [] for role in ('teacher', 'student')}
        cell = {}
        for arm in run.ARMS:
            folder = run.pair_location(dataset, rate, seed, arm)
            teacher, student, audit = reload_pair(gpu, folder)
            intervention, info, marked = run.make_data(dataset, rate, seed, arm)
            run.save_intervention(dataset, rate, seed, arm, intervention, info, marked)
            for role in initial:
                config = run.read(folder/role/'config.json')
                assert config['effective_input_hash'] == info['effective_input_hash']
                initial[role].append((config['initial_state_hash'], config['initial_rng_hash']))
            audit.update(dataset=dataset, rate=rate, seed=seed, arm=arm, folder=str(folder))
            audits.append(audit)
            row = dict(dataset=dataset, rate=rate, seed=seed, arm=arm, supervised=info['supervised'],
                  teacher_folder=str(folder/'teacher'), student_folder=str(folder/'student'),
                  **{'teacher_'+k: v for k, v in scores(teacher[test], truth).items()},
                  **{'student_'+k: v for k, v in scores(student[test], truth).items()})
            row['student_minus_teacher_pp'] = row['student_accuracy']-row['teacher_accuracy']
            cell[arm] = row
            rows.append(row)
            if arm == 'E_correct':
                base = torch.load(run.pair_location(dataset, rate, seed, 'A_noisy')/'teacher/probabilities.pt', weights_only=True)
                diagnostics.append(dict(dataset=dataset, rate=rate, seed=seed,
                      **diagnosis(protocol, base, marked, intervention.noisy_y)))
                marked_e = marked
            if arm == 'F_delete':
                assert torch.equal(marked_e, marked)
                assert set(intervention.train_index.tolist()) == set(protocol['train'].tolist())-set(marked.tolist())
        for role in initial:
            assert len(set(initial[role])) == 1, (dataset, rate, seed, role)
        pair = dict(dataset=dataset, rate=rate, seed=seed)
        for first, second, name in COMPARISONS:
            for role in ('teacher', 'student'):
                pair[name+'_'+role+'_pp'] = cell[first][role+'_accuracy']-cell[second][role+'_accuracy']
        pairs.append(pair)
        del gpu, data
        torch.cuda.empty_cache()
    export('results.csv', rows)
    export('paired_deltas.csv', pairs)
    export('label_diagnostics.csv', diagnostics)
    columns = [key for key in pairs[0] if key.endswith('_pp')]
    cell_summary = []
    for dataset in run.DATASETS:
        for rate in run.RATES:
            selected = [row for row in pairs if row['dataset'] == dataset and row['rate'] == rate]
            assert len(selected) == 3
            summary = dict(dataset=dataset, rate=rate, seeds=3)
            for key in columns:
                summary[key+'_mean'] = stats.mean(row[key] for row in selected)
                summary[key+'_std'] = stats.stdev(row[key] for row in selected)
            cell_summary.append(summary)
    export('cell_summary.csv', cell_summary)
    seed_summary = []
    for seed in run.SEEDS:
        selected = [row for row in pairs if row['seed'] == seed]
        seed_summary.append(dict(seed=seed, **{key: stats.mean(row[key] for row in selected) for key in columns}))
    export('seed_macro.csv', seed_summary)
    macro = {key: stats.mean(row[key] for row in seed_summary) for key in columns}
    macro_std = {key: stats.stdev(row[key] for row in seed_summary) for key in columns}
    arm_summary = []
    for arm in run.ARMS:
        for dataset in run.DATASETS:
            for rate in run.RATES:
                selected = [r for r in rows if r['arm'] == arm and r['dataset'] == dataset and r['rate'] == rate]
                entry = dict(dataset=dataset, rate=rate, arm=arm, seeds=3)
                for role in ('teacher', 'student'):
                    entry[role+'_mean'] = stats.mean(row[role+'_accuracy'] for row in selected)
                    entry[role+'_std'] = stats.stdev(row[role+'_accuracy'] for row in selected)
                arm_summary.append(entry)
    export('arm_summary.csv', arm_summary)
    def gate(key, threshold):
        return dict(mean_pp=macro[key], seed_macro_std_pp=macro_std[key],
              positive_cell_means=sum(row[key+'_mean'] > 0 for row in cell_summary),
              positive_seed_macros=sum(row[key] > 0 for row in seed_summary),
              passed=macro[key] >= threshold and sum(row[key+'_mean'] > 0 for row in cell_summary) >= 4
                     and all(row[key] > 0 for row in seed_summary))
    decision = dict(stage='three-seed paired confirmation, reused seed1 references',
          mean_deltas=macro, seed_macro_std=macro_std,
          oracle_recovery=gate('B_minus_C_student_pp', 1.),
          realistic_recovery=gate('E_minus_F_student_pp', .5),
          net_improvement=gate('E_minus_A_student_pp', .5),
          significance_established=False, external_confirmation=False, new_method_started=False)
    run.write(run.ROOT/'decision.json', decision)
    run.verify()
    run.write(run.ROOT/'audit.json', dict(passed=True, all_inputs_and_sources_unchanged=True,
         checkpoint_checks=audits, E_F_marked_sets_identical=True,
         paired_initialization_hashes_equal=True, validation_selection_verified=True))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
