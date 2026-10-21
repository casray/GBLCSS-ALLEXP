from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from protocol import ROOT, WORKSPACE, DATASETS, final_metrics, load_data, write_json
from run_grid import location, sources
import sim_probe
import tined_formal


def audit_record(method, dataset, rate, seed):
    folder = location(method, dataset, rate, seed)
    result = json.loads((folder / 'result.json').read_text())
    if result['status'] != 'ok' or result['smoke'] or result['source_hashes'] != sources():
        raise RuntimeError('Formal status or source identity failure: ' + str(folder))
    data, protocol = load_data(dataset, seed, rate)
    if result['protocol'] != protocol:
        raise RuntimeError('Protocol identity failure: ' + str(folder))
    for name, expected in result['artifact_hashes'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Artifact identity failure: ' + str(folder / name))
    probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location=data.x.device, weights_only=False)
    logs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
    if method == 'simmlp':
        model, head, _ = sim_probe.build_probe(data, seed)
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['classifier'])
        model.eval()
        head.eval()
        with torch.no_grad():
            reloaded = head(model.encode(data.x)).softmax(1).cpu()
        if sim_probe.state_hash(model.encoder) != result['frozen_encoder_hash']:
            raise RuntimeError('Encoder hash failure')
        student_logs = [r for r in logs if r['stage'] == 'probe']
        val_key = 'valid'
        teacher = None
    else:
        config = result['config']
        if checkpoint['config'] != config or config['seed'] != seed:
            raise RuntimeError('Checkpoint config or seed mismatch')
        graph = tined_formal.GCNGraph(data.edge_index, len(data.x), data.x.dtype)
        teacher = tined_formal.GCNTeacher(checkpoint['dims'], config['teacher_dropout']).to(data.x.device)
        teacher.load_state_dict(checkpoint['teacher_state'])
        teacher.eval()
        student = tined_formal.TINEDGCNStudent(teacher, config)
        student.load_state_dict(checkpoint['student_state'])
        student.eval()
        with torch.no_grad():
            reloaded = tined_formal._student_logits(student, data.x, config['batch_size']).softmax(1).cpu()
            teacher_prediction = teacher(graph, data.x).softmax(1).cpu()
        torch.testing.assert_close(teacher_prediction,
            torch.load(folder / 'teacher_probabilities.pt', map_location='cpu', weights_only=True),
            rtol=2e-5, atol=2e-6)
        teacher_logs = [r for r in logs if r['phase'] == 'teacher']
        if len(teacher_logs) != 200:
            raise RuntimeError('Incomplete teacher budget')
        maximum = max(r['val'] for r in teacher_logs)
        selected = [r['epoch'] for r in teacher_logs if r['val'] == maximum][-1]
        if selected != result['teacher_best_epoch']:
            raise RuntimeError('Teacher last maximum selection failure')
        student_logs = [r for r in logs if r['phase'] == 'student']
        val_key = 'val'
    torch.testing.assert_close(reloaded, probabilities, rtol=2e-5, atol=2e-6)
    if len(student_logs) != 200 or [r['epoch'] for r in student_logs] != list(range(1, 201)):
        raise RuntimeError('Incomplete student budget')
    maximum = max(r[val_key] for r in student_logs)
    selected = next(r['epoch'] for r in student_logs if r[val_key] == maximum)
    if result['best_epoch'] != selected or abs(result['best_val'] - maximum) > 1e-12:
        raise RuntimeError('Student first maximum selection failure')
    metrics = final_metrics(dataset, seed, probabilities)
    if abs(metrics['valid'] - result['valid']) > 1e-12 or abs(metrics['test'] - result['test']) > 1e-12:
        raise RuntimeError('Final metric mismatch')
    row = dict(method=method, dataset=dataset, noise_type='uniform', noise_rate=rate, seed=seed,
               test_percent=100*result['test'], valid_percent=100*result['valid'],
               best_epoch=result['best_epoch'], parameters=result['parameters'],
               final_valid_percent=100*student_logs[-1][val_key],
               teacher_test_percent=100*result['teacher_test'] if teacher is not None else '',
               wall_seconds=result['wall_seconds'], reused_run=result.get('reused_run', ''))
    torch.cuda.empty_cache()
    return row, dict(method=method, dataset=dataset, rate=rate, seed=seed, status='passed',
                     checkpoint_reload_max_abs_difference=float((reloaded-probabilities).abs().max()),
                     student_first_maximum=True, teacher_last_maximum=teacher is not None,
                     historical_split=True, original_NoiseGL_noise=True, test_labels_training=False)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for fast checkpoint audit')
    torch.set_num_threads(4)
    rows, audits = [], []
    for method, rates in [('simmlp', [.4]), ('tined', [.4, .6])]:
        for rate in rates:
            for dataset in DATASETS:
                for seed in range(5):
                    row, audit = audit_record(method, dataset, rate, seed)
                    rows.append(row)
                    audits.append(audit)
                print('Audited:', method, rate, dataset, flush=True)
    if len(rows) != 75:
        raise RuntimeError('Expected exactly 75 formal records')
    summary = []
    for method, rates in [('simmlp', [.4]), ('tined', [.4, .6])]:
        for rate in rates:
            for dataset in DATASETS:
                matched = [r for r in rows if r['method'] == method and r['noise_rate'] == rate and r['dataset'] == dataset]
                if sorted(r['seed'] for r in matched) != list(range(5)):
                    raise RuntimeError('Missing or duplicate seed')
                scores = np.array([r['test_percent'] for r in matched])
                teacher_scores = [r['teacher_test_percent'] for r in matched if r['teacher_test_percent'] != '']
                summary.append(dict(method=method, dataset=dataset, noise_rate=rate, n=5,
                                    mean_percent=float(scores.mean()), sample_std_percent=float(scores.std(ddof=1)),
                                    teacher_mean_percent=float(np.mean(teacher_scores)) if teacher_scores else '',
                                    teacher_sample_std_percent=float(np.std(teacher_scores, ddof=1)) if teacher_scores else '',
                                    minimum_percent=float(scores.min()), maximum_percent=float(scores.max())))
    save_csv(ROOT / 'per_seed_results.csv', rows)
    save_csv(ROOT / 'summary.csv', summary)
    write_json(ROOT / 'artifact_audit.json', audits)
    table_rows = []
    for method, rate, label in [('simmlp', .4, 'SimMLP (matched)'), ('tined', .4, 'TINED (GCN, Uniform 0.4)'), ('tined', .6, 'TINED (GCN, Uniform 0.6)')]:
        cells = []
        for dataset in DATASETS:
            record = next(r for r in summary if r['method'] == method and r['noise_rate'] == rate and r['dataset'] == dataset)
            cells.append(f"${record['mean_percent']:.2f}\\pm{record['sample_std_percent']:.2f}$")
        table_rows.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
    (ROOT / 'table_rows.tex').write_text('\n'.join(table_rows)+'\n', encoding='utf-8')
    report = ['# Five-seed baseline results and audit', '',
              '75/75 formal records passed protocol, source/artifact identity, checkpoint reload,',
              '200-epoch budget and clean-validation first/last maximum checks.',
              'Mean +/- sample standard deviation (ddof=1), in percentage points.', '',
              '| Method | Noise | Dataset | Test accuracy |', '|---|---|---|---|']
    for r in summary:
        report.append(f"| {r['method']} | {r['noise_rate']} | {r['dataset']} | {r['mean_percent']:.2f} +/- {r['sample_std_percent']:.2f} |")
    report += ['', '## Scope and interpretation', '',
               'SimMLP uses the exact common two-FC deployed backbone and a fixed label-free',
               'seed0 encoder, with new noisy-label probe heads for seeds1..4. Existing seed0',
               'runs were source/config/fingerprint checked and reused. This estimates',
               'downstream variability conditional on that encoder, not pretraining variance.', '',
               'TINED uses a common GCN teacher and its native four-FC transformation-mimicking',
               'student. Adam/budgets/data/teacher match the common protocol; native student',
               'depth, dropout, minibatching and objective are disclosed exceptions. Native',
               'SAGE method-knob presets were fixed before this run; no test-guided tuning.',
               'Its sorted CE minibatches correct an earlier port omission, so all student',
               'seed0 runs were regenerated. Prior unsorted student scores must not be mixed.', '',
               'A low score or late degradation is retained. Best clean-validation checkpoint',
               'selection does not prove the optimization is stable; inspect final_valid_percent',
               'and epochs.jsonl for degradation. Five seeds do not establish global superiority.', '',
               'GSSC is withheld because the present port has unmatched depth and supervised',
               'budget. DGD has no verified runnable adaptation. No fabricated/missing rows.', '',
               'Final decision: retain these audited adaptation rows as candidate additions;',
               'update manuscript method/backbone descriptions before treating TINED as Table8',
               'data. No manuscript or existing table was changed in this experiment.']
    (ROOT / 'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('\n'.join(table_rows), flush=True)
    print('PASS: 75/75 formal checkpoint/protocol audits; 15 complete five-seed cells', flush=True)


if __name__ == '__main__':
    main()
