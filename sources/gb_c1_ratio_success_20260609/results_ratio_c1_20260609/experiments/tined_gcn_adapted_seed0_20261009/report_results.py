from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'new_graph_mlp_seed0_20261009'))
from common import DATASETS, final_metrics, load_data, write_json
from gcn_adapter import GCNGraph, GCNTeacher, TINEDGCNStudent, _student_logits


def main():
    torch.set_num_threads(4)
    rows, audits = [], []
    for dataset in DATASETS:
        folder = ROOT / 'runs' / dataset / 'tined_gcn'
        result = json.loads((folder / 'result.json').read_text())
        config = json.loads((folder / 'config.json').read_text())
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        data, protocol = load_data(dataset, device='cpu')
        graph = GCNGraph(data.edge_index, len(data.x), data.x.dtype)
        teacher = GCNTeacher(checkpoint['dims'], config['teacher_dropout']).eval()
        teacher.load_state_dict(checkpoint['teacher_state'])
        student = TINEDGCNStudent(teacher, config).eval()
        student.load_state_dict(checkpoint['student_state'])
        probabilities = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        teacher_probabilities = torch.load(folder / 'teacher_probabilities.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            reloaded = _student_logits(student, data.x, config['batch_size']).softmax(1)
            reloaded_teacher = teacher(graph, data.x).softmax(1)
        torch.testing.assert_close(reloaded, probabilities, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(reloaded_teacher, teacher_probabilities, rtol=1e-4, atol=2e-6)
        metrics, teacher_metrics = final_metrics(dataset, probabilities), final_metrics(dataset, teacher_probabilities)
        for key in ('valid', 'test'):
            if metrics[key] != final_metrics(dataset, reloaded)[key] or abs(metrics[key]-result[key]) > 1e-10:
                raise ValueError(f'{dataset}: student reload metrics mismatch')
            if teacher_metrics[key] != final_metrics(dataset, reloaded_teacher)[key]:
                raise ValueError(f'{dataset}: teacher reload metrics mismatch')
        if abs(teacher_metrics['valid']-result['teacher_best_val']) > 1e-10:
            raise ValueError(f'{dataset}: teacher selected validation mismatch')
        if abs(teacher_metrics['test']-result['teacher_test']) > 1e-10:
            raise ValueError(f'{dataset}: teacher selected test mismatch')
        records = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        for phase in ('teacher', 'student'):
            stage = [r for r in records if r['phase'] == phase]
            if [r['epoch'] for r in stage] != list(range(1, config[phase+'_epochs']+1)):
                raise ValueError(f'{dataset}: incomplete {phase} logs')
            ordered = stage[::-1] if phase == 'teacher' else stage
            best = max(ordered, key=lambda r: r['val'])
            if best['epoch'] != checkpoint[phase+'_best']['epoch']:
                raise ValueError(f'{dataset}: {phase} tie rule mismatch')
        for key in ('split_hash', 'noise_hash'):
            if protocol[key] != result[key]:
                raise ValueError(f'{dataset}: main protocol fingerprint mismatch')
        for filename, digest in json.loads((folder / 'source_hashes.json').read_text()).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{dataset}: source changed {filename}')
        torch.testing.assert_close(student.DE_targets, torch.tensor(checkpoint['diagnostics']['teacher_log_de_targets']),
                                   rtol=0, atol=0)
        old = json.loads((ROOT.parent / 'new_graph_mlp_seed0_20261009/runs' / dataset / 'tined/result.json').read_text())
        final_student = [r for r in records if r['phase']=='student'][-1]
        rows.append(dict(dataset=dataset, seed=0, noise='uniform 0.4',
                         student_test_percent=100*result['test'], student_valid_percent=100*result['valid'],
                         teacher_test_percent=100*result['teacher_test'],
                         teacher_valid_percent=100*result['teacher_best_val'],
                         previous_student_test_percent=100*old['test'],
                         change_points=100*(result['test']-old['test']),
                         teacher_best_epoch=result['teacher_best_epoch'], student_best_epoch=result['best_epoch'],
                         final_student_valid_percent=100*final_student['val'],
                         student_dropout=config['dropout'], student_fc_layers=len(student.layers),
                         student_parameters=result['parameters'],
                         prediction_classes=int(probabilities.argmax(1).unique().numel()),
                         total_classes=data.num_classes, wall_seconds=result['wall_seconds'],
                         peak_allocated_gib=result['peak_allocated_gib']))
        audits.append(dict(dataset=dataset, status='passed', teacher_checkpoint_reload=True,
                           student_checkpoint_reload=True, teacher_last_validation_maximum=True,
                           student_first_validation_maximum=True, source_hashes=True,
                           main_protocol_hashes=True, student_energy_targets=True,
                           max_teacher_probability_error=float((reloaded_teacher-teacher_probabilities).abs().max()),
                           max_student_probability_error=float((reloaded-probabilities).abs().max())))
    with (ROOT / 'results.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / 'artifact_audit.json', audits)
    print(json.dumps(dict(results=rows, audit=audits), indent=2), flush=True)


if __name__ == '__main__':
    main()
