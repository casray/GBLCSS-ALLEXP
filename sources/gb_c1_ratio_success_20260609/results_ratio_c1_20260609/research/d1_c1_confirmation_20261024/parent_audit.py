"""Read-only independent review of the five-graph, three-seed confirmation."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.dont_write_bytecode = True
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
EXP = WORK / 'experiments/d1_c1_confirmation_20261024'
HISTORY = WORK / 'experiments/sc_support_channel_pilot_20261023/diagnostics/e1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(name):
    with (EXP / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def check_summaries(source, filename, identities, metrics, seeds):
    count = 0
    for summary in rows(filename):
        matched = [r for r in source if int(r['seed']) in seeds and all(
            (key == 'dataset' and summary[key] == 'ALL') or
            (key == 'condition' and summary[key] == 'NOISY_MACRO' and r[key] != 'clean') or
            r[key] == summary[key] for key in identities)]
        assert {int(r['seed']) for r in matched} == set(seeds)
        for metric in metrics:
            perseed = [np.mean([float(r[metric]) for r in matched if int(r['seed']) == s]) for s in seeds]
            assert abs(np.mean(perseed) - float(summary[metric + '_mean'])) < 1e-10
            assert abs(np.std(perseed, ddof=1) - float(summary[metric + '_std_ddof1'])) < 1e-10
        count += 1
    return count


def main():
    assert read(EXP / 'complete.json')['full_fits'] == 3600
    assert read(EXP / 'audit.json')['status'] == 'passed'
    assert not (EXP / 'run.lock').exists()
    result = rows('results.csv')
    assert Counter(r['stage'] for r in result) == {'D1': 1440, 'C1': 2160}
    assert len({r['folder'] for r in result}) == 3600
    for r in result:
        assert int(r['seed']) in (0, 1, 2)
    for path, digest in read(EXP / 'sources.json').items():
        assert sha(path) == digest, path
    freeze = read(EXP / 'selections_frozen.json')
    for path, digest in freeze['selections'].items():
        assert sha(EXP / path) == digest, path
    assert read(EXP / 'testing_started.json')['started'] >= freeze['frozen']

    selected = rows('selected_results.csv')
    lookup = {tuple(r[k] for k in ('stage', 'dataset', 'seed', 'backbone', 'condition', 'arm')): r for r in selected}
    assert len(lookup) == len(selected)
    paired = rows('paired_differences.csv')
    metrics = ['test_accuracy', 'validation', 'test_macro_f1']
    for r in paired:
        key = tuple(r[k] for k in ('stage', 'dataset', 'seed', 'backbone', 'condition'))
        a, b = lookup[key + (r['selected'],)], lookup[key + (r['reference'],)]
        for metric in metrics:
            assert abs(100*(float(a[metric])-float(b[metric])) - float(r[metric + '_diff_pp'])) < 1e-10
    summary_checks = 0
    for prefix, seeds in [('', (0, 1, 2)), ('seeds12_', (1, 2))]:
        summary_checks += check_summaries(selected, prefix + 'summary.csv',
            ['stage', 'dataset', 'backbone', 'condition', 'arm'], metrics, seeds)
        summary_checks += check_summaries(paired, prefix + 'paired_summary.csv',
            ['stage', 'dataset', 'backbone', 'condition', 'selected', 'reference'],
            [m + '_diff_pp' for m in metrics], seeds)

    project = WORK.parents[1]
    files = {
        'cora': project / 'external/NoisyGL/data/Cora/processed/data.pt',
        'citeseer': project / 'external/NoisyGL/data/CiteSeer/processed/data.pt',
        'corafull': project / 'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
        'coauthorcs': project / 'external/NoisyGL/data/CS/processed/data.pt',
        'amazonpho': project / 'external/NoisyGL/data/Photo/processed/data.pt',
    }
    checks = []
    for name, path in files.items():
        raw = torch.load(path, map_location='cpu', weights_only=False)[0]
        x, y = (raw['x'], raw['y']) if isinstance(raw, dict) else (raw.x, raw.y)
        x, y = x.float(), y.long()
        samples = [r for r in selected if r['dataset'] == name and r['condition'] == 'uniform0.6'
                   and r['arm'] in ('GROUP_SELECTED', 'PROBE')]
        assert len(samples) == 12
        for r in samples:
            folder = EXP / r['folder']
            p, ck = load(folder / 'probabilities.pt'), load(folder / 'checkpoint.pt')
            geo = load(HISTORY / name / ('seed' + r['seed']) / 'geometry.pt')
            te, vi = geo['test'], geo['valid']
            pred = p[te].argmax(1).numpy()
            acc = accuracy_score(y[te].numpy(), pred)
            f1 = f1_score(y[te].numpy(), pred, labels=np.arange(p.shape[1]), average='macro', zero_division=0)
            assert abs(acc - float(r['test_accuracy'])) < 1e-12
            assert abs(f1 - float(r['test_macro_f1'])) < 1e-12
            assert abs(accuracy_score(y[vi], p[vi].argmax(1)) - float(r['validation'])) < 1e-12
            if r['backbone'] == 'R2':
                assert sha(ck['encoder_path']) == ck['encoder_file_sha256']
                state = load(ck['encoder_path'])
                weight, bias = ck['model']['weight'], ck['model']['bias']
            else:
                state = ck['model']
                weight, bias = state['backbone.layers.1.weight'], state['backbone.layers.1.bias']
            hidden = F.relu(F.linear(x, state['backbone.layers.0.weight'], state['backbone.layers.0.bias']))
            bn = 'backbone.batch_norms.0.'
            hidden = F.batch_norm(hidden, state[bn+'running_mean'], state[bn+'running_var'],
                state[bn+'weight'], state[bn+'bias'], training=False, eps=1e-5)
            reloaded = F.linear(hidden, weight, bias).softmax(1)
            error = float((reloaded-p).abs().max())
            assert error < 1e-5, (r['folder'], error)
            assert torch.equal(reloaded[te].argmax(1), p[te].argmax(1))
            checks.append(dict(folder=r['folder'], max_cpu_probability_error=error))

    prefix_checks = 0
    for path in (EXP / 'selections/C1').rglob('*_full.json'):
        decision_path = path.with_name(path.name.replace('_full.json', '.json'))
        decision, full = read(decision_path), read(path)
        assert decision['frozen'] <= full['frozen']
        name, seed, backbone = path.parts[-4:-1]
        condition = path.stem[:-5]
        for step in decision['rounds']:
            assert 'S32_L1.0' in step['survivors']
            for score in step['scores']:
                folder = EXP / 'fits/C1' / name / seed / backbone / condition / score['arm']
                logs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()][:step['prefix']]
                first = max(logs, key=lambda row: row['validation_correct'])
                assert first['epoch'] == score['best_epoch']
                assert first['validation_correct'] == score['valid_correct']
                assert abs(first['validation_nll'] - score['valid_nll']) < 1e-12
                prefix_checks += 1
    assert prefix_checks == 120 * 24
    report = dict(status='passed', finished=datetime.now(timezone.utc).isoformat(), full_fits=3600,
        paired_rows_recomputed=len(paired), summary_rows_recomputed=summary_checks,
        independent_sklearn_metrics=len(checks), checkpoint_reloads=checks,
        causal_prefix_scores=prefix_checks, source_sha256=sha(Path(__file__)))
    (HERE / 'PARENT_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checkpoint_reloads'}, indent=2))


if __name__ == '__main__':
    main()
