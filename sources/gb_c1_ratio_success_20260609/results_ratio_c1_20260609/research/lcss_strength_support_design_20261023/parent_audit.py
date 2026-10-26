"""Independent read-only result checks; writes only to this design directory."""
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
EXP = WORK / 'experiments/lcss_strength_support_20261024'
HISTORY = WORK / 'experiments/sc_support_channel_pilot_20261023/diagnostics/e1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert read(EXP / 'complete.json')['unique_classification_fits'] == 1500
    assert not (EXP / 'run.lock').exists()
    data = rows(EXP / 'results.csv')
    counts = dict(Counter(r['stage'] for r in data))
    assert counts == dict(A=80, B1=336, B2=216, C1=576, C1_ANCHOR=40, D1=144, D2=108), counts
    assert len({r['folder'] for r in data}) == 1500
    assert all(r['seed'] == '0' and r['seed_std'] == '' for r in data)
    for path, digest in read(EXP / 'sources.json').items():
        assert sha(Path(path)) == digest, path
    for path, digest in read(EXP / 'selections_frozen.json')['selections'].items():
        assert sha(EXP / path) == digest, path

    # Independent metrics use sklearn, not the worker's confusion-matrix helper.
    project = WORK.parents[1]
    files = {
        'cora': project / 'external/NoisyGL/data/Cora/processed/data.pt',
        'citeseer': project / 'external/NoisyGL/data/CiteSeer/processed/data.pt',
        'corafull': project / 'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
        'coauthorcs': project / 'external/NoisyGL/data/CS/processed/data.pt',
        'amazonpho': project / 'external/NoisyGL/data/Photo/processed/data.pt',
    }
    graphs = {}
    for name, path in files.items():
        raw = torch.load(path, map_location='cpu', weights_only=False)[0]
        x, y = (raw['x'], raw['y']) if isinstance(raw, dict) else (raw.x, raw.y)
        graphs[name] = (x.float(), y.long(), load(HISTORY / name / 'seed0/geometry.pt'))

    samples = {}
    for r in data:
        if r['condition'] == 'uniform0.6':
            key = (r['stage'], r['dataset'], r['backbone'])
            if key not in samples or r['arm'] == 'GATE_LCSS':
                samples[key] = r
    errors = []
    reloads = []
    for r in samples.values():
        folder = EXP / r['folder']
        p = load(folder / 'probabilities.pt')
        ck = load(folder / 'checkpoint.pt')
        x, y, geo = graphs[r['dataset']]
        ids = geo['test']
        pred = p[ids].argmax(1).numpy()
        acc = accuracy_score(y[ids].numpy(), pred)
        f1 = f1_score(y[ids].numpy(), pred, labels=np.arange(p.shape[1]), average='macro', zero_division=0)
        assert abs(acc - float(r['test_accuracy'])) < 1e-12
        assert abs(f1 - float(r['test_macro_f1'])) < 1e-12
        valid = geo['valid']
        if r['stage'] == 'D2':
            valid = torch.tensor(read(EXP / 'meta_splits' / r['dataset'] / 'uniform0.6.json')['selection'])
        va = accuracy_score(y[valid].numpy(), p[valid].argmax(1).numpy())
        assert abs(va - float(r['validation'])) < 1e-12
        logs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert [l['epoch'] for l in logs] == list(range(1, 201))
        first = max(logs, key=lambda l: l['validation_correct'])
        assert first['epoch'] == int(r['best_epoch']) == ck['epoch']
        if r['backbone'] in ('R0', 'R2'):
            # Direct functional evaluation avoids importing the new fit adapter.
            state = ck['model']
            hidden = F.relu(F.linear(x, state['backbone.layers.0.weight'], state['backbone.layers.0.bias']))
            bn = 'backbone.batch_norms.0.'
            hidden = F.batch_norm(hidden, state[bn + 'running_mean'], state[bn + 'running_var'],
                                  state[bn + 'weight'], state[bn + 'bias'], training=False, eps=1e-5)
            predp = F.linear(hidden, state['backbone.layers.1.weight'], state['backbone.layers.1.bias']).softmax(1)
            diff = float((predp - p).abs().max())
            assert diff <= 1e-5, (r['folder'], diff)
            assert torch.equal(predp[ids].argmax(1), p[ids].argmax(1))
            reloads.append(dict(folder=r['folder'], cpu_max_probability_difference=diff))
        errors.append(dict(folder=r['folder'], accuracy_error=acc-float(r['test_accuracy']),
                           macro_f1_error=f1-float(r['test_macro_f1'])))

    # Recompute causal prefix scores from only the corresponding log prefix.
    prefix_checks = 0
    for path in (EXP / 'selections/C1').rglob('*_full.json'):
        name, backbone = path.parts[-3:-1]
        condition = path.stem[:-5]
        decision = read(path.with_name(condition + '.json'))
        assert decision['candidate_updates'] == 1000 and decision['ce_updates'] == 200
        assert decision['frozen'] <= read(path)['frozen']
        assert sha(path.with_name(condition + '.json')) == read(path)['probe_sha256']
        for step in decision['rounds']:
            prefix = step['prefix']
            assert 'S32_L1.0' in step['survivors']
            for score in step['scores']:
                folder = EXP / 'fits/C1' / name / backbone / condition / score['arm']
                logs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()][:prefix]
                first = max(logs, key=lambda r: r['validation_correct'])
                assert first['epoch'] == score['best_epoch']
                assert first['validation_correct'] == score['valid_correct']
                assert abs(first['validation_nll'] - score['valid_nll']) < 1e-12
                prefix_checks += 1

    c2 = read(EXP / 'selections/C2.json')
    assert len(c2['folds']) == 20
    features = read(EXP / 'c2_features.json')
    for fold in c2['folds']:
        assert fold['heldout'] not in fold['training_datasets']
        subset = [r for r in features if r['dataset'] != fold['heldout']]
        enhanced = fold['feature_set'].startswith('ENHANCED')
        matrix = np.asarray([r['stats'] + (r['enhanced'] if enhanced else []) for r in subset])
        assert np.allclose(matrix.mean(0), fold['feature_mean'], atol=1e-12, rtol=0)
        assert abs(np.mean([r['target'] for r in subset]) - fold['target_mean']) < 1e-12

    mass_checks = 0
    for name in ['cora', 'corafull', 'amazonpho']:
        geo = graphs[name][2]
        for condition in ['clean', 'uniform0.4', 'uniform0.6', 'pair0.4']:
            e = load(EXP / 'evidence' / name / condition / 'evidence.pt')['LEGACY']
            for backbone in ['R0', 'R2']:
                controls = load(EXP / 'evidence' / name / condition / ('d1_' + backbone + '.pt'))
                first = (e['m'] * controls['weights']).sum()
                second = (e['m'] * controls['shuffled']).sum()
                assert abs(float(first-second)) < 1e-8
                mass_checks += 1
            split = read(EXP / 'meta_splits' / name / (condition + '.json'))
            sets = [set(geo['train'].tolist()), set(split['meta']), set(split['selection']), set(geo['test'].tolist())]
            for i in range(4):
                for j in range(i):
                    assert not sets[i] & sets[j]
            assert sets[1] | sets[2] == set(geo['valid'].tolist())
            base = EXP / 'fits/D2' / name / 'R2' / condition
            actual = load(base / 'GATE_LCSS/weights_history.pt').double()
            for arm in ['MEAN_LCSS', 'SHUFFLE_LCSS']:
                control = load(base / arm / 'weights_history.pt').double()
                assert torch.allclose((actual * e['m']).sum(1), (control * e['m']).sum(1), atol=1e-8, rtol=1e-10)
                mass_checks += 200

    selected = rows(EXP / 'selected_results.csv')
    lookup = {(r['stage'], r['dataset'], r['backbone'], r['condition'], r['selector']): r for r in selected}
    diffs = rows(EXP / 'paired_differences.csv')
    for r in diffs:
        key = (r['stage'], r['dataset'], r['backbone'], r['condition'])
        a = lookup[key + (r['method'],)]
        reference_stage = 'C1' if r['stage'] == 'C2' else r['stage']
        b = lookup[(reference_stage,) + key[1:] + (r['reference'],)]
        for metric, column in [('test_accuracy', 'accuracy_pp'), ('test_macro_f1', 'macro_f1_pp'), ('validation', 'validation_pp')]:
            assert abs(100*(float(a[metric])-float(b[metric])) - float(r[column])) < 1e-10

    by_dataset = defaultdict(list)
    for r in diffs:
        if r['stage'] == 'D1' and r['reference'] == 'CELL_GLOBAL_VAL' and r['condition'] != 'clean':
            by_dataset[(r['backbone'], r['dataset'])].append(float(r['accuracy_pp']))
    result = dict(status='passed', time=datetime.now(timezone.utc).isoformat(), scientific_fits=counts,
                  metric_samples=errors, direct_cpu_checkpoint_reloads=reloads,
                  causal_prefix_scores_recomputed=prefix_checks, c2_scalers_recomputed=20,
                  exact_mass_checks=mass_checks, paired_rows_recomputed=len(diffs),
                  d1_dataset_deltas={b + '/' + n: float(np.mean(v)) for (b,n),v in by_dataset.items()},
                  run_lock_absent=True, own_source_sha256=sha(Path(__file__)))
    (HERE / 'PARENT_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k,v in result.items() if k not in ('metric_samples','direct_cpu_checkpoint_reloads')}, ensure_ascii=False, indent=2))
    print('metric_samples', len(errors), 'cpu_reload_samples', len(reloads),
          'max_cpu_probability_difference', max(r['cpu_max_probability_difference'] for r in reloads))


if __name__ == '__main__':
    main()
