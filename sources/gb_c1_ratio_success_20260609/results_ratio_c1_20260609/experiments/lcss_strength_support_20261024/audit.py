"""Recompute selection, hashes, target/mass and checkpoint probabilities."""
from collections import defaultdict
import json

import torch

from protocol import ROOT, EXP, CONFIG, read, load, sha, save, now, graph_inputs, cell_inputs, training_data, encoder_path
from fit import Student
from selection import probe_survivors, rank_key


def audit():
    errors, checks = [], defaultdict(int)
    for kind in ['sources', 'inputs']:
        for path, digest in read(ROOT/f'{kind}.json').items():
            if sha(path) != digest:
                errors.append('hash changed '+path)
            checks[kind+'_hashes'] += 1
    frozen = read(ROOT/'selections_frozen.json')
    for p, h in frozen['selections'].items():
        assert sha(ROOT/p) == h
        checks['selection_hashes'] += 1
    registry = read(ROOT/'fit_registry.json')
    folders = sorted({r['folder'] for r in registry})
    initial_groups = defaultdict(set)
    for folder in folders:
        f = ROOT/folder
        r = read(f/'result.json')
        assert r['epoch'] == 200
        assert sha(f/'probabilities.pt') == r['probabilities_sha256']
        assert sha(f/'checkpoint.pt') == r['checkpoint_sha256']
        logs = [json.loads(line) for line in (f/'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in logs] == list(range(1, 201))
        best = max(l['validation_correct'] for l in logs)
        first = next(l['epoch'] for l in logs if l['validation_correct'] == best)
        assert first == r['best_epoch'] and best == r['valid_correct']
        p = load(f/'probabilities.pt')
        assert torch.isfinite(p).all() and torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-6)
        initial_groups[(r['dataset'], r['backbone'])].add(r['initial_hash'])
        checks['complete_logs_probabilities_checkpoints'] += 1
    assert all(len(v) == 1 for v in initial_groups.values())
    checks['shared_initialization_groups'] = len(initial_groups)
    for path in (ROOT/'selections/C1').rglob('*_full.json'):
        cond = path.stem[:-5]
        probe = read(path.with_name(cond+'.json'))
        for r, count in zip(probe['rounds'], [6, 2]):
            assert probe_survivors(r['scores'], count, r['prefix']) == r['survivors']
        assert min(probe['final_scores'], key=rank_key)['arm'] == probe['PROBE']
        assert datetime_value(probe['frozen']) <= datetime_value(read(path)['frozen'])
        assert probe['candidate_updates'] == 1000 and probe['true_prefix_execution']
        checks['causal_probe_cells'] += 1
    c2 = read(ROOT/'selections/C2.json')
    assert c2['fits'] == len(c2['folds']) == 20
    for f in c2['folds']:
        assert f['heldout'] not in f['training_datasets'] and len(f['training_datasets']) == 4
        checks['heldout_graph_predictors'] += 1
    for name in CONFIG['datasets_mechanism']:
        for cond in CONFIG['conditions']:
            es = load(ROOT/'evidence'/name/cond/'evidence.pt')
            e = es['LEGACY']
            for b in ['R0', 'R2']:
                c = read(ROOT/'selections/D1'/name/b/f'{cond}.json')
                assert abs(c['total_mass']-c['shuffled_mass']) < 1e-8
                checks['d1_exact_mass'] += 1
            weights = load(ROOT/f'fits/D2/{name}/R2/{cond}/GATE_LCSS/weights_history.pt').double()
            m = e['m']
            for arm in ['MEAN_LCSS', 'SHUFFLE_LCSS']:
                ws = load(ROOT/f'fits/D2/{name}/R2/{cond}/{arm}/weights_history.pt').double()
                assert torch.allclose((weights*m).sum(1), (ws*m).sum(1), atol=1e-8, rtol=1e-10)
                checks['d2_trajectory_exact_mass'] += 200
            for arm in ['GATE_LCSS', 'GATE_MASS', 'GATE_KM']:
                r = read(ROOT/f'fits/D2/{name}/R2/{cond}/{arm}/result.json')
                assert r['mean_meta_gradient_norm'] > 0
                checks['meta_gradients'] += 1
            split = read(ROOT/'meta_splits'/name/f'{cond}.json')
            geo = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt')
            parts = [set(geo['train'].tolist()), set(split['meta']), set(split['selection']), set(geo['test'].tolist())]
            for i in range(4):
                for j in range(i):
                    assert not parts[i] & parts[j]
            assert parts[1] | parts[2] == set(geo['valid'].tolist())
            checks['meta_split_disjoint'] += 1
    # Deterministic reloading sample: one arm per stage/backbone/dataset, all nodes.
    samples = {}
    for r in registry:
        if r['stage'].endswith('_REF'):
            continue
        samples.setdefault((r['stage'], r['dataset'], r['backbone']), r)
    maximum = 0.
    from fit import sim_embedding
    for (stage, name, b), r in samples.items():
        graph = graph_inputs(name)
        pack = cell_inputs(name, r['condition'])
        geo = graph['geometry']
        values = training_data(dict(x=graph['x'].cuda(), train=geo['train'].cuda(), valid=geo['valid'].cuda(),
                   observed=pack['observed'].cuda(), valid_y=pack['valid_y'].cuda(), classes=pack['targets'][32].shape[1]))
        enc = load(encoder_path(name))
        student = Student(b, values, enc, sim_embedding(name, values['x']) if b == 'SIM' else None)
        ck = load(ROOT/r['folder']/'checkpoint.pt')
        student.model.load_state_dict(ck['model'])
        student.model.eval()
        if b == 'R2':
            with torch.no_grad():
                student.hidden = student.model.encode(values['x']).detach()
        with torch.no_grad():
            diff = float((student.forward().softmax(1).cpu()-load(ROOT/r['folder']/'probabilities.pt')).abs().max())
        maximum = max(maximum, diff)
        assert diff <= 3e-7
        checks['independent_checkpoint_reload_samples'] += 1
        del graph, student, values
        torch.cuda.empty_cache()
    assert len(folders) <= 1500
    save(ROOT/'audit.json', dict(status='passed' if not errors else 'failed', finished=now(), checks=dict(checks),
            errors=errors, unique_fits=len(folders), sampled_reload_max_probability_error=maximum,
            seed_std=None, external_parent_audit='separate_readonly_review'))
    if errors:
        raise RuntimeError('Audit errors: '+str(errors))


def datetime_value(s):
    from datetime import datetime
    return datetime.fromisoformat(s)
