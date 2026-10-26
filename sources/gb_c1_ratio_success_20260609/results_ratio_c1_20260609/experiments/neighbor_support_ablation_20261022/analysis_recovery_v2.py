"""Fixed-grid audits and paired summaries; test scoring is full-grid only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Import the runner first: it establishes the historical generator/kernel paths.
from run import (ROOT, BASE, FUSION, CONFIG, read, write, sha, csv_write,
                 initialize, verify, input_dir, encoder_path, reference_run,
                 reference_targets, load_dataset, raw_data, percent_split_indices,
                 MLP, state_hash, g, support_costs, cost_components,
                 count_matched_cut)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def logs(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def manifest_audit():
    verify()
    manifest = read(ROOT/'frozen_inputs.json')
    require(manifest['config'] == CONFIG, 'Frozen config differs')
    files = {Path(p).resolve(): digest for p, digest in manifest['sha256'].items()}
    for path in (ROOT/'config.json', ROOT/'PLAN.md', ROOT/'run.py', ROOT/'analyze.py'):
        require(path.resolve() in files, 'Missing frozen source: '+str(path))
    recovery = dict(original_manifest_sha256=sha(ROOT/'frozen_inputs.json'),
                    original_analysis_sha256=sha(ROOT/'analyze.py'),
                    recovery_analysis_sha256=sha(__file__),
                    reason='float32 softmax normalization roundoff; includes historical CE',
                    change='float64 accumulation, tolerance eight float32 eps; no renormalization',
                    predictions_modified=False, fits_rerun=0, scientific_gates_changed=False)
    recovery_path=ROOT/'audit_recovery_manifest.json'
    if recovery_path.exists():
        require(read(recovery_path)==recovery, 'Recovery source/manifest changed')
    else:
        write(recovery_path,recovery)
    return dict(frozen_manifest_sha256=sha(ROOT/'frozen_inputs.json'),
                frozen_files=len(files), config_sha256=sha(ROOT/'config.json'),
                analysis_sha256=sha(__file__), recovery_manifest_sha256=sha(recovery_path))


def source_chain(folder):
    chain, seen = [], set()
    while True:
        folder = Path(folder).resolve()
        require(folder not in seen, 'Cyclic source alias: '+str(folder))
        seen.add(folder)
        record = read(folder/'result.json')
        require(record['status'] == 'ok', 'Unfinished source: '+str(folder))
        chain.append((folder, record))
        if not record.get('reused', False):
            return chain
        folder = Path(record['source_folder'])


def check_log(record, history, epochs, valid_total):
    require(record['status'] == 'ok' and record['epochs'] == epochs,
            'Status/duration mismatch')
    require(len(history) == epochs and [r['epoch'] for r in history] == list(range(1, epochs+1)),
            'Incomplete or unordered epoch log')
    require(all(np.isfinite(r['loss']) and 0 <= r['validation_correct'] <= valid_total
                for r in history), 'Invalid epoch log')
    best = max(r['validation_correct'] for r in history)
    first = next(r['epoch'] for r in history if r['validation_correct'] == best)
    require(record['best_epoch'] == first and record['valid_correct'] == best,
            'Selection is not the FIRST maximum validation count')
    require(record['valid_total'] == valid_total and
            abs(record['validation']-best/valid_total) <= 1e-12, 'Validation metadata mismatch')
    return best, first


def smoke_audit():
    audit = manifest_audit()
    checks = []
    for seed in CONFIG['seeds']:
        geo = load(input_dir('cora', seed)/'geometry.pt')
        for route in CONFIG['routes']:
            for arm in ('CE', 'OLD64', 'SUPPORT', 'COUNT'):
                folder = ROOT/'smoke'/'cora'/f'seed{seed}'/'uniform0.4'/(route+'_'+arm)
                result, history = read(folder/'result.json'), logs(folder/'epochs.jsonl')
                check_log(result, history, 3, len(geo['valid']))
                require(result['seed'] == seed and result['route'] == route and
                        result['arm'] == arm and result['smoke'], 'Smoke job identity mismatch')
                anchor = read(reference_run('cora', seed, 'uniform0.4', route, 'CE')/'result.json')
                for key in ('initial_hash', 'initial_rng_hash'):
                    require(result[key] == anchor[key], 'Smoke initialization mismatch: '+key)
                delta = None
                if arm in ('CE', 'OLD64'):
                    chain = source_chain(reference_run('cora', seed, 'uniform0.4', route, arm))
                    expected = logs(chain[-1][0]/'epochs.jsonl')[:3]
                    require(len(expected) == 3, 'Historical smoke reference is incomplete')
                    require([r['validation_correct'] for r in history] ==
                            [r['validation_correct'] for r in expected], 'Historical smoke validation mismatch')
                    delta = max(abs(a['loss']-b['loss']) for a, b in zip(history, expected))
                    require(delta <= 1e-7, 'Historical smoke loss mismatch')
                checks.append(dict(seed=seed, route=route, arm=arm, loss_difference=delta,
                                   epochs=3, validation_verified=True))
    require(len(checks) == 24, 'Expected 24 smoke jobs')
    verify()
    write(ROOT/'smoke_audit.json', dict(passed=True, fits=24, historical_bridges=12,
                                      checks=checks, **audit))
    print(json.dumps(dict(smoke_passed=True, fits=24, historical_bridges=12, **audit)), flush=True)


def audit_geometry(name, seed, data):
    folder = input_dir(name, seed)
    geo, tree = load(folder/'geometry.pt'), load(folder/'tree.pt')
    historical = load(FUSION/'inputs'/name/f'seed{seed}'/'assets.pt')
    size = len(data['x'])
    for key, expected in zip(('train', 'valid', 'test'), percent_split_indices(size, seed)):
        np.testing.assert_array_equal(geo[key].numpy(), expected)
        require(torch.equal(geo[key], historical[key]), 'Historical split mismatch')
    train, near = geo['train'].numpy(), geo['near'].numpy()
    k = min(CONFIG['reference_k'], len(train)-1)
    require(geo['k'] == k and near.shape == (len(train), len(train)-1), 'Neighbor shape/k mismatch')
    for row, donors in enumerate(near):
        require(np.array_equal(np.sort(donors), np.delete(np.arange(len(train)), row)),
                'Self, missing or duplicate neighbors')
    nodes = [{**node, 'members': node['members'].numpy()} for node in tree['nodes']]
    for node in nodes:
        require(node['train_count'] == int(np.isin(train, node['members']).sum()), 'Tree train count mismatch')
        require(len(node['members']) >= CONFIG['min_nodes'] and
                node['train_count'] >= CONFIG['min_train'], 'Tree minimum constraint failed')
    full = g.node_costs(nodes, train, near[:, :k])
    support, identity = cost_components(nodes, train, near[:, :k])
    np.testing.assert_allclose(full, support+identity, rtol=0, atol=1e-10)
    np.testing.assert_allclose(support, support_costs(nodes, k), rtol=0, atol=1e-10)
    for key, actual in [('full_cost', full), ('size_cost', support), ('identity_cost', identity)]:
        np.testing.assert_allclose(tree[key].numpy(), actual, rtol=0, atol=1e-10)
        require(np.isfinite(actual).all() and (actual >= -1e-10).all(), 'Invalid cost component')
    cuts = dict(NP=g.select_cut(nodes, full)[0], SUPPORT=g.select_cut(nodes, support)[0])
    cuts['COUNT'] = count_matched_cut(nodes, len(cuts['NP']))
    for arm, cut in cuts.items():
        require(cut == tree[arm.lower()+'_leaves'], 'Cut mismatch: '+arm)
        np.testing.assert_array_equal(g.assignment(nodes, cut, size), geo['partitions'][arm].numpy())
    require(torch.equal(geo['partitions']['OLD64'], historical['assignment']), 'Seed OLD64 partition mismatch')
    for arm, part in geo['partitions'].items():
        a = part.numpy()
        require(a.shape == (size,) and (a >= 0).all(), 'Invalid partition: '+arm)
        require(np.array_equal(np.unique(a), np.arange(len(np.unique(a)))), 'Noncanonical region IDs')
    for arm, count in [('COUNT', len(cuts['NP'])), ('KMNP', len(cuts['NP'])),
                       ('KMOLD', len(torch.unique(historical['assignment'])))]:
        require(len(torch.unique(geo['partitions'][arm])) == count, 'Region count mismatch: '+arm)
    a = geo['partitions']['NP'].numpy()[train]
    expected_support = np.bincount(a)[a]-1
    np.testing.assert_array_equal(geo['support'].numpy(), expected_support)
    if seed == 0:
        previous = load(BASE/'inputs'/name/'geometry.pt')
        np.testing.assert_array_equal(near, previous['near'].numpy())
        for arm, old in [('NP', 'NP'), ('KMNP', 'KM')]:
            require(torch.equal(geo['partitions'][arm], previous['partitions'][old]), 'Seed0 bridge mismatch')
    return geo, dict(dataset=name, seed=seed, nodes=len(nodes), np_leaves=len(cuts['NP']),
                     support_leaves=len(cuts['SUPPORT']), count_leaves=len(cuts['COUNT']),
                     splits_verified=True, costs_verified=True, cuts_verified=True)


def reconstruct_counts(z, arm, geo, classes):
    counts = np.zeros((len(z), classes), dtype=np.float64)
    if arm == 'CE':
        return counts
    if arm == 'KNNMATCH':
        for row, size in enumerate(geo['support'].numpy()):
            donors = geo['near'][row, :int(size)].numpy()
            counts[row] = np.bincount(z[donors], minlength=classes)
    else:
        assignment = geo['partitions'][arm][geo['train']].numpy()
        for ball in np.unique(assignment):
            rows = np.flatnonzero(assignment == ball)
            counts[rows] = np.bincount(z[rows], minlength=classes)
        counts[np.arange(len(z)), z] -= 1
    require((counts >= 0).all(), 'Negative LOO count')
    return counts


def audit_targets(name, seed, condition, cache, geo, classes, y):
    original = load(FUSION/'inputs'/name/f'seed{seed}'/(condition+'.pt'))
    require(torch.equal(cache['observed'], original['observed']) and
            torch.equal(cache['valid_y'], original['valid_y']), 'Historical observed/validation labels differ')
    train = geo['train'].numpy()
    z = cache['observed'].numpy()
    require(z.shape == train.shape and (z >= 0).all() and (z < classes).all(), 'Invalid observed labels')
    np.testing.assert_array_equal(cache['valid_y'].numpy(), y[geo['valid'].numpy()])
    if condition == 'clean':
        np.testing.assert_array_equal(z, y[train])
    clean = z == y[train]
    truth, observed = np.eye(classes)[y[train]], np.eye(classes)[z]
    baseline = ((observed-truth)**2).sum(1)
    rows, checks = [], []
    for arm in CONFIG['arms']:
        counts = reconstruct_counts(z, arm, geo, classes)
        support = counts.sum(1)
        target = cache['targets'][arm].numpy().astype(float)
        require(target.shape == counts.shape and np.isfinite(target).all() and (target >= 0).all(),
                'Invalid target: '+arm)
        np.testing.assert_allclose(target.sum(1), 1, rtol=0, atol=3e-7)
        error = float(np.abs(reference_targets(z, counts)-target).max())
        require(error <= 3e-7, 'Independent target reference mismatch: '+arm)
        alpha = np.where(support >= CONFIG['min_votes'],
                         np.clip((classes*counts.max(1)/np.maximum(support, 1)-1)/(classes-1), 0, 1), 0)
        np.testing.assert_allclose(cache['alphas'][arm].numpy(), alpha, rtol=0, atol=3e-7)
        np.testing.assert_array_equal(cache['supports'][arm].numpy(), support)
        if arm == 'KNNMATCH':
            np.testing.assert_array_equal(support, geo['support'].numpy())
        if arm == 'OLD64':
            require(torch.equal(cache['targets'][arm], original['target']), 'OLD64 target bridge mismatch')
        brier = ((target-truth)**2).sum(1)
        for stratum, mask in [('all', np.ones(len(z), bool)), ('clean', clean), ('corrupted', ~clean)]:
            n = int(mask.sum())
            values = dict(brier=brier, brier_gain=baseline-brier, alpha=alpha, support=support,
                          argmax_accuracy=(target.argmax(1) == y[train]),
                          damaged_fraction=brier > baseline+1e-7,
                          tied_fraction=(target == target.max(1)[:, None]).sum(1) > 1)
            rows.append(dict(dataset=name, seed=seed, condition=condition, arm=arm, stratum=stratum, n=n,
                             **{key+'_mean' if key in ('alpha', 'support') else key:
                                float(value[mask].mean()) if n else None for key, value in values.items()}))
        checks.append(dict(dataset=name, seed=seed, condition=condition, arm=arm,
                           target_error=error, counts_verified=True, support_verified=True))
    return rows, checks


def macro_f1(y, pred, classes):
    cm = np.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes)
    denominator = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*cm.diagonal(), denominator, out=np.zeros(classes), where=denominator > 0).mean())


def audit_model(name, seed, condition, route, arm, data, geo, cache, y, verified):
    folder = ROOT/'runs'/name/f'seed{seed}'/condition/(route+'_'+arm)
    chain = source_chain(folder)
    source, stored = chain[-1]
    record = chain[0][1]
    epochs = CONFIG['epochs' if route == 'R0' else 'head_epochs']
    for key, expected in dict(dataset=name, seed=seed, condition=condition, route=route, arm=arm,
                              epochs=epochs, smoke=False).items():
        require(record[key] == expected, 'Run identity mismatch: '+str(folder)+' '+key)
    expected_hash = state_hash(dict(target=cache['targets'][arm]))
    require(record['target_hash'] == expected_hash, 'Actual target hash mismatch')
    require(record['input_sha256'] == sha(input_dir(name, seed)/(condition+'.pt')), 'Input hash mismatch')
    require(record.get('reused', False) == (reference_run(name, seed, condition, route, arm) is not None),
            'Unexpected reuse accounting')
    anchor = read(reference_run(name, seed, condition, route, 'CE')/'result.json')
    for key in ('initial_hash', 'initial_rng_hash'):
        require(record[key] == anchor[key], 'Initialization bridge mismatch: '+key)
    for _, alias in chain:
        for key in ('checkpoint_sha256', 'probabilities_sha256', 'best_epoch', 'valid_correct',
                    'valid_total', 'epochs', 'initial_hash', 'initial_rng_hash'):
            require(alias[key] == record[key], 'Alias/source mismatch: '+key)
    # CE historically hashed the loaded LCSS payload before its one-hot objective.
    historical_target = (load(FUSION/'inputs'/name/f'seed{seed}'/(condition+'.pt'))['target']
                         if record.get('reused', False) and arm == 'CE' else cache['targets'][arm])
    require(stored['target_hash'] == state_hash(dict(target=historical_target)), 'Source target hash mismatch')
    if record.get('reused', False):
        require(record['historical_target_hash'] == chain[1][1]['target_hash'] and
                record['historical_input_sha256'] == chain[1][1]['input_sha256'], 'Historical alias metadata mismatch')
    config = read(source/'config.json')
    for key, value in config.items():
        require(stored.get(key) == value, 'Source config/result mismatch: '+key)
    historical_input = ((FUSION/'inputs'/name/f'seed{seed}') if source.is_relative_to(FUSION.resolve()) else
                        (BASE/'inputs'/name) if source.is_relative_to(BASE.resolve()) else input_dir(name, seed))
    require(stored['input_sha256'] == sha(historical_input/(condition+'.pt')), 'Source input hash mismatch')
    valid, test = geo['valid'].numpy(), geo['test'].numpy()
    best, first = check_log(record, logs(source/'epochs.jsonl'), epochs, len(valid))
    for filename, key in [('student.pt', 'checkpoint_sha256'), ('probabilities.pt', 'probabilities_sha256')]:
        require(sha(source/filename) == record[key], 'Artifact hash mismatch: '+str(source/filename))
    if source not in verified:
        probabilities, state = load(source/'probabilities.pt'), load(source/'student.pt')
        require(probabilities.shape == (len(data['x']), data['classes']) and
                torch.isfinite(probabilities).all().item() and (probabilities >= 0).all().item(), 'Invalid probabilities')
        require(torch.allclose(probabilities.double().sum(1),
                               torch.ones(len(probabilities),dtype=torch.float64),
                               rtol=0,atol=8*torch.finfo(probabilities.dtype).eps),
                'Probabilities not normalized')
        model = MLP(data['x'].shape[1], CONFIG['hidden'], data['classes'], 2, CONFIG['dropout']).cuda().eval()
        model.load_state_dict(state)
        with torch.no_grad():
            reloaded = model(data['x']).softmax(1).cpu()
        require(torch.equal(reloaded, probabilities), 'X-only checkpoint probabilities differ')
        if route == 'R2':
            original = load(encoder_path(name, seed))
            for key, value in state.items():
                if not key.startswith('layers.1.'):
                    require(torch.equal(value, original['backbone.'+key]), 'R2 encoder/BN changed: '+key)
            require(record['frozen_encoder_unchanged'], 'Missing frozen encoder verification')
        verified[source] = probabilities.argmax(1).numpy()
        del model, state, reloaded
    pred = verified[source]
    require(int((pred[valid] == y[valid]).sum()) == best, 'Probability validation count mismatch')
    row = dict(dataset=name, seed=seed, condition=condition, route=route, arm=arm,
               accuracy=100*float((pred[test] == y[test]).mean()),
               macro_f1=100*macro_f1(y[test], pred[test], data['classes']),
               validation=100*best/len(valid), valid_correct=best, valid_total=len(valid), best_epoch=first,
               epochs=epochs, seconds=record['seconds'], reused=record.get('reused', False), source=str(source))
    check = dict(dataset=name, seed=seed, condition=condition, route=route, arm=arm,
                 source=str(source), alias_depth=len(chain)-1, checkpoint_sha256=record['checkpoint_sha256'],
                 probabilities_sha256=record['probabilities_sha256'], target_hash=expected_hash,
                 checkpoint_difference=0, selection_verified=True, initialization_verified=True,
                 historical_ce_payload_hash=stored['target_hash'] if arm == 'CE' and row['reused'] else None)
    return row, check


def mean_std(values):
    values = np.asarray(values, dtype=float)
    require(len(values) == len(CONFIG['seeds']) and np.isfinite(values).all(), 'Expected three finite paired seeds')
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)))


def summarize(rows):
    summaries, macros, paired, cell_pairs, paired_macros, gates = [], [], [], [], [], {}
    lookup = {(r['dataset'], r['condition'], r['route'], r['arm'], r['seed']): r for r in rows}
    require(len(lookup) == len(rows) == CONFIG['expected_cells'], 'Duplicate/missing result cells')
    contrasts = [('NP', ref) for ref in ('OLD64', 'SUPPORT', 'COUNT', 'KMNP', 'KNNMATCH', 'CE')]
    contrasts += [('KMNP', 'KMOLD'), ('SUPPORT', 'OLD64')]
    metrics = ('accuracy', 'macro_f1', 'validation', 'best_epoch')
    for route in CONFIG['routes']:
        for arm in CONFIG['arms']:
            for name in CONFIG['datasets']:
                for condition in CONFIG['conditions']:
                    items = [lookup[name, condition, route, arm, seed] for seed in CONFIG['seeds']]
                    summaries.append(dict(dataset=name, condition=condition, route=route, arm=arm, seeds=len(items),
                        **{metric+'_'+key: value for metric in metrics
                           for key, value in mean_std([r[metric] for r in items]).items()}))
            for domain in ('clean', 'noisy'):
                conditions = [c for c in CONFIG['conditions'] if (c == 'clean') == (domain == 'clean')]
                seed_means = {metric: [float(np.mean([lookup[n, c, route, arm, seed][metric]
                              for n in CONFIG['datasets'] for c in conditions])) for seed in CONFIG['seeds']]
                              for metric in metrics}
                macros.append(dict(route=route, arm=arm, domain=domain, seeds=len(CONFIG['seeds']),
                    cells_per_seed=len(CONFIG['datasets'])*len(conditions),
                    **{metric+'_'+key: value for metric in metrics for key, value in mean_std(seed_means[metric]).items()},
                    accuracy_seed_means=json.dumps(dict(zip(CONFIG['seeds'], seed_means['accuracy'])))))
        contrast_stats = {}
        for left, right in contrasts:
            contrast = left+'-'+right
            differences = {}
            for name in CONFIG['datasets']:
                for condition in CONFIG['conditions']:
                    values = []
                    for seed in CONFIG['seeds']:
                        a, b = lookup[name, condition, route, left, seed], lookup[name, condition, route, right, seed]
                        delta = a['accuracy']-b['accuracy']
                        differences[name, condition, seed] = delta
                        values.append(delta)
                        paired.append(dict(dataset=name, condition=condition, seed=seed, route=route,
                            contrast=contrast, left=left, right=right, accuracy_delta_pp=delta,
                            macro_f1_delta_pp=a['macro_f1']-b['macro_f1']))
                    stats = mean_std(values)
                    cell_pairs.append(dict(dataset=name, condition=condition, route=route, contrast=contrast,
                        delta_mean=stats['mean'], delta_std=stats['std'], positive_seeds=sum(v > 1e-10 for v in values),
                        nonnegative_seeds=sum(v >= -1e-10 for v in values),
                        sign='positive' if stats['mean'] > 1e-10 else 'negative' if stats['mean'] < -1e-10 else 'zero'))
            domains = {}
            for domain in ('clean', 'noisy'):
                cells = [(n, c) for n in CONFIG['datasets'] for c in CONFIG['conditions']
                         if (c == 'clean') == (domain == 'clean')]
                seed_means = [float(np.mean([differences[n, c, seed] for n, c in cells])) for seed in CONFIG['seeds']]
                cell_means = {n+'/'+c: float(np.mean([differences[n, c, s] for s in CONFIG['seeds']])) for n, c in cells}
                domains[domain] = dict(**mean_std(seed_means), seed_means=dict(zip(map(str, CONFIG['seeds']), seed_means)),
                    cell_means=cell_means, positive_seeds=sum(v > 1e-10 for v in seed_means),
                    nonnegative_seeds=sum(v >= -1e-10 for v in seed_means),
                    positive_cells=sum(v > 1e-10 for v in cell_means.values()),
                    nonnegative_cells=sum(v >= -1e-10 for v in cell_means.values()),
                    worst_cell_mean=min(cell_means.values()),
                    worst_seed_cell=min(differences[n, c, s] for n, c in cells for s in CONFIG['seeds']))
                stats = domains[domain]
                paired_macros.append(dict(route=route, contrast=contrast, domain=domain, cells_per_seed=len(cells),
                    delta_mean=stats['mean'], delta_std=stats['std'], positive_seeds=stats['positive_seeds'],
                    positive_cells=stats['positive_cells'], nonnegative_cells=stats['nonnegative_cells'],
                    worst_cell_mean=stats['worst_cell_mean'], worst_seed_cell=stats['worst_seed_cell'],
                    seed_means=json.dumps(stats['seed_means']), cell_means=json.dumps(cell_means)))
            contrast_stats[contrast] = domains
        repair = contrast_stats['NP-OLD64']
        clean_ce = contrast_stats['NP-CE']['clean']
        repair_items = dict(all_seed_noisy_macros_nonnegative=repair['noisy']['nonnegative_seeds'] == 3,
            at_least_six_nonnegative_cell_means=repair['noisy']['nonnegative_cells'] >= 6,
            no_noisy_cell_below_minus_one=repair['noisy']['worst_cell_mean'] >= -1-1e-10,
            each_clean_dataset_safe=clean_ce['worst_cell_mean'] >= -1-1e-10,
            corafull_clean_gain=repair['clean']['cell_means']['corafull/clean'] >= 1-1e-10)
        increments = {}
        for reference in ('SUPPORT', 'COUNT', 'KMNP'):
            stats = contrast_stats['NP-'+reference]['noisy']
            items = dict(noisy_macro_at_least_half=stats['mean'] >= .5-1e-10,
                         at_least_six_positive_cell_means=stats['positive_cells'] >= 6,
                         at_least_two_positive_seed_macros=stats['positive_seeds'] >= 2)
            increments[reference] = dict(passed=all(items.values()), items=items,
                                        no_material_increment_detected=abs(stats['mean']) < .5,
                                        formal_equivalence=False)
        support_macro = contrast_stats['NP-SUPPORT']['noisy']['mean']
        gates[route] = dict(repair_pass=all(repair_items.values()), repair_items=repair_items,
            increments=increments, neighbor_identity_supported=increments['SUPPORT']['passed'],
            fixed_count_placement_supported=increments['COUNT']['passed'],
            independent_ball_advantage_supported=increments['KMNP']['passed'],
            worst_seed_clean_vs_ce=clean_ce['worst_seed_cell'], differences=contrast_stats,
            mechanism_candidate='SUPPORT' if support_macro <= 0 else 'donor_identity' if increments['SUPPORT']['passed'] else 'unresolved')
    for filename, records in [('summary.csv', summaries), ('macro_summary.csv', macros),
                              ('paired_differences.csv', paired), ('paired_summary.csv', cell_pairs),
                              ('paired_macro.csv', paired_macros)]:
        csv_write(ROOT/filename, records)
    decision = dict(routes=gates, primary_route='R0', secondary_route='R2', seed_count=3,
                    sample_std_ddof=1, macro_std_basis='three paired seed-level means, not cell variability',
                    independent_grace_repeats=False, seed0_previously_explored=True, confirmatory=False,
                    causal_fraction_estimated=False,
                    decision='retain_repair_candidate' if gates['R0']['repair_pass'] else 'repair_gate_failed')
    write(ROOT/'decision.json', decision)
    table = ['# Fixed-Grid Results', '', 'Test accuracy (%), mean +/- sample std across seeds 0/1/2.', '',
             '| Dataset | Condition | Route | '+' | '.join(CONFIG['arms'])+' |',
             '|---|---|---|'+ '|'.join(['---']*len(CONFIG['arms']))+'|']
    index = {(r['dataset'], r['condition'], r['route'], r['arm']): r for r in summaries}
    for name in CONFIG['datasets']:
        for condition in CONFIG['conditions']:
            for route in CONFIG['routes']:
                values = [index[name, condition, route, arm] for arm in CONFIG['arms']]
                table.append('| '+ ' | '.join([name, condition, route]+[
                    f"{r['accuracy_mean']:.3f} \u00b1 {r['accuracy_std']:.3f}" for r in values])+' |')
    (ROOT/'RESULT_TABLES.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    return macros, decision


def full_audit():
    progress = read(ROOT/'progress.json')
    require(progress['status'] == 'completed' and progress['completed'] == CONFIG['expected_cells'] == 576,
            'Complete all 576 cells before loading test labels or scoring predictions')
    audit = manifest_audit()
    smoke = read(ROOT/'smoke_audit.json')
    require(smoke['passed'] and smoke['fits'] == 24 and
            smoke['frozen_manifest_sha256'] == audit['frozen_manifest_sha256'], 'Missing/stale smoke audit')
    initialize()
    rows, target_rows, checks, target_checks, geometry_checks = [], [], [], [], []
    verified = {}
    for name in CONFIG['datasets']:
        data, y = load_dataset(name), raw_data(name).y.numpy().astype(np.int64)
        for seed in CONFIG['seeds']:
            geo, geometry_check = audit_geometry(name, seed, data)
            geometry_checks.append(geometry_check)
            for condition in CONFIG['conditions']:
                cache = load(input_dir(name, seed)/(condition+'.pt'))
                diagnostics, targets = audit_targets(name, seed, condition, cache, geo, data['classes'], y)
                target_rows.extend(diagnostics)
                target_checks.extend(targets)
                for route in CONFIG['routes']:
                    for arm in CONFIG['arms']:
                        row, check = audit_model(name, seed, condition, route, arm, data, geo, cache, y, verified)
                        rows.append(row)
                        checks.append(check)
            print(f'AUDITED {name} seed{seed}: {len(rows)}/576', flush=True)
    require(len(rows) == 576 and sum(r['reused'] for r in rows) == CONFIG['expected_reused'] == 216 and
            sum(not r['reused'] for r in rows) == CONFIG['expected_new'] == 360, 'Grid/reuse counts differ')
    csv_write(ROOT/'results.csv', rows)
    csv_write(ROOT/'target_diagnostics.csv', target_rows)
    macros, decision = summarize(rows)
    verify()
    artifacts = ['results.csv', 'target_diagnostics.csv', 'summary.csv', 'macro_summary.csv',
                 'paired_differences.csv', 'paired_summary.csv', 'paired_macro.csv', 'decision.json', 'RESULT_TABLES.md']
    write(ROOT/'audit.json', dict(passed=True, total_cells=len(rows), reused=216, new=360,
        unique_models=len(verified), target_checks=target_checks, geometry_checks=geometry_checks, checks=checks,
        artifact_sha256={f: sha(ROOT/f) for f in artifacts}, **audit))
    print(json.dumps(dict(macro_summary=macros, decision=decision['decision'], routes={route: {
        key: value for key, value in gate.items() if key != 'differences'} for route, gate in decision['routes'].items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke', 'full'])
    args = parser.parse_args()
    smoke_audit() if args.phase == 'smoke' else full_audit()
