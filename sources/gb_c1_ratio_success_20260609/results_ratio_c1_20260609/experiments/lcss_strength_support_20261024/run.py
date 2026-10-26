"""Authorized seed0 queue. No test/clean-train labels are read here."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from protocol import (ROOT, EXP, WORKSPACE, CONFIG, INPUTS, save, read, load, track, sha, now,
                      append, csv_save, check_time, acquire_lock, freeze_sources,
                      graph_inputs, cell_inputs, training_data, encoder_path, split_validation,
                      state_hash, seed_all)
from targets import evidence, shrink, mass_direction, b2_targets, groups, global_mass, match_mass, observable_stats
from selection import rank_key, probe_survivors, balanced, constrained, dataset_fold
from fit import fit, sim_embedding

ALL = CONFIG['datasets_all']
MECH = CONFIG['datasets_mechanism']
CONDS = CONFIG['conditions']
RECORDS = []


def event(stage, **kwargs):
    value = dict(timestamp=now(), stage=stage, **kwargs)
    save(ROOT/'status.json', value)
    append(ROOT/'events.jsonl', value)
    print('STATUS '+json.dumps(value), flush=True)


def remember(stage, name, backbone, cond, arm, folder, result, **kwargs):
    item = dict(stage=stage, dataset=name, backbone=backbone, condition=cond, arm=arm,
                folder=str(folder.relative_to(ROOT)), alias=not str(folder.relative_to(ROOT)).startswith('fits/'+stage+'/'),
                **kwargs)
    key = tuple(item[k] for k in ('stage', 'dataset', 'backbone', 'condition', 'arm'))
    if key not in {tuple(r[k] for k in ('stage', 'dataset', 'backbone', 'condition', 'arm')) for r in RECORDS}:
        RECORDS.append(item)
        save(ROOT/'fit_registry.json', RECORDS)
    return result


def prepare(name, cond, graph, protocol='FULL'):
    pack = cell_inputs(name, cond)
    geo, z = graph['geometry'], pack['observed']
    classes = pack['targets'][32].shape[1]
    values = dict(x=graph['x'].cuda(), train=geo['train'].cuda(), valid=geo['valid'].cuda(),
                  observed=z.cuda(), valid_y=pack['valid_y'].cuda(), classes=classes)
    if protocol == 'META':
        mi, si = split_validation(pack['valid_y'])
        if len(mi) == 0:
            raise RuntimeError('Meta-fit empty')
        values.update(meta=geo['valid'][mi].cuda(), meta_y=pack['valid_y'][mi].cuda(),
                      valid=geo['valid'][si].cuda(), valid_y=pack['valid_y'][si].cuda())
        save(ROOT/'meta_splits'/name/f'{cond}.json', dict(meta=values['meta'].cpu().tolist(),
              selection=values['valid'].cpu().tolist(), seed=20261024,
              train_sha256=state_hash({'train': geo['train']}), test_sha256=state_hash({'test': geo['test']})))
    es = {s: evidence(z, part[geo['train']], classes) for s, part in geo['partitions'].items()}
    es['LEGACY'] = evidence(z, geo['legacy32'][geo['train']], classes)
    es['KM'] = evidence(z, graph['km'][geo['train']], classes)
    folder = ROOT/'evidence'/name/cond
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder/'evidence.pt').exists():
        torch.save(es, folder/'evidence.pt')
        profiles = []
        for s, e in es.items():
            profiles.append(dict(partition=str(s), mean=float(e['support'].mean()), median=float(e['support'].median()),
                                 q90=float(torch.quantile(e['support'], .9)), minimum=float(e['support'].min()),
                                 maximum=float(e['support'].max()), mass_mean=float(e['m'].mean()),
                                 tied_fraction=float(e['tied'].double().mean())))
        save(folder/'profiles.json', profiles)
        # Independent historical NumPy implementation is the latest R2 anchor.
        source = track(EXP/'r2_support_5dataset_20261022/run.py')
        nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'numpy_targets']
        ns = dict(np=np)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        differences = {}
        for label, part in [('LEGACY', geo['legacy32']), ('KM', graph['km'])]:
            old = torch.from_numpy(ns['numpy_targets'](z.numpy(), part[geo['train']].numpy(), classes, .5, 2)).float()
            diff = float((es[label]['native']-old).abs().max())
            if diff > 3e-7:
                raise RuntimeError('NumPy bridge mismatch')
            differences[label] = diff
        differences['deep32_vs_e1_float32_target'] = float((es[32]['native']-pack['targets'][32]).abs().max())
        differences['lambda1_reconstruction'] = max(float((shrink(e, 1)-e['native']).abs().max()) for e in es.values())
        save(folder/'bridge.json', differences)
    return training_data(values), es, pack


def folder_for(stage, name, backbone, cond, arm):
    return ROOT/'fits'/stage/name/backbone/cond/arm


def execute(stage, name, backbone, cond, arm, values, target, enc, *, until=200, alias=None, **kwargs):
    folder = alias or folder_for(stage, name, backbone, cond, arm)
    event(stage, dataset=name, backbone=backbone, condition=cond, arm=arm, until=until)
    result = fit(folder, backbone, values, target, enc, until=until,
                 metadata=dict(dataset=name, condition=cond, protocol='META' if 'meta' in values else 'FULL'), **kwargs)
    remember(stage, name, backbone, cond, arm, folder, result)
    return result


def score(arm, result, support=0, strength=0):
    return dict(arm=arm, support=support, strength=strength, valid_correct=result['valid_correct'],
                valid_total=result['valid_total'], valid_nll=result['valid_nll'],
                best_epoch=result['best_epoch'], epoch=result['epoch'])


def anchors(stage, name, backbone, cond, values, es, enc):
    support = es['LEGACY']
    ce = support['own']
    if name in MECH:
        ce_folder = folder_for('B1', name, backbone, cond, 'CE')
        gce = {q: folder_for('B1', name, backbone, cond, f'GCE_Q{q}') for q in [.3, .5, .7, .9]}
        support_folder = folder_for('B1', name, backbone, cond, 'SUPPORT_L1.0')
    else:
        ce_folder = folder_for('C1_ANCHOR', name, backbone, cond, 'CE')
        gce = {q: folder_for('C1_ANCHOR', name, backbone, cond, f'GCE_Q{q}') for q in [.3, .5, .7, .9]}
        support_folder = None
    execute(stage+'_REF', name, backbone, cond, 'CE', values, ce, enc, alias=ce_folder)
    for q, folder in gce.items():
        execute(stage+'_REF', name, backbone, cond, f'GCE_Q{q}', values, ce, enc, q=q, alias=folder)
    return ce_folder, gce, support_folder


def stage_a():
    for name in ALL:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        sim = sim_embedding(name, graph['x'].cuda())
        for cond in CONDS:
            values, es, _ = prepare(name, cond, graph)
            for arm, target, q in [('CE', es['LEGACY']['own'], None), ('GCE', es['LEGACY']['own'], .7),
                                   ('SUPPORT', es['LEGACY']['native'], None), ('KM', es['KM']['native'], None)]:
                execute('A', name, 'SIM', cond, arm, values, target, enc, sim=sim, q=q)
        del graph, sim, values
        torch.cuda.empty_cache()
    finish_stage('A', 80)


def stage_b1():
    choices = []
    for name in MECH:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        for backbone in ['R0', 'R2']:
            all_results = {f: {p: [] for p in ([.3, .5, .7, .9] if f == 'GCE' else [0., .25, .5, 1.])}
                           for f in ['SUPPORT', 'KM', 'MASS', 'GCE']}
            ce_results = []
            for cond in CONDS:
                values, es, _ = prepare(name, cond, graph)
                ce = execute('B1', name, backbone, cond, 'CE', values, es['LEGACY']['own'], enc)
                ce_results.append(ce)
                for family in ['SUPPORT', 'KM', 'MASS']:
                    all_results[family][0.].append(ce)
                    e = es['KM'] if family == 'KM' else es['LEGACY']
                    for strength in [.25, .5, 1.]:
                        t = shrink(e, strength, mass_direction(e) if family == 'MASS' else None)
                        r = execute('B1', name, backbone, cond, f'{family}_L{strength}', values, t, enc)
                        all_results[family][strength].append(r)
                for q in [.3, .5, .7, .9]:
                    r = execute('B1', name, backbone, cond, f'GCE_Q{q}', values, es['LEGACY']['own'], enc, q=q)
                    all_results['GCE'][q].append(r)
            for family, candidates in all_results.items():
                scores = {p: [r['validation'] for r in rows] for p, rows in candidates.items()}
                parameter = balanced(scores)
                secondary, feasible = constrained(candidates, ce_results, family)
                choices.append(dict(dataset=name, backbone=backbone, family=family,
                                    BALANCED=parameter, CONSTRAINED=secondary, constrained_feasible=feasible,
                                    candidate_validation_scores=scores))
        del graph, values
        torch.cuda.empty_cache()
    save(ROOT/'selections/B1.json', dict(frozen=now(), choices=choices))
    finish_stage('B1', 336)


def stage_b2():
    for name in MECH:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        for backbone in ['R0', 'R2']:
            for cond in CONDS:
                values, es, _ = prepare(name, cond, graph)
                ts, j = b2_targets({s: es[s] for s in [8, 32, 128]})
                torch.save(j, ROOT/'evidence'/name/cond/'common_J.pt')
                for arm, target in ts.items():
                    execute('B2', name, backbone, cond, arm, values, target, enc)
        del graph, values
        torch.cuda.empty_cache()
    finish_stage('B2', 216)


def stage_c1():
    features = []
    for name in ALL:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        for backbone in (['R0', 'R2'] if name in MECH else ['R0']):
            for cond in CONDS:
                values, es, pack = prepare(name, cond, graph)
                ce_folder, gce_folders, legacy = anchors('C1', name, backbone, cond, values, es, enc)
                candidates = {f'S{k}_L{strength}': (k, strength) for k in CONFIG['C1']['support_grid']
                              for strength in CONFIG['C1']['lambda_grid']}
                decision_path = ROOT/'selections/C1'/name/backbone/f'{cond}.json'
                if decision_path.exists():
                    decisions = read(decision_path)
                else:
                    survivors = list(candidates)
                    rounds = []
                    for prefix, count in [(25, 6), (75, 2), (200, 1)]:
                        saved_round = decision_path.with_name(cond+f'_prefix{prefix}.json')
                        if prefix < 200 and saved_round.exists():
                            completed_round = read(saved_round)
                            rounds.append(completed_round)
                            survivors = completed_round['survivors']
                            continue
                        rows = []
                        for arm in survivors:
                            k, strength = candidates[arm]
                            r = execute('C1', name, backbone, cond, arm, values, shrink(es[k], strength), enc, until=prefix)
                            rows.append(score(arm, r, k, strength))
                        if prefix < 200:
                            kept = probe_survivors(rows, count, prefix)
                            rounds.append(dict(prefix=prefix, scores=rows, survivors=kept, frozen=now()))
                            save(decision_path.with_name(cond+f'_prefix{prefix}.json'), rounds[-1])
                            survivors = kept
                        else:
                            rows.append(score('CE', read(ce_folder/'result.json')))
                            winner = min(rows, key=rank_key)['arm']
                            decisions = dict(frozen=now(), PROBE=winner, rounds=rounds, final_scores=rows,
                                             candidate_updates=1000, ce_updates=200,
                                             true_prefix_execution=True, b2_alias=False)
                            save(decision_path, decisions)
                freeze_sha = sha(decision_path)
                # Only after the causal PROBE choice exists do eliminated arms see epoch > prefix.
                all_scores = [score('CE', read(ce_folder/'result.json'))]
                for arm, (k, strength) in candidates.items():
                    r = execute('C1', name, backbone, cond, arm, values, shrink(es[k], strength), enc)
                    all_scores.append(score(arm, r, k, strength))
                    if backbone == 'R0':
                        folder = folder_for('C1', name, backbone, cond, arm)
                        logs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
                        dynamic = []
                        for ep in [5, 10, 25]:
                            row = logs[ep-1]
                            dynamic.extend([row['validation_correct']/r['valid_total'], row['validation_nll'], row['loss']])
                        features.append(dict(dataset=name, condition=cond, arm=arm, support=k, strength=strength,
                                             stats=observable_stats(graph['x'], graph['edge_index'], graph['geometry']['train'], es[k], k, strength),
                                             enhanced=dynamic, target=r['validation']-read(ce_folder/'result.json')['validation']))
                if sha(decision_path) != freeze_sha:
                    raise RuntimeError('PROBE freeze changed')
                full = min(all_scores, key=rank_key)['arm']
                only_k = min([r for r in all_scores if r['arm'] == 'CE' or r['strength'] == 1.], key=rank_key)['arm']
                only_strength = min([r for r in all_scores if r['arm'] == 'CE' or r['support'] == 32], key=rank_key)['arm']
                gce_rows = {q: read(f/'result.json') for q, f in gce_folders.items()}
                best_q = min(gce_rows, key=lambda q: (-gce_rows[q]['validation'], gce_rows[q]['valid_nll'], q))
                save(decision_path.with_name(cond+'_full.json'), dict(frozen=now(), probe_sha256=freeze_sha,
                    FULL_VAL=full, ONLY_K=only_k, ONLY_STRENGTH=only_strength, FIXED_S32='S32_L1.0',
                    SC=f'S{pack["selected_support"]}_L1.0', GCE_Q=best_q,
                    LEGACY_SUPPORT32=str(legacy.relative_to(ROOT)) if legacy else None,
                    scores=all_scores, audit_complete=True))
        del graph, values
        torch.cuda.empty_cache()
    save(ROOT/'c2_features.json', features)
    finish_stage('C1', 576, additional_anchors=40)


def stage_c2():
    from sklearn.linear_model import Ridge
    rows = read(ROOT/'c2_features.json')
    output_rows, folds = [], []
    for name in ALL:
        train, held = dataset_fold(rows, name)
        for enhanced in [False, True]:
            getx = lambda r: r['stats']+(r['enhanced'] if enhanced else [])
            x = np.asarray([getx(r) for r in train], dtype=np.float64)
            tx = np.asarray([getx(r) for r in held], dtype=np.float64)
            y = np.asarray([r['target'] for r in train], dtype=np.float64)
            mean, sd = x.mean(0), x.std(0)
            sd[sd < 1e-8] = 1
            ym, ys = float(y.mean()), max(float(y.std()), 1e-8)
            xn, txn, yn = (x-mean)/sd, (tx-mean)/sd, (y-ym)/ys
            for model_name in ['RIDGE', 'MLP']:
                label = ('ENHANCED' if enhanced else 'STATS')+'_'+model_name
                if model_name == 'RIDGE':
                    model = Ridge(alpha=10.).fit(xn, yn)
                    predicted = model.predict(txn)*ys+ym
                    state = dict(coef=model.coef_.tolist(), intercept=float(model.intercept_))
                else:
                    seed_all(20261024)
                    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
                    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.01)
                    xt, yt = torch.tensor(xn).float(), torch.tensor(yn).float()
                    for _ in range(300):
                        opt.zero_grad(set_to_none=True)
                        loss = F.mse_loss(model(xt).squeeze(1), yt)
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        predicted = model(torch.tensor(txn).float()).squeeze(1).numpy()*ys+ym
                    state = {'model': model.state_dict()}
                    torch.save(state, ROOT/f'c2_{name}_{label}.pt')
                    state = {'checkpoint': f'c2_{name}_{label}.pt'}
                folds.append(dict(heldout=name, feature_set=label, training_datasets=sorted({r['dataset'] for r in train}),
                                  train_rows=len(train), heldout_rows=len(held), feature_mean=mean.tolist(),
                                  feature_sd=sd.tolist(), target_mean=ym, target_sd=ys, model=state))
                for cond in CONDS:
                    indices = [i for i, r in enumerate(held) if r['condition'] == cond]
                    best = min(indices, key=lambda i: (-float(predicted[i]), held[i]['strength'], -held[i]['support']))
                    arm = held[best]['arm'] if predicted[best] > 0 else 'CE'
                    val_gain = held[best]['target'] if arm != 'CE' else 0.
                    from scipy.stats import spearmanr
                    corr = spearmanr([predicted[i] for i in indices], [held[i]['target'] for i in indices]).statistic
                    output_rows.append(dict(dataset=name, condition=cond, selector=label, selected=arm,
                         predicted_gain=float(predicted[best]), validation_gain=val_gain,
                         rank_spearman=float(corr) if np.isfinite(corr) else None,
                         predictions=[dict(arm=held[i]['arm'], gain=float(predicted[i])) for i in indices]))
    save(ROOT/'selections/C2.json', dict(frozen=now(), fits=20, folds=folds, choices=output_rows))
    finish_stage('C2', 0, cpu_predictor_fits=20)


def stage_d1():
    choices = []
    for name in MECH:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        for backbone in ['R0', 'R2']:
            for cond in CONDS:
                values, es, _ = prepare(name, cond, graph)
                e = es['LEGACY']
                global_candidates = {0.: read(folder_for('B1', name, backbone, cond, 'CE')/'result.json')}
                global_candidates.update({l: read(folder_for('B1', name, backbone, cond, f'SUPPORT_L{l}')/'result.json')
                                          for l in [.25, .5, 1.]})
                best_global = min(global_candidates, key=lambda l: (-global_candidates[l]['validation'], l))
                group = groups(e)
                rows, weights = [], {}
                for a, b in [(0., 1.), (.25, 1.), (1., .25), (1., 0.)]:
                    arm = f'GROUP_A{a}_C{b}'
                    w = torch.where(group == 0, a, torch.where(group == 1, b, best_global)).double()
                    r = execute('D1', name, backbone, cond, arm, values, shrink(e, w), enc)
                    rows.append(score(arm, r))
                    weights[arm] = w
                winner = min(rows, key=rank_key)['arm']
                w = weights[winner]
                mean = global_mass(e['m'], w)
                perm = torch.randperm(len(w), generator=torch.Generator().manual_seed(20261024))
                shuffled = match_mass(e['m'], w, w[perm])
                choice = dict(dataset=name, backbone=backbone, condition=cond, selected=winner,
                              best_cell_global_lambda=best_global, mean_lambda=mean, frozen=now(),
                              total_mass=float((e['m']*w).sum()), shuffled_mass=float((e['m']*shuffled).sum()),
                              groups=[dict(group=g, count=int((group == g).sum()),
                                           original_m_mean=float(e['m'][group == g].mean()) if (group == g).any() else None,
                                           selected_lambda=float(w[group == g][0]) if (group == g).any() else None,
                                           observed_class_counts=torch.bincount(e['z'][group == g], minlength=values['classes']).tolist()) for g in range(3)])
                save(ROOT/'selections/D1'/name/backbone/f'{cond}.json', choice)
                torch.save(dict(weights=w, shuffled=shuffled, groups=group), ROOT/'evidence'/name/cond/f'd1_{backbone}.pt')
                for arm, target in [('EXACT_MASS_GLOBAL', shrink(e, mean)), ('MASS_MATCHED_SHUFFLE', shrink(e, shuffled))]:
                    execute('D1', name, backbone, cond, arm, values, target, enc)
                choices.append(choice)
        del graph, values
        torch.cuda.empty_cache()
    save(ROOT/'selections/D1.json', dict(frozen=now(), choices=choices))
    finish_stage('D1', 144)


def stage_d2():
    for name in MECH:
        graph = graph_inputs(name)
        enc = load(encoder_path(name))
        for cond in CONDS:
            values, es, _ = prepare(name, cond, graph, 'META')
            e = es['LEGACY']
            for arm in CONFIG['D2']['arms']:
                target = shrink(e, .5) if arm == 'GLOBAL_LCSS' else e['own']
                ev = es['KM'] if arm == 'GATE_KM' else e
                replay = None
                if arm in ['MEAN_LCSS', 'SHUFFLE_LCSS']:
                    replay = torch.load(folder_for('D2', name, 'R2', cond, 'GATE_LCSS')/'weights_history.pt', weights_only=True)
                execute('D2', name, 'R2', cond, arm, values, target, enc, mode=arm, evidence=ev, replay=replay)
        del graph, values
        torch.cuda.empty_cache()
    finish_stage('D2', 108)


def finish_stage(stage, planned, **kwargs):
    save(ROOT/f'{stage}_complete.json', dict(finished=now(), planned_classification_fits=planned, **kwargs))
    event(stage+'_COMPLETE', planned_classification_fits=planned, **kwargs)


def smoke():
    name, cond = 'cora', 'uniform0.4'
    graph = graph_inputs(name)
    enc = load(encoder_path(name))
    values, es, _ = prepare(name, cond, graph)
    sim = sim_embedding(name, values['x'])
    for kind in ['R0', 'R2', 'SIM']:
        fit(ROOT/'smoke'/kind, kind, values, es['LEGACY']['native'], enc, until=2, sim=sim if kind == 'SIM' else None)
    fit(ROOT/'smoke/R0_resume', 'R0', values, es['LEGACY']['native'], enc, until=1)
    fit(ROOT/'smoke/R0_resume', 'R0', values, es['LEGACY']['native'], enc, until=2)
    a = torch.load(ROOT/'smoke/R0/probabilities.pt', weights_only=True)
    b = torch.load(ROOT/'smoke/R0_resume/probabilities.pt', weights_only=True)
    diff = float((a-b).abs().max())
    if diff > 1e-7:
        raise RuntimeError('Per-arm RNG/optimizer resume mismatch')
    meta_values, mes, _ = prepare(name, cond, graph, 'META')
    fit(ROOT/'smoke/META', 'R2', meta_values, mes['LEGACY']['own'], enc, until=2,
        mode='GATE_LCSS', evidence=mes['LEGACY'])
    save(ROOT/'smoke_complete.json', dict(fits=5, epochs_per_fit=2, resume_max_probability_difference=diff,
         note='Exact meta-gradient path separately exercised by unit test beyond warmup', finished=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    check_time()
    if (ROOT/'complete.json').exists():
        print('Already complete; no duplicate launch', flush=True)
        return
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4*1024**3:
        save(ROOT/'gpu_wait.json', dict(timestamp=now(), reason='GPU unavailable or occupied; no task terminated'))
        raise RuntimeError('GPU unavailable or occupied')
    lock = acquire_lock()
    try:
        freeze_sources()
        if (ROOT/'inputs.json').exists():
            INPUTS.update(read(ROOT/'inputs.json'))
        if (ROOT/'fit_registry.json').exists():
            RECORDS.extend(read(ROOT/'fit_registry.json'))
        save(ROOT/'environment.json', dict(python=sys.executable, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
              free_total_bytes=list(torch.cuda.mem_get_info()), pid=os.getpid(), started=now(), model='gpt-6.1-sol', effort='high'))
        if args.smoke:
            smoke()
            return
        started = time.perf_counter()
        for stage, function in [('A', stage_a), ('B1', stage_b1), ('B2', stage_b2), ('C1', stage_c1),
                                ('C2', stage_c2), ('D1', stage_d1), ('D2', stage_d2)]:
            if not (ROOT/f'{stage}_complete.json').exists():
                function()
        save(ROOT/'selections_frozen.json', dict(frozen=now(), selections={str(p.relative_to(ROOT)): sha(p)
             for p in (ROOT/'selections').rglob('*.json')}, all_stages_complete=True))
        from analyze import analyze
        analyze()
        from audit import audit
        audit()
        unique = len({r['folder'] for r in RECORDS})
        save(ROOT/'complete.json', dict(finished=now(), unique_classification_fits=unique,
             planned_upper_bound=1500, cpu_predictors=20, smoke_fits=5, seconds_this_invocation=time.perf_counter()-started,
             stages=CONFIG['stages'], audit=read(ROOT/'audit.json')['status']))
        event('COMPLETE', unique_classification_fits=unique)
    except BaseException:
        failure = dict(timestamp=now(), traceback=traceback.format_exc(), state=read(ROOT/'status.json') if (ROOT/'status.json').exists() else None)
        append(ROOT/'failures.jsonl', failure)
        save(ROOT/'failure_latest.json', failure)
        output = ROOT/'HANDOFF.md'
        output.write_text('# 可恢复交接\n\n状态：中断，未宣告完成。\n恢复命令：指定Python -B experiments/lcss_strength_support_20261024/run.py\n'
                          '逐阶段标记和逐臂resume.pt保留；请先检查failure_latest.json及run.lock所有者。\n'+failure['traceback'], encoding='utf-8')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
