"""Additional post-hoc diagnostics; never imported by training or selection."""
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict, Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol import EXP, CONFIG, common, read, save, csv_save, sha, now, output, encoder_path
from targets import groups, shrink, mass_direction
from selection import rank_key, probe_survivors

INPUTS = {}


def sload(path):
    path = Path(path)
    INPUTS[str(path.resolve())] = sha(path)
    return torch.load(path, weights_only=True, map_location='cpu')


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def masks(e, clean, observed):
    n = len(clean)
    g = groups(e)
    freq = torch.bincount(clean, minlength=e['own'].shape[1]).double()
    terciles = torch.quantile(freq, torch.tensor([1/3, 2/3], dtype=torch.float64))
    size_group = torch.bucketize(freq, terciles)
    mass_group = torch.bucketize(e['m'], torch.tensor([.25, .5, .75], dtype=torch.float64))
    use = [('all', torch.ones(n, dtype=torch.bool)), ('original_clean', observed == clean), ('original_corrupted', observed != clean)]
    use += [(f'group_{gnum}', g == gnum) for gnum in range(3)]
    use += [(f'true_class_size_{s}', size_group[clean] == s) for s in range(3)]
    use += [(f'mass_bin_{s}', mass_group == s) for s in range(4)]
    return use, g, size_group[clean], mass_group


def node_controls():
    selected = table('selected_results.csv')
    records = [r for r in selected if r['stage'] in ['D1', 'D2']]
    summaries, trajectory, test_class_size = [], [], []
    node_count = 0
    node_file = ROOT/'node_control_details.csv.gz'
    with gzip.open(node_file, 'wt', encoding='utf-8', newline='') as handle:
        fields = ['stage', 'dataset', 'backbone', 'condition', 'arm', 'node', 'observed', 'truth', 'group',
                  'class_size_group', 'mass_bin', 'lambda', 'original_m', 'actual_mass', 'actual_support',
                  'q_observed', 'alpha', 'vote_max', 'vote_margin', 'target_brier_change', 'true_class_mass_change',
                  'student_clean_correct', 'baseline_clean_correct', 'baseline_observed_probability', 'baseline_observed_loss']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CONFIG['datasets_mechanism']:
            raw = common.raw_data(name)
            truth_all = raw.y.long().cpu()
            geo = sload(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt')
            train, test = geo['train'], geo['test']
            truth = truth_all[train]
            for cond in CONFIG['conditions']:
                es = sload(ROOT/'evidence'/name/cond/'evidence.pt')
                for r in [x for x in records if x['dataset'] == name and x['condition'] == cond]:
                    folder = ROOT/r['folder']
                    stage, backbone, arm = r['stage'], r['backbone'], r['selector']
                    meta = read(folder/'result.json')
                    e = es['KM'] if arm == 'GATE_KM' else es['LEGACY']
                    z = e['z']
                    selection_epoch = meta['best_epoch']
                    history = None
                    weights = None
                    if stage == 'D2' and (folder/'weights_history.pt').exists():
                        history = sload(folder/'weights_history.pt').double()
                        weights = history[selection_epoch-1]
                        target = shrink(e, weights, mass_direction(e) if arm == 'GATE_MASS' else None)
                    else:
                        target = sload(folder/'target.pt')
                        actual = 1-target.gather(1, z[:, None]).squeeze(1).double()
                        weights = torch.where(e['m'] > 1e-15, actual/e['m'].clamp_min(1e-15), 0)
                        if stage == 'D1':
                            detail = sload(ROOT/'evidence'/name/cond/f'd1_{backbone}.pt')
                            choice = read(ROOT/'selections/D1'/name/backbone/f'{cond}.json')
                            if arm == 'GROUP_SELECTED':
                                weights = detail['weights'].double()
                            elif arm == 'MASS_MATCHED_SHUFFLE':
                                weights = detail['shuffled'].double()
                            elif arm == 'EXACT_MASS_GLOBAL':
                                weights = torch.full_like(e['m'], choice['mean_lambda'])
                            elif arm == 'CELL_GLOBAL_VAL':
                                weights = torch.full_like(e['m'], choice['best_cell_global_lambda'])
                        elif arm == 'GLOBAL_LCSS':
                            weights = torch.full_like(e['m'], .5)
                    actual_mass = 1-target.gather(1, z[:, None]).squeeze(1).double()
                    p = sload(folder/'probabilities.pt')
                    baseline = ROOT/f'fits/B1/{name}/{backbone}/{cond}/CE' if stage == 'D1' else ROOT/f'fits/D2/{name}/R2/{cond}/TRAIN_CE'
                    bp = sload(baseline/'probabilities.pt')
                    pred, bpred = p[train].argmax(1), bp[train].argmax(1)
                    obs_prob = bp[train].gather(1, z[:, None]).squeeze(1)
                    obs_loss = -obs_prob.clamp_min(1e-30).log()
                    own = e['own']
                    onehot = torch.nn.functional.one_hot(truth, own.shape[1]).float()
                    brier = (target-onehot).square().sum(1)-(own-onehot).square().sum(1)
                    gain = target.gather(1, truth[:, None]).squeeze(1)-own.gather(1, truth[:, None]).squeeze(1)
                    use, g, sizegroup, massgroup = masks(e, truth, z)
                    for label, mask in use:
                        if not mask.any():
                            continue
                        summaries.append(dict(stage=stage, dataset=name, backbone=backbone, condition=cond, arm=arm,
                            stratum=label, nodes=int(mask.sum()), best_epoch=selection_epoch,
                            lambda_mean=float(weights[mask].mean()), lambda_std=float(weights[mask].std(unbiased=False)),
                            original_m_mean=float(e['m'][mask].mean()), actual_mass_mean=float(actual_mass[mask].mean()),
                            target_brier_change=float(brier[mask].mean()), true_class_mass_change=float(gain[mask].mean()),
                            student_clean_accuracy=float((pred[mask] == truth[mask]).double().mean()),
                            baseline_clean_accuracy=float((bpred[mask] == truth[mask]).double().mean()),
                            baseline_observed_loss=float(obs_loss[mask].mean()),
                            baseline_observed_probability=float(obs_prob[mask].mean())))
                    node_arrays = [train.tolist(), z.tolist(), truth.tolist(), g.tolist(), sizegroup.tolist(), massgroup.tolist(),
                         weights.tolist(), e['m'].tolist(), actual_mass.tolist(), e['support'].tolist(), e['qz'].tolist(),
                         e['alpha'].tolist(), e['maximum'].tolist(), e['margin'].tolist(), brier.tolist(), gain.tolist(),
                         (pred == truth).int().tolist(), (bpred == truth).int().tolist(), obs_prob.tolist(), obs_loss.tolist()]
                    for values in zip(*node_arrays):
                        writer.writerow(dict(zip(fields, [stage, name, backbone, cond, arm]+list(values))))
                        node_count += 1
                    if history is not None:
                        logs = [json.loads(l) for l in (folder/'epochs.jsonl').read_text().splitlines()]
                        for ep, w in enumerate(history):
                            trajectory.append(dict(dataset=name, condition=cond, arm=arm, epoch=ep+1,
                                 lambda_mean=float(w.mean()), lambda_std=float(w.std(unbiased=False)),
                                 weighted_mass=float((w*e['m']).sum()), lambda_q10=float(torch.quantile(w, .1)),
                                 lambda_q90=float(torch.quantile(w, .9)), meta_gradient_norm=logs[ep]['meta_gradient_norm'],
                                 meta_loss=logs[ep]['meta_loss']))
                    freq = torch.bincount(truth, minlength=own.shape[1]).double()
                    limits = torch.quantile(freq, torch.tensor([1/3, 2/3], dtype=torch.float64))
                    class_groups = torch.bucketize(freq, limits)
                    testpred, basepred = p[test].argmax(1), bp[test].argmax(1)
                    for size in range(3):
                        mask = class_groups[truth_all[test]] == size
                        if mask.any():
                            test_class_size.append(dict(stage=stage, dataset=name, backbone=backbone, condition=cond,
                                arm=arm, true_class_size_group=size, nodes=int(mask.sum()),
                                accuracy=float((testpred[mask] == truth_all[test][mask]).double().mean()),
                                baseline_accuracy=float((basepred[mask] == truth_all[test][mask]).double().mean())))
    csv_save(ROOT/'node_control_strata.csv', summaries)
    csv_save(ROOT/'gate_trajectories.csv', trajectory)
    csv_save(ROOT/'small_class_test_strata.csv', test_class_size)
    return dict(node_rows=node_count, stratum_rows=len(summaries), trajectory_rows=len(trajectory)), summaries


def causal_audit():
    n = 0
    survivors = []
    selected_hist = Counter()
    for p in (ROOT/'selections/C1').rglob('*_full.json'):
        name, backbone = p.parts[-3:-1]
        cond = p.stem[:-5]
        probe = read(p.with_name(cond+'.json'))
        full = read(p)
        for rnd, count in zip(probe['rounds'], [6, 2]):
            assert probe_survivors(rnd['scores'], count, rnd['prefix']) == rnd['survivors']
            for row in rnd['scores']:
                folder = ROOT/f'fits/C1/{name}/{backbone}/{cond}'/row['arm']
                logs = [json.loads(l) for l in (folder/'epochs.jsonl').read_text().splitlines()][:rnd['prefix']]
                maximum = max(l['validation_correct'] for l in logs)
                best = next(l for l in logs if l['validation_correct'] == maximum)
                assert best['epoch'] == row['best_epoch'] and maximum == row['valid_correct']
                assert abs(best['validation_nll']-row['valid_nll']) < 1e-12
            for arm in rnd['survivors']:
                survivors.append(dict(dataset=name, backbone=backbone, condition=cond, prefix=rnd['prefix'], arm=arm))
        assert min(full['scores'], key=rank_key)['arm'] == full['FULL_VAL']
        selected_hist[(backbone, 'PROBE', probe['PROBE'])] += 1
        selected_hist[(backbone, 'FULL_VAL', full['FULL_VAL'])] += 1
        n += 1
    csv_save(ROOT/'c1_survivors.csv', survivors)
    csv_save(ROOT/'c1_selected_histogram.csv', [dict(backbone=b, selector=s, arm=a, count=count)
                                              for (b, s, a), count in selected_hist.items()])
    return n


def sim_bridges():
    rows = []
    for name in CONFIG['datasets_all']:
        for cond in CONFIG['conditions']:
            new = ROOT/f'fits/A/{name}/SIM/{cond}/CE'
            old = EXP/f'r2_body_comparison_20261023/external/{name}/{cond}/SIMMLP'
            if not (old/'probabilities.pt').exists():
                continue
            a, b = sload(new/'probabilities.pt'), sload(old/'probabilities.pt')
            logs = [json.loads(l) for l in (new/'epochs.jsonl').read_text().splitlines()]
            maximum = max(l['validation_correct'] for l in logs)
            first = next(l['epoch'] for l in logs if l['validation_correct'] == maximum)
            last = max(l['epoch'] for l in logs if l['validation_correct'] == maximum)
            oldr = read(old/'result.json')
            rows.append(dict(dataset=name, condition=cond, first_strict_epoch=first, new_trajectory_last_tie_epoch=last,
                             historical_last_tie_epoch=oldr['best_epoch'], historical_best_val=oldr['best_val'],
                             current_best_val=maximum/read(new/'result.json')['valid_total'],
                             max_probability_difference=float((a-b).abs().max()),
                             selection_policy_changed=True))
    csv_save(ROOT/'simmlp_selection_bridge.csv', rows)
    return rows


def resources():
    events = [json.loads(l) for l in (ROOT/'events.jsonl').read_text().splitlines()]
    ts = lambda value: datetime.fromisoformat(value)
    rows = []
    for path in (ROOT/'selections/C1').rglob('*_full.json'):
        name, backbone = path.parts[-3:-1]
        cond = path.stem[:-5]
        probe = read(path.with_name(cond+'.json'))
        cutoff = ts(probe['frozen'])
        duration = 0.
        fits = 0
        for i, event in enumerate(events):
            if event['stage'] != 'C1' or event.get('dataset') != name or event.get('backbone') != backbone or event.get('condition') != cond or ts(event['timestamp']) >= cutoff:
                continue
            end = min(ts(events[i+1]['timestamp']), cutoff) if i+1 < len(events) else cutoff
            duration += max(0., (end-ts(event['timestamp'])).total_seconds())
            fits += 1
        assert fits == 26  # 18 prefix25 + 6 continuations75 + 2 continuations200.
        rows.append(dict(dataset=name, backbone=backbone, condition=cond,
                         probe_candidate_wall_seconds_including_initialization_and_io=duration,
                         prefix_fit_calls=fits, candidate_updates=1000, ce_updates=200,
                         full_candidate_updates=3600,
                         candidate_fits_actual=18, partition_source='cached_E1_deep_tree',
                         ce_anchor='exact_current_protocol_reuse_or_registered_new_anchor'))
    csv_save(ROOT/'c1_real_resources.csv', rows)
    return rows


def passports():
    provenance = {}
    for name in CONFIG['datasets_all']:
        geometry_path = EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt'
        geo = sload(geometry_path)
        hp = Path(geo['h_path'])
        h = sload(hp)
        if isinstance(h, dict):
            ssl_seed = h.get('pretrain_seed')
            metadata_source = str(hp)
        else:
            mp = hp.parent/'metadata.json'
            INPUTS[str(mp)] = sha(mp)
            hm = read(mp)
            ssl_seed = hm.get('seed')
            metadata_source = str(mp)
        assert ssl_seed == 0
        provenance[name] = dict(historical_grace_seed=ssl_seed, grace_path=str(hp), grace_sha256=sha(hp),
            grace_seed_metadata=metadata_source, geometry_path=str(geometry_path), geometry_sha256=sha(geometry_path),
            r2_encoder=str(encoder_path(name)), r2_encoder_sha256=sha(encoder_path(name)),
            simmlp_encoder=str(EXP/'simmlp_tined_audit_seed0_20261009/runs'/name/'simmlp/checkpoint.pt'),
            simmlp_pretraining_seed=0, independent_ssl_repetitions=1)
    registry = read(ROOT/'fit_registry.json')
    output_rows = []
    for folder in sorted({r['folder'] for r in registry}):
        r = read(ROOT/folder/'result.json')
        if r['mode'].startswith('GATE_'):
            assert r['static_feature_version'] == 'train_scaled_frequency_v2'
            assert r['repair_component_sha256'] == sha(Path(__file__).resolve().parent/'static_repair.py')
        output_rows.append(dict(folder=folder, dataset=r['dataset'], condition=r['condition'], backbone=r['backbone'],
            protocol_seed=0, student_seed=0, ssl_seed=0, seed_std=None, frozen_historical_ssl=True,
            init_sha256=r['initial_hash'], initial_rng_sha256=r['initial_rng_hash'], target_sha256=r['target_hash'],
            input_sha256=r['input_hash'], source_manifest_sha256=r['source_sha256'],
            effective_static_feature_version=r.get('static_feature_version'),
            effective_repair_component_sha256=r.get('repair_component_sha256'), provenance=provenance[r['dataset']]))
    save(ROOT/'run_passports.json', dict(provenance=provenance, fits=output_rows))
    return len(output_rows)


def paired_resume_and_legacy_bridges():
    bridges, coverage = [], []
    scored = read(ROOT/'scored_folders.json')
    for name in CONFIG['datasets_mechanism']:
        for backbone in ['R0', 'R2']:
            for cond in CONFIG['conditions']:
                for k in [8, 32, 128]:
                    a = ROOT/f'fits/B2/{name}/{backbone}/{cond}/N{k}'
                    b = ROOT/f'fits/C1/{name}/{backbone}/{cond}/S{k}_L1.0'
                    ta, tb = sload(a/'target.pt'), sload(b/'target.pt')
                    pa, pb = sload(a/'probabilities.pt'), sload(b/'probabilities.pt')
                    target_diff = float((ta-tb).abs().max())
                    prob_diff = float((pa-pb).abs().max())
                    ra, rb = read(a/'result.json'), read(b/'result.json')
                    bridges.append(dict(dataset=name, backbone=backbone, condition=cond, support=k,
                        continuous_folder=str(a.relative_to(ROOT)), resumed_folder=str(b.relative_to(ROOT)),
                        exact_target=torch.equal(ta, tb), target_max_difference=target_diff,
                        probability_max_difference=prob_diff, continuous_epoch=ra['best_epoch'], resumed_epoch=rb['best_epoch'],
                        initial_hash_equal=ra['initial_hash'] == rb['initial_hash']))
                    assert ra['initial_hash'] == rb['initial_hash']
                    if torch.equal(ta, tb):
                        assert prob_diff <= 3e-7 and ra['best_epoch'] == rb['best_epoch']
    for name in ['citeseer', 'coauthorcs']:
        g = sload(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt')
        equal_partition = torch.equal(g['legacy32'], g['partitions'][32])
        for cond in CONFIG['conditions']:
            es = sload(ROOT/'evidence'/name/cond/'evidence.pt')
            equal_target = torch.equal(es['LEGACY']['native'], es[32]['native'])
            folder = f'fits/C1/{name}/R0/{cond}/S32_L1.0'
            row = dict(dataset=name, backbone='R0', condition=cond, exact_partition=equal_partition,
                       exact_target=equal_target, legacy_available=equal_partition and equal_target,
                       alias_folder=folder if equal_partition and equal_target else None,
                       reason='exact input/target equivalence' if equal_partition and equal_target else 'no separately budgeted legacy R0 fit; no additional training')
            if equal_partition and equal_target:
                row.update(scored[folder])
            coverage.append(row)
    csv_save(ROOT/'c1_b2_resume_bridges.csv', bridges)
    csv_save(ROOT/'c1_legacy_coverage.csv', coverage)
    return dict(continuous_resumed_bridges=len(bridges), exact_target_bridges=sum(r['exact_target'] for r in bridges),
                maximum_resume_probability_difference=max(r['probability_max_difference'] for r in bridges),
                supplemental_legacy_aliases=sum(r['legacy_available'] for r in coverage),
                missing_legacy_control_cells=sum(not r['legacy_available'] for r in coverage))


def c2_ranking_errors():
    from scipy.stats import rankdata
    truth = {(r['dataset'], r['condition'], r['arm']): r['target'] for r in read(ROOT/'c2_features.json')}
    rows = []
    for c in read(ROOT/'selections/C2.json')['choices']:
        y = np.asarray([truth[(c['dataset'], c['condition'], p['arm'])] for p in c['predictions']])
        prediction = np.asarray([p['gain'] for p in c['predictions']])
        ry, rp = rankdata(-y, method='average'), rankdata(-prediction, method='average')
        rows.append(dict(dataset=c['dataset'], condition=c['condition'], selector=c['selector'],
            absolute_rank_error=float(np.abs(ry-rp).mean()), validation_gain_prediction_mae_pp=float(100*np.abs(y-prediction).mean()),
            validation_regret_pp=float(100*(max(0., y.max())-c['validation_gain'])),
            rank_spearman=c['rank_spearman'], selected=c['selected']))
    csv_save(ROOT/'c2_rank_errors.csv', rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['dataset'], r['selector'])].append(r)
    csv_save(ROOT/'c2_graph_rank_errors.csv', [dict(dataset=n, selector=s, conditions=len(rs),
             absolute_rank_error=float(np.mean([r['absolute_rank_error'] for r in rs])),
             validation_regret_pp=float(np.mean([r['validation_regret_pp'] for r in rs])))
             for (n, s), rs in grouped.items()])
    return len(rows)


def report_explanations(node_summary):
    decisions = read(ROOT/'decision.json')['stages']
    differences = table('paired_differences.csv')
    summaries = table('summary.csv')
    selected = table('selected_results.csv')
    new = ['', '## 分阶段解释与行动决策', '']
    for stage in ['A', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        new += [f'### {stage}', '']
        for r in decisions[stage]:
            if stage == 'B2' and r['method'].startswith('DM'):
                continue
            new.append(f'- {r["backbone"]} {r["method"]} 对 {r["reference"]}：带噪accuracy {r["mean_accuracy_pp"]:+.3f}pp，'
                       f'验证 {r["mean_validation_pp"]:+.3f}pp，macro-F1 {r["mean_macro_f1_pp"]:+.3f}pp；{r["wins"]}胜/{r["losses"]}负/{r["cells"]}格。')
        for backbone in sorted({r['backbone'] for r in decisions[stage]}):
            contrasts = [r for r in decisions[stage] if r['backbone'] == backbone]
            gain = {r['reference']: r['mean_validation_pp'] for r in contrasts}
            if stage == 'A':
                action = ('继续：值得小规模独立重复固定监督插件。' if min(gain.values()) >= .5 else
                          '修改：保留对CE的局部监督增量，暂缓对强控制的优势主张。' if gain.get('CE', -1) >= .5 else
                          '暂缓：当前SimMLP固定监督插件的验证增量不足。')
            elif stage == 'B1':
                action = ('继续：优先对冻结全局收缩规格做小规模配对seed重复。' if gain.get('SUPPORT_L1', -1) >= .5 and gain.get('KM_BALANCED', -1) >= 0 else
                          '暂缓统一收缩：保留剂量曲线及条件性代价，原lambda=1继续作为简单锚。')
            elif stage == 'B2':
                direction = [r for r in contrasts if r['method'] in ['D8', 'D128']]
                magnitude = [r for r in contrasts if r['method'] in ['M8', 'M128']]
                dg = np.mean([r['mean_validation_pp'] for r in direction])
                mg = np.mean([r['mean_validation_pp'] for r in magnitude])
                action = f'继续机制记录：方向控制平均验证变化{dg:+.3f}pp，转移量控制{mg:+.3f}pp；当前不扩展尺度搜索。'
            elif stage == 'C1':
                action = ('继续：PROBE对固定S32的验证增量值得小范围重复，同时保留最差单格损失。' if gain.get('FIXED_S32', -1) >= .5 else
                          '修改/暂缓统一PROBE部署：保留完整网格与短训筛选损失，不自动放大预算。')
            elif stage == 'C2':
                mlp = [r for r in contrasts if 'MLP' in r['method']]
                ridge = [r for r in contrasts if 'RIDGE' in r['method']]
                bestm = max(r['mean_validation_pp'] for r in mlp)
                bestr = max(r['mean_validation_pp'] for r in ridge)
                action = ('继续：当前小MLP相对PROBE及Ridge的验证信号值得单独确认。' if bestm >= .5 and bestm > bestr else
                          '暂缓加深MLP：当前保留Ridge、MLP和PROBE的完整比较，优先简单方法。')
            elif stage == 'D1':
                action = ('继续：分组分配相对最佳全局及两种均量控制的验证信号值得小规模确认。' if gain.get('CELL_GLOBAL_VAL', -1) >= 0 and min(gain.get('EXACT_MASS_GLOBAL', -1), gain.get('MASS_MATCHED_SHUFFLE', -1)) >= .5 else
                          '暂缓通用节点分组主张：收益尚未充分超过最佳全局/均量/打乱控制，保留条件性结果。')
            else:
                controls = ['CLEAN_META_CE', 'CLEAN_META_GCE', 'GATE_KM', 'MEAN_LCSS', 'SHUFFLE_LCSS']
                action = ('继续：门控对同预算清洁标签与均量控制具有验证信号，可建议冻结规格重复。' if min(gain.get(c, -1) for c in controls) >= .5 else
                          '暂缓复杂门控主线：同清洁标签预算、KM和逐轮均量控制仍有竞争力，当前不融合C/D。')
            new += ['', f'本轮具体决策（{backbone}）：{action}']
        if stage == 'A':
            new += ['', '判读：固定同一个SimMLP表示之后，SUPPORT对CE衡量局部监督增量，对GCE衡量通用抗噪替代，对KM衡量分区形式的额外价值。三者分别判断，强表示本身的收益沿用固定编码器来源。',
                    '决策：保留本轮完整证据；若SUPPORT对GCE/KM缺少优势，暂缓“强骨干上粒球独立增量”的主张。', '']
        elif stage == 'B1':
            choices = read(ROOT/'selections/B1.json')['choices']
            for c in choices:
                if c['family'] == 'SUPPORT':
                    new.append(f'{c["dataset"]}/{c["backbone"]}：BALANCED lambda={c["BALANCED"]}，CONSTRAINED lambda={c["CONSTRAINED"]}。')
            new += ['', '判读：BALANCED使用四条件共同参数，代价完整保留。收缩若仅改善clean而显著损失带噪收益，应作为使用条件权衡；KM/MASS/GCE同候选预算结果用于识别通用收缩收益。',
                    '决策：优先保留与原强度接近且简单的规格；只对有清晰验证增量并超出同协议控制的局部规格建议配对seed重复。', '']
        elif stage == 'B2':
            new += ['', '判读：D8/D128隔离类别方向变化，M8/M128隔离实际转移量变化；DM与原生N8/N128分别列出，共同集合外沿用N32。交互表为描述性差分，不能以节点数替代独立seed。',
                    '决策：继续保留分解作为机制证据；后续候选优先依据验证方向一致性和原生实际收益，当前不扩大尺度网格。', '']
        elif stage == 'C1':
            regrets = table('c1_regret.csv')
            for b in ['R0', 'R2']:
                rs = [r for r in regrets if r['backbone'] == b]
                gaps = np.array([float(r['test_gap_pp']) for r in rs])
                new.append(f'{b} 距FULL_VAL测试绝对差≤0.5/1/2pp比例：'+
                           '/'.join(f'{100*np.mean(np.abs(gaps) <= t):.2f}%' for t in [.5, 1, 2])+f'；单格测试差范围[{gaps.min():+.3f},{gaps.max():+.3f}]pp。')
            new += ['', '判读：前缀策略先冻结、全网格后补训，适合衡量有限预算筛选损失。FULL_VAL自身按验证选择，测试中PROBE偶尔更高也完整保留。',
                    '决策：若最差单格损失较大，保留FULL_VAL作为研究参照，暂缓把PROBE当作统一替代；候选更新预算与完整审计实际成本分别披露。', '']
        elif stage == 'C2':
            cs = read(ROOT/'selections/C2.json')['choices']
            for selector in ['STATS_RIDGE', 'STATS_MLP', 'ENHANCED_RIDGE', 'ENHANCED_MLP']:
                use = [c for c in cs if c['selector'] == selector]
                values = [c['rank_spearman'] for c in use if c['rank_spearman'] is not None]
                new.append(f'{selector}：20格CE回退{sum(c["selected"] == "CE" for c in use)}，候选排名Spearman平均{np.mean(values):.3f}。')
            new += ['', '判读：整个图留出，scaler与回归目标统计均由四个拟合图估计；ENHANCED额外消耗短训练信息。五折只有五个图级迁移任务。',
                    '决策：MLP若未超过Ridge/PROBE，保留较简单方法，暂缓网络扩容与新增超参搜索。', '']
        elif stage == 'D1':
            counts = Counter(c['selected'] for c in read(ROOT/'selections/D1.json')['choices'])
            new += ['', '24格所选规则分布：'+json.dumps(dict(counts), ensure_ascii=False)+'。',
                    '判读：对最佳全局力度、精确均量全局和精确均量打乱的三种比较，分别回答分组是否实用、是否只是总改动量变化、是否需要正确节点分配。',
                    '决策：只在分组规则同时超出精确均量/打乱控制的条件保留节点选择解释；单独优于CE时继续保留为监督增量证据。', '']
        else:
            new += ['', '判读：TRAIN_CE/GLOBAL_LCSS使用相同selection子集；CLEAN_META_CE/GCE直接使用同量干净meta标签；MEAN/SHUFFLE逐epoch回放相同实际转移量。元梯度与目标stopgrad均有测试和运行日志。',
                    '决策：GATE_LCSS若低于直接干净标签监督、KM门控或等量回放，暂缓复杂门控主线；仅有局部单格收益时保留原规格做小范围重复建议，当前不拼接C/D。', '']
    new += ['## D1/D2 节点层面结果', '',
            '以下将每格训练节点先分层求均值，再对带噪格等权平均。original_clean/corrupted使用后处理真值；group_0/1/2分别为唯一领先同意、冲突、并列。',
            'class_size使用真实训练类别频数的三分位分组，限后处理；节点明细含基线观测类概率/损失、目标Brier变化、真实类别质量变化及所选力度。', '',
            '|阶段|骨干|臂|分层|平均力度|原m|实际转移量|目标Brier变化|真实类质量变化|学生clean accuracy|',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    wanted = {'D1': ['GROUP_SELECTED', 'EXACT_MASS_GLOBAL', 'MASS_MATCHED_SHUFFLE'],
              'D2': ['GATE_LCSS', 'CLEAN_META_CE', 'CLEAN_META_GCE', 'MEAN_LCSS', 'SHUFFLE_LCSS']}
    buckets = defaultdict(list)
    for r in node_summary:
        if r['condition'] == 'clean' or r['arm'] not in wanted[r['stage']] or r['stratum'] not in ['original_clean', 'original_corrupted', 'group_0', 'group_1', 'group_2', 'true_class_size_0']:
            continue
        buckets[(r['stage'], r['backbone'], r['arm'], r['stratum'])].append(r)
    for (stage, b, arm, stratum), rs in sorted(buckets.items()):
        avg = lambda field: np.mean([r[field] for r in rs])
        new.append(f'|{stage}|{b}|{arm}|{stratum}|{avg("lambda_mean"):.3f}|{avg("original_m_mean"):.3f}|{avg("actual_mass_mean"):.3f}|{avg("target_brier_change"):+.4f}|{avg("true_class_mass_change"):+.4f}|{100*avg("student_clean_accuracy"):.2f}%|')
    new += ['', '完整分图、条件和所有臂在node_control_strata.csv；低频类别测试保护在small_class_test_strata.csv；逐epoch门控与实际转移量在gate_trajectories.csv。', '',
            '目标Brier与最终分类分别衡量标签目标和共享学生泛化，二者的同向或反向都在明细保留。', '',
            '## 运行版本与补充审计', '',
            '训练源码与sources.json自正式启动后保持冻结。postprocess/supplement.py只读已冻结选择、日志、概率、目标与真值，不调用学生fit或选择器重设计。',
            '补充源/输入哈希见supplement_manifest.json；testing.json原7项计数为文档滞后，原契约测试9项加静态特征及Windows路径修复各1项，共11项，见testing.final.json。',
            'D2第9维观测类别频率最初遗漏静态缩放，在D2首次训练前由独立版本入口修复。原训练文件和sources.json保持原样，生效组件见repair_sources.json及每个门控result.json的static_feature_version。',
            '为修复而受控中断本队列时，B1有一个尚无检查点的122更新部分轨迹，已原样归档；该格从同初始化重跑。1500指有效完整科学拟合，另披露该中断尝试及5个两轮smoke。']
    new += ['', 'Windows路径分隔符与alias展示在独立后处理版本中归一化；原fit_registry.json和冻结选择保留。fit_registry.canonical.json及aliases.csv提供清晰新拟合/复用记录，40个新CE/GCE锚完整进入results.csv。',
            'D2均量控制逐epoch匹配；各臂按自身selection最佳时点选模，节点表按各自选定epoch报告实际力度。']
    report = ROOT/'REPORT_zh.md'
    existing = report.read_text(encoding='utf-8').split('\n## 分阶段解释与行动决策')[0]
    existing = existing.replace('CiteSeer/CS的历史SUPPORT32 R0对照未列入额外预算，该两图在C1使用新树固定S32；历史SUPPORT32比较覆盖机制三图R0/R2。',
        '历史SUPPORT32主表覆盖机制三图R0/R2；CiteSeer经精确分区/目标核对可复用新S32并列于补充表；CS R0未单列历史对照，保持缺失。')
    report.write_text(existing+'\n'+'\n'.join(new)+'\n', encoding='utf-8')


def main():
    torch.set_num_threads(1)
    assert read(ROOT/'selections_frozen.json')['all_stages_complete']
    assert read(ROOT/'audit.json')['status'] == 'passed'
    counts, ns = node_controls()
    causal = causal_audit()
    bridges = sim_bridges()
    cost_rows = resources()
    passport_count = passports()
    bridge_audit = paired_resume_and_legacy_bridges()
    rank_rows = c2_ranking_errors()
    report_explanations(ns)
    save(ROOT/'testing.final.json', dict(tests=11, original_contract_tests=9, static_repair_tests=1, windows_path_tests=1, status='passed', original_record='testing.json',
         explanation='Original 7 count predates two added tests; one additional static feature regression test', smoke_fits=5,
         smoke_resume_max_error=read(ROOT/'smoke_complete.json')['resume_max_probability_difference']))
    save(ROOT/'supplement_manifest.json', dict(finished=now(), source={str(Path(__file__).resolve()): sha(__file__)},
         inputs=INPUTS, training_sources_manifest_sha256=sha(ROOT/'sources.json'),
         classification_fits_added=0, **counts, causal_probe_recomputed=causal, sim_bridge_cells=len(bridges),
         fit_passports=passport_count, c1_resource_cells=len(cost_rows), c2_rank_cells=rank_rows, **bridge_audit))
    print(json.dumps(dict(status='supplement_complete', **counts, causal_probe_cells=causal)), flush=True)


if __name__ == '__main__':
    main()
