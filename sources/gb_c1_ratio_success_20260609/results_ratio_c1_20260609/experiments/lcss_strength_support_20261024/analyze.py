"""Postprocessing only: opened after all validation choices are frozen."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from protocol import ROOT, EXP, CONFIG, common, load, read, save, csv_save, sha, now, output
from fit import f1_score


def result(folder):
    return read(ROOT/folder/'result.json')


def metrics(prob, y, ids, classes):
    p, truth = prob[ids], y[ids]
    pred = p.argmax(1)
    accuracy = float((pred == truth).double().mean())
    f1 = f1_score(pred, truth, classes)
    recalls = [float((pred[truth == c] == c).double().mean()) if (truth == c).any() else None for c in range(classes)]
    return accuracy, f1, recalls


def analyze():
    frozen = read(ROOT/'selections_frozen.json')
    assert frozen['all_stages_complete']
    for p, digest in frozen['selections'].items():
        assert sha(ROOT/p) == digest
    registry = read(ROOT/'fit_registry.json')
    unique = sorted({r['folder'] for r in registry})
    by_folder, class_rows, diagnostics = {}, [], []
    grouped = defaultdict(list)
    for folder in unique:
        meta = result(folder)
        grouped[meta['dataset']].append((folder, meta))
    for name, fits in grouped.items():
        # Training/scoring are separate entry points; this is the first clean-truth access.
        raw = common.raw_data(name)
        y = raw.y.long().cpu()
        geo = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0/geometry.pt')
        for folder, meta in fits:
            probabilities = load(ROOT/folder/'probabilities.pt')
            acc, f1, recalls = metrics(probabilities, y, geo['test'], probabilities.shape[1])
            va, vf, _ = metrics(probabilities, y, geo['valid'], probabilities.shape[1])
            trainacc, trainf1, _ = metrics(probabilities, y, geo['train'], probabilities.shape[1])
            by_folder[folder] = dict(test_accuracy=acc, test_macro_f1=f1, full_validation_accuracy=va,
                                     train_clean_accuracy=trainacc, train_clean_macro_f1=trainf1,
                                     validation=meta['validation'], valid_macro_f1=meta['valid_macro_f1'],
                                     best_epoch=meta['best_epoch'], seconds=meta['seconds'])
            for c, recall in enumerate(recalls):
                class_rows.append(dict(folder=folder, dataset=name, condition=meta['condition'],
                                       backbone=meta['backbone'], class_id=c, recall=recall))
        for cond in CONFIG['conditions']:
            pack = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/'seed0'/f'{cond}.pt')
            es = load(ROOT/'evidence'/name/cond/'evidence.pt')
            z, truth = pack['observed'], y[geo['train']]
            freq = torch.bincount(z, minlength=es['LEGACY']['own'].shape[1])
            boundaries = torch.quantile(freq.double(), torch.tensor([1/3, 2/3], dtype=torch.float64))
            class_group = torch.bucketize(freq.double(), boundaries, right=False)
            for label, e in es.items():
                from targets import groups
                own, native = e['own'], e['native']
                truth_onehot = torch.nn.functional.one_hot(truth, native.shape[1]).float()
                brier = (native-truth_onehot).square().sum(1)-(own-truth_onehot).square().sum(1)
                gain = native.gather(1, truth[:, None]).squeeze(1)-own.gather(1, truth[:, None]).squeeze(1)
                g = groups(e)
                mass_bin = torch.bucketize(e['m'], torch.tensor([.25, .5, .75], dtype=torch.float64))
                masks = [('all', torch.ones(len(z), dtype=torch.bool)), ('original_clean', z == truth), ('original_corrupted', z != truth)]
                masks += [(f'group_{v}', g == v) for v in range(3)]
                masks += [(f'frequency_{v}', class_group[z] == v) for v in range(3)]
                masks += [(f'mass_bin_{v}', mass_bin == v) for v in range(4)]
                for stratum, mask in masks:
                    if not mask.any():
                        continue
                    maxima = native.max(1).values
                    strict = (native == maxima[:, None]).sum(1) == 1
                    diagnostics.append(dict(dataset=name, condition=cond, partition=str(label), stratum=stratum,
                        count=int(mask.sum()), mean_support=float(e['support'][mask].mean()), mass=float(e['m'][mask].mean()),
                        q_observed=float(e['qz'][mask].mean()), alpha=float(e['alpha'][mask].mean()),
                        raw_vote_max=float(e['maximum'][mask].mean()), raw_vote_margin=float(e['margin'][mask].mean()),
                        target_entropy=float((-(native.clamp_min(1e-30)*native.clamp_min(1e-30).log()).sum(1))[mask].mean()),
                        brier_change=float(brier[mask].mean()), true_class_mass_change=float(gain[mask].mean()),
                        true_mass_increased=float((gain[mask] > 1e-8).double().mean()),
                        true_mass_decreased=float((gain[mask] < -1e-8).double().mean()),
                        strict_wrong_target=float(((native.argmax(1) != truth) & strict)[mask].double().mean())))
    results = []
    for r in registry:
        if r['stage'].endswith('_REF'):
            continue
        results.append(dict(r, seed=0, seed_std=None, **by_folder[r['folder']]))
    csv_save(ROOT/'results.csv', results)
    save(ROOT/'scored_folders.json', by_folder)
    csv_save(ROOT/'per_class_recall.csv', class_rows)
    csv_save(ROOT/'node_diagnostics.csv', diagnostics)
    selection_rows = []

    def add(stage, name, backbone, cond, selector, folder):
        selection_rows.append(dict(stage=stage, dataset=name, backbone=backbone, condition=cond,
            selector=selector, folder=folder, seed=0, seed_std=None, **by_folder[folder]))

    def fitpath(stage, name, backbone, cond, arm):
        return f'fits/{stage}/{name}/{backbone}/{cond}/{arm}'

    for r in results:
        if r['stage'] in ['A', 'B2', 'D2']:
            add(r['stage'], r['dataset'], r['backbone'], r['condition'], r['arm'], r['folder'])
    for choice in read(ROOT/'selections/B1.json')['choices']:
        name, b, family = choice['dataset'], choice['backbone'], choice['family']
        for cond in CONFIG['conditions']:
            for policy in ['BALANCED', 'CONSTRAINED']:
                param = choice[policy]
                arm = f'GCE_Q{param}' if family == 'GCE' else ('CE' if param == 0 else f'{family}_L{float(param)}')
                add('B1', name, b, cond, family+'_'+policy, fitpath('B1', name, b, cond, arm))
            if family != 'GCE':
                add('B1', name, b, cond, family+'_L1', fitpath('B1', name, b, cond, family+'_L1.0'))
        for cond in CONFIG['conditions']:
            if family == 'SUPPORT':
                add('B1', name, b, cond, 'CE', fitpath('B1', name, b, cond, 'CE'))
    regret_rows = []
    for p in (ROOT/'selections/C1').rglob('*_full.json'):
        name, b = p.parts[-3:-1]
        cond = p.stem[:-5]
        full, probe = read(p), read(p.with_name(cond+'.json'))
        ce_stage = 'B1' if name in CONFIG['datasets_mechanism'] else 'C1_ANCHOR'
        ce = fitpath(ce_stage, name, b, cond, 'CE')
        def resolve(arm):
            return ce if arm == 'CE' else fitpath('C1', name, b, cond, arm)
        choices = {k: full[k] for k in ['FULL_VAL', 'ONLY_K', 'ONLY_STRENGTH', 'FIXED_S32', 'SC']}
        choices.update(PROBE=probe['PROBE'], CE='CE')
        for label, arm in choices.items():
            add('C1', name, b, cond, label, resolve(arm))
        add('C1', name, b, cond, 'GCE_VAL', fitpath(ce_stage, name, b, cond, f'GCE_Q{full["GCE_Q"]}'))
        if full['LEGACY_SUPPORT32']:
            add('C1', name, b, cond, 'LEGACY_SUPPORT32', full['LEGACY_SUPPORT32'])
        ps, fs = by_folder[resolve(probe['PROBE'])], by_folder[resolve(full['FULL_VAL'])]
        regret_rows.append(dict(dataset=name, backbone=b, condition=cond, PROBE=probe['PROBE'], FULL_VAL=full['FULL_VAL'],
             validation_gap_pp=100*(ps['validation']-fs['validation']), test_gap_pp=100*(ps['test_accuracy']-fs['test_accuracy']),
             candidate_updates=1000, ce_updates=200, full_grid_candidate_updates=3600,
             prefix25_survivors=probe['rounds'][0]['survivors'], prefix75_survivors=probe['rounds'][1]['survivors']))
    for c in read(ROOT/'selections/C2.json')['choices']:
        n, cond, arm = c['dataset'], c['condition'], c['selected']
        ce_stage = 'B1' if n in CONFIG['datasets_mechanism'] else 'C1_ANCHOR'
        folder = fitpath(ce_stage if arm == 'CE' else 'C1', n, 'R0', cond, arm)
        add('C2', n, 'R0', cond, c['selector'], folder)
    for c in read(ROOT/'selections/D1.json')['choices']:
        n, b, cond = c['dataset'], c['backbone'], c['condition']
        for label, arm in [('GROUP_SELECTED', c['selected']), ('EXACT_MASS_GLOBAL', 'EXACT_MASS_GLOBAL'), ('MASS_MATCHED_SHUFFLE', 'MASS_MATCHED_SHUFFLE')]:
            add('D1', n, b, cond, label, fitpath('D1', n, b, cond, arm))
        gl = c['best_cell_global_lambda']
        for label, arm in [('CE', 'CE'), ('CELL_GLOBAL_VAL', 'CE' if gl == 0 else f'SUPPORT_L{float(gl)}')]:
            add('D1', n, b, cond, label, fitpath('B1', n, b, cond, arm))
    csv_save(ROOT/'selected_results.csv', selection_rows)
    csv_save(ROOT/'c1_regret.csv', regret_rows)
    summary = []
    cells = defaultdict(list)
    for r in selection_rows:
        cells[(r['stage'], r['backbone'], r['selector'])].append(r)
    for (stage, b, selector), rows in cells.items():
        for condition in ['all', 'noisy', 'clean', 'uniform0.4', 'uniform0.6', 'pair0.4']:
            use = [r for r in rows if condition == 'all' or (r['condition'] != 'clean' if condition == 'noisy' else r['condition'] == condition)]
            if not use:
                continue
            summary.append(dict(stage=stage, backbone=b, selector=selector, condition=condition, cells=len(use), seed_count=1, seed_std=None,
                test_accuracy=100*np.mean([r['test_accuracy'] for r in use]),
                test_macro_f1=100*np.mean([r['test_macro_f1'] for r in use]),
                validation=100*np.mean([r['validation'] for r in use])))
    csv_save(ROOT/'summary.csv', summary)
    comparisons = {
        'A': [('SUPPORT', v) for v in ['CE', 'GCE', 'KM']],
        'B1': [('SUPPORT_BALANCED', v) for v in ['CE', 'SUPPORT_L1', 'KM_BALANCED', 'MASS_BALANCED', 'GCE_BALANCED']],
        'B2': [(a, 'N32') for a in ['N8', 'N128', 'D8', 'M8', 'DM8', 'D128', 'M128', 'DM128']],
        'C1': [('PROBE', v) for v in ['FULL_VAL', 'CE', 'FIXED_S32', 'LEGACY_SUPPORT32', 'SC', 'GCE_VAL']],
        'C2': [(s, 'PROBE') for s in ['STATS_RIDGE', 'STATS_MLP', 'ENHANCED_RIDGE', 'ENHANCED_MLP']],
        'D1': [('GROUP_SELECTED', v) for v in ['CE', 'CELL_GLOBAL_VAL', 'EXACT_MASS_GLOBAL', 'MASS_MATCHED_SHUFFLE']],
        'D2': [('GATE_LCSS', v) for v in ['TRAIN_CE', 'CLEAN_META_CE', 'CLEAN_META_GCE', 'GLOBAL_LCSS', 'GATE_MASS', 'GATE_KM', 'MEAN_LCSS', 'SHUFFLE_LCSS']]}
    lookup = {(r['stage'], r['dataset'], r['backbone'], r['condition'], r['selector']): r for r in selection_rows}
    differences, warnings = [], []
    for stage, pairs in comparisons.items():
        for a, b in pairs:
            for r in selection_rows:
                if r['stage'] != stage or r['selector'] != a:
                    continue
                refstage = 'C1' if stage == 'C2' else stage
                ref = lookup.get((refstage, r['dataset'], r['backbone'], r['condition'], b))
                if ref is None:
                    continue
                row = dict(stage=stage, dataset=r['dataset'], backbone=r['backbone'], condition=r['condition'],
                           method=a, reference=b, accuracy_pp=100*(r['test_accuracy']-ref['test_accuracy']),
                           macro_f1_pp=100*(r['test_macro_f1']-ref['test_macro_f1']),
                           validation_pp=100*(r['validation']-ref['validation']), seed=0, seed_std=None)
                differences.append(row)
                if (r['condition'] == 'clean' and row['accuracy_pp'] < -2) or row['macro_f1_pp'] < -1:
                    warnings.append(row)
    csv_save(ROOT/'paired_differences.csv', differences)
    csv_save(ROOT/'clean_macro_f1_warnings.csv', warnings)
    interactions = []
    for name in CONFIG['datasets_mechanism']:
        for b in ['R0', 'R2']:
            for cond in CONFIG['conditions']:
                for s in [8, 128]:
                    get = lambda arm: lookup[('B2', name, b, cond, arm)]
                    interactions.append(dict(dataset=name, backbone=b, condition=cond, support=s,
                        accuracy_interaction_pp=100*(get('DM'+str(s))['test_accuracy']-get('D'+str(s))['test_accuracy']-get('M'+str(s))['test_accuracy']+get('N32')['test_accuracy'])))
    csv_save(ROOT/'b2_interactions.csv', interactions)
    costs = [dict(folder=p, epoch=result(p)['epoch'], seconds=result(p)['seconds'],
                  virtual_seconds=result(p)['virtual_seconds'], **{k: result(p)[k] for k in ['backbone', 'dataset', 'condition']}) for p in unique]
    csv_save(ROOT/'costs.csv', costs)
    bridge_rows = []
    for name in CONFIG['datasets_mechanism']:
        for cond in CONFIG['conditions']:
            for arm, newarm in [('CE', 'CE'), ('SUPPORT', 'SUPPORT_L1.0'), ('KM', 'KM_L1.0')]:
                old = EXP/'r2_body_comparison_20261023/fits'/name/'seed0'/cond/arm
                new = ROOT/fitpath('B1', name, 'R2', cond, newarm)
                if (old/'probabilities.pt').exists():
                    oldmeta, newmeta = read(old/'result.json'), read(new/'result.json')
                    diff = float((load(old/'probabilities.pt')-load(new/'probabilities.pt')).abs().max())
                    bridge_rows.append(dict(dataset=name, condition=cond, arm=arm, max_probability_difference=diff,
                         selected_epoch_old=oldmeta['best_epoch'], selected_epoch_new=newmeta['best_epoch'],
                         old_valid_correct=oldmeta['valid_correct'], new_valid_correct=newmeta['valid_correct']))
    csv_save(ROOT/'historical_probability_bridges.csv', bridge_rows)
    decisions = {}
    for stage in comparisons:
        primary = [r for r in differences if r['stage'] == stage and r['condition'] != 'clean']
        agg = defaultdict(list)
        for r in primary:
            agg[(r['backbone'], r['method'], r['reference'])].append(r)
        decisions[stage] = [dict(backbone=b, method=a, reference=ref, cells=len(rs),
             mean_accuracy_pp=float(np.mean([r['accuracy_pp'] for r in rs])),
             mean_validation_pp=float(np.mean([r['validation_pp'] for r in rs])),
             mean_macro_f1_pp=float(np.mean([r['macro_f1_pp'] for r in rs])),
             wins=sum(r['accuracy_pp'] > 1e-9 for r in rs), losses=sum(r['accuracy_pp'] < -1e-9 for r in rs))
             for (b, a, ref), rs in agg.items()]
    save(ROOT/'decision.json', dict(seed=0, seed_std=None, exploration_only=True, automatic_expansion=False, stages=decisions))
    lines = ['# GB-LCSS 支持量与监督强度首轮实验报告', '',
        '## 执行与协议', '',
        f'完成时间：{now()}。全部预注册阶段A/B1/B2/C1/C2/D1/D2完成；分类唯一拟合{len(unique)}，CPU预测器20，工程冒烟5。',
        '协议/学生seed均为0；GRACE、R2与SimMLP使用固定历史预训练。seed标准差留空。所有选择先冻结，再统一读取测试真值。',
        '五图为Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；机制三图为Cora、CoraFull、Amazon-Photo。四条件全部保留。',
        'R0：原始64隐藏MLP、dropout/BN正常训练；R2：历史100轮编码器及BN固定、线性头Adam；SimMLP：作者固定编码器/线性头AdamW。分类200轮、首次严格最佳验证检查点。',
        '粒球用于共享留一类别统计，beta=.5、最少2票、并列不强化；历史SUPPORT32与深树S32分别存储。', '',
        '## 假设与改动', '',
        'A检查固定SimMLP上的监督增量。B1固定历史分区扫描全局强度与GCE，BALANCED为四条件共同参数、.2pp近优取较小值；旧约束为补充表。',
        'B2在深树8/32/128的共同可识别集合上分解方向、实际转移量与交互。C1用18→6→2真实前缀筛选支持量/强度，CE可回退；C2测试跨整图泛化。',
        'D1比较同意/冲突分组力度与精确sum(m*lambda)全局/打乱控制。D2使用validation内部meta-fit/selection，9输入线性sigmoid门控及精确可微一步虚拟SGD，真实头Adam、目标stopgrad。',
        '源码只新增本实验目录；训练入口以明确白名单排除测试/干净训练真值。D2所有对照重跑同子协议，干净meta标签预算通过直接CE/GCE对照评估。', '',
        '## 成功与失败判读', '',
        '首轮预设探索参考：带噪验证约+.5pp或至少两预列单元+.5pp且代价有限；clean或macro-F1约1pp改善也可作为条件性信号。约±.3pp列为待判断。',
        '对同协议简单控制多数退化，或收益主要依赖额外清洁标签且低于直接监督控制，则暂缓。clean下降>2pp、macro-F1下降>1pp见单独警示表。',
        '本轮未扩种子、图、模块组合，也未修改稿件或远端。', '']
    for stage in ['A', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']:
        lines += [f'## {stage} 结果', '', '| 骨干 | 方法 | 参照 | 带噪测试Δpp | 验证Δpp | F1Δpp | 胜/负/格 |',
                  '|---|---|---|---:|---:|---:|---|']
        for r in decisions[stage]:
            lines.append(f'|{r["backbone"]}|{r["method"]}|{r["reference"]}|{r["mean_accuracy_pp"]:+.3f}|{r["mean_validation_pp"]:+.3f}|{r["mean_macro_f1_pp"]:+.3f}|{r["wins"]}/{r["losses"]}/{r["cells"]}|')
        lines += ['']
    lines += ['## C1 成本与最差单格', '',
        '每格PROBE为1000次候选更新加CE200次；完整网格为3600次候选更新。此次真实执行了完整网格，节省预算仅描述冻结策略的候选更新，实际成本包含审计补训。',
        'B2完整200轮结果未用于C1筛选或候选复用。C2 STATS使用可观测统计；ENHANCED额外使用5/10/25轮验证信息，需18×25=450次候选更新/格。',
        'CiteSeer/CS的历史SUPPORT32 R0对照未列入额外预算，该两图在C1使用新树固定S32；历史SUPPORT32比较覆盖机制三图R0/R2。', '']
    for b in ['R0', 'R2']:
        rows = [r for r in regret_rows if r['backbone'] == b]
        gaps = np.asarray([r['test_gap_pp'] for r in rows])
        vgap = np.asarray([r['validation_gap_pp'] for r in rows])
        worst = min(rows, key=lambda r: r['test_gap_pp'])
        lines += [f'{b} PROBE−FULL_VAL：平均测试{gaps.mean():+.3f}pp，验证{vgap.mean():+.3f}pp；测试损失≤0.5/1/2pp比例分别为'+
                  '/'.join(f'{100*np.mean(gaps >= -threshold):.1f}%' for threshold in [.5, 1, 2])+f'；最差{worst["dataset"]}/{worst["condition"]} {worst["test_gap_pp"]:+.3f}pp。', '']
    lines += ['## 负结果与代价', '', f'警示表共{len(warnings)}条比较，详见clean_macro_f1_warnings.csv。下列为按clean准确率损害排序的前12条：', '']
    for r in sorted([w for w in warnings if w['condition'] == 'clean'], key=lambda r: r['accuracy_pp'])[:12]:
        lines.append(f'- {r["stage"]}/{r["dataset"]}/{r["backbone"]} {r["method"]} 对 {r["reference"]}：accuracy {r["accuracy_pp"]:+.3f}pp，F1 {r["macro_f1_pp"]:+.3f}pp。')
    lines += ['', '## 证据分类与最终决策', '',
        '已被本次实验支持的结论：上表为完整seed0配对结果；选择、概率和检查点均可追溯。各方法的收益/代价应结合分图四条件表，优先用同协议控制判断。',
        '合理解释：支持量、类别方向、目标幅度与学生表示可能共同影响泛化；B2交互及D1/D2精确均量控制为进一步区分提供线索。',
        '尚未验证：独立seed重复、独立SSL、新图迁移及C/D组合。五图留一只提供五个迁移任务。',
        '被负结果削弱的具体对比见各阶段负Δpp与警示表，继续保留完整结果。最终决策：本轮结束；有条件性信号的最简规格可建议小规模配对seed确认，当前不自动续跑。', '',
        '## 追溯与工程记录', '',
        f'分类拟合累计墙钟（逐fit求和）{sum(r["seconds"] for r in costs):.2f}秒；D2虚拟更新累计{sum(r["virtual_seconds"] for r in costs):.2f}秒；初始化/分区输入读取及后处理另计，环境事件时间见events.jsonl。',
        'sources.json/inputs.json保存源与输入SHA256；每臂target、原始日志、优化器/RNG恢复状态、checkpoint和全节点probabilities均保留。选择冻结时间/哈希与完整网格时序见selections及selections_frozen.json。',
        '测试最初缺少实现为预期红灯；一处测试float32/float64断言类型修复保留在testing记录。任何运行故障见failures.jsonl；历史数值桥见evidence/*/*/bridge.json和historical_probability_bridges.csv。',
        '独立本地audit见audit.json；主会话另做只读抽样复核。']
    output(ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(ROOT/'analysis_complete.json', dict(finished=now(), fits_scored=len(unique), result_rows=len(results), selected_rows=len(selection_rows)))
