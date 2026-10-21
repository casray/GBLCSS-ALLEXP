"""Recompute frozen artifacts, paired differences and the complete Chinese report."""
import json
import inspect
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import run as r

ROOT = r.ROOT


def independent_ball_target(groups, observed, classes):
    _, ids = np.unique(groups, return_inverse=True)
    counts = np.bincount(ids)
    hist = np.zeros((len(counts), classes))
    np.add.at(hist, (ids, observed), 1)
    support = counts[ids] - 1
    onehot = np.eye(classes)[observed]
    dist = np.full(onehot.shape, 1 / classes)
    valid = support > 0
    dist[valid] = (hist[ids[valid]] - onehot[valid]) / support[valid, None]
    maximum = dist.max(1)
    q = dist.copy()
    unique = (np.abs(dist - maximum[:, None]) <= 1e-12).sum(1) == 1
    q[unique] = .5 * dist[unique] + .5 * np.eye(classes)[dist[unique].argmax(1)]
    alpha = np.clip((classes * maximum - 1) / (classes - 1), 0, 1)
    alpha[support < 2] = 0
    return dict(r=dist, q=q, alpha=alpha, support=support,
                target=(1-alpha[:, None])*onehot + alpha[:, None]*q)


def paired(frame, metric):
    records = []
    lookup = {(x.dataset, x.seed, x.noise, x.arm): getattr(x, metric) for x in frame.itertuples()}
    for x in frame.itertuples():
        references = ['CE', 'GRACE_OLD64', 'GRACE_ADAPT64', 'GRACE_SQRT16', 'GREET_OLD64',
                      'GREET_KM_MATCH', 'GREET_KNN_MATCH', 'GRACE_KM_MATCH', 'GRACE_KNN_MATCH']
        for base in references:
            if base == x.arm:
                continue
            other = lookup[(x.dataset, x.seed, x.noise, base)]
            records.append(dict(dataset=x.dataset, seed=x.seed, noise=x.noise, arm=x.arm,
                                reference=base, delta=(getattr(x, metric)-other)*(100 if metric == 'test_accuracy' else 1)))
    return pd.DataFrame(records)


def audit_and_report():
    r.verify_sources()
    errors = []; protocols = artifacts = students = partitions = 0
    historical_checks = []
    assert list(inspect.signature(r.build_partition).parameters) == ['h', 'train', 'budget', 'seed']
    target_frame = pd.read_csv(ROOT / 'target_results.csv')
    student_frame = pd.read_csv(ROOT / 'student_results.csv')
    for name in r.CONFIG['datasets']:
        for seed in r.CONFIG['protocol_seeds']:
            graph, _ = r.load_case(name, seed, 'clean'); train = graph.train_index.numpy()
            assignments = {}; knn = {}
            for space in r.CONFIG['spaces']:
                folder = ROOT / 'partitions' / name / f'seed{seed}' / space
                with np.load(folder / 'assignments.npz') as a:
                    assignments[space] = {key: a[key].copy() for key in a.files}
                stats = json.loads((folder / 'stats.json').read_text(encoding='utf-8'))
                frozen = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
                assert r.d.sha(folder/'assignments.npz') == frozen['assignment_sha256']
                assert r.d.sha(folder/'knn.npz') == frozen['knn_sha256']
                assert r.d.sha(r.H_PATHS[space][name]) == frozen['h_sha256']
                for rule, groups in assignments[space].items():
                    assert groups.shape == (graph.num_nodes,)
                    np.testing.assert_array_equal(np.unique(groups), np.arange(len(np.unique(groups))))
                    if rule in ['ADAPT64', 'SQRT16']:
                        summary = stats[rule]
                        assert len(np.unique(groups)) <= summary['feasible_target'] <= summary['requested_target']
                        assert np.bincount(groups[train], minlength=summary['num_balls']).min() >= 16
                        trace = json.loads((folder / (rule+'_trace.json')).read_text(encoding='utf-8'))
                        accepted = [x for x in trace if x['accepted']]
                        assert len(accepted) == summary['num_balls']-1
                        assert all(x['sse_gain'] > 0 and min(x['child_train_counts']) >= 16 for x in accepted)
                        assert all(min(x['child_sizes']) >= summary['min_leaf_size'] for x in accepted)
                assert len(np.unique(assignments[space]['KM_MATCH'])) == len(np.unique(assignments[space]['SQRT16']))
                w = sparse.load_npz(folder / 'knn.npz'); knn[space] = w
                count = np.bincount(assignments[space]['SQRT16'][train])[assignments[space]['SQRT16'][train]]-1
                assert not w.diagonal().any()
                np.testing.assert_array_equal(np.diff(w.indptr), count)
                partitions += 1
            for condition in r.CONFIG['conditions']:
                graph, protocol = r.load_case(name, seed, condition)
                assert protocol['val_clean'] and protocol['test_clean']
                folder = ROOT / 'targets' / name / f'seed{seed}' / condition
                assert json.loads((folder/'protocol.json').read_text(encoding='utf-8')) == protocol
                frozen = json.loads((folder/'frozen_targets.json').read_text(encoding='utf-8'))
                observed = graph.noisy_y[train].numpy(); classes = graph.num_classes
                truth = graph.y[train].numpy(); onehot = np.eye(classes)[observed]
                np.testing.assert_array_equal(observed, np.load(folder/'observed_train.npy'))
                for arm in r.ARMS:
                    file = folder/(arm+'.npz')
                    assert r.d.sha(file) == frozen[arm]
                    with np.load(file) as a:
                        saved = {key:a[key].copy() for key in a.files}
                    if arm == 'CE':
                        expected = dict(target=onehot, q=onehot, r=onehot, alpha=np.zeros(len(train)),
                                        support=np.zeros(len(train)))
                    else:
                        space, rule = arm.split('_', 1)
                        if rule == 'KNN_MATCH':
                            # Direct sparse class histograms, independent of the production vote routine.
                            w = knn[space]; count = np.diff(w.indptr); hist = w @ onehot
                            dist = hist / count[:, None]
                            maxval = dist.max(1); q = dist.copy()
                            unique = (np.abs(dist-maxval[:, None])<=1e-12).sum(1)==1
                            q[unique] = .5*dist[unique]+.5*np.eye(classes)[dist[unique].argmax(1)]
                            alpha = np.clip((classes*maxval-1)/(classes-1),0,1); alpha[count<2]=0
                            expected = dict(r=dist,q=q,alpha=alpha,support=count,
                                            target=(1-alpha[:,None])*onehot+alpha[:,None]*q)
                        else:
                            expected = independent_ball_target(assignments[space][rule][train], observed, classes)
                    for key in expected:
                        np.testing.assert_allclose(saved[key], expected[key], atol=2e-12, rtol=0)
                    np.testing.assert_allclose(saved['target'].sum(1), 1, atol=1e-12)
                    assert saved['target'].min() >= -1e-12 and saved['target'].max() <= 1+1e-12
                    row = target_frame[(target_frame.dataset==name)&(target_frame.seed==seed)&
                                       (target_frame.noise==condition)&(target_frame.arm==arm)].iloc[0]
                    brier = np.square(saved['target']-np.eye(classes)[truth]).sum(1).mean()
                    assert abs(brier-row.target_brier) < 2e-12
                    if arm == 'GRACE_OLD64':
                        old = r.MECHANISM/'runs'/name/'grace'/f'seed{seed}'/condition/'GB64.npz'
                        with np.load(old) as prior:
                            np.testing.assert_allclose(saved['target'], prior['target'], atol=2e-6, rtol=0)
                    if arm == 'GREET_OLD64':
                        old = r.PREVIOUS/'targets'/name/'A'/f'seed{seed}'/condition/'GREET_GB.npz'
                        with np.load(old) as prior:
                            np.testing.assert_allclose(saved['target'], prior['target'], atol=2e-6, rtol=0)
                    artifacts += 1
                    student = ROOT/'students'/name/f'seed{seed}'/condition/arm
                    config = json.loads((student/'config.json').read_text(encoding='utf-8'))
                    result = json.loads((student/'result.json').read_text(encoding='utf-8'))
                    assert config['training'] == r.CONFIG['training'] and config['seed'] == seed
                    assert config['target_sha256'] == result['target_sha256'] == frozen[arm]
                    assert config['split_hash'] == protocol['split_hash'] and config['noise_hash'] == protocol['noise_hash']
                    logits = torch.load(student/'logits.pt', weights_only=True)
                    assert logits.shape == (graph.num_nodes, classes) and torch.isfinite(logits).all()
                    for key, index, labels in [('test_accuracy',graph.test_index,graph.y),
                                               ('validation_accuracy',graph.val_index,graph.noisy_y),
                                               ('train_accuracy',graph.train_index,graph.noisy_y)]:
                        accuracy = float((logits[index].argmax(1)==labels[index]).float().mean())
                        assert abs(accuracy-result[key]) < 1e-7
                    saved_row = student_frame[(student_frame.dataset==name)&(student_frame.seed==seed)&
                        (student_frame.noise==condition)&(student_frame.arm==arm)].iloc[0]
                    assert abs(saved_row.test_accuracy-result['test_accuracy']) < 1e-12
                    assert 1 <= result['best_epoch'] <= r.CONFIG['training']['epochs']
                    if arm in ['CE', 'GRACE_OLD64']:
                        historic_arm = 'CE' if arm=='CE' else 'GB64'
                        source = (ROOT.parent/'heterophily_original_baseline_20261014'/'students'/name/f'seed{seed}'/condition/historic_arm
                                  if name=='roman' else r.MECHANISM/'students'/name/'grace'/f'seed{seed}'/condition/historic_arm)
                        if (source/'result.json').exists():
                            old_score = json.loads((source/'result.json').read_text(encoding='utf-8'))['test_accuracy']
                            delta = 100*(result['test_accuracy']-old_score)
                            historical_checks.append(dict(dataset=name, seed=seed, noise=condition, arm=arm, accuracy_delta_pp=delta))
                            assert abs(delta) < 1e-5
                    students += 1
                protocols += 1
    assert protocols == r.CONFIG['expected_target_protocols']
    assert artifacts == students == len(target_frame) == len(student_frame) == r.CONFIG['expected_targets_and_students']
    r.verify_sources()
    audit = dict(status='passed', partition_protocols=partitions, target_protocols=protocols,
                 target_artifacts=artifacts, student_logits_checked=students,
                 historical_student_checks=len(historical_checks), errors=errors)
    r.write(ROOT/'audit.json',audit)
    r.write(ROOT/'historical_student_checks.json',historical_checks)
    tpair = paired(target_frame,'target_brier'); spair = paired(student_frame,'test_accuracy')
    for kind, frame in [('target',tpair),('student',spair)]:
        frame.to_csv(ROOT/(kind+'_paired_deltas.csv'),index=False,encoding='utf-8-sig')
        frame.groupby(['dataset','noise','arm','reference']).delta.agg(['count','mean','std']).reset_index().to_csv(
            ROOT/(kind+'_paired_summary.csv'),index=False,encoding='utf-8-sig')
    report(target_frame, student_frame, tpair, spair, audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


def delta_values(pairs, name, arm, reference, noise=None):
    chosen = pairs[(pairs.dataset==name)&(pairs.arm==arm)&(pairs.reference==reference)]
    return chosen[chosen.noise==noise].delta if noise else chosen[chosen.noise!='clean'].delta


def report(targets, students, tp, sp, audit):
    lines = ['# GREET / GRACE 自适应粒球重实验结论','',
             '2026-10-15。新规则及全部对照预先固定，结果由冻结目标与保存logits复核。', '',
             '## 实际完成与可比范围','',
             f"三图×三个协议seed×三条件×11臂，{audit['target_artifacts']}目标、{audit['student_logits_checked']}次200轮X-only MLP；独立审计通过。{audit['partition_protocols']}套分区协议、{audit['target_protocols']}套标签协议；{audit['historical_student_checks']}项可获得的历史学生准确率重合。", '',
             '复用同一SSL seed0的冻结H。三seed联合改变划分/人工噪声/分裂与MLP初始化，不等于三独立SSL训练。GREET portable MC 400epoch与旧GRACE预算不同，属于已有空间的受控下游比较。只有训练集合位置参与分球，不用类别；验证/测试标签不入目标。', '',
             '## 数量与支持','',
             '| 图 | 空间 | 规则 | 实际球数范围 | 每球最少训练支持范围 |',
             '|---|---|---|---:|---:|']
    count_rows = []
    for name in r.CONFIG['datasets']:
        for space in r.CONFIG['spaces']:
            for rule in ['OLD64','ADAPT64','SQRT16','KM_MATCH']:
                stat = [json.loads((ROOT/'partitions'/name/f'seed{s}'/space/'stats.json').read_text(encoding='utf-8'))[rule]
                        for s in r.CONFIG['protocol_seeds']]
                counts = [x['actual_balls'] for x in stat]; supports = [x['min_train_support'] for x in stat]
                lines.append(f'| {name} | {space} | {rule} | {min(counts)}–{max(counts)} | {min(supports)}–{max(supports)} |')
                count_rows.append(dict(dataset=name,space=space,rule=rule,min_balls=min(counts),max_balls=max(counts),
                                       min_support=min(supports),max_support=max(supports)))
    pd.DataFrame(count_rows).to_csv(ROOT/'ball_count_summary.csv',index=False,encoding='utf-8-sig')
    lines += ['', 'SQRT16预算为ceil(sqrt(N))，支持16与几何可行性会限制数量；SSE优先/投影fallback/float64等共同变化。数量达标并不证明语义正确；KM仅匹配数量，kNN逐训练节点匹配唯一其他投票数。', '',
              '## 分类准确率：三seed均值±样本标准差，%','',
              '| 图 | 条件 | '+' | '.join(r.ARMS)+' |',
              '|---|---|'+'---:|'*len(r.ARMS)]
    for name in r.CONFIG['datasets']:
        for noise in r.CONFIG['conditions']:
            values=[]
            for arm in r.ARMS:
                v=students[(students.dataset==name)&(students.noise==noise)&(students.arm==arm)].test_accuracy*100
                values.append(f'{v.mean():.2f}±{v.std(ddof=1):.2f}')
            lines.append('| '+name+' | '+noise+' | '+' | '.join(values)+' |')
    lines += ['', '## 主规则的配对差','',
              'Brier差为新−参照，负数有利；准确率差为新−参照百分点，正数有利。每个条件n=3；两种噪声合并的6配对共享3个协议seed，不能视为6次完全独立实验。', '',
              '| 图 | 候选 | 参照 | 条件 | Brier差均值±std | 准确率差均值±std，pp |',
              '|---|---|---|---|---:|---:|']
    comparison=[('GRACE_SQRT16','GRACE_OLD64'),('GREET_SQRT16','GREET_OLD64'),
                ('GREET_ADAPT64','GRACE_ADAPT64'),('GREET_SQRT16','GRACE_SQRT16'),
                ('GREET_SQRT16','GRACE_OLD64'),('GREET_SQRT16','GREET_KM_MATCH'),
                ('GREET_SQRT16','GREET_KNN_MATCH')]
    for name in r.CONFIG['datasets']:
        for arm,base in comparison:
            for noise in r.CONFIG['conditions']:
                a=delta_values(tp,name,arm,base,noise);b=delta_values(sp,name,arm,base,noise)
                lines.append(f'| {name} | {arm} | {base} | {noise} | {a.mean():+.6f}±{a.std(ddof=1):.6f} | {b.mean():+.3f}±{b.std(ddof=1):.3f} |')
    decisions=[]
    lines += ['', '## 预设推进判据逐项判断','',
              '带噪Brier平均≤−.005、≥5/6配对改善；准确率平均≥+.5pp、≥5/6改善；clean退化不超过1pp。它是研究预算筛查，不是统计显著性结论。', '',
              '| 图 | 候选 | 参照 | 带噪Brier差 | 带噪准确率差，pp | 正向准确率配对 | clean准确率差，pp | 筛查 |',
              '|---|---|---|---:|---:|---:|---:|---|']
    for name in r.CONFIG['datasets']:
        for arm,base in [('GRACE_SQRT16','GRACE_OLD64'),('GREET_SQRT16','GRACE_OLD64'),
                         ('GREET_SQRT16','GRACE_SQRT16'),('GREET_SQRT16','GREET_KM_MATCH'),('GREET_SQRT16','GREET_KNN_MATCH')]:
            a=delta_values(tp,name,arm,base);b=delta_values(sp,name,arm,base);c=delta_values(sp,name,arm,base,'clean')
            passed=bool(a.mean()<=-.005 and (a<0).sum()>=5 and b.mean()>=.5 and (b>0).sum()>=5 and c.mean()>=-1)
            item=dict(dataset=name,arm=arm,reference=base,noisy_brier_delta=float(a.mean()),
                      noisy_accuracy_delta_pp=float(b.mean()),positive_accuracy_pairs=int((b>0).sum()),
                      clean_accuracy_delta_pp=float(c.mean()),screen_pass=passed)
            decisions.append(item)
            lines.append(f'| {name} | {arm} | {base} | {a.mean():+.6f} | {b.mean():+.3f} | {int((b>0).sum())}/6 | {c.mean():+.3f} | {passed} |')
    primary=[x for x in decisions if x['arm']=='GREET_SQRT16' and x['reference']=='GRACE_SQRT16']
    final='modify' if any(x['screen_pass'] for x in primary) else 'defer'
    r.write(ROOT/'decision.json',dict(status='completed',decision=final,comparisons=decisions,
              statistical_significance_claimed=False,more_training_pending=False))
    lines += ['', '## 结论边界与最终决策','',
              '已支持：新规则的覆盖、数量、训练支持与目标/分类结果均有审计记录；收益和损害应逐图逐噪声阅读上述配对表。固定64对照区分规则变化，sqrt对照再检查预算改变。', '',
              '合理解释但未证明：若细分后投票边际改善，说明全局1球是旧规则的重要限制；若新规则GREET仍落后同规则GRACE，表示空间的类别组织可能仍有问题。不能仅用SSE或球数确定语义原因。', '',
              '尚未验证：独立SSL seed、多计算预算公平预训练、更广异配图与HALF原方法；不把本轮描述成新对比损失成功。匹配kNN/KM更优时，不宣称粒球必要性。', '',
              f'最终决策：**{final}**。只有同规则GREET相对GRACE达到上述筛查才保留局部继续信号；未达标不再按测试集搜索支持数、球数或阈值。本轮所有297学生已结束，无待续队列。', '',
              '复查入口：config.json、PLAN.md、source_manifest.json、raw.log、target_results.csv、student_results.csv、两类paired_summary.csv、ball_count_summary.csv、audit.json。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    audit_and_report()
