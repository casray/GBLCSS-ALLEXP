"""Independent frozen-artifact audits and descriptive ball-count statistics."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_target(groups, observed, classes):
    """Recompute leave-one-out LCSS without importing production vote code."""
    groups = np.asarray(groups); observed = np.asarray(observed)
    if groups.shape != observed.shape or observed.ndim != 1 or classes < 2:
        raise ValueError('invalid target dimensions/classes')
    if np.any(observed < 0) or np.any(observed >= classes):
        raise ValueError('observed label outside class range')
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    hist = np.zeros((len(counts), classes), dtype=np.float64)
    np.add.at(hist, (inverse, observed), 1)
    onehot = np.eye(classes)[observed]
    support = counts[inverse] - 1
    dist = np.full(onehot.shape, 1/classes)
    valid = support > 0
    dist[valid] = (hist[inverse[valid]]-onehot[valid])/support[valid, None]
    maximum = dist.max(axis=1)
    q = dist.copy()
    unique = (np.abs(dist-maximum[:, None]) <= 1e-12).sum(axis=1) == 1
    q[unique] = .5*dist[unique] + .5*np.eye(classes)[dist[unique].argmax(axis=1)]
    alpha = np.clip((classes*maximum-1)/(classes-1), 0, 1)
    alpha[support < 2] = 0
    return dict(r=dist, q=q, alpha=alpha, support=support,
                target=(1-alpha[:, None])*onehot + alpha[:, None]*q)


def validate_nested(assignments):
    """Validate refinement as set membership, independent of numeric ball IDs."""
    previous = None
    for key, value in assignments.items():
        value = np.asarray(value)
        if value.ndim != 1 or not value.size or value.dtype.kind not in 'iu':
            raise ValueError(f'invalid partition {key}')
        if previous is not None:
            if value.shape != previous.shape:
                raise ValueError('nested partitions have different node coverage')
            for leaf in np.unique(value):
                if np.unique(previous[value == leaf]).size != 1:
                    raise ValueError(f'partition {key} is not nested')
        previous = value
    return True


def target_digest(target):
    """Exact student input identity: contiguous float32 bytes."""
    return hashlib.sha256(np.ascontiguousarray(target, dtype=np.float32).tobytes()).hexdigest()


def paired_deltas(frame, metric):
    records = []; scale = 100 if metric in ['test_accuracy', 'test_macro_f1'] else 1
    for (name, seed, noise), part in frame.groupby(['dataset', 'seed', 'noise'], sort=False):
        by_arm = part.set_index('arm')
        for row in part.itertuples():
            references = ['CE']
            if row.space in ['GRACE', 'GREET']:
                references += [row.space+'_B64', row.space+'_OLD64']
            for reference in references:
                if reference == row.arm or reference not in by_arm.index:
                    continue
                records.append(dict(dataset=name, seed=seed, noise=noise, arm=row.arm,
                    reference=reference, comparison='fixed_anchor',
                    actual_balls=row.actual_balls,
                    delta=(getattr(row, metric)-by_arm.loc[reference, metric])*scale))
        fine = part[part.kind == 'B'].sort_values('requested_balls')
        # A capped trajectory repeats the same region partition; keep one per seed.
        fine = fine.drop_duplicates(['space', 'actual_balls'])
        grace = fine[fine.space == 'GRACE'].set_index('actual_balls')
        greet = fine[fine.space == 'GREET'].set_index('actual_balls')
        for count in grace.index.intersection(greet.index):
            a, b = greet.loc[count], grace.loc[count]
            records.append(dict(dataset=name, seed=seed, noise=noise, arm=a.arm,
                reference=b.arm, comparison='matched_actual_count_cross_space', actual_balls=count,
                delta=(a[metric]-b[metric])*scale))
    return pd.DataFrame(records, columns=['dataset', 'seed', 'noise', 'arm', 'reference',
                                         'comparison', 'actual_balls', 'delta'])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_trace(assignments, stats, trace, train, normalized_h):
    """Use final leaves to reconstruct all snapshots and the accepted trace prefix."""
    budgets = sorted(int(key[1:]) for key in assignments if key.startswith('B'))
    ordered = {b: assignments['B'+str(b)] for b in budgets}
    validate_nested(ordered)
    accepted = [row for row in trace if row['accepted']]
    leaves = {0}; ancestry = {}; snapshots = {1: {0}}
    for index, row in enumerate(accepted, 2):
        parent = row['parent_id']; children = row['child_ids']
        check(parent in leaves and len(children) == 2, 'trace parent/children inconsistent')
        check(not any(child in ancestry or child == 0 for child in children), 'trace reuses leaf IDs')
        check(row['actual_count_after'] == index, 'trace count is not split prefix length')
        check(min(row['child_sizes']) >= 16 and min(row['child_train_counts']) >= 16,
              'accepted split violates fixed leaf/support16')
        check(row['sse_gain'] > 0, 'accepted split has nonpositive SSE gain')
        np.testing.assert_allclose(row['parent_sse']-sum(row['child_sse']), row['sse_gain'], rtol=1e-9, atol=1e-9)
        leaves.remove(parent); leaves.update(children)
        ancestry.update({child: parent for child in children}); snapshots[index] = set(leaves)
    final = ordered[budgets[-1]]
    final_ids = np.asarray(sorted(leaves))
    np.testing.assert_array_equal(np.unique(final), np.arange(len(final_ids)))
    check(len(leaves) == len(accepted)+1, 'trace does not account for final leaf count')
    node_leaf = final_ids[final]
    previous_sse = np.inf
    for budget, partition in ordered.items():
        key = 'B'+str(budget); summary = stats[key]
        count = len(np.unique(partition)); active = snapshots[count]
        reconstructed = node_leaf.copy()
        for final_id in final_ids:
            parent = int(final_id)
            while parent not in active:
                parent = ancestry[parent]
            reconstructed[node_leaf == final_id] = parent
        expected_ids = {leaf: i for i, leaf in enumerate(sorted(active))}
        expected = np.array([expected_ids[int(leaf)] for leaf in reconstructed])
        np.testing.assert_array_equal(partition, expected)
        check(count <= budget and count == summary.get('actual_balls', summary.get('num_balls')), 'snapshot count mismatch')
        sizes = np.bincount(partition); supports = np.bincount(partition[train], minlength=count)
        if len(train) >= 16 and len(partition) >= 16:
            check(sizes.min() >= 16 and supports.min() >= 16, 'snapshot violates fixed support/size16')
        sse = sum(float(np.square(z-z.mean(0)).sum()) for leaf in range(count)
                  for z in [normalized_h[partition == leaf]])
        np.testing.assert_allclose(sse, summary['normalized_input_sse'], atol=1e-7, rtol=1e-9)
        check(sse <= previous_sse+1e-7, 'nested SSE increases')
        previous_sse = sse
        np.testing.assert_allclose(supports.min(), summary['min_train_support'])
        np.testing.assert_allclose((supports[partition[train]]-1).mean(), summary['mean_other_votes'])
    # Verify actual split statistics from descendants, independent of trace arithmetic.
    descendants = {leaf: np.flatnonzero(node_leaf == leaf) for leaf in final_ids}
    for row in reversed(accepted):
        children = [descendants[c] for c in row['child_ids']]
        parent_nodes = np.concatenate(children); descendants[row['parent_id']] = parent_nodes
        check([len(x) for x in children] == row['child_sizes'], 'trace child membership size mismatch')
        check([int(np.isin(x, train).sum()) for x in children] == row['child_train_counts'], 'trace support mismatch')
        costs = [float(np.square(normalized_h[x]-normalized_h[x].mean(0)).sum()) for x in children]
        np.testing.assert_allclose(costs, row['child_sse'], rtol=1e-9, atol=1e-7)
    return len(ordered)


def independent_metrics(bundle, truth, observed, clean_bundle=None):
    t, q = bundle['target'], bundle['q']; clean_mask = truth == observed
    prediction = t.argmax(1)
    def masked(value, mask):
        return float(value[mask].mean()) if mask.any() else None
    result = dict(target_brier=float(np.square(t-np.eye(t.shape[1])[truth]).sum(1).mean()),
                  q_accuracy=float((q.argmax(1) == truth).mean()),
                  clean_harm_rate=masked(prediction != truth, clean_mask),
                  noise_recovery_rate=masked(prediction == truth, ~clean_mask),
                  mean_support=float(bundle['support'].mean()), min_support=int(bundle['support'].min()),
                  support_coverage=float((bundle['support'] >= 2).mean()))
    if clean_bundle is not None:
        dist = clean_bundle['r']; other = dist.copy(); other[np.arange(len(truth)), truth] = -np.inf
        margin = dist[np.arange(len(truth)), truth]-other.max(1)
        valid = bundle['support'] >= 2
        result.update(positive_clean_margin_fraction=float((valid & (margin > 0)).mean()),
                      mean_clean_margin=masked(margin, valid))
    return result


def compare_metric(actual, expected, label, tolerance=2e-12):
    if expected is None or pd.isna(expected):
        check(actual is None or pd.isna(actual), label+' missing-value mismatch')
    else:
        check(actual is not None and np.isfinite(actual) and abs(float(actual)-float(expected)) <= tolerance,
              f'{label}: {actual} != {expected}')


def audit_and_report(make_plots=True):
    import data_tools as dt
    import run
    import torch
    from sklearn.metrics import f1_score
    from gblcss.granular_ball_trajectory import build_ball_trajectory
    from gblcss.representation import load_frozen_representation

    root = dt.ROOT; config = dt.CONFIG
    errors = []; history = []; identities = {}; input_models = {}
    counters = dict(partition_protocols=0, snapshots_checked=0, target_protocols=0,
                    target_artifacts=0, student_results_checked=0, training_executed=0)
    def safe(label, function):
        try:
            return function()
        except Exception as error:
            errors.append(dict(scope=label, error=f'{type(error).__name__}: {error}'))
            return None
    safe('source_manifest_before', run.verify_sources)
    check('labels' not in inspect.signature(build_ball_trajectory).parameters,
          'trajectory API must not accept class labels')
    targets = pd.read_csv(root/'target_results.csv', keep_default_na=False)
    students = pd.read_csv(root/'student_results.csv', keep_default_na=False)
    check(not targets.duplicated(['dataset', 'seed', 'noise', 'arm']).any(), 'duplicate target rows')
    check(not students.duplicated(['dataset', 'seed', 'noise', 'arm']).any(), 'duplicate student rows')
    expected_rows = 0
    for name in config['datasets']:
        for seed in config['protocol_seeds']:
            graph, _ = dt.load_case(name, seed, 'clean')
            train = graph.train_index.numpy(); assignments = {}; statistics = {}
            arms = dt.arms_for(graph.num_nodes)
            expected_rows += len(arms)*len(config['conditions'])
            for space in config['spaces']:
                def audit_partition():
                    folder = root/'partitions'/name/f'seed{seed}'/space
                    with np.load(folder/'assignments.npz') as saved:
                        local = {key: saved[key].copy() for key in saved.files}
                    assignments[space] = local; statistics[space] = read_json(folder/'stats.json')
                    frozen = read_json(folder/'complete.json')
                    check(dt.d.sha(folder/'assignments.npz') == frozen['assignment_sha256'], 'assignment hash mismatch')
                    check(dt.d.sha(dt.H_PATHS[space][name]) == frozen['h_sha256'], 'embedding hash mismatch')
                    with np.load(folder/'split.npz') as split:
                        for key, index in [('train', graph.train_index), ('val', graph.val_index), ('test', graph.test_index)]:
                            np.testing.assert_array_equal(split[key], index.numpy())
                    h = load_frozen_representation(dt.H_PATHS[space][name], graph.num_nodes).numpy().astype(np.float64)
                    scale = np.maximum(np.max(np.abs(h), axis=1, keepdims=True), np.finfo(np.float64).tiny)
                    h = h/scale; h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), np.finfo(np.float64).tiny)
                    check(set(local) == {'OLD64'} | {'B'+str(b) for b in dt.budgets_for(graph.num_nodes)}, 'snapshot budget coverage mismatch')
                    for key, partition in local.items():
                        check(partition.shape == (graph.num_nodes,), 'partition lacks complete node coverage')
                        np.testing.assert_array_equal(np.unique(partition), np.arange(len(np.unique(partition))))
                    counters['snapshots_checked'] += verify_trace(local, statistics[space], read_json(folder/'trajectory_trace.json'), train, h)
                    counters['partition_protocols'] += 1
                safe(f'partition/{name}/{seed}/{space}', audit_partition)
            for noise in config['conditions']:
                graph, protocol = dt.load_case(name, seed, noise)
                observed = graph.noisy_y[train].numpy(); truth = graph.y[train].numpy()
                folder = root/'targets'/name/f'seed{seed}'/noise
                def audit_protocol():
                    check(protocol['val_clean'] and protocol['test_clean'], 'evaluation labels changed')
                    check(read_json(folder/'protocol.json') == protocol, 'split/noise protocol mismatch')
                    np.testing.assert_array_equal(np.load(folder/'observed_train.npy'), observed)
                    check(protocol['split_hash'] == dt.d.fingerprint(train, graph.val_index.numpy(), graph.test_index.numpy()), 'ordered split hash mismatch')
                    check(protocol['noise_hash'] == dt.d.fingerprint(observed), 'noise hash mismatch')
                    counters['target_protocols'] += 1
                safe(f'protocol/{name}/{seed}/{noise}', audit_protocol)
                frozen = read_json(folder/'frozen_targets.json')
                for arm in arms:
                    def audit_arm():
                        path = folder/(arm+'.npz')
                        check(dt.d.sha(path) == frozen[arm], 'frozen target file hash mismatch')
                        with np.load(path) as saved:
                            bundle = {key: saved[key].copy() for key in saved.files}
                        onehot = np.eye(graph.num_classes)[observed]; clean_bundle = None
                        if arm == 'CE':
                            expected = dict(r=onehot, q=onehot, alpha=np.zeros(len(train)),
                                            support=np.zeros(len(train)), target=onehot)
                        else:
                            space, rule = arm.split('_', 1)
                            expected = audit_target(assignments[space][rule][train], observed, graph.num_classes)
                            clean_bundle = audit_target(assignments[space][rule][train], truth, graph.num_classes)
                        for key in expected:
                            np.testing.assert_allclose(bundle[key], expected[key], atol=2e-12, rtol=0, err_msg=key)
                        check(np.isfinite(bundle['target']).all(), 'nonfinite target probability')
                        check(bundle['target'].min() >= 0 and bundle['target'].max() <= 1, 'target outside probability simplex')
                        np.testing.assert_allclose(bundle['target'].sum(1), 1, atol=2e-12, rtol=0)
                        row = targets[(targets.dataset == name)&(targets.seed == seed)&(targets.noise == noise)&(targets.arm == arm)].iloc[0]
                        for key, value in independent_metrics(bundle, truth, observed, clean_bundle).items():
                            actual = row[key]; actual = np.nan if actual == '' else actual
                            compare_metric(actual, value, key)
                        counters['target_artifacts'] += 1
                        student_dir = root/'students'/name/f'seed{seed}'/noise/arm
                        cfg = read_json(student_dir/'config.json'); result = read_json(student_dir/'result.json')
                        check(cfg['training'] == config['training'], 'student training config mismatch')
                        check(cfg['protocol'] == protocol, 'student protocol mismatch')
                        check(cfg['target_hash'] == result['target_hash'] == frozen[arm], 'student target hash mismatch')
                        check(cfg['target_semantic_hash'] == result['target_semantic_hash'] == target_digest(bundle['target']),
                              'student float32 target identity mismatch')
                        logit_path = Path(result['logits_path']); check(logit_path.is_absolute(), 'logits path is not absolute')
                        logits = torch.load(logit_path, weights_only=True, map_location='cpu')
                        check(tuple(logits.shape) == (graph.num_nodes, graph.num_classes) and bool(torch.isfinite(logits).all()), 'invalid logits')
                        for key, index in [('test_accuracy', graph.test_index), ('validation_accuracy', graph.val_index), ('train_accuracy', graph.train_index)]:
                            labels = graph.noisy_y[index] if key == 'train_accuracy' else graph.y[index]
                            accuracy = float((logits[index].argmax(1) == labels).float().mean())
                            compare_metric(result[key], accuracy, key, 1e-12)
                        f1 = f1_score(graph.y[graph.test_index].numpy(), logits[graph.test_index].argmax(1).numpy(),
                                      labels=list(range(graph.num_classes)), average='macro', zero_division=0)
                        compare_metric(result['test_macro_f1'], f1, 'test_macro_f1', 1e-12)
                        check(1 <= result['best_epoch'] <= config['training']['epochs'], 'best epoch outside training budget')
                        saved_row = students[(students.dataset == name)&(students.seed == seed)&(students.noise == noise)&(students.arm == arm)].iloc[0]
                        for key in ['test_accuracy', 'validation_accuracy', 'train_accuracy', 'test_macro_f1']:
                            compare_metric(saved_row[key], result[key], 'CSV/'+key)
                        input_key = (name, seed, noise, target_digest(bundle['target']))
                        unique_id = result['unique_model_id']
                        check(input_models.setdefault(input_key, unique_id) == unique_id, 'identical float32 target trained under different model identities')
                        identity = (input_key, str(logit_path.resolve()))
                        check(identities.setdefault(unique_id, identity) == identity, 'model ID reused for different target/protocol/logits')
                        if result['training_executed']:
                            check(logit_path.resolve() == (student_dir/'logits.pt').resolve(), 'canonical student must save own logits')
                            counters['training_executed'] += 1
                        else:
                            check(logit_path.resolve() != (student_dir/'logits.pt').resolve(), 'alias has own canonical logits path')
                        if name in ['roman', 'corafull', 'coauthorcs'] and arm in ['CE', 'GRACE_OLD64', 'GREET_OLD64']:
                            source = root.parent/'greet_adaptive_ball_20261015'/'students'/name/f'seed{seed}'/noise/arm/'result.json'
                            check(source.exists(), 'required historical student result missing')
                            old = read_json(source)
                            delta = 100*(result['test_accuracy']-old['test_accuracy'])
                            history.append(dict(dataset=name, seed=seed, noise=noise, arm=arm,
                                                accuracy_delta_pp=delta, exact_match=delta == 0, source=str(source)))
                            compare_metric(result['test_accuracy'], old['test_accuracy'], 'historical score (dtype differences are audit errors)', 0)
                        counters['student_results_checked'] += 1
                    safe(f'arm/{name}/{seed}/{noise}/{arm}', audit_arm)
    safe('source_manifest_after', run.verify_sources)
    safe('complete_target_matrix', lambda: check(counters['target_artifacts'] == len(targets) == expected_rows, 'incomplete target matrix'))
    safe('complete_student_matrix', lambda: check(counters['student_results_checked'] == len(students) == expected_rows, 'incomplete student matrix'))
    safe('one_canonical_per_model', lambda: check(counters['training_executed'] == len(identities), 'canonical training count != unique model count'))
    historical_expected = len(set(config['datasets']) & {'roman', 'corafull', 'coauthorcs'})*len(config['protocol_seeds'])*len(config['conditions'])*3
    safe('historical_coverage', lambda: check(len(history) == historical_expected, 'incomplete historical anchors'))
    audit = dict(status='passed' if not errors else 'failed', **counters, unique_student_models=len(identities),
                 alias_results=counters['student_results_checked']-counters['training_executed'],
                 historical_student_checks=len(history), errors=errors,
                 uncertainty='sample standard deviation across protocol seeds; not confidence intervals')
    write_json(root/'audit.json', audit); write_json(root/'historical_student_checks.json', history)
    summarize_and_report(root, config, targets, students, audit)
    if make_plots and not errors:
        import plot_curves
        plot_curves.make_plots()
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} audit failures; see audit.json')
    return audit


def summarize_and_report(root, config, targets, students, audit):
    outputs = []
    for label, frame, metrics in [('target', targets, ['target_brier', 'q_accuracy', 'mean_support', 'positive_clean_margin_fraction']),
                                   ('student', students, ['test_accuracy', 'test_macro_f1'])]:
        for metric in metrics:
            values = frame.copy(); values[metric] = pd.to_numeric(values[metric], errors='coerce')
            summary = values.groupby(['dataset', 'noise', 'space', 'kind', 'arm'], dropna=False)[metric].agg(['count', 'mean', 'std']).reset_index()
            summary.to_csv(root/f'{label}_{metric}_summary.csv', index=False, encoding='utf-8-sig')
            pairs = paired_deltas(values, metric)
            pairs.to_csv(root/f'{label}_{metric}_paired_deltas.csv', index=False, encoding='utf-8-sig')
            pair_summary = pairs.groupby(['dataset', 'noise', 'arm', 'reference', 'comparison'], dropna=False).delta.agg(['count', 'mean', 'std']).reset_index()
            pair_summary.to_csv(root/f'{label}_{metric}_paired_summary.csv', index=False, encoding='utf-8-sig')
            outputs.append((label, metric, summary, pair_summary))
    lines = ['# 粒球数量响应诊断：统计结果与证据边界', '',
             f"审计状态：**{audit['status']}**。检查{audit['partition_protocols']}套分区、{audit['snapshots_checked']}个树快照、{audit['target_artifacts']}份目标、{audit['student_results_checked']}份学生结果。实际训练{audit['training_executed']}次，独立模型身份{audit['unique_student_models']}个，复用别名{audit['alias_results']}份。历史分数核对{audit['historical_student_checks']}项。", '',
             '本报告仅自动整理冻结统计。工程审计通过表示轨迹、目标和分类产物可复核，并不表示新方法改善。审计失败时不得将表内数字作为已验证结论。', '',
             '## 协议与统计口径', '',
             '固定每图每空间每协议seed的一棵树，min_leaf_size=16、min_train_per_ball=16；请求球数与实际球数分开记录。固定末轮无标签表示复用，三个协议seed共同改变划分/噪声/分球/MLP初始化，不是三个独立SSL seed。原64生成器是历史锚；新B64同时改变了分裂约束，因此新树内部曲线才提供数量归因。', '',
             '每个条件分别报告三seed均值±样本标准差（ddof=1），不是置信区间。准确率和macroF1配对差为百分点，新−参照；Brier差为新−参照，负值有利。同实际球数跨空间比较只保留每seed每空间一个可达分区，重复预算别名不增加样本量。', '',
             '## 实际球数及分类结果', '',
             '| 数据 | 条件 | 方法 | 实际球数范围 | accuracy，% | macroF1，% |',
             '|---|---|---|---:|---:|---:|']
    for (name, noise, arm), part in students.groupby(['dataset', 'noise', 'arm'], sort=False):
        acc = part.test_accuracy.astype(float)*100; f1 = part.test_macro_f1.astype(float)*100
        counts = part.actual_balls.astype(int)
        lines.append(f'| {name} | {noise} | {arm} | {counts.min()}–{counts.max()} | {acc.mean():.3f}±{acc.std(ddof=1):.3f} | {f1.mean():.3f}±{f1.std(ddof=1):.3f} |')
    lines += ['', '## 相对历史64、同树64与CE的配对差', '',
              '| 数据 | 条件 | 候选 | 参照 | accuracy差，pp | n |', '|---|---|---|---|---:|---:|']
    for label, metric, summary, paired in outputs:
        if metric == 'test_accuracy':
            for row in paired.itertuples():
                lines.append(f'| {row.dataset} | {row.noise} | {row.arm} | {row.reference} | {row.mean:+.3f}±{row.std:.3f} | {row.count} |')
    lines += ['', '## 机制阅读与结论边界', '',
              '已被产物支持的事实：上表及CSV中的实际数量、目标质量与学生成绩，须以audit.json通过为前提。相同float32目标共享同一学生，不能当独立重复。', '',
              '合理解释：数量上升且干净投票边际上升、Brier下降、分类同步改善时，支持“粗区域混类限制监督”的解释；数量上升但支持耗尽、边际或恢复率下降时，支持“细分削弱抗噪证据”的解释。这些关联不能单独证明因果中介。', '',
              '尚未验证：用无标签局部几何和可观察噪声稳健信号决定自适应停止；更多独立SSL seed；相同SSL预算；独立kNN/k-means对照。本轮没有开展这些验证。', '',
              '被负结果削弱的方向：任何仅在单seed、单噪声或单数据集出现的test峰值都不能支持统一最优球数；如果Brier改善与分类无对应，不能把监督统计改善直接称为分类成功。', '',
              '后续设计仅可由可观察的几何收益、支持和稳定性构成停止准则；真实干净标签边际及test表现只用于事后诊断。不得将事后test最高点写成部署算法或据此继续阈值搜索。', '',
              '最终研究决策和逐数据机制解释由综合分析补充；此处不自动宣称方法有效或统计显著。复核文件包括config.json、PLAN.md、source_manifest.json、raw.log、分裂轨迹、冻结目标、学生logits、两类结果CSV和各项paired_summary.csv。']
    if audit['errors']:
        lines += ['', '## 审计失败记录', '', *[f"- {item['scope']}: {item['error']}" for item in audit['errors']]]
    (root/'实验结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    audit_and_report()
