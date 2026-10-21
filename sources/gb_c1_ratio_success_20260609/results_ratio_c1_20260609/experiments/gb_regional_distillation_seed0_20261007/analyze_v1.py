from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_regional as run
from regional_loss import RegionalKL, residual_components

ROOT = Path(__file__).resolve().parent


def save_csv(name, rows):
    columns = list(rows[0])
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compare(rows, left, right):
    differences = np.array([row[left] - row[right] for row in rows])
    return dict(mean=float(differences.mean()), wins=int((differences > 1e-10).sum()),
                ties=int((np.abs(differences) <= 1e-10).sum()),
                losses=int((differences < -1e-10).sum()), worst=float(differences.min()),
                best=float(differences.max()))


def analyze():
    run.require_gpu()
    run.verify_frozen()
    pairs, diagnostics, audit_rows = [], [], []
    per_class = []
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            data, protocol = run.previous.load_data(dataset, rate)
            frozen = run.read(run.cache_folder(dataset, rate) / 'config.json')
            assert protocol == frozen['protocol']
            cache = torch.load(run.cache_folder(dataset, rate) / 'regional.pt',
                               map_location='cpu', weights_only=True)
            teacher = cache['teacher']
            predictions = {}
            x = data.x.cuda()
            teacher_pred = teacher.argmax(1)
            test = data.test_index
            teacher_correct = int((teacher_pred[test] == data.y[test]).sum())
            pair = dict(dataset=dataset, noise_rate=rate, teacher=100*teacher_correct/test.numel())
            initial_hashes = set()
            for arm in run.ARMS:
                folder = run.run_folder(dataset, rate, arm)
                result = run.read(folder / 'result.json')
                assert result['status'] == 'ok'
                assert run.read(folder / 'config.json')['protocol'] == protocol
                probabilities = torch.load(folder / 'student_probabilities.pt',
                                           map_location='cpu', weights_only=True)
                assert probabilities.shape == teacher.shape
                assert torch.isfinite(probabilities).all()
                assert (probabilities.sum(1)-1).abs().max() < 1e-5
                metrics = run.exact_metrics(probabilities, data)
                for key, value in metrics.items():
                    assert abs(result[key] - value) < 1e-12, (dataset, rate, arm, key)
                rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
                assert [row['epoch'] for row in rows] == list(range(1, 201))
                best_value = max(row['valid'] for row in rows)
                first_best = next(row['epoch'] for row in rows if row['valid'] == best_value)
                assert first_best == result['best_epoch'], (dataset, rate, arm, 'selection')
                model = run.previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
                state = torch.load(folder / 'student.pt', map_location='cuda', weights_only=True)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    restored = model(x).softmax(1).cpu()
                checkpoint_error = float((restored-probabilities).abs().max())
                assert checkpoint_error <= 1e-6
                assert torch.equal(restored.argmax(1), probabilities.argmax(1))
                if arm == 'R0':
                    assert result['replay_passed']
                initial_hashes.add(result['initial_state_hash'])
                predictions[arm] = probabilities
                pair[arm] = 100*metrics['student_test']
                pred = probabilities.argmax(1)
                energy = residual_components(probabilities, teacher, cache['assignments']['R1'])
                assert energy['decomposition_error'] < 1e-12
                point_kl = F.kl_div(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
                                    teacher, reduction='batchmean')
                region_values = {key: float(RegionalKL(teacher, ids)(probabilities))
                                 for key, ids in cache['assignments'].items()}
                assert all(value <= float(point_kl)+1e-5 for value in region_values.values())
                train = data.train_index
                truth = F.one_hot(data.y[train], data.num_classes).float()
                diagnostics.append(dict(dataset=dataset, noise_rate=rate, arm=arm,
                    student_test=pair[arm], pointwise_kl=float(point_kl),
                    real_region_kl=region_values['R1'], random_region_kl=region_values['R2'],
                    global_region_kl=region_values['R3'], residual_total=energy['total'],
                    residual_between=energy['between'], residual_within=energy['within'],
                    decomposition_error=energy['decomposition_error'],
                    clean_train_brier=float((probabilities[train]-truth).square().sum(1).mean()),
                    teacher_clean_train_brier=float((teacher[train]-truth).square().sum(1).mean()),
                    teacher_wrong_student_right=int(((teacher_pred[test] != data.y[test])
                                                     & (pred[test] == data.y[test])).sum()),
                    teacher_right_student_wrong=int(((teacher_pred[test] == data.y[test])
                                                     & (pred[test] != data.y[test])).sum()),
                    student_seconds=result['student_seconds'],
                    peak_allocated_gib=result['peak_allocated_gib'], best_epoch=result['best_epoch']))
                for label in range(data.num_classes):
                    selected = test[data.y[test] == label]
                    if selected.numel():
                        correct = int((pred[selected] == label).sum())
                        per_class.append(dict(dataset=dataset, noise_rate=rate, arm=arm, label=label,
                                              correct=correct, total=selected.numel(),
                                              accuracy=correct/selected.numel()))
                audit_rows.append(dict(dataset=dataset, noise_rate=rate, arm=arm,
                                       checkpoint_max_abs_difference=checkpoint_error,
                                       replay_max_abs_difference=result.get('replay_max_abs_difference'),
                                       replay_bit_exact=result.get('replay_bit_exact'),
                                       epoch_selection_passed=True, metrics_passed=True))
                del model, state, restored
            assert len(initial_hashes) == 1, 'student initialization differs between arms'
            for arm in run.ARMS[1:]:
                pair[f'{arm}_minus_R0'] = pair[arm] - pair['R0']
            for arm in ['R2', 'R3', 'R4']:
                pair[f'R1_minus_{arm}'] = pair['R1'] - pair[arm]
            pair['R1_minus_teacher'] = pair['R1'] - pair['teacher']
            baseline = predictions['R0'].argmax(1)
            candidate = predictions['R1'].argmax(1)
            pair['R1_repairs_R0_test'] = int(((baseline[test] != data.y[test])
                                            & (candidate[test] == data.y[test])).sum())
            pair['R1_harms_R0_test'] = int(((baseline[test] == data.y[test])
                                          & (candidate[test] != data.y[test])).sum())
            pairs.append(pair)
            del x
            torch.cuda.empty_cache()
    assert len(pairs) == 10 and len(diagnostics) == 50 and len(audit_rows) == 50
    comparisons = {arm: compare(pairs, 'R1', arm) for arm in ['R0', 'R2', 'R3', 'R4', 'teacher']}
    primary = comparisons['R0']
    gates = dict(
        meaningful_gain_over_KD=primary['mean'] >= .5 and primary['wins'] >= 6 and primary['worst'] >= -1,
        real_over_random=comparisons['R2']['mean'] >= .3 and comparisons['R2']['wins'] >= 7,
        real_over_global=comparisons['R3']['mean'] > 0 and comparisons['R3']['wins'] >= 6,
        real_over_loss_scaling=comparisons['R4']['mean'] > 0 and comparisons['R4']['wins'] >= 6,
        preserves_teacher_accuracy=comparisons['teacher']['worst'] >= -2)
    mean_accuracy = {arm: float(np.mean([row[arm] for row in pairs]))
                     for arm in ['teacher']+run.ARMS}
    mechanism = {}
    lookup = {(r['dataset'], r['noise_rate'], r['arm']): r for r in diagnostics}
    for metric in ['real_region_kl', 'residual_between', 'pointwise_kl', 'clean_train_brier']:
        differences = [lookup[(row['dataset'], row['noise_rate'], 'R1')][metric]
                       - lookup[(row['dataset'], row['noise_rate'], 'R0')][metric] for row in pairs]
        mechanism[metric] = dict(mean_difference=float(np.mean(differences)),
                                lower_cells=int(np.sum(np.array(differences) < 0)))
    summary = dict(status='complete', date='2026-10-07', seed=0, cells=10, runs=50,
                   mean_accuracy=mean_accuracy, R1_comparisons=comparisons, gates=gates,
                   passed=all(gates.values()), mechanism=mechanism,
                   result_scope='single-seed development screen; no across-seed standard deviation',
                   checkpoint_checks=50, replay_checks=10, selection_checks=50)
    save_csv('paired_results.csv', pairs)
    save_csv('diagnostics.csv', diagnostics)
    save_csv('per_class_results.csv', per_class)
    run.save_results()
    run.write(ROOT / 'summary.json', summary)
    run.write(ROOT / 'output_audit.json', dict(status='passed', checks=audit_rows,
                                             frozen_inputs_unchanged=True))
    run.verify_frozen()
    write_report(summary, pairs)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def write_report(summary, pairs):
    lines = [
        '# 固定 ERASE 教师的区域分布蒸馏：实验结论', '',
        '日期：2026-10-07。五个数据集，训练标签 uniform 0.4/0.6，seed 0。',
        '50 次学生训练全部完成，其中 10 次普通蒸馏逐格回放原缓存。',
        '本轮是开发阶段的单种子筛查，不能估计跨种子标准差或声称稳定提升。', '',
        '## 1. 假设、机制和实验配置', '',
        '假设：普通蒸馏剩余的师生概率差异具有区域结构，额外匹配粒球平均类别分布可能改善知识传递。',
        '粒球提供互斥的共享区域统计，允许成员有不同类别和预测。训练区域项不读取观测或真实类别。',
        '固定教师输出 P，原逐节点软交叉熵记为 CE；区域项为按球大小加权的 KL(P_bar || p_bar)。', '',
        '- R0：原全节点 CE。',
        '- R1：CE + 真实 GRACE 粒球区域项。',
        '- R2：CE + 全节点随机大小匹配区域项，shuffle seed 137。',
        '- R3：CE + 全图单区域项。',
        '- R4：2 * CE，整体损失尺度对照。', '',
        'lambda=1 固定，未按本轮测试结果调参。学生为原两层属性 MLP，hidden=64、BN、dropout=0.5，',
        'Adam lr=0.01、weight_decay=0.0005、200 epochs、temperature=1。每组重置 seed 0。',
        '所有节点参加蒸馏，只按干净验证精度严格提升选首个最优检查点。',
        '全节点概率可用于原传导协议；测试标签只用于选模后的评估。',
        '输入、源码和配置均预先冻结并核验 SHA-256；GPU 使用原 noisygl-rocm721 和 RX 7700 XT。', '',
        '## 2. 十个设置的配对结果', '',
        '精度单位为百分比，差值单位为百分点。数值按整数正确数计算，表中精度保留两位小数。', '',
        '| 数据集 | 噪声 | 教师 | R0 普通 KD | R1 粒球 | R2 随机 | R3 全局 | R4 放大 CE | R1-R0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in pairs:
        lines.append(f'| {row["dataset"]} | {row["noise_rate"]} | {row["teacher"]:.2f} | '
                     f'{row["R0"]:.2f} | {row["R1"]:.2f} | {row["R2"]:.2f} | '
                     f'{row["R3"]:.2f} | {row["R4"]:.2f} | {row["R1_minus_R0"]:+.3f} |')
    means = summary['mean_accuracy']
    lines.append('| 十格等权均值 | | ' + ' | '.join(f'{means[a]:.3f}' for a in ['teacher']+run.ARMS)
                 + f' | {summary["R1_comparisons"]["R0"]["mean"]:+.3f} |')
    lines.extend(['', '## 3. 配对比较与预先约定的门槛', '',
                  '| R1 相对对象 | 平均差值 | 胜/平/负 | 最差 |',
                  '|---|---:|---:|---:|'])
    for arm, values in summary['R1_comparisons'].items():
        lines.append(f'| {arm} | {values["mean"]:+.4f} | '
                     f'{values["wins"]}/{values["ties"]}/{values["losses"]} | {values["worst"]:+.4f} |')
    lines.extend(['', '门槛来自训练前冻结的 PLAN.md；不是统计显著性或非劣检验。', '',
                  '| 门槛 | 结果 |', '|---|---|'])
    labels = dict(meaningful_gain_over_KD='对 R0：平均至少 +0.5，至少 6 胜，最差不低于 -1',
                  real_over_random='对 R2：平均至少 +0.3，至少 7 胜',
                  real_over_global='对 R3：平均为正，至少 6 胜',
                  real_over_loss_scaling='对 R4：平均为正，至少 6 胜',
                  preserves_teacher_accuracy='每格相对教师损失不超过 2')
    for key, passed in summary['gates'].items():
        lines.append(f'| {labels[key]} | {"通过" if passed else "未通过"} |')
    lines.extend(['', '## 4. 机制诊断', '',
                  '| R1 相对 R0 的指标 | 十格平均变化 | 下降的设置数 |',
                  '|---|---:|---:|'])
    for key, values in summary['mechanism'].items():
        lines.append(f'| {key} | {values["mean_difference"]:+.6f} | {values["lower_cells"]}/10 |')
    repairs = sum(row['R1_repairs_R0_test'] for row in pairs)
    harms = sum(row['R1_harms_R0_test'] for row in pairs)
    lines.extend(['', f'相对普通 KD，R1 在测试节点上合计纠正 {repairs} 个错误，同时新增 {harms} 个错误。',
                  '这是异质数据集上的合计，不等同于十格等权平均增益。',
                  'real_region_kl 与 residual_between 衡量师生区域差异，下降只说明学生更接近教师；',
                  '必须结合测试精度、随机/全局/损失尺度对照判断是否有实际收益。',
                  'clean_train_brier 是选模完成后的干净训练标签诊断，未用于训练目标或选参数。', '',
                  '## 5. 最终决策与证据边界', ''])
    if summary['passed']:
        lines.extend(['**本固定候选通过全部 seed 0 筛查门槛，建议进入独立种子验证。**',
                      '当前结果仍不能证明稳定性、统计显著性或粒球优于非粒球几何分组。',
                      '后续应冻结参数，比较同表示空间 k-means/kNN、类别分组及 DGKD/KRD，再更换教师。'])
    else:
        lines.extend(['**本固定候选未通过全部 seed 0 门槛，停止直接扩大到多种子。**',
                      '这次检验的是 lambda=1 的区域平均分布匹配，不能据此否定所有区域或关系蒸馏。',
                      '本轮不按测试结果改变 lambda，也不继续堆加机制。'])
    lines.extend(['', '已支持的结论以十个配对结果和机制对照为限；优化偏差、教师偏差或模型容量的解释仍需进一步验证。',
                  '如果真实粒球优于普通 KD 却不能胜过随机或全局项，不能把增益归因于粒球成员。',
                  '如果只缩小区域差异而没有精度提升，应区分“更像教师”和“更准确”。', '',
                  '## 6. 核验与文件', '',
                  '- 50 组完整 200 轮日志，首个验证最优 epoch 全部核对。',
                  '- 50 个保存检查点在 GPU 恢复并核对预测概率及类别。',
                  '- 50 行 train/valid/test 精度按整数正确数重新核算。',
                  '- 10 个 R0 与原缓存回放比较通过；详见 output_audit.json。',
                  '- 每个设置五组学生初始化哈希一致，教师与分组输入指纹保持一致。',
                  '- 残差的区域/球内精确分解及 KL 聚合不等式均通过数值核验。',
                  '- 源码与原始输入 SHA-256 在执行前后保持不变。', '',
                  '`paired_results.csv`：十格配对精度与差值；`diagnostics.csv`：50 行机制、训练时间与显存。',
                  '`per_class_results.csv`：逐类正确数；`summary.json`：均值、胜负和门槛；`output_audit.json`：完整核验。',
                  '`runs/`：config、epochs.jsonl、console.log、student.pt、student_probabilities.pt、result.json。',
                  '`cache/`：冻结教师概率、真实/随机/全局分组与教师区域均值。',
                  '`frozen_inputs.json`：全部来源指纹；`environment.json`：环境；`execution.json`：运行状态与时间。'])
    (ROOT / '实验结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    analyze()
