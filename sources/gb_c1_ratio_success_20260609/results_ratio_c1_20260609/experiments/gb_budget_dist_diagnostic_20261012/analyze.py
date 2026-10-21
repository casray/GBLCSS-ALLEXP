from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
ARMS = ('Uniform', 'GB_size', 'GB_adaptive', 'KM_adaptive', 'BKM_adaptive', 'Random_adaptive')


def load(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def mean(rows, field):
    return statistics.mean(float(r[field]) for r in rows)


def main():
    execution = json.loads((ROOT/'execution.json').read_text(encoding='utf-8'))
    assert execution['status'] == 'ok' and execution['metric_rows'] == 14400
    raw, biases = load('replicates.csv'), load('aggregates.csv')
    assert len(raw) == 14400 and len(biases) == 720
    per_state = []
    for bias in biases:
        fields = ('dataset', 'rate', 'seed', 'state', 'fraction', 'arm')
        rows = [r for r in raw if all(r[f] == bias[f] for f in fields)]
        assert len(rows) == 20 and len({r['repeat'] for r in rows}) == 20
        per_state.append(dict(**{f: bias[f] for f in fields},
            gradient_rmse=mean(rows, 'grad_relative_mse')**.5,
            mc_gradient_relative_error_std=statistics.stdev(float(r['grad_relative_mse'])**.5 for r in rows),
            mean_gradient_relative_bias=float(bias['mean_gradient_relative_bias']),
            gradient_cosine=mean(rows, 'gradient_cosine'),
            loss_rmse=statistics.mean(float(r['loss_error'])**2 for r in rows)**.5,
            loss_bias=mean(rows, 'loss_error')))
    with (ROOT/'state_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_state[0]))
        writer.writeheader()
        writer.writerows(per_state)
    overall = {}
    for fraction in (.1, .25, .5):
        overall[str(fraction)] = {}
        for arm in ARMS:
            rows = [r for r in per_state if r['arm'] == arm and float(r['fraction']) == fraction]
            assert len(rows) == 40
            overall[str(fraction)][arm] = {f: mean(rows, f) for f in
                ('gradient_rmse', 'mc_gradient_relative_error_std', 'mean_gradient_relative_bias',
                 'gradient_cosine', 'loss_rmse', 'loss_bias')}
    cells, wins_uniform, wins_best_cluster = [], 0, 0
    for dataset in sorted({r['dataset'] for r in raw}):
        for rate in ('0.4', '0.6'):
            rows = [r for r in per_state if r['dataset'] == dataset and r['rate'] == rate
                    and float(r['fraction']) == .25]
            values = {arm: mean([r for r in rows if r['arm'] == arm], 'gradient_rmse') for arm in ARMS}
            wins_uniform += values['GB_adaptive'] < values['Uniform']
            wins_best_cluster += values['GB_adaptive'] < min(values['KM_adaptive'], values['BKM_adaptive'])
            cells.append(dict(dataset=dataset, rate=rate, **values))
    primary = overall['0.25']
    uniform_ratio = primary['GB_adaptive']['gradient_rmse']/primary['Uniform']['gradient_rmse']
    size_ratio = primary['GB_adaptive']['gradient_rmse']/primary['GB_size']['gradient_rmse']
    cluster_ratio = primary['GB_adaptive']['gradient_rmse']/min(
        primary['KM_adaptive']['gradient_rmse'], primary['BKM_adaptive']['gradient_rmse'])
    stages = {}
    for state in ('initial', 'epoch200'):
        rows = [r for r in per_state if r['state'] == state and r['arm'] == 'GB_adaptive'
                and float(r['fraction']) == .25]
        stages[state] = dict(bias=mean(rows, 'mean_gradient_relative_bias'),
                             cosine=mean(rows, 'gradient_cosine'))
    gates = dict(D1=uniform_ratio <= .8 and wins_uniform >= 7,
                 D2=size_ratio <= .9,
                 D3=all(v['bias'] <= .1 and v['cosine'] >= .9 for v in stages.values()),
                 D4=cluster_ratio <= .95 and wins_best_cluster >= 7)
    proceed = bool(gates['D1'] and gates['D2'] and gates['D3'])
    summary = dict(overall=overall, cells=cells, stages=stages, gates=gates,
                   gb_uniform_ratio=uniform_ratio, gb_size_ratio=size_ratio,
                   gb_best_cluster_ratio=cluster_ratio, gb_lower_than_uniform_cells=wins_uniform,
                   gb_lower_than_best_cluster_cells=wins_best_cluster,
                   proceed_training=proceed, execution=execution)
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
                                    encoding='utf-8')
    lines = ['# 区域预算DIST：损失与分类头梯度诊断结论', '',
             '本轮没有训练学生。使用五数据集、uniform0.4/0.6、协议种子1/2的20组既有输入；'
             '每组检查初始化及epoch200两个固定学生状态。每配置20次配对无放回采样。'
             '所有输入散列及epoch200预测缓存核对通过，不读取真实标签进行评价。', '',
             '## 25%预算：主要结果', '',
             '梯度相对RMSE先在每个状态内取20次相对平方误差的均值再开方，'
             '再对40状态等权平均。偏差为20次平均梯度相对全量梯度的距离，'
             '包含有限MC误差，不是无偏性检验。', '',
             '| 方法 | 分类头梯度相对RMSE | 平均梯度相对偏差 | 平均方向余弦 | DIST损失RMSE |',
             '|---|---:|---:|---:|---:|']
    for arm, values in primary.items():
        lines.append(f"| {arm} | {values['gradient_rmse']:.5f} | {values['mean_gradient_relative_bias']:.5f} | "
                     f"{values['gradient_cosine']:.5f} | {values['loss_rmse']:.6f} |")
    lines += ['', '## 预登记门槛', '',
              f"- D1：GB_adaptive/Uniform梯度RMSE={uniform_ratio:.4f}；要求<=0.80且至少7/10单元更低，"
              f"实际{wins_uniform}/10，{'通过' if gates['D1'] else '未通过'}。",
              f"- D2：GB_adaptive/GB_size={size_ratio:.4f}；要求<=0.90，{'通过' if gates['D2'] else '未通过'}。",
              f"- D3：初始化偏差{stages['initial']['bias']:.4f}、余弦{stages['initial']['cosine']:.4f}；"
              f"epoch200偏差{stages['epoch200']['bias']:.4f}、余弦{stages['epoch200']['cosine']:.4f}。"
              f"分别要求偏差<=0.10且余弦>=0.90，{'通过' if gates['D3'] else '未通过'}。",
              f"- D4：GB_adaptive/较优聚类={cluster_ratio:.4f}；要求<=0.95且至少7/10单元更低，"
              f"实际{wins_best_cluster}/10，{'通过' if gates['D4'] else '未通过'}。", '',
              '## 10个数据集-噪声单元', '',
              '| 数据集 | 噪声率 | Uniform | GB规模 | GB自适应 | KM自适应 | 二分KM自适应 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in cells:
        lines.append(f"| {row['dataset']} | {row['rate']} | {row['Uniform']:.4f} | {row['GB_size']:.4f} | "
                     f"{row['GB_adaptive']:.4f} | {row['KM_adaptive']:.4f} | {row['BKM_adaptive']:.4f} |")
    lines += ['', '## 预算曲线', '',
              '| 方法 | 10%梯度RMSE | 25%梯度RMSE | 50%梯度RMSE |',
              '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {overall['0.1'][arm]['gradient_rmse']:.4f} | "
                     f"{overall['0.25'][arm]['gradient_rmse']:.4f} | {overall['0.5'][arm]['gradient_rmse']:.4f} |")
    lines += ['', '## 决策', '']
    if proceed:
        lines.append('D1-D3通过，允许设计同协议的预算学生训练；本轮尚无最终分类精度、非劣效性或实际加速证据。'
                     + ('D4通过，仍需独立确认。' if gates['D4'] else 'D4失败，只能按一般区域预算机制推进，不以粒球独有优势立论。'))
    else:
        lines.append('D1-D3至少一项失败，按预登记规则停止本分支，不启动学生训练，不追加梯度权重或阈值调参。'
                     '知识均值估计有改善，但该信号不足以支持当前预算DIST实现。')
    lines += ['', '## 解释边界', '',
              '本轮只检查固定eval模式下的末层分类头梯度；不是完整网络梯度，也不是训练模式BN。'
              '初始化和epoch200结果分别保留。梯度与损失估计依赖同一个教师，不能证明教师正确。'
              '加权相关系数是非线性估计量，不能由样本权重和为1推断无偏。', '',
              '全量属性前向用于缓存固定隐层。教师全图推理、GRACE、分区和预算准备均未省略；'
              '本轮不报告学生训练速度或最终准确率改善。MC标准差见state_metrics.csv，'
              '这些重复不是独立训练种子。', '',
              f"GPU：{execution['device']}；诊断用时{execution['seconds']:.2f}秒。"
              'PLAN.md、frozen_inputs.json、environment.json、原始freeze/run日志、replicates.csv、'
              'aggregates.csv、allocations.csv、references.csv、state_metrics.csv、summary.json保留完整配置和结果。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('gates', 'gb_uniform_ratio', 'gb_size_ratio',
                                            'gb_best_cluster_ratio', 'proceed_training')}, indent=2))


if __name__ == '__main__':
    main()
