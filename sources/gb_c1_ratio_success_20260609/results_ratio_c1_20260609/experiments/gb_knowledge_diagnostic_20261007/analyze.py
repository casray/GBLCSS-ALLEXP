from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def main():
    execution = json.loads((ROOT / 'execution.json').read_text(encoding='utf-8'))
    assert execution['status'] == 'ok' and execution['protocol_cases'] == 20
    metrics, sampling, curves = [load(name) for name in
                               ('group_metrics.csv', 'sampling_metrics.csv', 'compression_curves.csv')]
    assert len(metrics) == 100 and len(sampling) == 600 and len(curves) == 500
    overall = {}
    for method in ('GB', 'KM', 'Random'):
        rows = [r for r in metrics if r['method'] == method]
        overall[method] = {field: mean(rows, field) for field in
                           ('mse', 'kl', 'agreement_pct', 'explained_teacher_variance',
                            'macro_teacher_class_agreement_pct', 'fp16_teacher_agreement_pct')}
    wins, cell_rows = 0, []
    for dataset in sorted({r['dataset'] for r in metrics}):
        for rate in ('0.4', '0.6'):
            rows = [r for r in metrics if r['dataset'] == dataset and r['rate'] == rate]
            values = {m: mean([r for r in rows if r['method'] == m], 'mse')
                      for m in overall}
            wins += values['GB'] < values['KM']
            cell_rows.append(dict(dataset=dataset, rate=rate, **values))
    budget = {}
    for method in overall:
        budget[method] = {}
        for scheme in ('size', 'adaptive'):
            rows = [r for r in sampling if r['method'] == method
                    and r['scheme'] == scheme and float(r['fraction']) == .25]
            budget[method][scheme] = dict(mse=mean(rows, 'sampling_mean_mse'),
                                        ratio_to_uniform=mean(rows, 'ratio_to_uniform'))
    gb_random = overall['GB']['mse'] / overall['Random']['mse']
    gb_km = overall['GB']['mse'] / overall['KM']['mse']
    adaptive_size = budget['GB']['adaptive']['mse'] / budget['GB']['size']['mse']
    gates = dict(G1=gb_random <= .8, G2=gb_km <= .95 and wins >= 7,
                 G3=adaptive_size <= .8)
    compressed = []
    for method in overall:
        for fraction in (0., .1, .25, .5, 1.):
            rows = [r for r in curves if r['method'] == method
                    and float(r['exception_fraction']) == fraction]
            compressed.append(dict(method=method, exception_fraction=fraction,
                                   **{f: mean(rows, f) for f in
                                      ('mse', 'agreement_pct', 'ratio_to_fp32', 'ratio_to_fp16')}))
    random_std = statistics.stdev(mean([r for r in metrics if r['method'] == 'Random'
                                        and int(r['random_seed']) == seed], 'mse')
                                 for seed in (100, 101, 102))
    summary = dict(execution=execution, overall=overall, random_macro_mse_std=random_std,
                   cells=cell_rows, sampling_25pct=budget, compression=compressed,
                   gb_random_mse_ratio=gb_random, gb_km_mse_ratio=gb_km,
                   gb_lower_than_km_cells=wins, adaptive_size_mse_ratio=adaptive_size,
                   gates=gates, proceed_generic_dist=bool(gates['G1'] and gates['G3']))
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False), encoding='utf-8')
    lines = ['# 教师知识压缩与预算分配：实验结论', '',
             '这是探索性离线诊断，未训练模型、未计算测试准确率。五数据集、两噪声率、'
             '两个既有协议种子，共20组。原始125个输入和冻结脚本散列核验通过。', '',
             '## 区域均值压缩', '',
             '| 分组 | 区域MSE | KL | 教师最高概率类别保留率 | 解释教师方差 |',
             '|---|---:|---:|---:|---:|']
    for method, result in overall.items():
        lines.append(f"| {method} | {result['mse']:.8f} | {result['kl']:.6f} | "
                     f"{result['agreement_pct']:.3f}% | {result['explained_teacher_variance']:.4f} |")
    lines += ['', f'Random的三次随机分组宏平均MSE样本标准差：{random_std:.8f}。'
              '随机重复不是新增独立数据集或协议种子。', '',
              '## 预登记门槛', '',
              f"- G1：GB/Random区域MSE={gb_random:.4f}，要求不超过0.80；{'通过' if gates['G1'] else '未通过'}。",
              f"- G2：GB/KM区域MSE={gb_km:.4f}，要求不超过0.95且至少7/10单元更低；"
              f"实际{wins}/10，{'通过' if gates['G2'] else '未通过'}。",
              f"- G3：25%预算，GB自适应/规模分配教师均值估计MSE={adaptive_size:.4f}，"
              f"要求不超过0.80；{'通过' if gates['G3'] else '未通过'}。", '',
              '## 25%节点预算', '',
              '| 分组 | 规模分配MSE | 自适应分配MSE | 自适应/均匀采样MSE平均比值 |',
              '|---|---:|---:|---:|']
    for method, result in budget.items():
        lines.append(f"| {method} | {result['size']['mse']:.10f} | {result['adaptive']['mse']:.10f} | "
                     f"{result['adaptive']['ratio_to_uniform']:.4f} |")
    lines += ['', '## 保留例外节点的缓存', '',
              '按教师输出残差从大到小保存例外节点。该排序不使用真实标签，但要求先有全量教师输出；'
              '这是缓存构造，不等于减少教师图推理。字节统计包括区域索引、区域均值及例外索引。', '',
              '| 分组 | 例外节点比例 | MSE | 教师类别保留率 | 相对FP32字节 | 相对FP16字节 |',
              '|---|---:|---:|---:|---:|---:|']
    for row in compressed:
        lines.append(f"| {row['method']} | {row['exception_fraction']:.0%} | {row['mse']:.8f} | "
                     f"{row['agreement_pct']:.3f}% | {row['ratio_to_fp32']:.3f} | {row['ratio_to_fp16']:.3f} |")
    lines += ['', '## 决策与边界', '']
    if summary['proceed_generic_dist']:
        lines.append('G1和G3通过，可继续一次DIST损失/梯度近似诊断；当前没有学生训练证据。'
                     + ('G2也通过，但仍不足以证明最终精度或总成本优势。' if gates['G2'] else
                        'G2未通过，不能把预算分配机制当作粒球独有优势；后续必须保留KM和二分KM对照。'))
    else:
        lines.append('G1或G3未通过，按预登记规则暂缓预算蒸馏实现，不启动学生训练或追加调参。')
    lines += ['', '均值估计方差有解析公式；DIST包含跨节点非线性关系，不能由该结果推断DIST梯度无偏。'
              '中心、半径和分区都不保证教师预测正确。缓存压缩也必须对照FP16；'
              '预处理、GRACE和教师开销尚未被本轮计入训练成本收益。', '',
              f"运行用时：{execution['seconds']:.2f}秒。原脚本freeze时出现日志退出异常，"
              '冻结内容随后逐项核验；run采用外层日志包装，保留原冻结脚本不变。', '',
              '完整结果：group_metrics.csv、compression_curves.csv、ball_metrics.csv、'
              'sampling_metrics.csv、allocations.csv、summary.json；配置和日志：PLAN.md、'
              'frozen_inputs.json、environment.json、freeze.log、run.log、execution.json。']
    (ROOT / '实验结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('gates', 'gb_random_mse_ratio', 'gb_km_mse_ratio',
                                            'gb_lower_than_km_cells', 'adaptive_size_mse_ratio',
                                            'proceed_generic_dist')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
