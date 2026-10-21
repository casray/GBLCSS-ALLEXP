"""Compare fixed support16 follow-up against frozen support4 and historical old64."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_screen as screen

ROOT=screen.ROOT;PREVIOUS=ROOT.parent/'adaptive_sqrt_ball_20261014';d=screen.d


def main():
    for path,digest in json.loads((ROOT/'comparison_source_manifest.json').read_text(encoding='utf-8')).items():
        assert d.sha(path)==digest,path
    old=pd.read_csv(PREVIOUS/'student_results.csv');new=pd.read_csv(ROOT/'student_results.csv')
    targets_old=pd.read_csv(PREVIOUS/'target_metrics.csv');targets_new=pd.read_csv(ROOT/'target_metrics.csv')
    old=old[old.arm=='SQRT_SUPPORT'];new=new[new.arm=='SQRT_SUPPORT']
    rows=[];lines=['# 支持数16与根号粒度的后续对照','',
        '2026-10-14。仅seed0；每个学生200轮，输入、噪声、初始化和原LCSS保持一致。第一轮五seed统计不能当作本轮学生std。',
        '','## 实际数量与支持','',
        '| 图 | 请求sqrt数 | 支持上限 | 实际球数 | 最少训练节点 |',
        '|---|---:|---:|---:|---:|']
    for name in screen.NAMES:
        stats=json.loads((ROOT/'runs'/name/'seed0/partition_stats.json').read_text(encoding='utf-8'))['SQRT_SUPPORT']
        lines.append(f"| {name} | {stats['requested_target']} | {stats['feasible_target']} | {stats['actual_balls']} | {stats['min_train_support']} |")
    lines+=['','## 同seed分类比较：%','',
        '| 图 | 条件 | 支持4 | 支持16 | 支持16−4，pp | 支持16−旧64，pp |',
        '|---|---|---:|---:|---:|---:|']
    full=pd.read_csv(ROOT/'student_results.csv')
    for row in new.itertuples():
        previous=old[(old.dataset==row.dataset)&(old.noise==row.noise)].iloc[0]
        historical=full[(full.dataset==row.dataset)&(full.noise==row.noise)&(full.arm=='OLD64')].iloc[0]
        delta=100*(row.test_accuracy-previous.test_accuracy);base_delta=100*(row.test_accuracy-historical.test_accuracy)
        lines.append(f'| {row.dataset} | {row.noise} | {100*previous.test_accuracy:.2f} | {100*row.test_accuracy:.2f} | {delta:+.2f} | {base_delta:+.2f} |')
        key=dict(dataset=row.dataset,noise=row.noise)
        a=targets_old[(targets_old.dataset==row.dataset)&(targets_old.seed==0)&(targets_old.noise==row.noise)&(targets_old.arm=='SQRT_SUPPORT')].iloc[0]
        b=targets_new[(targets_new.dataset==row.dataset)&(targets_new.seed==0)&(targets_new.noise==row.noise)&(targets_new.arm=='SQRT_SUPPORT')].iloc[0]
        rows.append(dict(**key,accuracy_support16_minus4_pp=delta,accuracy_support16_minus_old64_pp=base_delta,
            brier_support16_minus4=float(b.target_brier-a.target_brier)))
    d.csv_write(ROOT/'comparison.csv',rows)
    lines+=['','## 目标Brier变化：负值有利于支持16','',
        '| 图 | 条件 | 支持16−4 Brier |','|---|---|---:|']
    for row in rows:lines.append(f"| {row['dataset']} | {row['noise']} | {row['brier_support16_minus4']:+.5f} |")
    lines+=['','## 判读','',
        '数量由全图规模、训练样本预算及几何可行性共同限制。16是预先固定的原频率波动约束，并非从测试集选出的最优支持数；不检验多个阈值。',
        '原投票比例条件标准差上界不等于标签正确概率，也不直接控制多数类增强q或最终准确率。固定几何下的类别混合仍可能导致稳定错误。',
        '只一个学生/SSL seed；任何支持16改善或退化均需后续配对确认，不把单次结果主张为统一成功。',
        '完整匹配KM/KNN结果见实验结论.md；必须保留这些对照，不能只与旧64比较。']
    (ROOT/'后续对照结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparisons=len(rows),no_parameter_search=True),ensure_ascii=True))


if __name__=='__main__':main()
