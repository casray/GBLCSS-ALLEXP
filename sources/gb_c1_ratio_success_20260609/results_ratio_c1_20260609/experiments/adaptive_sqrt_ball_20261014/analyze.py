"""Report frozen adaptive-ball evidence without tuning post-test parameters."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import run_screen as screen

ROOT=screen.ROOT;d=screen.d


def main():
    frame=pd.read_csv(ROOT/'target_metrics.csv')
    pairs=pd.read_csv(ROOT/'paired_deltas.csv')
    lines=['# 自适应根号粒球：机制与下游实验','',
        '2026-10-14。方法和门槛预先冻结于PLAN.md；当前报告只依据已完成记录。',
        f"已完成{len(frame[['dataset','seed','noise']].drop_duplicates())}个目标诊断单元、{len(frame)}行方法结果。",
        '每图仅一个SSL预训练seed；五协议seed不等于五次独立表示训练。分球不读取类别，support模式使用训练集合位置。',
        '','## 修改内容','',
        '目标ceil(sqrt(N))，N为全图节点数；最小球大小ceil(N/(4B))；按总SSE优先分裂，取消固定5%平均距离下降门槛；约束失败时尝试满足大小/训练支持的投影边界；相同几何不硬切。主候选每球至少4个训练节点。原LCSS和MLP不变。',
        '该方法是带数据规模及支持约束的预算分裂，尚不能声称自动找到最优球数或完全无参数。',
        '','## 实际球数及支持','',
        '| 数据集 | sqrt目标 | 旧64 | 仅改旧预算 | 新64 | 新sqrt几何 | 新sqrt支持 | 支持版最少训练节点 |',
        '|---|---:|---|---|---|---|---|---:|']
    partitions={}
    for name in screen.NAMES:
        records=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((ROOT/'runs'/name).glob('seed*/partition_stats.json'))]
        if not records:continue
        partitions[name]=records
        values=[','.join(str(r[arm]['actual_balls']) for r in records)
            for arm in ['OLD64','OLD_SQRT','ADAPT64','SQRT_GEOM','SQRT_SUPPORT']]
        lines.append(f"| {name} | {records[0]['SQRT_SUPPORT']['requested_target']} | "+' | '.join(values)+
            f" | {min(r['SQRT_SUPPORT']['min_train_support'] for r in records)} |")
    lines+=['','## 目标质量：五协议seed mean ± sample std','',
        '| 数据集 | 条件 | 旧64 Brier | 新sqrt支持 Brier | 匹配KM | 匹配KNN |',
        '|---|---|---:|---:|---:|---:|']
    for (name,noise),group in frame.groupby(['dataset','noise']):
        values=[]
        for arm in ['OLD64','SQRT_SUPPORT','KM_MATCH','KNN_MATCH']:
            v=group[group.arm==arm].target_brier
            values.append(f'{v.mean():.5f} ± {v.std(ddof=1):.5f}' if len(v)>1 else f'{v.mean():.5f} (n=1)')
        lines.append(f'| {name} | {noise} | '+' | '.join(values)+' |')
    comparisons=[('SQRT_SUPPORT','OLD64'),('OLD_SQRT','OLD64'),('ADAPT64','OLD64'),
        ('SQRT_GEOM','OLD_SQRT'),('SQRT_SUPPORT','SQRT_GEOM'),('SQRT_SUPPORT','KM_MATCH'),('SQRT_SUPPORT','KNN_MATCH')]
    decisions={};rows=[]
    lines+=['','## 配对机制增量','',
        '先对每个seed平均Uniform/Pair，再对五个配对seed计算mean/std。负差有利于前者。',
        '','| 数据集 | 比较 | Brier差 mean ± 配对std | 改善seed | .005探索门槛 |',
        '|---|---|---:|---:|---|']
    noisy=pairs[pairs.noise!='clean']
    for name in screen.NAMES:
        decisions[name]={}
        for arm,reference in comparisons:
            group=noisy[(noisy.dataset==name)&(noisy.arm==arm)&(noisy.reference==reference)]
            if group.empty:continue
            v=group.groupby('seed').brier_delta.mean()
            item=dict(dataset=name,comparison=arm+' - '+reference,mean_delta=float(v.mean()),
                paired_std=float(v.std(ddof=1)) if len(v)>1 else None,n=len(v),improved_seeds=int((v<0).sum()),
                screening_pass=bool(len(v)==5 and v.mean()<=-.005 and (v<0).sum()>=4))
            rows.append(item);decisions[name][item['comparison']]=item
            std=f'{item["paired_std"]:.5f}' if item['paired_std'] is not None else 'n=1'
            lines.append(f'| {name} | {item["comparison"]} | {item["mean_delta"]:+.5f} ± {std} | {item["improved_seeds"]}/{len(v)} | {item["screening_pass"]} |')
    d.csv_write(ROOT/'paired_summary.csv',rows)
    d.write(ROOT/'decision.json',dict(decisions=decisions,significance_claim=False,
        criterion='paired per-seed pooled Uniform/Pair Brier delta<=-.005, >=4/5 improved; only an exploratory signal',
        quantity_is_not_quality=True))
    lines+=['','## 首轮分类器结果','']
    path=ROOT/'student_results.csv'
    if path.exists():
        students=pd.read_csv(path)
        lines+=[f'已完成{len(students)}次原200轮MLP训练，仅seed0，准确率单位%；无虚构std或稳定性结论。',
            '','| 数据集 | 条件 | CE | 旧64 | 新sqrt支持 | 匹配KM | 匹配KNN | 新−旧pp |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for (name,noise),group in students.groupby(['dataset','noise']):
            values={row.arm:row.test_accuracy for row in group.itertuples()}
            rendered=[f'{100*values[arm]:.2f}' if arm in values else '未完成'
                for arm in ['CE','OLD64','SQRT_SUPPORT','KM_MATCH','KNN_MATCH']]
            delta=100*(values['SQRT_SUPPORT']-values['OLD64']) if all(k in values for k in ['SQRT_SUPPORT','OLD64']) else None
            lines.append(f'| {name} | {noise} | '+' | '.join(rendered)+f' | {delta:+.2f} |' if delta is not None else
                         f'| {name} | {noise} | '+' | '.join(rendered)+' | 未完成 |')
    else:lines.append('尚未完成学生训练；不能将数量达标视为分类增益。')
    lines+=['','## 结论边界与接续','',
        '- 实际达到根号尺度，只支持数量适配成功；不能证明该数量最优，也不能证明粒球局部监督更好。',
        '- 仅改预算和新64两个控制分别隔离数量与分裂规则；KM/KNN是普通分组强对照，必须一起报告。',
        '- 缩小球能改善局部性，同时减少抗噪投票支持；训练数量约束不是纯度证明，也不是置信度估计。',
        '- 所有条件报告，包括clean误修正；不从test选择球数、支持阈值或表示。',
        '- 目标改善与学生准确率未必同向；pilot只是转化检查，后续五seed训练需另行记录确认范围。',
        '- 源码与旧数据保留；source_manifest/config、逐seed partition_stats/balls/split_trace/assignments/relations、冻结target/protocol、raw日志、CSV与审计可追溯。',
        '- 审计是否通过及深度覆盖以audit/report.json为准，不因脚本存在就声称验证完成。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(cells=len(frame[['dataset','seed','noise']].drop_duplicates()),rows=len(frame),
                         candidate_signals=[name for name,c in decisions.items() if c.get('SQRT_SUPPORT - OLD64',{}).get('screening_pass')]),ensure_ascii=True))


if __name__=='__main__':main()
