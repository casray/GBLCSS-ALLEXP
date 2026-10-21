"""Compact, reproducible summaries of the audited count intervention."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import data_tools as dt


def main():
    import torch
    torch.set_num_threads(dt.CONFIG['threads'])
    root=dt.ROOT;audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    if audit['status']!='passed':raise ValueError('Complete independent audit first')
    students=pd.read_csv(root/'student_results.csv');targets=pd.read_csv(root/'target_results.csv')
    keys=['dataset','seed','noise','arm']
    frame=students.merge(targets[keys+['target_brier','mean_support','positive_clean_margin_fraction','normalized_sse']],on=keys,validate='one_to_one')
    metrics=['test_accuracy','test_macro_f1','target_brier','mean_support','positive_clean_margin_fraction']
    summary=frame.groupby(['dataset','noise','space','kind','arm','requested_balls'])[metrics].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
    counts=frame.groupby(['dataset','noise','space','kind','arm','requested_balls']).actual_balls.agg(['min','max']).reset_index()
    summary=summary.merge(counts,on=['dataset','noise','space','kind','arm','requested_balls'],validate='one_to_one')
    summary.to_csv(root/'count_response_summary.csv',index=False,encoding='utf-8-sig')
    capacities=[];node_counts={}
    for name in dt.CONFIG['datasets']:
        graph,_=dt.load_case(name,0,'clean');node_counts[name]=graph.num_nodes;del graph
        for space in dt.CONFIG['spaces']:
            stats=[json.loads((root/'partitions'/name/f'seed{s}'/space/'stats.json').read_text(encoding='utf-8')) for s in dt.CONFIG['protocol_seeds']]
            last=[v['B256'] for v in stats];old=[v['OLD64'] for v in stats]
            capacities.append(dict(dataset=name,space=space,sqrt_request=math.ceil(math.sqrt(node_counts[name])),
                requested_max=256,actual_max_min=min(x['actual_balls'] for x in last),actual_max_max=max(x['actual_balls'] for x in last),
                old_actual_min=min(x['actual_balls'] for x in old),old_actual_max=max(x['actual_balls'] for x in old),
                min_train_support=min(x['min_train_support'] for x in last),
                stop_reasons=';'.join(sorted(set(x['stop_reason'] for x in last)))))
    pd.DataFrame(capacities).to_csv(root/'capacity_summary.csv',index=False,encoding='utf-8-sig')
    # A peak table is descriptive only. It is never read by any model or selector.
    peak=summary[summary.kind=='B'].sort_values(['dataset','noise','space','test_accuracy_mean','min','requested_balls'],
        ascending=[True,True,True,False,True,True]).drop_duplicates(['dataset','noise','space']).copy()
    peak['posthoc_test_peak_not_deployment_rule']=True
    peak.to_csv(root/'posthoc_peak_summary.csv',index=False,encoding='utf-8-sig')
    lines=['# GREET数量响应：便于阅读的完整表','',
        '以下列为请求预算，实际数量及可达范围见capacity_summary.csv。所有值为三个协议seed均值±样本std，accuracy单位%。高预算遇到平台时重复同一分区，不能当作更多独立实验。', '',
        'sqrt是ceil(sqrt(N))，各数据集的实际请求不同。表中完整保留clean/U/Pair及低分条件；本表不以test选择部署预算。']
    for noise in dt.CONFIG['conditions']:
        lines+=['',f'## {noise}','',
            '| 数据 | CE | 原GRACE旧规则 | GREET旧规则 | 1 | 8 | 16 | 32 | 64 | 128 | sqrt | 256 |',
            '|---|'+'---:|'*11]
        for name in dt.CONFIG['datasets']:
            base=frame[(frame.dataset==name)&(frame.noise==noise)]
            arms=['CE','GRACE_OLD64','GREET_OLD64']+['GREET_B'+str(b) for b in [1,8,16,32,64,128,math.ceil(math.sqrt(node_counts[name])),256]]
            values=[]
            for arm in arms:
                v=base[base.arm==arm].test_accuracy*100
                if len(v)!=3:raise ValueError('Incomplete curve cell')
                values.append(f'{v.mean():.2f}±{v.std(ddof=1):.2f}')
            lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','## 原始统计与边界','',
        'GRACE全部预算与macroF1、Brier、支持、边际见count_response_summary.csv；完整逐seed结果见student_results.csv/target_results.csv。',
        'posthoc_peak_summary.csv只是事后观测峰值，不是已验证最优参数，不进入生成器或验证集选择。validation_budget_diagnostic目录使用另一套只读验证分数的选择规则，二者不能混称。',
        '本轮按用户要求暂不做独立kNN/k-means比较，因此只回答数量和生成规则的机制问题。']
    (root/'粒球数量响应表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    source=[Path(__file__),root/'audit.json',root/'student_results.csv',root/'target_results.csv']
    dt.d.write(root/'summary_manifest.json',dict(status='passed',sources={str(p.resolve()):dt.d.sha(p) for p in source},
        rows=len(frame),summary_cells=len(summary),test_peaks_are_descriptive_only=True,
        unique_student_models=audit['unique_student_models'],alias_results=audit['alias_results']))
    print({'status':'passed','rows':len(frame),'summary_cells':len(summary)})


if __name__=='__main__':main()
