"""Post-hoc diagnostic using validation scores only, never test-driven ball counts."""
from pathlib import Path
import pandas as pd
import data_tools as dt


def select_by_validation(frame):
    # Deliberately exclude test scores from the selection function's working table.
    columns=['dataset','seed','noise','space','kind','arm','requested_balls','actual_balls','validation_accuracy']
    candidates=frame.loc[frame.kind=='B',columns].copy()
    if candidates.empty or not candidates.validation_accuracy.notna().all():raise ValueError('Missing validation candidates')
    groups=['dataset','seed','noise','space']
    candidates=candidates.sort_values(groups+['validation_accuracy','actual_balls','requested_balls'],
        ascending=[True]*4+[False,True,True],kind='stable')
    return candidates.drop_duplicates(groups).reset_index(drop=True)


def main():
    import torch
    torch.set_num_threads(dt.CONFIG['threads'])
    root=dt.ROOT;path=root/'student_results.csv';digest=dt.d.sha(path)
    node_counts={n:dt.load_case(n,0,'clean')[0].num_nodes for n in dt.CONFIG['datasets']}
    frame=pd.read_csv(path);expected=sum(len(dt.arms_for(node_counts[n]))*9 for n in dt.CONFIG['datasets'])
    if len(frame)!=expected:raise ValueError('Wait for complete frozen result matrix')
    selected=select_by_validation(frame)
    scores=frame[['dataset','seed','noise','arm','test_accuracy','test_macro_f1']]
    selected=selected.merge(scores,on=['dataset','seed','noise','arm'],validate='one_to_one')
    pairs=[]
    for row in selected.itertuples():
        n=node_counts[row.dataset]
        import math
        for reference in ['CE',row.space+'_OLD64',row.space+'_B64',row.space+'_B'+str(math.ceil(math.sqrt(n)))]:
            base=frame[(frame.dataset==row.dataset)&(frame.seed==row.seed)&(frame.noise==row.noise)&(frame.arm==reference)].iloc[0]
            pairs.append(dict(dataset=row.dataset,seed=row.seed,noise=row.noise,space=row.space,reference=reference,
                accuracy_delta_pp=100*(row.test_accuracy-base.test_accuracy),macro_f1_delta_pp=100*(row.test_macro_f1-base.test_macro_f1)))
    out=root/'validation_budget_diagnostic';out.mkdir(exist_ok=True)
    selected.to_csv(out/'selected_budgets.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(out/'paired_deltas.csv',index=False,encoding='utf-8-sig')
    lines=['# 验证集选粒度：事后补充诊断','',
        '选择函数只读取原模型的validation_accuracy；并列时先选实际球数较少者，再选请求预算较少者。test分数不会进入选择。每个空间单独选择，不选择编码器。', '',
        '这是在数量实验之后定义的补充诊断，不是预注册的新方法成绩，也不是粒球生成算法创新。各候选已用同一验证集选择epoch，再扫球数增加了验证选择预算；与固定64的训练成本不同。未来正式验证需要在独立设置下冻结规则。', '',
        '| 图 | 空间 | 条件 | 选中实际球数范围 | accuracy，% | macroF1，% |','|---|---|---|---:|---:|---:|']
    for (name,space,noise),part in selected.groupby(['dataset','space','noise']):
        a=part.test_accuracy*100;f=part.test_macro_f1*100;b=part.actual_balls
        lines.append(f'| {name} | {space} | {noise} | {b.min()}–{b.max()} | {a.mean():.3f}±{a.std(ddof=1):.3f} | {f.mean():.3f}±{f.std(ddof=1):.3f} |')
    (out/'验证集选粒度诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if dt.d.sha(path)!=digest:raise ValueError('Student matrix changed during selection analysis')
    dt.d.write(out/'manifest.json',dict(status='passed',type='post-hoc validation-only diagnostic',source_result_sha256=digest,
        script_sha256=dt.d.sha(__file__),test_sha256=dt.d.sha(root/'test_budget_selection.py'),selected_cells=len(selected),
        selection_columns=['validation_accuracy','actual_balls','requested_balls'],test_used_to_select=False,
        no_new_model_trained=True,not_preregistered_method_claim=True))
    print({'status':'passed','selected_cells':len(selected)})


if __name__=='__main__':main()
