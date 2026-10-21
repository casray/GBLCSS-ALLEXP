"""Descriptive paired summaries; no best-test arm is selected."""
import numpy as np
import pandas as pd
import common as c


COMPARISONS=[('PRUNE','CF_FINE'),('PRUNE','CF_64'),('PRUNE_ZERO','PRUNE'),('PRUNE_ALL','PRUNE'),
             ('PRUNE','GREET_B64'),('PRUNE','GREET_B256'),('PRUNE_ZERO','GREET_B64'),('PRUNE_ALL','GREET_B64')]


def paired(frame,metric):
    rows=[]
    for (name,seed,noise),part in frame.groupby(['dataset','seed','noise']):
        values=part.set_index('arm')[metric]
        for left,right in COMPARISONS:
            rows.append(dict(dataset=name,seed=int(seed),noise=noise,left=left,right=right,
                delta=float(values[left]-values[right])*(100 if metric in ['test_accuracy','test_macro_f1'] else 1)))
    return pd.DataFrame(rows)


def make_report():
    target=pd.read_csv(c.ROOT/'target_results.csv');student=pd.read_csv(c.ROOT/'student_results.csv')
    assert len(target)==len(student)==540
    target.groupby(['dataset','noise','arm'])[['target_brier','mean_alpha','mean_support','clean_harm_rate']].agg(['count','mean','std']).to_csv(c.ROOT/'target_summary.csv',encoding='utf-8-sig')
    summary=student.groupby(['dataset','noise','arm'])[['test_accuracy','test_macro_f1','validation_accuracy']].agg(['count','mean','std'])
    assert (summary[('test_accuracy','count')]==3).all()
    summary.to_csv(c.ROOT/'student_summary.csv',encoding='utf-8-sig')
    pairs={metric:paired(frame,metric) for metric,frame in [('target_brier',target),('test_accuracy',student),('test_macro_f1',student)]}
    for metric,frame in pairs.items():
        frame.to_csv(c.ROOT/f'paired_{metric}.csv',index=False,encoding='utf-8-sig')
        frame.groupby(['dataset','noise','left','right']).delta.agg(['count','mean','std',lambda x:int((x>0).sum())]).rename(columns={'<lambda_0>':'positive_count'}).to_csv(c.ROOT/f'paired_summary_{metric}.csv',encoding='utf-8-sig')
    decisions=[]
    for name in c.CONFIG['datasets']:
        bp=pairs['target_brier'];bp=bp[(bp.dataset==name)&(bp.noise!='clean')&(bp.left=='PRUNE')&(bp.right=='CF_FINE')]
        ap=pairs['test_accuracy'];ap=ap[(ap.dataset==name)&(ap.left=='PRUNE')&(ap.right=='GREET_B64')]
        noisy=ap[ap.noise!='clean'];clean=ap[ap.noise=='clean']
        target_pass=bool(bp.delta.mean()<=-.005 and (bp.delta<0).sum()>=4)
        student_pass=bool(noisy.delta.mean()>=.5 and (noisy.delta>0).sum()>=4 and clean.delta.mean()>=-.5)
        decisions.append(dict(dataset=name,target_signal=target_pass,classifier_signal=student_pass,
            noisy_brier_delta_vs_cf_fine=float(bp.delta.mean()),brier_improved_pairs=int((bp.delta<0).sum()),
            noisy_accuracy_delta_pp_vs_old_b64=float(noisy.delta.mean()),accuracy_positive_pairs=int((noisy.delta>0).sum()),
            clean_accuracy_delta_pp=float(clean.delta.mean()),not_significance_test=True))
    audit=c.read(c.ROOT/'audit.json')
    text=['# 训练内证据剪枝实验：完整结果','',
        '2026-10-15。六图、三个协议种子、clean/Uniform0.4/Pair0.4。324新目标/结果＋216历史锚；当前数据曾用于方法探索，本轮不是独立确认性验证。',
        f"实际新增训练{audit['new_training_executed']}个，新增结果中的复用{audit['new_alias_records']}条；历史216条另计。所有方案完整报告，不按真值Brier或test成绩选择训练。",'',
        '## 方法与读表方式','',
        'CF_ROOT/CF_64/CF_FINE在同一树的粗/中/细区域独立分折投票；PRUNE以独立训练评分及固定置信惩罚剪枝；PRUNE_ZERO去掉惩罚；PRUNE_ALL沿用PRUNE选区但改为全部训练票留一投票。CF区域遇到V票不足16则回退祖先。每个查询折交换S/V两次平均概率目标。',
        '旧锚CE、GREET_OLD64、GREET_B64、GREET_B256只读复用；旧规则OLD64并不保证实际64球。不同种子划分/噪声/学生变化，但每空间仅一个SSLseed。std为三个协议种子的样本标准差，不是CI。',
        '支持与区域数量为每查询折/角色估计的诊断，不能解释为所有节点共享一个最终全局分区。PRUNE_ALL的选择与投票有关联，不能沿用严格独立投票理论。','',
        '## 预设判据（仅探索信号）','',
        '目标：PRUNE相对CF_FINE的六带噪配对Brier均值≤−.005且≥4/6改善。分类：PRUNE相对原GREET_B64六带噪配对平均≥+.5pp、≥4/6为正，clean平均损害≤.5pp。不能仅超过受分折损害的基线就判成功。','',
        '| 数据 | Brier差 vs CF_FINE | 改善/6 | 目标准则 | 带噪准确率差 vs B64，pp | 正向/6 | clean差，pp | 分类准则 |',
        '|---|---:|---:|---|---:|---:|---:|---|']
    for row in decisions:
        text.append(f"| {row['dataset']} | {row['noisy_brier_delta_vs_cf_fine']:+.5f} | {row['brier_improved_pairs']} | {row['target_signal']} | {row['noisy_accuracy_delta_pp_vs_old_b64']:+.3f} | {row['accuracy_positive_pairs']} | {row['clean_accuracy_delta_pp']:+.3f} | {row['classifier_signal']} |")
    for name in c.CONFIG['datasets']:
        text += ['',f'## {name}','', '| 条件 | 方法 | accuracy % | macroF1 % | target Brier | alpha | 平均票数 | 选区数诊断 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for noise in c.CONFIG['conditions']:
            for arm in c.CONFIG['historical_arms']+c.CONFIG['arms']:
                a=student[(student.dataset==name)&(student.noise==noise)&(student.arm==arm)]
                b=target[(target.dataset==name)&(target.noise==noise)&(target.arm==arm)]
                def score(col):return f'{a[col].mean()*100:.3f}±{a[col].std(ddof=1)*100:.3f}'
                region=b.region_count_mean.mean()
                text.append(f'| {noise} | {arm} | {score("test_accuracy")} | {score("test_macro_f1")} | {b.target_brier.mean():.5f} | {b.mean_alpha.mean():.4f} | {b.mean_support.mean():.2f} | {region:.2f} |')
        text += ['', '### 配对增量：均值±std，正向seed数/3','',
                 '| 条件 | 比较 | accuracy差，pp | macroF1差，pp | Brier差（负为好） |','|---|---|---:|---:|---:|']
        for noise in c.CONFIG['conditions']:
            for left,right in COMPARISONS:
                cells=[]
                for metric in ['test_accuracy','test_macro_f1','target_brier']:
                    f=pairs[metric];v=f[(f.dataset==name)&(f.noise==noise)&(f.left==left)&(f.right==right)].delta
                    positive=int((v<0).sum()) if metric=='target_brier' else int((v>0).sum())
                    cells.append(f'{v.mean():+.4f}±{v.std(ddof=1):.4f}（{positive}/3）')
                text.append(f'| {noise} | {left}−{right} | '+' | '.join(cells)+' |')
    text += ['', '## 审计与结论边界','',
        'audit.json独立重开文件核验：54个协议、324目标、每种估计的评分/DP/投票、540条预测结果、216历史锚以及冻结文件哈希。模型只接收X；验证选择epoch，测试不参与区域、参数或模型选择。',
        '11类统计误读检查：描述与推断区分；无p值或显著性声称；std非CI；重复使用同树/seed不当独立样本；多方案完整披露；不选test最佳；划分和噪声配对；历史复用另计；不因目标提升推断分类提升；机制解释不当因果结论；单SSL和探索数据范围明确。',
        '完整结果的机制解释与最终建议另见综合结论与接续.md；本自动表不代替逐图负结果分析。']
    (c.ROOT/'实验结论.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    c.write(c.ROOT/'decision_signals.json',dict(status='completed',criteria_frozen_in='PLAN.md',datasets=decisions,formal_significance_test=False))


if __name__=='__main__':make_report()
