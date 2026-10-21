"""Independently recompute saved targets, fold losses, predictions, and paired reports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
from prepare_greet import ROOT,sha,write


def summarize(frame,keys,value):
    return frame.groupby(keys)[value].agg(['count','mean','std']).reset_index()


def main():
    import run_pipeline as run
    sources=json.loads((ROOT/'execution_sources.json').read_text(encoding='utf-8'))
    for filename,digest in sources.items():
        if sha(ROOT/'source_snapshot'/filename)!=digest or sha(ROOT/filename)!=digest:
            raise ValueError('Execution source/config changed: '+filename)
    expected=len(run.CONFIG['datasets'])*len(run.CONFIG['protocol_seeds'])*len(run.CONFIG['conditions'])
    protocols=0; targets=0; folds_verified=0; errors=[]
    for name in run.CONFIG['datasets']:
        rep=ROOT/'representations'/name
        rep_summary=json.loads((rep/'summary.json').read_text(encoding='utf-8'))
        if sha(rep/'H.pt')!=rep_summary['H_sha256'] or sha(rep/'edge_weights.pt')!=rep_summary['weights_sha256']:
            raise ValueError('Representation integrity failure')
        for seed in run.CONFIG['protocol_seeds']:
            for condition in run.CONFIG['conditions']:
                graph,assignment,base,protocol=run.graph_protocol(name,seed,condition)
                tr=graph.train_index.numpy(); observed=graph.noisy_y[tr].numpy()
                for stage in ['A','B']:
                    folder=ROOT/'targets'/name/stage/f'seed{seed}'/condition
                    saved=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
                    if saved!=protocol:raise ValueError('Protocol changed')
                    frozen=json.loads((folder/'frozen_targets.json').read_text(encoding='utf-8'))
                    for arm,digest in frozen.items():
                        path=folder/f'{arm}.npz'
                        if sha(path)!=digest:raise ValueError('Frozen target modified')
                        with np.load(path) as a:
                            q,alpha,t=a['q'],a['alpha'],a['target']
                            if not np.isfinite(t).all() or np.any(t<-.000001):raise ValueError('Invalid targets')
                            if not np.allclose(t.sum(1),1,atol=1e-6):raise ValueError('Targets not normalized')
                            rebuilt=(1-alpha[:,None])*np.eye(graph.num_classes)[observed]+alpha[:,None]*q
                            if not np.allclose(rebuilt,t,atol=2e-6):raise ValueError('Target formula inconsistent')
                            if np.any(alpha<0) or np.any(alpha>1):raise ValueError('Invalid alpha')
                            if stage=='B' and not np.array_equal(q,base['q']):raise ValueError('HF changed original q')
                        targets+=1
                    protocols+=1
                bfolder=ROOT/'targets'/name/'B'/f'seed{seed}'/condition
                with np.load(bfolder/'HF_ADAPTIVE.npz') as a:budget=float(a['alpha'].sum())
                for arm in ['HF_CONST','HF_SHUFFLE']:
                    with np.load(bfolder/f'{arm}.npz') as a:
                        if abs(float(a['alpha'].sum())-budget)>1e-7:raise ValueError('Correction budget mismatch')
                aux=ROOT/'auxiliary'/name/f'seed{seed}'/condition
                with np.load(aux/'scores.npz') as scores:
                    for channel in ['attribute','low','high']:
                        held_all=[]
                        for fold in range(3):
                            parent=aux/f'fold{fold}'/channel
                            with np.load(parent/'fold.npz') as a:
                                fit=a['fit_indices']; held=a['hold_indices']; fit_labels=a['fit_labels']
                            if np.intersect1d(fit,held).size or not np.isin(fit,tr).all() or not np.isin(held,tr).all():
                                raise ValueError('Crossfit contamination')
                            if not np.array_equal(fit_labels,graph.noisy_y[fit].numpy()):raise ValueError('Fit labels incorrect')
                            held_all.extend(held.tolist())
                            positions={node:i for i,node in enumerate(tr.tolist())}
                            local=np.asarray([positions[node] for node in held])
                            for epoch in [10,20,30]:
                                logits=np.load(parent/f'hold_logits_epoch{epoch}.npy')
                                loss=logsumexp(logits,axis=1)-logits[np.arange(len(held)),graph.noisy_y[held].numpy()]
                                if not np.allclose(loss,scores[f'{channel}_{epoch}'][local],atol=2e-6):
                                    raise ValueError('Auxiliary score reconstruction failed')
                            folds_verified+=1
                        if sorted(held_all)!=sorted(tr.tolist()):raise ValueError('Holdout coverage mismatch')
    if protocols!=2*expected:raise ValueError('Incomplete target protocols')
    student_rows=[]
    for path in (ROOT/'students').glob('*/seed*/*/*/result.json'):
        row=json.loads(path.read_text(encoding='utf-8')); parent=path.parent
        if sha(parent/'targets.pt')!=row['targets_sha256'] or sha(parent/'logits.pt')!=row['logits_sha256']:
            raise ValueError('Student integrity failure')
        graph=run.graph_protocol(row['dataset'],row['seed'],row['noise'])[0]
        logits=torch.load(parent/'logits.pt',weights_only=True)
        accuracy=float((logits[graph.test_index].argmax(1)==graph.y[graph.test_index]).float().mean())
        if accuracy!=row['test_accuracy']:raise ValueError('Student accuracy mismatch')
        student_rows.append(row)
    frame=pd.DataFrame(run.aggregate_targets())
    if student_rows:
        students=pd.DataFrame(student_rows); students.to_csv(ROOT/'student_results.csv',index=False,encoding='utf-8')
        summarize(students,['dataset','noise','arm'],'test_accuracy').to_csv(ROOT/'student_summary.csv',index=False,encoding='utf-8')
        lookup=students.pivot(index=['dataset','seed','noise'],columns='arm',values='test_accuracy')
        differences=lookup.subtract(lookup['CE'],axis=0)*100
        differences.to_csv(ROOT/'student_paired_deltas.csv',encoding='utf-8')
    report=dict(status='passed',target_protocols=protocols,expected_protocols=2*expected,
                target_artifacts=targets,crossfit_models_checked=folds_verified,student_logits_checked=len(student_rows),
                source_files_checked=len(sources),errors=errors)
    write(ROOT/'audit.json',report)
    result=json.loads((ROOT/'decision.json').read_text(encoding='utf-8')); result['student_runs_completed']=len(student_rows)
    write(ROOT/'decision.json',result)
    lines=['# GREET与HF-proxy首轮实验结论','',
           '日期2026-10-15。结果由冻结目标、交叉拟合记录与保存的学生logits重算，完整原始记录保留。', '',
           '## 范围与限制','',
           f'- 三图、三个协议seed、三噪声；{protocols}套目标协议、{targets}目标文件；{folds_verified}个交叉拟合模型，{len(student_rows)}次学生训练。',
           '- 每图一个GREET SSLseed0；三个协议seed不是三个独立预训练。GREET结构编码使用1024条轨迹的Monte Carlo近似，400epoch固定终点；旧GRACE缓存为原预算，不是同计算预算的算法排名。',
           '- HF-proxy为本项目简化探索，未复现HALF完整原式。折外标签训练、留出标签评分；真值只在产物冻结后诊断。原split和训练标签污染经历史hash复核。',
           '- 正负Brier差均指前者减后者，负数较好。推进门槛为预算规则，不是统计显著性；不隐藏单图或Pair负结果。','',
           '## 独立路线的配对结果','',
           '| 数据集 | GREET−GRACE GB Brier | GREET−KM | GREET−kNN | A门槛 | HF−原GB | HF−同预算常数 | HF−打乱 | B门槛 |',
           '|---|---:|---:|---:|---|---:|---:|---:|---|']
    for row in result['datasets']:
        b=row['hf_differences']
        lines.append(f"| {row['dataset']} | {row['greet_minus_grace_brier']:+.6f} | {row['greet_minus_km']:+.6f} | {row['greet_minus_knn']:+.6f} | {row['greet_pass']} | {b['GRACE_GB']:+.6f} | {b['HF_CONST']:+.6f} | {b['HF_SHUFFLE']:+.6f} | {row['hf_pass']} |")
    paired=frame.pivot(index=['dataset','seed','noise','stage'],columns='arm',values='target_brier')
    pairs=[]
    for index,row in paired.iterrows():
        name,seed,noise,stage=index
        if stage=='A':
            for other in ['GRACE_GB','GREET_KM','GREET_KNN']:
                pairs.append(dict(dataset=name,seed=seed,noise=noise,comparison='GREET_GB minus '+other,difference=row['GREET_GB']-row[other]))
        else:
            for other in ['GRACE_GB','HF_ATTR','HF_LOW','HF_HIGH','HF_FIXED','HF_CONST','HF_SHUFFLE']:
                pairs.append(dict(dataset=name,seed=seed,noise=noise,comparison='HF_ADAPTIVE minus '+other,difference=row['HF_ADAPTIVE']-row[other]))
    pairs=pd.DataFrame(pairs); pairs.to_csv(ROOT/'target_paired_deltas.csv',index=False,encoding='utf-8')
    stats=summarize(pairs,['dataset','noise','comparison'],'difference'); stats.to_csv(ROOT/'target_paired_summary.csv',index=False,encoding='utf-8')
    lines+=['','## 三seed均值与标准差','', '| 数据集 | 噪声 | 配对比较 | n | 差值均值±样本std |','|---|---|---|---:|---:|']
    for _,row in stats.iterrows():
        lines.append(f"| {row['dataset']} | {row['noise']} | {row['comparison']} | {int(row['count'])} | {row['mean']:+.6f}±{row['std']:.6f} |")
    lines+=['','## 决策与解释边界','',
            f"GREET路线门槛：{result['A_pass']}；HF路线门槛：{result['B_pass']}。",'',
            '若GREET改变目标质量但未胜KM/kNN，只支持空间路线，不支持粒球不可替代。若HF未胜常数/打乱或属性小损失，只支持减少修正或通用难度信号，不支持局部双频新机制。',
            '两路线均未过门时停止学生/融合大搜索；一条过门时训练相应预定候选与必要对照；两条过门时才运行固定2×2融合。门槛失败削弱本配置，不证明完整GREET/HALF无效。','',
            '边AUROC/AP仅为冻结后诊断，不决定预训练终点；同类边基率、precision-coverage与分球支持均需一起看。学生准确率见student_results/student_summary（若有）；目标改善不自动等于分类改善。','',
            '审计检查：'+json.dumps(report,ensure_ascii=False),'',
            '后续只在明确机制信号上继续；不按测试集改球数、30epoch门控或融合参数。新SSLseed/新图用于确认，首次探索统计不包装为稳定论文增益。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
