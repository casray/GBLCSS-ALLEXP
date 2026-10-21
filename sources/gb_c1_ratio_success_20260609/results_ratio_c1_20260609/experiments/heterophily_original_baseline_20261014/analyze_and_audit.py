"""Audit completed original-baseline artifacts and report paired split evidence."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import run_original_baseline as runner

ROOT=runner.HERE
d=runner.d
sys.path.insert(0,str(runner.PREVIOUS))
import audit_diagnostic as audit


def main():
    torch.set_num_threads(4)
    runner.configure_diagnostic()
    manifest=json.loads((ROOT/'adapter_manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest.items():
        if d.sha(path)!=digest:raise AssertionError('Adapter/source changed: '+path)
    audit.ROOT=ROOT;audit.OUT=ROOT/'audit';audit.OUT.mkdir(exist_ok=True)
    targets=[]
    with threadpool_limits(limits=4):
        for metrics in sorted((ROOT/'runs').glob('*/*/seed*/metrics.json')):
            folder=metrics.parent
            targets.append(audit.audit_run(folder,folder.name=='seed0','cuda'))
    results=[]
    for file in sorted((ROOT/'students').glob('*/seed*/*/*/result.json')):
        row=json.loads(file.read_text(encoding='utf-8'))
        config=json.loads((file.parent/'config.json').read_text(encoding='utf-8'))
        graph=runner.load_graph(row['dataset'],row['seed'])
        logits=torch.load(file.parent/'logits.pt',map_location='cpu',weights_only=True)
        assert d.sha(runner.FILES[row['dataset']])==config['input_hash']
        assert d.sha(config['target_path'])==config['target_hash']==row['target_hash']
        assert d.sha(runner.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py')==config['trainer_hash']
        assert d.sha(runner.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py')==config['model_hash']
        for index,key in [(graph.val_index,'validation_accuracy'),(graph.test_index,'test_accuracy')]:
            actual=(logits[index].argmax(1)==graph.y[index]).float().mean().item()
            assert abs(actual-row[key])<1e-8
        assert 1<=row['best_epoch']<=200
        results.append(row)
    assert len(results)==135
    expected={(name,seed,noise,arm) for name in ['texas','wisconsin','roman'] for seed in range(5)
              for noise in ['clean','uniform','pair'] for arm in ['CE','GB64','KNN64']}
    assert {(r['dataset'],r['seed'],r['noise'],r['arm']) for r in results}==expected
    d.write(ROOT/'audit'/'report.json',dict(status='passed',target_protocol_runs=len(targets),
        deep_target_seeds=[0],student_runs=len(results),target_runs=targets,
        coverage='Adapter manifest; ten new target protocols (seed0 deep per new graph); 135 saved-logit accuracies/input/target/trainer/model hashes and expected cells.',
        roman_target_audit=str(runner.PREVIOUS/'audit/report.json'),
        source_independence='Formal GB64 scalar formula checked per condition before each student batch; this audit independently reconstructs saved logits.'))
    frame=pd.DataFrame(results)
    pairs=pd.read_csv(ROOT/'student_paired_deltas.csv')
    lines=['# 原版GB-LCSS异配图基线：已完成结果','',
           '2026-10-14。3图×5官方split×3条件×3方法，135次200轮学生训练完成。来源与结果核查见audit/report.json。',
           '','## 统一设置','',
           '原GB64/min_child32/split_improvement0.05、原留一LCSS(min_votes2/beta0.5)、原属性MLP/验证选点未修改。KNN64逐节点匹配原GB64的其他训练票数。每图一个SSL seed，五个split/噪声/学生协议seed。新图GRACE是固定fallback，非官方最佳配置。',
           '','## 准确率：五划分mean ± sample std，%','',
           '| 图 | 条件 | CE | GB64 | 匹配KNN64 | GB−CE配对增量pp ± std | GB改善划分 |','|---|---|---:|---:|---:|---:|---:|']
    decisions={}
    for name in ['texas','wisconsin','roman']:
        decisions[name]={}
        clean_delta=pairs[(pairs.dataset==name)&(pairs.noise=='clean')&(pairs.arm=='GB64')].accuracy_minus_ce_pp.mean()
        for noise in ['clean','uniform','pair']:
            values=[]
            for arm in ['CE','GB64','KNN64']:
                v=100*frame[(frame.dataset==name)&(frame.noise==noise)&(frame.arm==arm)].test_accuracy
                values.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
            diff=pairs[(pairs.dataset==name)&(pairs.noise==noise)&(pairs.arm=='GB64')].accuracy_minus_ce_pp
            positive=int((diff>1e-5).sum())
            lines.append(f'| {name} | {noise} | '+' | '.join(values)+f' | {diff.mean():+.2f} ± {diff.std(ddof=1):.2f} | {positive}/5 |')
            decisions[name][noise]=dict(mean_accuracy_gain_pp=float(diff.mean()),paired_std_pp=float(diff.std(ddof=1)),
                improved_splits=positive,clean_mean_gain_pp=float(clean_delta),
                screening_pass=bool(noise!='clean' and diff.mean()>=1 and positive>=4 and clean_delta>=-1))
    lines+=['','## 数据、实际球数与证据质量','',
            '| 图 | 节点/类别 | split0训练/验证/测试 | 一测试节点对应pp | 边同配率/校正 | GB64实际球数，seed0–4 | 最大训练球占比mean |',
            '|---|---|---|---:|---|---|---:|']
    for name in ['texas','wisconsin','roman']:
        base=(runner.PREVIOUS if name=='roman' else ROOT)/'runs'/name/'grace'
        stats=json.loads((base/'graph_stats.json').read_text(encoding='utf-8'))
        graph=runner.load_graph(name,0);counts=[];maxfractions=[]
        for seed in range(5):
            graph_seed=runner.load_graph(name,seed)
            with np.load(base/f'seed{seed}'/'assignments.npz') as a:
                groups=a['GB64'];counts.append(len(np.unique(groups)))
                maxfractions.append(float(np.unique(groups[graph_seed.train_index],return_counts=True)[1].max()/len(graph_seed.train_index)))
        splits=f'{len(graph.train_index)}/{len(graph.val_index)}/{len(graph.test_index)}'
        lines.append(f'| {name} | {graph.num_nodes}/{graph.num_classes} | {splits} | {100/len(graph.test_index):.2f} | '+
            f"{stats['edge_homophily']:.4f}/{stats['adjusted_homophily']:.4f} | {counts} | {100*np.mean(maxfractions):.2f}% |")
    lines+=['','## 目标Brier：五split均值','',
            '| 图 | 条件 | CE | GB64 | 匹配KNN64 | GB64正干净margin节点比例 |',
            '|---|---|---:|---:|---:|---:|']
    for name in ['texas','wisconsin','roman']:
        src=(runner.PREVIOUS if name=='roman' else ROOT)/'target_metrics.csv'
        metrics=pd.read_csv(src);metrics=metrics[(metrics.dataset==name)&(metrics.representation=='grace')]
        for noise in ['clean','uniform','pair']:
            vals=[metrics[(metrics.noise==noise)&(metrics.arm==arm)].target_brier.mean() for arm in ['CE','GB64','KNN64']]
            margin=metrics[(metrics.noise==noise)&(metrics.arm=='GB64')].positive_clean_margin_fraction.mean()
            lines.append(f'| {name} | {noise} | '+' | '.join(f'{v:.5f}' for v in vals)+f' | {100*margin:.2f}% |')
    lines+=['','## 结论与决策','',
       '- Texas：Uniform下有均值改善，但Pair几乎无增量，干净条件明显退化；不支持不分条件迁移原共识。',
       '- Wisconsin：带噪均值有改善，但划分波动和干净条件损害必须一起看，不能只报有利均值。',
       '- Roman：将单seed观察扩为五split；按配对结果判读小增量，仍只有一个SSL seed。',
       '- 本轮没有GNN分类器对照，不能据此声称MLP优于异配GNN；也没有OFA消融，不能声称OFA有效。',
       '- 整体决策：继续分层诊断、修改下一阶段研究假设；暂缓多模块整合。先比较原X/GRACE/属性自监督空间，固定GB与软标签及学生，再看分裂/局部margin是否改善。',
       '- OFA优先考虑标签无关的独立预训练表示消融，冻结表示后再沿用LCSS；联合训练OFA与带噪分类另立实验，避免混淆优化冲突。',
       '- 本次WebKB来自同一个领域，不能支撑宽泛异配泛化。未来再加Amazon-ratings等不同领域大图，但不将其写为已完成。',
       '- 官方划分、模型与SSL预算不同于HOG-GCN/HeterGCL论文，文献准确率仅作机制背景，不能横向排名。',
       '','## 追溯','',
       'PLAN.md、data/*_source.json、adapter_manifest.json、representations/*/grace/config/training/summary、runs/*/grace/*的源/划分/目标/诊断、students/*的config/result/logits、raw.log、student_results/summary/paired_deltas、decision.json、audit/report.json。']
    d.write(ROOT/'decision.json',dict(decisions=decisions,threshold_scope='Exploratory screen; not significance testing',
        overall='continue mechanism diagnosis; no OFA efficacy claim'))
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit='passed',student_runs=len(results),decisions=decisions),ensure_ascii=True))


if __name__=='__main__':main()
