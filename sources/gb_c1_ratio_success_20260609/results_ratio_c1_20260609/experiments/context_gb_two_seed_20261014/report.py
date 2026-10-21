"""Post-run descriptive analysis; never selects a training configuration."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from pingda import build_generators

LABELS={'CE':'观测标签CE','grace_h':'GRACE-H','pingda_h':'PiNGDA-H',
        'pingda_mean_r':'PiNGDA固定均值R','context_h':'上下文训练后的H',
        'context_r':'上下文训练后的R','hreg_h':'节点VICReg对照H'}


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();output=ROOT/'runs'
    status=json.loads((output/'status.json').read_text(encoding='utf-8'))
    complete=status['status']=='completed' and status.get('student_files')==684
    if not complete and not args.partial:raise ValueError('all registered experiments must finish')
    r=rows(output/'results.csv');summary=rows(output/'summary.csv');delta=rows(output/'paired_deltas.csv')
    audit_path=output/'audit/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
    verified=complete and audit.get('coverage_complete') is True and not audit.get('errors',['missing audit'])
    lookup={(v['dataset'],v['noise'],v['head'],v['space'],int(v['budget'])):v for v in summary}
    lines=['# Context-GB 两seed实验结论','',
        '日期：2026-10-14。状态：'+('完整队列已完成，严格独立审计通过。' if verified else '完整队列已完成，审计状态待确认。' if complete else '中途快照；不作为最终决策。'),'',
        f'实际已汇总{len(r)}次分类器训练；预训练完成文件{len(list((output/"pretrain").glob("*/seed*/*/complete.json")))}份。',
        '预注册范围：Cora/CiteSeer/CoraFull，独立完整seed0/1，clean/Uniform0.4/Pair0.4，MLP/GCN，预算64/128/256。',
        '所有编码器统一200轮；CoraFull不同于旧论文1000轮。PiNGDA为统一骨干的增强适配。用户本轮排除了kNN/K-means比较，因此不能判断粒球独特优势。',
        '表中均值±样本标准差单位为百分比/百分点；两个seed仅作探索，不作显著性结论。64是预注册主预算，128/256完整保留敏感性结果，不能看test后逐图挑预算。','']
    for head in ['mlp','gcn']:
        lines += [f'## {head.upper()}，主预算64','',
                  '| 数据/条件 | CE | GRACE-H | PiNGDA-H | 固定均值R | 上下文H | 上下文R | 节点VICReg-H |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for dataset in ['cora','citeseer','corafull']:
            for noise in ['clean','uniform','pair']:
                values=[]
                for space in LABELS:
                    v=lookup.get((dataset,noise,head,space,0 if space=='CE' else 64))
                    if v and v['sample_std_pp']:values.append(f"{float(v['mean_accuracy_pct']):.2f}±{float(v['sample_std_pp']):.2f}")
                    elif v:values.append(f"{float(v['mean_accuracy_pct']):.2f}（n={v['seed_count']}）")
                    else:values.append('待完成')
                lines.append('| '+dataset+'/'+noise+' | '+' | '.join(values)+' |')
        lines += ['','### 上下文R的配对变化（预算64）','',
                  '| 数据/条件 | 相对PiNGDA-H | 相对固定均值R | 相对节点VICReg-H | 相对GRACE-H |','|---|---:|---:|---:|---:|']
        for dataset in ['cora','citeseer','corafull']:
            for noise in ['clean','uniform','pair']:
                values=[]
                for ref in ['pingda_h','pingda_mean_r','hreg_h','grace_h']:
                    x=[float(v['accuracy_delta_pp']) for v in delta if v['dataset']==dataset and v['noise']==noise and
                       v['head']==head and v['space']=='context_r' and int(v['budget'])==64 and v['reference']==ref]
                    values.append(f'{np.mean(x):+.2f}±{np.std(x,ddof=1):.2f}' if len(x)==2 else '待完成')
                lines.append('| '+dataset+'/'+noise+' | '+' | '.join(values)+' |')
        lines += ['']
    decisions={}
    for head in ['mlp','gcn']:
        data=[v for v in delta if v['head']==head and v['space']=='context_r' and int(v['budget'])==64 and
              v['reference']=='pingda_h' and v['noise'] in ['uniform','pair']]
        if len(data)==12:
            values=[float(v['accuracy_delta_pp']) for v in data]
            ds={name:float(np.mean([float(v['accuracy_delta_pp']) for v in data if v['dataset']==name]))
                for name in ['cora','citeseer','corafull']}
            decisions[head]={'mean_noisy_delta_pp':float(np.mean(values)),'dataset_noisy_delta_pp':ds,
                             'meets_numeric_gate':bool(np.mean(values)>=.5 and min(ds.values())>=-.5),
                             'gate_scope':'relative to PiNGDA-H, B64; additional controls still required'}
    lines += ['## 预注册门槛','','```json',json.dumps(decisions,ensure_ascii=False,indent=2),'```','',
              '上述门槛只判断上下文R相对PiNGDA-H的实用筛查；还需检查固定均值与节点VICReg对照、干净副作用和两个seed方向。不能把跨条件平均视为独立统计样本。','',
              '## 表示与分球机制','',
              '| 数据/seed | 预训练 | H有效秩 | C有效秩 | H均值范数 | C均值范数 | 峰值GiB |',
              '|---|---|---:|---:|---:|---:|---:|']
    for p in sorted((output/'pretrain').glob('*/seed*/*/complete.json')):
        info=json.loads(p.read_text(encoding='utf-8'))
        lines.append(f"| {p.parents[2].name}/{p.parents[1].name} | {p.parent.name} | {info['h_geometry']['effective_rank']:.2f} | {info['c_geometry']['effective_rank']:.2f} | {info['h_geometry']['mean_norm']:.4f} | {info['c_geometry']['mean_norm']:.4f} | {info['peak_gpu_gib']:.2f} |")
    lines += ['','C在pingda_hreg记录中为普通均值读出，不是Hreg投影头；不能把该列当作Hreg头的秩。训练VICReg抑制常数输出，但低秩与语义过平滑仍可出现。','',
              '| 数据/seed | 分球空间 | 上限 | 实际球数 | 有效节点/总数 |','|---|---|---:|---:|---:|']
    for p in sorted((output/'partitions').glob('*/seed*/*/*/summary.json')):
        info=json.loads(p.read_text(encoding='utf-8'))
        lines.append(f"| {p.parents[3].name}/{p.parents[2].name} | {p.parents[1].name} | {p.parent.name} | {info['summary']['num_balls']} | {info['valid_nodes']}/{info['total_nodes']} |")
    lines += ['','## 证据聚合质量','',
              '| 数据/seed/噪声 | 空间 | Brier | 支持覆盖 | 正clean-vote margin覆盖 | 回退节点 |',
              '|---|---|---:|---:|---:|---:|']
    exact_path=output/'audit/exact_evidence.csv'
    exact={(v['dataset'],int(v['seed']),v['noise'],v['space'],int(v['budget'])):v for v in rows(exact_path)} if exact_path.exists() else {}
    for p in sorted((output/'targets').glob('*/seed*/*/*/B64/metrics.json')):
        info=json.loads(p.read_text(encoding='utf-8'))
        precise=exact.get((info['dataset'],info['seed'],info['noise'],info['space'],64))
        if precise:
            for field in ['voting_support_coverage','positive_clean_vote_margin_coverage']:
                info[field]=float(precise[field])
        lines.append(f"| {info['dataset']}/{info['seed']}/{info['noise']} | {LABELS[info['space']]} | {info['soft_target_brier']:.4f} | {info['voting_support_coverage']:.3f} | {info['positive_clean_vote_margin_coverage']:.3f} | {info['fallback_nodes']} |")
    lines += ['',('本表支持率和clean-margin覆盖优先使用audit/exact_evidence.csv，逐节点按实际投票集合重算，含完整H回退。原targets目录metrics.json中的混合assignment统计仍保留，不能混淆两者。Brier直接重算自保存目标。' if exact else '当前未生成exact_evidence.csv；R的混合assignment统计在回退节点上可能与实际投票集合不同，待独立公式审计补齐。'),'',
              '## 局限、结果归因与决策','',
              '- 固定两seed、固定λ=0.1；没有逐图优化超参，没有独立KM/KNN必要性检验。','- 所有预算结果保留，不能按test逐图挑选最优空间/球数。',
              '- 仅loss下降、球数增加或Brier改善不足以推出分类有效。新方案不强拉同球，不让球决定增强，也未用noisy CE联合优化编码器。',
              '- 新上下文来自全图H的邻居池化；它不是独立诱导子图编码，LS-GCL/CL-GCL完整基座仍未执行。',
              '- 本轮最终解释、局部正信号及暂缓决策见[综合结论与接续](综合结论与接续.md)。数值门槛不能独立替代关键消融和与原GRACE的比较。','',
              '完整结果：runs/results.csv、summary.csv、paired_deltas.csv；独立审核：runs/audit/report.json。所有训练config/logits、source hashes及输入图hash已保存。']
    (ROOT/('中途结果.md' if args.partial else '实验结论.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'decision_metrics.json').write_text(json.dumps({'complete':complete,'registered_gate':decisions},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'complete':complete,'students':len(r),'decisions':decisions},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
