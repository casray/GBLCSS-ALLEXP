"""Publish audited numerical findings, retaining failed target/classification gates."""
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import common as c


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])


def fmt(values,scale=1,digits=2):
    a=np.asarray(values,dtype=float)*scale
    return f'{a.mean():.{digits}f}±{a.std(ddof=1):.{digits}f}'


def main():
    audit=c.read(c.ROOT/'audit/report.json')
    if audit['status']!='passed':raise ValueError('Final audit must pass')
    c.verify_manifest(c.ROOT/'source_manifest.json');c.verify_manifest(c.ROOT/'student_source_manifest.json');c.verify_manifest(c.ROOT/'preparation_manifest.json')
    targets=pd.read_csv(c.ROOT/'target_metrics.csv');students=pd.read_csv(c.ROOT/'student_results.csv');gate=c.read(c.ROOT/'gate.json')
    data=c.read(c.ROOT/'dataset_manifest.json')['datasets']
    rows=[]
    for name in c.CONFIG['datasets']:
        for noise in c.conditions(name):
            v=students[(students.dataset==name)&(students.noise==noise)]
            rows.append([name,noise,'ROC-AUC' if name=='tolokers' else 'Accuracy']+[fmt(v[v.arm==a].test_primary,100) if (v.arm==a).any() else '未过目标门槛，未训练' for a in c.ARMS])
    lines=['# 属性与结构差异互补：完整实验结论','','2026-10-15。基于冻结PLAN/config、原始日志、保存目标/预测及独立审计。',
        '',f'完成四图五协议seed三条件七臂，共{len(targets)}条目标诊断、{len(students)}次新X-only MLP训练。新两图Amazon-ratings/Tolokers各完成1000轮GRACE；Roman/CS复用原冻结表示。',
        '',f"目标门槛通过图：{', '.join(gate['selected_datasets']) or '无'}。R/SHUFFLE候选分类仅按该门槛推进，未训练的格子不是缺失结果或失败运行。",
        '', '## 研究问题、角色、改动及机制', '',
        '原属性区域能否替代预训练或纯拓扑区域组织带噪监督？固定R=[rowL2(X),rowL2(PX−X)]/sqrt2是否提供超过X及残差置换的增量？粒球只组织原LCSS留一软监督，不作为硬对比标签。',
        '', '全部新代码和产物在本目录；原分球/targets/models默认入口未改。TREE统一固定minleaf16/mintrain16、ceil√N上限，协议seed；OLD64保留原64/32/5%规则；BFS为既有确定性拓扑规则。球数和支持可能不同，不能把整体差异全部归因于结构距离。',
        '', '残差通道用去环去重无向邻接的邻居和除度数，CPU float64计算后转float32，避免常量特征被float32归一化误差放大。孤立P行=0，残差=−X；零范数行保持0。shuffle_seed20261015固定置换残差行，原属性不变。',
        '', '## 数据与协议', '',
        table(['图','节点','属性维度','类别','边同配率','校正同配率','类别数量'],[[n,data[n]['nodes'],data[n]['features'],data[n]['classes'],f"{data[n]['edge_homophily']:.4f}",f"{data[n]['adjusted_homophily']:.4f}",data[n]['class_counts']] for n in c.CONFIG['datasets']]),
        '', '新两图与Roman保留官方前五split，CS保留原百分比split（整数截断的既有未分配节点也保留）。仅污染训练标签，验证/测试干净。Tolokers二分类只报告对称翻转.2/.4，不把等价Uniform与Pair重复计入；其余clean/U.4/Pair.4。',
        '', '每图仅一个SSLseed；五协议seed共同改变split/noise/分球/学生初始化，不能称五次独立预训练。Tolokers两输出softmax软CE，按clean val正类概率AUC选epoch，区别于官方单logitBCE；其余按accuracy。所有学生200轮、原hidden64/2层BN/dropout.5/Adam.01/wd.0005，严格>选最早最佳epoch。',
        '', '## 目标门槛', '', 'R相对X及SHUFFLE，在两种噪声下均需平均Brier≤−.005且至少4/5改善。门槛只读冻结后训练真值的目标诊断，不读验证/测试分类成绩；它是探索筛查，不是无真值部署规则。', '']
    gaterows=[]
    for n,v in gate['datasets'].items():
        for k in v['checks']:gaterows.append([n,k['noise'],k['reference'],f"{k['mean_delta']:+.6f}±{k['std']:.6f}",f"{k['improved_seeds']}/5",k['passed']])
    lines += [table(['图','条件','R−参照','Brier配对差','改善seed','通过'],gaterows),'','## 最终主指标：百分比，均值±样本标准差','','不同图的Accuracy与ROC-AUC不跨图混为同一平均分。', '',table(['图','条件','主指标']+c.ARMS,rows),'','## 配对分类变化：百分点','']
    raw=[c.read(p) for p in sorted((c.ROOT/'students').glob('*/seed*/*/*/result.json'))]
    lookup={(r['dataset'],r['seed'],r['noise'],r['arm']):r for r in raw}
    paired=[]
    for key,r in lookup.items():
        if r['arm'] not in ['R_TREE','X_TREE','BFS']:continue
        for ref in ['CE','OLD64','H_TREE','X_TREE','SHUFFLE_TREE']:
            other=lookup.get((*key[:3],ref))
            if other and ref!=r['arm']:
                paired.append(dict(dataset=r['dataset'],seed=r['seed'],noise=r['noise'],arm=r['arm'],reference=ref,delta_pp=100*(r['test_primary']-other['test_primary'])))
    pairs=pd.DataFrame(paired)
    pairrows=[]
    decisions={}
    for name in c.CONFIG['datasets']:
        for (noise,arm,ref),frame in pairs[pairs.dataset==name].groupby(['noise','arm','reference']):
            pairrows.append([name,noise,arm+'−'+ref,fmt(frame.delta_pp,1,3),f'{int((frame.delta_pp>0).sum())}/5'])
        checks=[]
        if name in gate['selected_datasets']:
            for noise in c.conditions(name):
                p=pairs[(pairs.dataset==name)&(pairs.noise==noise)&(pairs.arm=='R_TREE')&(pairs.reference=='X_TREE')].delta_pp
                passed=bool(p.mean()>=-1) if noise=='clean' else bool(p.mean()>=.5 and (p>0).sum()>=4)
                checks.append(dict(noise=noise,delta_pp_mean=float(p.mean()),std=float(p.std(ddof=1)),positive=int((p>0).sum()),passed=passed))
        decisions[name]=dict(target_gate=gate['datasets'][name]['passed'],classification_checks=checks,classification_passed=bool(checks and all(x['passed'] for x in checks)))
    lines += [table(['图','条件','比较','配对均值±std','正增量seed'],pairrows),'','## 监督质量与结构增量','']
    quality=[]
    for name in c.CONFIG['datasets']:
        for noise in c.conditions(name):
            for arm in ['OLD64','H_TREE','X_TREE','R_TREE','SHUFFLE_TREE','BFS']:
                f=targets[(targets.dataset==name)&(targets.noise==noise)&(targets.arm==arm)]
                quality.append([name,noise,arm,fmt(f.target_brier,1,6),fmt(f.eligible_rate,100),fmt(f.eligible_q_accuracy,100),fmt(f.positive_clean_margin_fraction,100),fmt(f.clean_harm_rate,100)])
    lines += [table(['图','条件','臂','Brier','≥2票覆盖%','合格q准确率%','正干净边际/全部训练%','原正确标签误改%'],quality)]
    if (c.ROOT/'posthoc_prior_diagnostic.csv').exists():
        prior=pd.read_csv(c.ROOT/'posthoc_prior_diagnostic.csv')
        pr=[]
        for (name,noise,arm),f in prior[prior.arm.isin(['CE','X_TREE','R_TREE','GLOBAL_POSTHOC'])].groupby(['dataset','noise','arm']):
            pr.append([name,noise,arm,fmt(f.micro_brier,1,6),fmt(f.balanced_class_brier,1,6),fmt(f.worst_class_brier,1,6)])
        lines += ['', '### 事后类别先验诊断', '', '以下为额外解释性分析，未改变预设gate或训练集合，GLOBAL_POSTHOC没有训练学生。全局共识以全部其他训练节点的观测标签构造原LCSS；balanced Brier对真实训练类别等权，区别于按节点数平均的micro Brier。该分析不能用来事后更改成功标准。', '',table(['图','条件','臂','micro Brier','类别等权Brier','最差类别Brier'],pr)]
    lines += [
        '', 'q准确率限定合格投票节点；正边际要求真实类别严格超过所有竞争类别。整体q_accuracy含无支持均匀分布，不单独解释为语义质量。Brier改善不保证最终分类改善；原本正确/错误标签分组的Brier在原CSV完整保留。',
        '', '## 已支持、解释与未验证', '',
        '已支持的具体数值见上述配对表。只有R超过X和SHUFFLE才支持本候选结构对应关系的独立增量；超过BFS本身不足以立论。目标门槛或分类判据失败必须保留，不包装成普遍异配适配。',
        '', '合理解释：结构残差可能提供中心相对邻域的上下文，也可能加入无关距离维度；支持覆盖和区域边界的改变可能影响监督。这些属于与结果相容的解释，未经过单变量因果证明。',
        '', '尚未验证：独立SSLseed、Minesweeper/Questions、有图学生/蒸馏、支持感知合并及训练内树剪枝。不从本轮test峰值改权重、阈值或每图预算。干净MLP仅经验参照，不是X-only数学上界。',
        '', '若R与SHUFFLE相当或不如X，削弱本固定结构差异候选；若目标改善而分类不变，不能直接断言学生信息上界。先定位学生输入与监督可学习性，再设计新的有图路线。',
        '', '## 验证、运行与复现', '',
        '单元测试覆盖无标签残差、手算图、常量特征数值边界、置换、独立目标公式、指标选点与原trainer精确等价。独立审计核数据/H/source清单、五split/noise、分区支持、目标完整键、420冻结目标、gate与CSV，以及全部新学生logits和最佳epoch。具体计数以audit/report.json为准。',
        '', '预训练与学生均GPU队列串行；Roman/CS CPU目标与新图GPU预训练并行，计时不当作独占端到端效率比较。GRACE学习的表示在三个噪声条件/五协议种子间复用，不能把其成本重复加到每个学生上。',
        '', '运行命令依PLAN顺序prepare.py→run.py→audit.py --targets-only→run_students.py→audit.py→finish_report.py，解释器D:/codex_temp/noisygl-rocm721/Scripts/python.exe -X utf8。raw日志、config、data/representation/source manifests、views、partitions/trace、targets、students/config/history/result/logits、CSV和mean/std完整保存。',
        '', '## 最终决策', '', '逐图判定见decision.json。目标门槛未过的图暂缓结构候选；分类判据未过不扩大阈值搜索。即便某图通过探索判据，也仅支持该图条件下的候选信号，正式主张仍需新seed与强普通邻域对照。']
    c.write(c.ROOT/'decision.json',dict(status='completed',target_rows=len(targets),new_students=len(students),selected_datasets=gate['selected_datasets'],datasets=decisions,
        research_decision='保留局部信号，需独立确认' if any(v['classification_passed'] for v in decisions.values()) else '暂缓本固定结构残差候选；保留基础对照与负结果'))
    (c.ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    history=['# 属性与结构差异实验历史','',f'2026-10-15完成。{len(targets)}目标、{len(students)}新学生，审计通过。',
        '', '研究假设：属性局部区域提供可靠监督，正确对应的结构残差应超过纯属性及打乱残差。粒球角色：留一软证据组织。动机：BFS连通性不保证同类多数。',
        '', '代码改动：仅独立实验目录。配置：四图五协议seed、三条件、七目标臂，固定支持16、sqrt上限、原LCSS和MLP；Tolokers按AUC。预期机制、成功/失败标准及完整结果在PLAN与实验结论。',
        '', '最终决策：'+c.read(c.ROOT/'decision.json')['research_decision']+'。原始config/raw logs/targets/logits/seed/mean/std/audit均保留，不以未训练候选作已验证结论。']
    (c.ROOT/'02_experiment_history_and_lessons.md').write_text('\n'.join(history)+'\n',encoding='utf-8')
    c.write(c.ROOT/'final_validation.json',dict(status='passed',target_rows=len(targets),student_rows=len(students),audit_sha256=c.sha(c.ROOT/'audit/report.json'),report_sha256=c.sha(c.ROOT/'实验结论.md'),decision_sha256=c.sha(c.ROOT/'decision.json'),reporting_script_sha256=c.sha(__file__)))
    print(c.read(c.ROOT/'decision.json'))


if __name__=='__main__':main()
