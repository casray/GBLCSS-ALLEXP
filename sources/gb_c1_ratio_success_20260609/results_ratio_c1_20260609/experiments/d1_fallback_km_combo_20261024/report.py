"""Chinese descriptive results; no posthoc training decisions."""
import csv
from collections import Counter
import statistics
from pathlib import Path

from protocol import ROOT,read,save,sha,now,csv_save


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def report():
    audit=read(ROOT/'audit.json')
    assert audit['status']=='passed'
    summaries=table('paired_summary.csv')
    new=table('seeds12_paired_summary.csv')
    physical=read(ROOT/'physical_fits.json')
    selections=[read(p) for p in (ROOT/'selections').rglob('*.json')]
    lines=['# D1回退、匹配KM与同树C1+D1实验报告','',
        f'完成：{now()}。本轮新科学fit {len(physical)}，每个200epoch；4个两轮smoke。全部旧结果只读引用，目标内容寻址去重。',
        '五图三seed、clean/U.4/U.6/Pair.4。零fit回退覆盖120格；新KM与同树组合仅60格R0。固定历史H/SSL与对应seed的split、noise、初始化。',
        '本轮沿用已经开发和查看过的历史结果，归为预先固定规则的回顾性探索。规则、源码和复用清单冻结，全部选择冻结后才评分测试集；验证集同时用于checkpoint与配置选择。','',
        '## 假设与控制','',
        '1. CE及全局力度回退可缓解强制GROUP的clean/F1代价。accuracy主selector；balanced次selector=(validation accuracy+macro-F1)/2，仅使用每臂既定最佳准确率checkpoint，平票回原accuracy/NLL/epoch/稳定臂序。两个selector同时应用于基线和扩展池。',
        '2. KM与粒球均采用四全局候选及八扩展候选；同一归一化H、相同区域数、相同学生协议。比较预算匹配的选择结果，并保存固定臂结果。',
        '3. 深树固定S32/历史C1 PROBE选支持量 × global-only/expanded四格。PROBE若为CE，自适应分支保持CE。原C1所选配置单列，same-support global refinement单列；支持量不会依据新GROUP结果重选。',
        '粒球用于局部留一软标签证据。beta=.5/min_votes2、四种分组力度及平票规则保持历史定义。全部条件完成，无新增成功硬门。','',
        '## 带噪宏均值','',
        '单位pp。先在seed内对五图三噪声等权平均，再计算seed均值和样本SD(ddof=1)。新增seed1/2单列。',
        '|selector|骨干|方法−对照|三seed accuracy±SD|新seed accuracy±SD|三seed macro-F1差|',
        '|---|---|---|---:|---:|---:|']
    for r in summaries:
        if r['dataset']!='ALL' or r['condition']!='NOISY_MACRO':
            continue
        n=next(v for v in new if all(v[k]==r[k] for k in ['stage','dataset','backbone','condition','selector','method','reference']))
        lines.append(f'|{r["selector"]}|{r["backbone"]}|{r["method"]}−{r["reference"]}|'+
            f'{float(r["test_accuracy_diff_pp_mean"]):+.3f} ± {float(r["test_accuracy_diff_pp_std_ddof1"]):.3f}|'+
            f'{float(n["test_accuracy_diff_pp_mean"]):+.3f} ± {float(n["test_accuracy_diff_pp_std_ddof1"]):.3f}|{float(r["test_macro_f1_diff_pp_mean"]):+.3f}|')
    lines+=['','## 主selector逐图结果','','|骨干|图|方法−对照|三seed带噪accuracy±SD|新seedaccuracy差|','|---|---|---|---:|---:|']
    wanted={('LEGACY_EXPANDED','LEGACY_GLOBAL'),('LEGACY_EXPANDED','KM_EXPANDED'),('ADAPTIVE_EXPANDED','ADAPTIVE_GLOBAL'),
            ('ADAPTIVE_EXPANDED','FIXED_EXPANDED'),('ADAPTIVE_EXPANDED','SINGLE_MODULE_VAL'),('ADAPTIVE_GLOBAL','ORIGINAL_C1')}
    for r in summaries:
        if r['selector']!='accuracy' or r['dataset']=='ALL' or r['condition']!='NOISY_MACRO' or (r['method'],r['reference']) not in wanted:
            continue
        n=next(v for v in new if all(v[k]==r[k] for k in ['stage','dataset','backbone','condition','selector','method','reference']))
        lines.append(f'|{r["backbone"]}|{r["dataset"]}|{r["method"]}−{r["reference"]}|{float(r["test_accuracy_diff_pp_mean"]):+.3f} ± {float(r["test_accuracy_diff_pp_std_ddof1"]):.3f}|{float(n["test_accuracy_diff_pp_mean"]):+.3f}|')
    lines+=['','## Clean与F1代价','','所有clean配对结果均在paired_summary；下表突出clean accuracy下降>2pp或macro-F1下降>1pp。','',
            '|selector|seed范围|骨干|图/条件|方法−对照|accuracy差|F1差|','|---|---|---|---|---|---:|---:|']
    warnings=[]
    for subset,rows in [('0,1,2',summaries),('1,2',new)]:
        for r in rows:
            if r['dataset']=='ALL' or r['condition']=='NOISY_MACRO':
                continue
            a,f=float(r['test_accuracy_diff_pp_mean']),float(r['test_macro_f1_diff_pp_mean'])
            if (r['condition']=='clean' and a < -2) or f < -1:
                warnings.append(dict(subset=subset,**r))
                lines.append(f'|{r["selector"]}|{subset}|{r["backbone"]}|{r["dataset"]}/{r["condition"]}|{r["method"]}−{r["reference"]}|{a:+.3f}|{f:+.3f}|')
    csv_save(ROOT/'warnings.csv',warnings)
    lines+=['','## 回退与搜索成本','']
    countrows=[]
    for b in ['R0','R2']:
        for selector in ['accuracy','balanced']:
            for condition in ['clean','uniform0.4','uniform0.6','pair0.4']:
                cells=[v for v in selections if v['backbone']==b and v['condition']==condition and v['seed'] in [1,2]]
                counts=Counter(v['choices'][selector]['LEGACY_EXPANDED'].split('/')[-1] for v in cells)
                countrows.extend(dict(backbone=b,selector=selector,condition=condition,seeds='1,2',arm=a,count=c) for a,c in counts.items())
                lines.append(f'- {b}/{selector}/{condition} 新seed回退分布：{dict(counts)}。')
    csv_save(ROOT/'fallback_counts.csv',countrows)
    cost=table('deployment_costs.csv')
    cost=[r for r in cost if r['selector']=='accuracy']
    seconds=sum(read(ROOT/r['folder']/'result.json')['seconds'] for r in physical.values())
    lines+=['',f'本轮实际新学生更新{len(physical)*200}次，训练循环计时{seconds:.3f}s，输入、哈希、序列化与分析时间另计。历史完整C1网格已支付432000候选更新，作为上游既有研究成本披露。',
        '预算匹配KM/GB的global-only各800更新，expanded各1600；本轮增量复用CE/KMλ1与全部legacy。深树固定支持global-only800、expanded1600。',
        f'自适应分支部署成本逐cell含历史PROBE1000+CE200，再补齐同支持三λ未完成轮数，最后四group800：global总更新范围{min(int(r["global_total_updates"]) for r in cost)}–{max(int(r["global_total_updates"]) for r in cost)}，expanded范围{min(int(r["expanded_total_updates"]) for r in cost)}–{max(int(r["expanded_total_updates"]) for r in cost)}。',
        '缓存中的完整λ候选在本次GPU上通过别名复用；部署估计仍补计原PROBE尚未支付的更新。CE分支保留上游1200成本。共同执行四格时共享支持、CE和相同目标；单分支成本与实际新拟合数量分别报告。',
        'per_class_recall.csv与strata.csv提供逐类召回、干净/错标及同意/冲突/平票交叉诊断；真值仅用于冻结后的分析。','',
        '## 审计与决策说明','',f'审计passed：{audit["new_physical_fits"]}新fit，{audit["logical_references"]}逻辑引用，{audit["cells"]}冻结cell，{audit["reloads"]}重载样本最大概率误差{audit["reload_max_error"]}。',
        '研究判断依赖新seed的方向、跨图覆盖、与匹配强对照的差及clean/F1成本。+.3–.5pp作组合量级参照，无自动成功硬门；所有负结果保留。',
        '已支持事实为表中配对结果及审计；分组改变纠错/保留权衡是机制解释；独立SSL随机性、新图迁移与额外规则仍待验证。后续结论补充于报告末尾，当前脚本不会发起下一轮实验。']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'report_complete.json',dict(finished=now(),actual_new_fits=len(physical),seconds=seconds,warnings=len(warnings)))
