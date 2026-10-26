"""Chinese report from frozen, audited tables; no new scientific thresholds."""
import csv
from collections import Counter

from protocol import ROOT,read,save,now


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def report():
    audit=read(ROOT/'audit.json');analysis=read(ROOT/'analysis_complete.json')
    paired=table('paired_summary.csv');new=table('seeds12_paired_summary.csv')
    selected=table('selected_results.csv');costs=table('costs.csv')
    text=['# R2 + SUPPORT + D1 + C1 集成实验报告','',
        '## 研究问题与协议','',
        '在冻结历史 GRACE 与每seed R2编码器上，检验支持量、分组力度与短训联合选择的增量。'
        '粒球提供 LOO 局部软标签证据，R2冻结编码器/BN，只训练属性学生头。'
        '五图、clean/U.4/U.6/Pair.4、seed0/1/2，共60格；所有图与seed已有开发历史，定位为回顾性集成验证。','',
        'beta=.5、min_votes=2、平票不锐化；D1使用原始投票唯一赢家定义同意/冲突。'
        '本轮明确修改平票力度为λ1，固定S32与联合候选完全一致。'
        '训练目标仅由observed标签构建，选择仅用clean validation；真值诊断与test评分在全部选择冻结后进行。','',
        '42候选=6支持×(3全局力度+4分组规则)，42→14→4，25/75/200轮。'
        '强制S32λ1包含在存活名额内。排名依次为验证accuracy、最佳checkpoint NLL、'
        '全训练节点平均施加λ、更大支持、更早best epoch、冻结ID（CE/global优先）。'
        '18→6→2全局对照沿用原排名，逐格验证旧PROBE和两轮survivors。','',
        '## 实现与复用','',
        'fit.py与上轮验证内核逐字节一致；checkpoint格式head_with_encoder_reference。'
        '历史全局候选通过完整轨迹的因果前缀回放复用，当前前缀之外的成绩不进入排名。'
        '所有新D1真实训练，逐arm保存及恢复optimizer/RNG，25/75前缀快照保留。'
        '联合决策冻结后补齐固定S32控制。OLD64旧格式缺完整input/RNG契约，60格按共同内核重训并在末尾做概率桥。'
        '未训练R0、SSL、KM/kNN、教师或额外seed。','',
        '## 带噪总体结果','',
        '先对每seed内五图×三噪声条件等权平均，再报三seed均值和样本SD(ddof1)，单位pp。'
        'seed1/2单列用于观察方向；本轮三seed均有先前开发历史。','',
        '|比较|accuracy均值±SD|macroF1均值±SD|seed1/2 accuracy均值±SD|',
        '|---|---:|---:|---:|']
    key=lambda r:(r['dataset'],r['condition'],r['method'],r['reference'])
    ni={key(r):r for r in new}
    for r in paired:
        if r['dataset']!='ALL' or r['condition']!='NOISY_MACRO':continue
        a=float(r['test_accuracy_diff_pp_mean']);sd=float(r['test_accuracy_diff_pp_std_ddof1'])
        f=float(r['test_macro_f1_diff_pp_mean']);fs=float(r['test_macro_f1_diff_pp_std_ddof1'])
        nn=ni[key(r)]
        text.append(f"|{r['method']}−{r['reference']}|{a:+.3f}±{sd:.3f}|{f:+.3f}±{fs:.3f}|{float(nn['test_accuracy_diff_pp_mean']):+.3f}±{float(nn['test_accuracy_diff_pp_std_ddof1']):.3f}|")
    text+=['','## 逐图主比较及clean/F1代价','','|图/条件|参照|accuracy差±SD|macroF1差±SD|','|---|---|---:|---:|']
    for r in paired:
        if r['dataset']=='ALL' or r['method']!='ADAPTIVE_EXPANDED':continue
        if r['condition']=='NOISY_MACRO' and r['reference']=='SINGLE_MODULE_VAL' or r['condition']=='clean' and r['reference'] in ('CE','FIXED_GLOBAL','SINGLE_MODULE_VAL'):
            text.append(f"|{r['dataset']}/{r['condition']}|{r['reference']}|{float(r['test_accuracy_diff_pp_mean']):+.3f}±{float(r['test_accuracy_diff_pp_std_ddof1']):.3f}|{float(r['test_macro_f1_diff_pp_mean']):+.3f}±{float(r['test_macro_f1_diff_pp_std_ddof1']):.3f}|")
    primary=next(r for r in paired if key(r)==('ALL','NOISY_MACRO','ADAPTIVE_EXPANDED','SINGLE_MODULE_VAL'))
    seeds=[float(primary[f'test_accuracy_diff_pp_seed{s}']) for s in (0,1,2)]
    mean=float(primary['test_accuracy_diff_pp_mean'])
    decision='保留探索性增量，停止本轮训练并结合成本评估' if mean>0 else '暂缓集成扩展，优先保留较简单单模块'
    text+=['','## 机制诊断与选择分布','',
        f"主比较三seed宏差：{seeds} pp。决策：{decision}。没有新增硬门槛，不自动扩实验。",'',
        f"逐类召回{analysis['per_class_rows']}行；干净/错标及同意/冲突/平票分层{analysis['strata_rows']}行。"
        '原表和三seed、seed1/2汇总均已保存；稀缺分层记录实际可用seed数。GCE的one-hot target仅表示其observed输入，GCE损失仍按q=.7计算。','']
    for c in ('clean','uniform0.4','uniform0.6','pair0.4'):
        counts=Counter(r['arm'] for r in selected if r['method']=='ADAPTIVE_EXPANDED' and r['condition']==c)
        text.append(f'- {c} 联合选择：{dict(counts)}。')
    updates=sum(int(r['new_updates']) for r in costs);seconds=sum(float(r['new_training_seconds']) for r in costs)
    text+=['','## 成本与审计','',
        f"实际新物理候选{audit['physical_unique']}，其中完整200轮{audit['physical_full']}；"
        f"科学更新{updates}，训练循环计时{seconds:.3f}s（I/O、哈希、审计另计）。4个2epoch smoke单列。",'',
        '联合race候选2250更新+CE200=2450；原全局1000+CE200=1200。固定global800、固定expanded1600。'
        '较强单模块验证选择还需支付两个分支的搜索成本，其并集上界2600更新（已扣重复CE）。'
        '逐格实际新训练、历史别名与部署预算在costs.csv分列。历史全局完整网格每格已付3600候选更新，'
        '当前回放新增更新为0。联合/全局搜索预算有差异，机制与预算贡献需共同解释。','',
        f"本地审计{audit['status']}：{audit['targets']}目标重建、{audit['races']}race重算、"
        f"{audit['prefix_resume_snapshots']}前缀状态、{audit['reload_count']}checkpoint重载，最大误差{audit['reload_max_error']}。"
        '源码及输入哈希未变。工程失败如有保留failures.jsonl；正式状态以complete.json为准。','',
        '已支持结论为上述配对结果及审计。分组力度改变保留/纠错权衡属于机制解释；独立SSL随机性和新图迁移尚待验证。'
        '此前R0匹配KM较强的历史结论继续保留，本轮按授权省略KM/kNN；R0只读参考另存，搜索/tie协议差异已注明。'
        '父侧独立审计由research目录维护，交付时另核其完成状态。','',
        '## 文件索引','',
        'config.json / sources.json / inputs.json / reuse_manifest.json / physical_fits.json / races / selections / '
        'selections_frozen.json / fits / targets / results.csv / selected_results.csv / summary.csv / paired_summary.csv / '
        'seeds12_* / per_class_recall* / strata* / costs.csv / old64_bridges.csv / audit.json。']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save(ROOT/'decision.json',dict(time=now(),decision=decision,primary_mean_pp=mean,primary_seedmeans=seeds,
        automatic_further_training=False,hard_new_threshold=False))
