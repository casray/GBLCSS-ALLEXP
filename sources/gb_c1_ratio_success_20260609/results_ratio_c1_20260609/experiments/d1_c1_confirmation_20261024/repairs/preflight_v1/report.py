"""Evidence-bound Chinese report from frozen tables; never triggers training."""
import csv
from collections import Counter
import json

import numpy as np

from protocol import ROOT, CONFIG, read, save, now, output, sha, csv_save
from analysis_stats import mean_std


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def report():
    audit = read(ROOT/'audit.json')
    if audit['status']!='passed':
        raise RuntimeError('Report requires successful full audit')
    pairs = table('paired_differences.csv')
    selected = table('selected_results.csv')
    regrets = table('c1_regret.csv')
    summary = table('paired_summary.csv')
    newsummary = table('seeds12_paired_summary.csv')
    lines = ['# GB-LCSS D1/C1 五图三 seed 确认报告','',
        '## 完成情况','',
        f'完成时间：{now()}。全部 3600 个科学拟合完成：D1 1440、C1 2160；另有 6 个两轮工程 smoke。',
        '五图：Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；四条件：clean、Uniform .4、Uniform .6、Pair .4；R0 主分析、冻结 R2 对照；协议/学生 seed 0/1/2。',
        '所有候选与参数均按验证集选择。120 个 PROBE 在各自完整网格补训前冻结；全部选择统一冻结后读取测试标签进行评分。',
        'seed0 是开发重放；seed1/2 的结果另列。历史 GRACE 固定，每个 seed 复用对应的 100 轮历史 R2 编码器。',
        '本轮采用完整新执行，没有整 cell 历史复用，没有 C2/D2/SimMLP、额外 seed 或模块组合。','',
        '## 核心结果','',
        '以下为三种带噪条件、五图等权；先在每个 seed 内平均，再计算 seed 间样本标准差（ddof=1）。单位 pp。','',
        '|阶段|骨干|对照|seed0|seed1|seed2|三seed均值±SD|seed1/2均值±SD|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    selected_refs = {'D1':['CELL_GLOBAL_VAL','EXACT_MASS_GLOBAL','MASS_MATCHED_SHUFFLE','CE','SUPPORT_L1.0','GCE_Q0.7','KM_L1.0'],
                     'C1':['FULL_VAL','FIXED_S32','LEGACY_SUPPORT32','SC','GCE_Q0.7','KM_L1.0']}
    for stage,refs in selected_refs.items():
        for backbone in CONFIG['backbones']:
            for ref in refs:
                r = next(r for r in summary if r['stage']==stage and r['backbone']==backbone and r['reference']==ref and r['dataset']=='ALL' and r['condition']=='NOISY_MACRO')
                n = next(n for n in newsummary if all(r[k]==n[k] for k in ('stage','backbone','reference','dataset','condition')))
                key = 'test_accuracy_diff_pp'
                values = '|'.join(f'{float(r[key+f"_seed{s}"]):+.3f}' for s in [0,1,2])
                lines.append(f'|{stage}|{backbone}|{ref}|{values}|{float(r[key+"_mean"]):+.3f} ± {float(r[key+"_std_ddof1"]):.3f}|{float(n[key+"_mean"]):+.3f} ± {float(n[key+"_std_ddof1"]):.3f}|')
    decisions = {}
    lines += ['', '## D1：分组力度','',
        '研究假设：同意/冲突分组能带来超过逐格最佳全局力度、精确均量全局和精确均量打乱的分类增量。粒球保持历史 SUPPORT32，只组织留一标签证据。',
        '四候选固定为 (agree,conflict)=(0,1)、(.25,1)、(1,.25)、(1,0)。平票组沿用本格验证最佳全局力度。',
        '先比较验证准确率，再比较所选 checkpoint 的 NLL，再比较较早 best_epoch，剩余平票保持 pilot 臂顺序。选定 GROUP 后才训练两个均量控制。','',
        '|骨干|数据集|GROUP−最佳全局 accuracy|macro-F1|新seed1/2 accuracy|',
        '|---|---|---:|---:|---:|']
    for b in CONFIG['backbones']:
        dataset_means = []
        for name in CONFIG['datasets']:
            r = next(r for r in summary if r['stage']=='D1' and r['backbone']==b and r['reference']=='CELL_GLOBAL_VAL' and r['dataset']==name and r['condition']=='NOISY_MACRO')
            n = next(n for n in newsummary if all(r[k]==n[k] for k in ('stage','backbone','reference','dataset','condition')))
            dataset_means.append(float(n['test_accuracy_diff_pp_mean']))
            lines.append(f'|{b}|{name}|{float(r["test_accuracy_diff_pp_mean"]):+.3f} ± {float(r["test_accuracy_diff_pp_std_ddof1"]):.3f}|{float(r["test_macro_f1_diff_pp_mean"]):+.3f}|{float(n["test_accuracy_diff_pp_mean"]):+.3f}|')
        rr = [r for r in pairs if r['stage']=='D1' and r['backbone']==b and r['reference']=='CELL_GLOBAL_VAL' and r['condition']!='clean']
        seedmeans = {s:float(np.mean([float(r['test_accuracy_diff_pp']) for r in rr if int(r['seed'])==s])) for s in [0,1,2]}
        controls = {ref:float(np.mean([float(r['test_accuracy_diff_pp']) for r in pairs if r['stage']=='D1' and r['backbone']==b and r['reference']==ref and r['condition']!='clean' and int(r['seed']) in [1,2]])) for ref in ['EXACT_MASS_GLOBAL','MASS_MATCHED_SHUFFLE']}
        positive = sum(v>0 for v in dataset_means)
        direction = '继续' if seedmeans[1]>0 and seedmeans[2]>0 and positive>=3 and min(controls.values())>0 else '修改/暂缓'
        decisions['D1_'+b] = dict(decision=direction,paired_seed_means_pp=seedmeans,newseed_positive_datasets=positive,newseed_control_differences_pp=controls)
        lines += ['',f'**{b} 决策：{direction}。** 新 seed 五图均值中 {positive}/5 为正；seed1/2 宏差 {seedmeans[1]:+.3f}/{seedmeans[2]:+.3f}pp；相对均量全局/打乱 {controls["EXACT_MASS_GLOBAL"]:+.3f}/{controls["MASS_MATCHED_SHUFFLE"]:+.3f}pp。']
    choices = [read(p) for p in (ROOT/'selections/D1').rglob('*.json')]
    lines += ['', '全部 120 格所选规则：`'+json.dumps(dict(Counter(r['selected'] for r in choices)),ensure_ascii=False)+'`。',
        '支持量、同意/冲突/平票比例及实际力度保存在 evidence 与 selections；训练真值仅在选择冻结后用于原干净/原错标、频数组与交叉分层诊断，见 d1_strata.csv。',
        '同意/冲突分配的解释由两种精确均量控制共同检查；目标 Brier、真实类质量与最终学生分类分别报告。','',
        '## C1：真实短训筛选','',
        '研究假设：同树六档支持量×三档力度的 18→6→2、25/75/200 真正续训，可在较低候选更新预算下接近完整验证网格。每轮强制 S32/lambda1 存活。',
        '每个候选独立恢复模型、Adam、BN、CPU/GPU/NumPy/Python RNG。PROBE 与共享 CE200 比较后冻结，再完成淘汰候选至 200 轮，得到 FULL_VAL。','',
        '|骨干|测试差≤0.5/1/2pp|最差/最好测试差|平均验证差|候选实际计时比 PROBE/FULL|',
        '|---|---:|---:|---:|---:|']
    for b in CONFIG['backbones']:
        rs = [r for r in regrets if r['backbone']==b]
        gaps = [float(r['test_gap_pp']) for r in rs]
        close = [float(np.mean([abs(g)<=t for g in gaps])) for t in [.5,1,2]]
        ratios = [float(r['candidate_time_ratio']) for r in rs]
        direction = '继续' if close[1]>=.9 and min(gaps)>=-2 else '修改'
        decisions['C1_'+b] = dict(decision=direction,closeness=close,worst_test_gap_pp=min(gaps),best_test_gap_pp=max(gaps))
        lines.append(f'|{b}|'+ '/'.join(f'{100*v:.2f}%' for v in close)+f'|{min(gaps):+.3f}/{max(gaps):+.3f}|{np.mean([float(r["validation_gap_pp"]) for r in rs]):+.3f}|{np.mean(ratios):.4f}|')
        lines += [f'']
    lines += [
        '每格 PROBE 候选更新 1000，另共享 CE200；完整候选网格 3600 更新。1000/3600=27.78%，加入 CE 后为 1200/3800=31.58%。本次实际支付完整网格审计费用，未将理论部署节省写成此次实际节省。',
        f'完整 C1 候选实际训练计时 {sum(float(r["full_candidate_seconds"]) for r in regrets):.3f}s；PROBE 因果前缀合计 {sum(float(r["probe_candidate_seconds"]) for r in regrets):.3f}s。序列化、输入与后处理耗时另在 events/环境和完整控制台轨迹。',
        'PROBE−FULL_VAL 测试差保留正负号。FULL_VAL 依据验证选择，测试上 PROBE 更高的格全部保留。逐格选择、差值、最差格、更新预算和计时见 c1_regret.csv/c1_costs.csv。','',
        '## Clean 与小类代价','',
        '以下列出所选方法对 CE 的三seed均值 clean 损失超过2pp，或任意条件 macro-F1 损失超过1pp的条目。',
        '|阶段|骨干|数据集|条件|accuracy差|macro-F1差|','|---|---|---|---|---:|---:|']
    warnings = []
    for r in summary:
        if r['reference']!='CE' or r['dataset']=='ALL' or r['condition']=='NOISY_MACRO':
            continue
        acc,f1 = float(r['test_accuracy_diff_pp_mean']),float(r['test_macro_f1_diff_pp_mean'])
        if (r['condition']=='clean' and acc < -2) or f1 < -1:
            warnings.append(r)
            lines.append(f'|{r["stage"]}|{r["backbone"]}|{r["dataset"]}|{r["condition"]}|{acc:+.3f}|{f1:+.3f}|')
    if not warnings:
        lines.append('|本项阈值未触发||||||')
    csv_save(ROOT/'clean_f1_warnings.csv',warnings)
    lines += ['', '每类召回完整保存在 per_class_recall.csv；按真实训练频数分成三组的测试表现见 frequency_group_test.csv，所有频数分组均为冻结后的描述性分析。',
        '## 验证与存储','',
        f'本地完整审计状态：{audit["status"]}。{audit["checks"]["full_logs_artifacts_targets"]} 个完整日志/目标/检查点/概率哈希核验；{audit["checks"]["causal_C1_cells"]} 格因果 race；{audit["checks"]["exact_mass_D1_cells"]} 格精确均量。',
        f'独立重载 {audit["checks"]["independent_checkpoint_reload_samples"]} 个 checkpoint，全节点概率最大误差 {audit["reload_max_error"]:.9g}；历史 seed0 概率桥最大误差 {audit["historical_seed0_max_error"]}。',
        '15 个历史 split/noise/geometry/encoder 护照均已验证；固定 H 的归一化遵循历史 F.normalize(float,dim=1,eps=1e-12)，15 个 KM 按 n_init10 和各 seed 精确重算核对。',
        '目标按内容哈希共享；R2 checkpoint 保存分类头及带哈希历史编码器引用。25/75 前缀保留独立完整恢复状态，达到200轮并核验最佳权重/概率/结果后删除本轮终态临时恢复副本，storage_cleanup.jsonl 留有路径/大小/哈希。历史文件保持只读。',
        '磁盘安全余量固定3GiB。空间不足时停止并记录 storage_pause.json，旧结果与历史目录保持。',
        '首次红灯测试逐项驱动实现；正式单测见 tests.log，工程 smoke 不参与科学选择。全部工程故障按 failures.jsonl 保留，任何恢复与版本修订见对应 receipt。','',
        '## 证据解释与后续','',
        '已获得的实验事实：本报告表格、完整三seed结果、各条件配对差、分组选择、race差距及代价。决策只针对当前冻结协议。',
        '合理解释：若分组同时超出最佳全局与两种均量控制，可将收益与节点分配联系起来；若只超过 CE，则保留局部软监督的分类增量。机制解释仍需结合目标分层与学生结果。',
        '尚待验证：独立 SSL 随机性、新图迁移、更多独立 seed、替换粒球生成规则、其他模型和更宽搜索空间。',
        '负结果：所有负配对、clean损害、小类代价与最差race格完整保存。接近±.3pp的差作为不确定描述，+.5pp为实用量级参照。三seed是重复单位，节点和条件用于分层描述。',
        '固定 H 和历史 beta=.5 具有开发来源；保留普通 KM 与 GCE 强对照。后续实验须另行授权，本脚本到此结束。','',
        '## 交付文件','',
        'PLAN/config、sources/inputs/passports、status/HANDOFF、tests/smoke、逐臂 epochs/config/segments/checkpoint/probabilities、内容寻址 targets、results/summary/seeds12_summary、paired_differences/paired_summary/seeds12_paired_summary、selections、c1_regret/costs、d1_strata、per_class_recall、frequency_group_test、failures、audit 和本报告。']
    save(ROOT/'decision.json',dict(finished=now(),decisions=decisions,warnings=len(warnings),automatic_further_training=False))
    output(ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('run.lock','console.log','status.json','HANDOFF.md','DELIVERY_SHA256.csv')]
    csv_save(ROOT/'DELIVERY_SHA256.csv',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
