"""Close the completed experiment; no training imports or further searches."""
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def table(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def csvsave(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def summarize_diagnostics(rows,ids,metrics,seeds):
    groups=defaultdict(list)
    for row in rows:
        if int(row['seed']) in seeds:
            groups[tuple(row[k] for k in ids)].append(row)
    out=[]
    for key,rs in groups.items():
        result=dict(zip(ids,key),seeds=','.join(map(str,seeds)),n_available_seeds=len({r['seed'] for r in rs}))
        for m in metrics:
            values=[float(r[m]) for r in rs if r[m]!='']
            result[m+'_mean']=statistics.mean(values) if values else None
            result[m+'_std_ddof1']=statistics.stdev(values) if len(values)>1 else None
        out.append(result)
    return out


def main():
    assert not (OUT/'delivery_complete.json').exists()
    assert read(ROOT/'complete.json')['scientific_fits']==684
    assert read(ROOT/'audit.json')['status']=='passed'
    assert not (ROOT/'run.lock').exists()
    for p,h in read(ROOT/'sources.json').items():
        assert sha(Path(p))==h
    frozen=read(ROOT/'selections_frozen.json')
    assert sha(ROOT/'reuse_manifest.json')==frozen['reuse_manifest_sha256']
    for p,h in frozen['choices'].items():
        assert sha(ROOT/p)==h
    parent=ROOT.parents[1]/'research/d1_fallback_km_combo_20261024/PARENT_AUDIT.json'
    pa=read(parent)
    assert pa['status']=='passed' and pa['physical_fits']==684
    recalls=table('per_class_recall.csv')
    strata=table('strata.csv')
    for prefix,seeds in [('',[0,1,2]),('seeds12_',[1,2])]:
        csvsave(OUT/(prefix+'per_class_recall_summary.csv'),summarize_diagnostics(recalls,
            ['dataset','backbone','condition','selector','method','class_id'],['recall','train_true_count','test_count'],seeds))
        csvsave(OUT/(prefix+'strata_summary.csv'),summarize_diagnostics(strata,
            ['dataset','backbone','condition','selector','method','partition','stratum'],
            ['count','support','brier_change','true_class_mass_change','train_accuracy'],seeds))
    selected=table('selected_results.csv')
    by=defaultdict(dict)
    for row in selected:
        by[tuple(row[k] for k in ['dataset','seed','backbone','condition','selector'])][row['method']]=row
    comparisons=[]
    for (n,s,b,c,selector),methods in by.items():
        if b!='R0':
            continue
        a,d=methods['ADAPTIVE_EXPANDED'],methods['KM_EXPANDED']
        comparisons.append(dict(dataset=n,seed=int(s),backbone=b,condition=c,selector=selector,
            method='ADAPTIVE_EXPANDED',reference='KM_EXPANDED',
            test_accuracy_diff_pp=100*(float(a['test_accuracy'])-float(d['test_accuracy'])),
            test_macro_f1_diff_pp=100*(float(a['test_macro_f1'])-float(d['test_macro_f1']))))
    csvsave(OUT/'combo_vs_km_paired.csv',comparisons)
    means=[statistics.mean(r['test_accuracy_diff_pp'] for r in comparisons if r['seed']==s and r['selector']=='accuracy' and r['condition']!='clean') for s in [0,1,2]]
    for actual,expected in zip(means,[-.5887686,-.4169406,-.0791946]):
        assert math.isclose(actual,expected,abs_tol=1e-7)
    mean,sd=statistics.mean(means),statistics.stdev(means)
    ns,nd=statistics.mean(means[1:]),statistics.stdev(means[1:])
    physical=read(ROOT/'physical_fits.json')
    count=Counter('KM' if v['partition']=='KM' else 'DEEP_GROUP' for v in physical.values())
    diagnostic_info=dict(per_class_rows=len(recalls),stratum_rows=len(strata),
        policy='summaries are descriptive per-seed means and ddof1 SD; unavailable strata retain n_available_seeds')
    decisions=dict(fallback='retain validation fallback; CiteSeer clean F1 cost remains',
        matched_km='KM stronger after budget-matched expansion; retain as primary strong control',
        combo='retain as modest within-tree exploratory increment; defer further training',
        auto_training=False,combo_vs_km_seed_means_pp=means,combo_vs_km_mean_pp=mean,combo_vs_km_sd=sd,
        newseeds_combo_vs_km_mean_pp=ns,newseeds_combo_vs_km_sd=nd)
    supplement=f'''
## 最终结论与交付核验

### 1. 回退：缓解强制分组代价，保留局部残余损害

R0 accuracy主selector的带噪扩展池相对global-only为+0.777±0.151pp，新增seed1/2为+0.714±0.146pp；相对强制GROUP仅−0.006±0.023pp。clean相对强制GROUP改善accuracy +0.265pp、macro-F1 +0.584pp。相对同selector全局池，clean五图宏差为−0.034pp/−0.279pp（accuracy/F1）。
逐图clean中，Cora、CoraFull、CoauthorCS、AmazonPhoto回到与全局池相同结果；CiteSeer仍有accuracy −0.172pp、macro-F1 −1.396pp。结论：保留验证回退，CiteSeer F1代价需明确披露。balanced为预设描述性次selector，保持完整报告，不依据测试挑选selector。

### 2. 公平KM：预算匹配后普通KM更强

R0带噪LEGACY_GLOBAL−KM_GLOBAL为−0.523±0.024pp；LEGACY_EXPANDED−KM_EXPANDED为−0.713±0.270pp，新增seed为−0.696±0.380pp。KM自身expanded−global为+0.968±0.378pp。分组力度对普通KM同样有效，当前结果削弱粒球特有优势的解释。决策：将匹配KM扩展池保留为强对照，不继续默认放大粒球特异性主张。

### 3. 同树组合：有增量，但最强对照限制整体收益

R0 accuracy主selector带噪组合相对固定支持global-only为+1.229±0.221pp；相对同支持global-only/原C1为+0.753±0.186pp；相对固定支持expanded为+0.326±0.290pp；相对验证选择的最强单模块为+0.249±0.082pp（新seed +0.246±0.115pp）。same-support global-refinement相对原C1的带噪差为0，故此处增量来自新增分组候选。
**组合相对KM_EXPANDED：seed0/1/2={means[0]:+.7f}/{means[1]:+.7f}/{means[2]:+.7f}pp，均值{mean:+.7f}±{sd:.7f}pp；新增seed1/2为{ns:+.7f}±{nd:.7f}pp。三seed均为负。** 该对照由主侧提出并已用冻结selected_results重新计算，完整配对另存postprocess/combo_vs_km_paired.csv。
组合clean相对adaptive global-only为accuracy −0.115pp、F1 −0.286pp；相对fixed expanded为−0.059pp/+0.082pp。决策：保留同树组合的探索性增量记录，当前停止训练；主线比较须同时呈现更强的KM扩展池及逐图/F1代价。+.3–.5pp沿用描述性量级参照，未添加硬门。

### 类诊断与成本完整性

逐类召回原始表共{len(recalls)}行，干净/错标及同意/冲突/平票分层原始表共{len(strata)}行；三seed与新增seed1/2汇总均在postprocess目录。分层缺失时记录实际可用seed数量，保持原始样本数和诊断口径。所有真值诊断在选择冻结后执行。
本轮精确物理fit为684（KM {count['KM']}、深树group {count['DEEP_GROUP']}），低于支持去重上限768和批准上限840。所有完整fit各200轮，共136800新更新；smoke为4个fit各2轮。部署成本已计入C1上游1200更新、同支持三λ未完成预算和group800更新，使用历史完整缓存没有将额外全长候选算作免费。旧文件未删除或复制大产物。

### 双侧审计与失败记录

本地审计：120cell、2040selector检查、2256目标重建、27个新checkpoint重载，最大概率误差0。主侧独立审计：2040selector/paired、3672summary、30个CPU重载及sklearn核验，全部passed。主侧CPU最大误差约7.75e-7；主侧文件只读引用，未改写。
唯一工程失败为科学训练前inputs.json原子替换遭Windows PermissionError；原源码/日志/失败与红灯测试已保存在repairs/preflight_01。有界重试修复后11项单测通过，4个smoke通过；正式训练没有失败记录。新训练源码自启动后保持冻结。此次结论属于固定历史SSL上的回顾性探索，seed0与新增seed1/2分别报告。
全部训练已完成，无待续任务，不自动创建下一轮实验。
'''
    report=ROOT/'REPORT_zh.md'
    shutil.copy2(report,OUT/'REPORT_generated_original_zh.md')
    report.write_text(report.read_text(encoding='utf-8')+supplement,encoding='utf-8')
    (OUT/'decision.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'HANDOFF.md').write_text('# COMPLETE: D1 fallback / matched KM / same-tree combo\n\n'
        '684 scientific fits (200 epochs), 4 smoke fits. No active training or lock.\n'
        'Local and parent audits passed. Read REPORT_zh.md final section and postprocess/decision.json.\n'
        f'Physical counts: {dict(count)}. Runtime sources remain frozen.\n'
        f'Combo minus KM_EXPANDED noisy accuracy: {mean:+.7f} +/- {sd:.7f} pp; all three seeds negative.\n'
        'Retain fallback with residual CiteSeer clean F1 cost; matched KM is stronger; combo is modest exploratory increment.\n'
        'Class/stratum raw diagnostics and both seed summaries complete. C1 upstream/refinement cost fully disclosed.\n'
        'No further experiment launched. Parent research files read-only.\n',encoding='utf-8')
    info=dict(finished=datetime.now(timezone.utc).isoformat(),scientific_fits=684,smoke_fits=4,
        diagnostic_info=diagnostic_info,physical_counts=dict(count),source_manifest_sha256=sha(ROOT/'sources.json'),
        parent_audit_path=str(parent),parent_audit_sha256=sha(parent),parent_audit_status='passed',
        free_bytes=shutil.disk_usage(ROOT).free,no_further_training=True)
    (OUT/'delivery_complete.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.csv')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    csvsave(ROOT/'DELIVERY_SHA256.csv',manifest)
    bad=[r['path'] for r in manifest if sha(ROOT/r['path'])!=r['sha256']]
    assert not bad
    print(json.dumps(dict(info,delivery_files=len(manifest),hash_mismatches=bad),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
