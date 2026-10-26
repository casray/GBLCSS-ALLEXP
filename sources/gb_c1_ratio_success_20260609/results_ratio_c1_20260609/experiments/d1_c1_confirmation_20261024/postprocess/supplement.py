"""Post-completion reporting only; frozen scientific sources stay unchanged."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def writecsv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert read(ROOT / 'complete.json')['audit'] == 'passed'
    assert not (ROOT / 'run.lock').exists()
    assert not (OUT / 'supplement_complete.json').exists()
    for path, expected in read(ROOT / 'sources.json').items():
        assert sha(Path(path)) == expected
    summary = table('paired_summary.csv')
    newsummary = table('seeds12_paired_summary.csv')
    notes = ['## 主对照代价与新增 Seed 补充', '',
             '本节在全部训练、选择冻结和正式审计完成后生成；训练源码、候选集合、排序和已冻结选择均保持原样。', '',
             '**决策口径：原 PLAN 使用新 seed 的平均增益、跨图覆盖与 clean/F1 代价综合判断。生成器中的“至少3图为正”“1pp接近率≥90%”“最差差值≥−2pp”等是额外的描述性建议阈值，未作为原计划新增硬门或训练停止门。+.5pp及±.3pp均按计划作量级参照。**', '',
             '### GROUP 相对 CELL_GLOBAL_VAL 的 Clean 代价', '',
             '单位为百分点；每格先做同 seed 配对差，再计算 seed 均值及样本标准差。新增 seed1/2 单列。', '',
             '|骨干|图|三seed accuracy差±SD|三seed macro-F1差±SD|新seed accuracy差±SD|新seed macro-F1差±SD|',
             '|---|---|---:|---:|---:|---:|']
    clean, warnings = [], []
    metrics = ['test_accuracy_diff_pp', 'test_macro_f1_diff_pp']
    fmt = lambda r, k: f'{float(r[k+"_mean"]):+.3f} ± {float(r[k+"_std_ddof1"]):.3f}'
    for r in summary:
        if r['stage'] != 'D1' or r['reference'] != 'CELL_GLOBAL_VAL' or r['dataset'] == 'ALL' or r['condition'] != 'clean':
            continue
        n = next(n for n in newsummary if all(n[k] == r[k] for k in ['stage', 'reference', 'dataset', 'backbone', 'condition']))
        notes.append(f'|{r["backbone"]}|{r["dataset"]}|'+ '|'.join(fmt(v, m) for v in [r, n] for m in metrics)+'|')
        clean.extend([dict(subset='0,1,2', **r), dict(subset='1,2', **n)])
    # Separate schemas preserve every per-seed value without empty-key ambiguity.
    writecsv('primary_clean_allseeds.csv', [r for r in clean if r['subset'] == '0,1,2'])
    writecsv('primary_clean_newseeds.csv', [r for r in clean if r['subset'] == '1,2'])
    notes += ['', '### 主对照 F1/clean 警示', '',
              '以下按原 PLAN 的披露阈值筛选：clean accuracy均值下降超过2pp，或任一条件 macro-F1均值下降超过1pp。', '',
              '|seed范围|骨干|图|条件|accuracy差|macro-F1差|', '|---|---|---|---|---:|---:|']
    for subset, rows in [('0,1,2', summary), ('1,2', newsummary)]:
        for r in rows:
            if r['stage'] != 'D1' or r['reference'] != 'CELL_GLOBAL_VAL' or r['dataset'] == 'ALL' or r['condition'] == 'NOISY_MACRO':
                continue
            a, f = [float(r[m + '_mean']) for m in metrics]
            if (r['condition'] == 'clean' and a < -2) or f < -1:
                warnings.append(dict(subset=subset, backbone=r['backbone'], dataset=r['dataset'], condition=r['condition'], accuracy_diff_pp=a, macro_f1_diff_pp=f))
                notes.append(f'|{subset}|{r["backbone"]}|{r["dataset"]}|{r["condition"]}|{a:+.3f}|{f:+.3f}|')
    if warnings:
        writecsv('primary_control_warnings.csv', warnings)
    choices = [read(p) for p in (ROOT / 'selections/D1').rglob('*.json')]
    arms = ['GROUP_A0.0_C1.0', 'GROUP_A0.25_C1.0', 'GROUP_A1.0_C0.25', 'GROUP_A1.0_C0.0']
    arms = sorted({r['selected'] for r in choices} | {s['arm'] for r in choices for s in r['scores']})
    winnerrows = []
    notes += ['', '### D1 验证赢家分布', '',
              '计数为验证集所选规则，覆盖每个噪声类型下的五张图。合并 seed1/2 时每行10格；单独 seed 每行5格。clean 与带噪分开，完整表含 seed0/1/2及新增seed合计。', '',
              '|骨干|seed范围|条件|'+'|'.join(arms)+'|', '|---|---|---|'+'---:|'*len(arms)]
    for b in ['R0', 'R2']:
        for subset in [(0,), (1,), (2,), (1, 2)]:
            for cond in ['clean', 'uniform0.4', 'uniform0.6', 'pair0.4']:
                chosen = [r for r in choices if r['backbone'] == b and r['seed'] in subset and r['condition'] == cond]
                counts = Counter(r['selected'] for r in chosen)
                assert len(chosen) == 5 * len(subset)
                row = dict(backbone=b, seeds=','.join(map(str, subset)), condition=cond, cells=len(chosen), **{a: counts[a] for a in arms})
                winnerrows.append(row)
                if subset != (0,):
                    notes.append(f'|{b}|{row["seeds"]}|{cond}|'+ '|'.join(str(counts[a]) for a in arms)+'|')
    writecsv('d1_winner_counts.csv', winnerrows)
    writecsv('d1_winner_cells.csv', [dict(dataset=r['dataset'], seed=r['seed'], backbone=r['backbone'], condition=r['condition'], selected=r['selected'], global_lambda=r['best_cell_global_lambda']) for r in choices])
    regrets = table('c1_regret.csv')
    notes += ['', '### C1 新 Seed 与最差格', '',
              '以下覆盖clean及三种噪声条件。接近率按绝对测试差计算，符号保留在最差/最好格中。', '',
              '|骨干|seed范围|格数|≤0.5pp|≤1pp|≤2pp|最差 PROBE−FULL|最好 PROBE−FULL|', '|---|---|---:|---:|---:|---:|---:|---:|']
    race = []
    for b in ['R0', 'R2']:
        for seeds in [(0, 1, 2), (1, 2)]:
            rows = [r for r in regrets if r['backbone'] == b and int(r['seed']) in seeds]
            gaps = [float(r['test_gap_pp']) for r in rows]
            within = [sum(abs(g) <= threshold for g in gaps) / len(gaps) for threshold in [.5, 1, 2]]
            worst = min(rows, key=lambda r: float(r['test_gap_pp']))
            notes.append(f'|{b}|{",".join(map(str,seeds))}|{len(rows)}|'+ '|'.join(f'{v:.2%}' for v in within)+f'|{min(gaps):+.3f}|{max(gaps):+.3f}|')
            race.append(dict(backbone=b, seeds=list(seeds), cells=len(rows), closeness=within, worst=worst))
    notes += ['', '最差格（每骨干全部seed）：']
    for entry in race:
        if len(entry['seeds']) == 3:
            w = entry['worst']
            notes.append(f'- {entry["backbone"]}：{w["dataset"]} seed{w["seed"]} {w["condition"]}，PROBE={w["PROBE"]}，FULL_VAL={w["FULL_VAL"]}，测试差{float(w["test_gap_pp"]):+.3f}pp。')
    recovery = ROOT / 'recovery/interruption_01'
    receipt = read(recovery / 'receipt.json')
    for name, expected in receipt['archive_hashes'].items():
        assert sha(recovery / 'original' / name) == expected
    arm = ROOT / receipt['arm']
    old = (recovery / 'original/arm/epochs.jsonl').read_bytes().splitlines()
    new = (arm / 'epochs.jsonl').read_bytes().splitlines()
    assert len(old) == 163 and len(new) == 200 and old == new[:163]
    registry = read(ROOT / 'fit_registry.json')
    full = [read(ROOT / row['folder'] / 'result.json') for row in registry]
    assert len(full) == 3600 and all(r['epoch'] == 200 for r in full)
    assert sum(r['finished'] < receipt['time'] for r in full) == 2797
    assert all(not (ROOT / row['folder'] / 'resume.pt').exists() for row in registry)
    recovery_check = dict(archived_hashes_verified=True, replayed_epochs_76_163_byte_identical=True,
                          full_fits=3600, preinterruption_completed_fits_unchanged_finish_times=2797,
                          nominal_scientific_updates=720000, observed_extra_updates=88,
                          actual_scientific_updates_lower_bound=720088,
                          extra_update_count_upper_bound='unknown: unlogged updates possible',
                          extra_training_seconds='unknown: interrupted segment timer was not persisted')
    (OUT / 'recovery_verification.json').write_text(json.dumps(recovery_check, indent=2), encoding='utf-8')
    notes += ['', '### 中断恢复与实际成本', '',
              '外部进程中断发生在CoauthorCS seed0 R0 Pair.4的S8λ1完整网格补跑阶段，PROBE此前已冻结。锁PID退出码为1。原日志到epoch163，该臂自身完整模型/Adam/RNG恢复点为epoch75。归档所有中断文件及哈希后，活动日志恢复为前75行，并从自身状态继续。',
              '重跑epoch76–163的逐行日志与中断现场完全一致；2797个先前完成fit的结束时间保持在恢复之前。D1/C1各唯一fit总数仍为1440/2160，未重复完整fit或更换候选。',
              '**实际科学更新至少720088次：计划720000次加88次已记录中断重复更新。未落盘更新数量和中断段精确训练计时未知；原cost表中的seconds仅覆盖成功保存的段。** C1本次支付完整432000次候选更新及至少88次额外更新；理论PROBE为120000候选更新，另共享CE。存储中的prefix在对应臂完成后按既定哈希校验清理，原恢复点另留中断归档。', '',
              '### 综合解读', '',
              'D1：新增seed1/2复现了相对逐格最佳全局力度与两种精确均量控制的正向平均差，支持继续研究分组分配。R0 clean的Cora/CiteSeer macro-F1代价需要和带噪平均收益一起权衡；后续优先改进这一代价并另行验证。',
              '普通KM仍是强对照：D1相对KM的平均增益较小，R2对GCE的平均差也较小。当前数据支持冻结协议内的分组增量，粒球独有优势仍需更强证据。',
              'C1：大部分cell的短训选择接近完整网格，且少量cell的PROBE测试值更高。候选更新预算优势清晰；完整确认实验已实际支付全网格成本。按最差格、F1代价与新增seed表现综合决定后续调整，生成器附加阈值仅作阅读辅助。',
              '三seed均值/SD与新增两seed结果均为描述性重复证据；固定H下的协议/学生随机性已覆盖，独立SSL随机性留待后续授权实验。', '']
    original = ROOT / 'REPORT_zh.md'
    shutil.copy2(original, OUT / 'REPORT_generated_original_zh.md')
    content = original.read_text(encoding='utf-8')
    content = content.replace('**R0 决策：', '**R0 生成器描述性建议：').replace('**R2 决策：', '**R2 生成器描述性建议：')
    content = content.replace('\n|R2|cora|', '\n\n|骨干|数据集|GROUP−最佳全局 accuracy|macro-F1|新seed1/2 accuracy|\n|---|---|---:|---:|---:|\n|R2|cora|')
    content = content.replace('|0.2871|\n\n|R2|', '|0.2871|\n|R2|')
    content = content.replace('完整 C1 候选实际训练计时', '完整 C1 候选成功落盘段训练计时（中断额外耗时另述）')
    content = content.replace('## 核心结果', '\n'.join(notes) + '\n## 核心结果', 1)
    original.write_text(content, encoding='utf-8')
    (OUT / 'supplement_complete.json').write_text(json.dumps(dict(finished=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(ROOT / 'sources.json'), warnings=warnings, race=race,
        original_plan_no_new_hard_gates=True, recovery=recovery_check), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(warnings=warnings, race=race, recovery=recovery_check), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
