"""Finalize complete audited experiments and render detailed evidence-bound reports."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_screen as screen
import run_students as students

ROOT = screen.ROOT
d = screen.d


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def summarize(values, percentage=False):
    values = np.asarray(values, dtype=np.float64)*(100 if percentage else 1)
    return f'{values.mean():.2f}±{values.std(ddof=1):.2f}' if percentage else f'{values.mean():.6f}±{values.std(ddof=1):.6f}'


def verify_history(records):
    checks = []
    for record in records:
        if 'historical_accuracy_delta_pp' not in record:
            continue
        name, seed, noise, arm = (record[k] for k in ['dataset', 'seed', 'noise', 'arm'])
        old_arm = 'GB64' if arm == 'OLD64' else arm
        locations = [ROOT.parent/'heterophily_original_baseline_20261014/students'/name/f'seed{seed}'/noise/old_arm,
                     screen.MECHANISM/'students'/name/'grace'/f'seed{seed}'/noise/old_arm]
        folder = next(p for p in locations if (p/'result.json').exists())
        config = screen.read(folder/'config.json')
        current_config = screen.read(ROOT/'students'/name/f'seed{seed}'/noise/arm/'config.json')
        assert config['training'] == students.TRAINING
        assert config['split_hash'] == current_config['split_hash']
        assert config['noise_hash'] == current_config['noise_hash']
        assert config.get('student_seed', config.get('seed', config.get('protocol_seed'))) == seed
        assert config.get('trainer_hash', config.get('training_source_hash')) == current_config['trainer_hash']
        assert config.get('model_hash', config.get('model_source_hash')) == current_config['model_hash']
        assert d.sha(config['target_path']) == config['target_hash']
        with np.load(config['target_path']) as prior, np.load(current_config['target_path']) as current:
            assert np.array_equal(prior['target'], current['target'])
        import torch
        graph = d.load_graph(name, seed)
        logits = torch.load(folder/'logits.pt', map_location='cpu', weights_only=True)
        prior_accuracy = float((logits[graph.test_index].argmax(1) == graph.y[graph.test_index]).float().mean())
        prior_result = screen.read(folder/'result.json')
        assert abs(prior_accuracy-prior_result['test_accuracy']) < 1e-8
        checks.append(dict(dataset=name, seed=seed, noise=noise, arm=arm, source=str(folder),
            config_hash=d.sha(folder/'config.json'), old_logits_hash=d.sha(folder/'logits.pt'),
            target_values_identical=True, accuracy_delta_pp=100*(record['test_accuracy']-prior_accuracy)))
    d.write(ROOT/'audit/historical_baseline_checks.json', dict(status='passed', checks=checks,
        exact_accuracy_matches=sum(abs(c['accuracy_delta_pp']) < 1e-8 for c in checks)))
    return checks


def main():
    audit = screen.read(ROOT/'audit/report.json')
    status = screen.read(ROOT/'student_status.json')
    gate = screen.read(ROOT/'gate.json')
    integrity = screen.read(ROOT/'gate_integrity.json')
    assert audit['status'] == integrity['status'] == 'passed'
    assert integrity['gate_hash'] == d.sha(ROOT/'gate.json')
    assert status['status'] in ['completed', 'skipped_by_gate']
    # Rebuild all derived CSVs unconditionally, including after any resumed run.
    screen.summarize()
    students.summarize()
    records = [screen.read(p) for p in sorted((ROOT/'students').glob('*/seed*/*/*/result.json'))]
    keys = {(r['dataset'], r['seed'], r['noise'], r['arm']) for r in records}
    expected = set(itertools.product(gate['selected_datasets'], screen.SEEDS, screen.NOISE, screen.ARMS))
    assert len(keys) == len(records) == len(audit['students']) and keys == expected
    history = verify_history(records)
    target = pd.read_csv(ROOT/'target_metrics.csv')
    summary = pd.read_csv(ROOT/'target_summary.csv')
    conditional = pd.read_csv(ROOT/'conditional_q_summary.csv')
    frame = pd.DataFrame(records)
    pair_frame = pd.read_csv(ROOT/'paired_student_deltas.csv') if records else pd.DataFrame()
    student_decisions = []
    paired_rows = []
    for name in gate['selected_datasets']:
        checks = []
        for noise in screen.NOISE:
            for reference in ['CE', 'OLD64', 'H_GEOM', 'X_GEOM', 'RANDOM', 'H_KNN']:
                values = pair_frame[(pair_frame.dataset == name) & (pair_frame.noise == noise)
                                    & (pair_frame.reference == reference)].accuracy_delta_pp
                assert len(values) == 5
                check = dict(noise=noise, reference=reference, mean_pp=float(values.mean()),
                    sample_std_pp=float(values.std(ddof=1)), improved_seeds=int((values > 0).sum()),
                    equal_seeds=int((values == 0).sum()))
                checks.append(check)
                paired_rows.append(dict(dataset=name, **check))
        close_to_old = all(c['mean_pp'] >= -1. for c in checks if c['reference'] == 'OLD64')
        robust_vs_ce = all(c['mean_pp'] > 0 and c['improved_seeds'] >= 4 for c in checks
                          if c['reference'] == 'CE' and c['noise'] != 'clean')
        student_decisions.append(dict(dataset=name, classification_checks=checks,
            all_condition_mean_within_1pp_of_old=close_to_old,
            both_noise_conditions_improve_ce_at_least4_seeds=robust_vs_ce,
            exploratory_simplification_supported=close_to_old and robust_vs_ce))
    d.csv_write(ROOT/'paired_student_summary.csv', paired_rows)
    decision = dict(status='completed', research_decision='修改：暂缓纯BFS作为异配主线',
        target_rows=315, newly_trained_students=len(records), protocol_seeds=screen.SEEDS,
        selected_datasets=gate['selected_datasets'], target_gate=gate['datasets'], student_decisions=student_decisions,
        tested_scope='label-free residual-degree connected BFS, original LCSS and original attribute MLP',
        roman_students_trained=False, no_test_based_rule_changes=True,
        formal_noninferiority_proven=False, independent_ssl_repeats=1,
        target_audit_hash=d.sha(ROOT/'audit/target_report.json'), student_audit_hash=d.sha(ROOT/'audit/report.json'))
    d.write(ROOT/'decision.json', decision)

    lines = ['# 无标签BFS→LCSS：完整实验结论', '', '2026-10-15。冻结方案见PLAN.md；本报告基于实际保存结果及独立审计。', '',
        '## 实验范围与核心结论', '',
        f'三图五个配对协议seed，clean/Uniform .4/Pair .4，共315目标记录。CoraFull与Coauthor-CS通过目标门槛，完成{len(records)}次新MLP训练；Roman目标不优于支持匹配随机分组，按预设门槛未训练新学生。', '',
        '直接拓扑分组有低成本监督信号，但尚未建立通用异配优势。BFS移除区域后形成大量碎片；结构区域和语义监督区域并不等价。最终是否支持接近旧框架的简化，见下方预设1pp探索判据，不能称统计非劣证明。', '',
        '## 假设、粒球角色、方法与代码改动', '',
        '假设为拓扑区域可替代表示预训练提供抗噪证据。区域只组织留一投票，原alpha/beta=.5/min_other_votes2/并列处理不变。新增granular_ball_topology.py及独立实验脚本；原GRACE、原分球、目标和训练源码未改。BFS只读取A/N，不读取X/Y/训练mask；随机对照只读取训练位置。', '',
        'BFS臂没有对比预训练，全部学生仅优化原软交叉熵；H相关对照使用既有冻结表示。本轮不包含对比/分类联合目标，因此不能以两种损失冲突解释此次负结果。', '',
        '对照H_GEOM与X_GEOM采用同一BisectingKMeans、相同实际球数预算、L2归一化、seed0/n_init1/max_iter50；前者用既有H，后者原始X稀疏矩阵。同球数不等于同大小或支持分布。RANDOM逐球保留BFS大小和训练支持，H_KNN逐节点匹配其他票数。', '',
        '## 分球和证据覆盖', '']
    rows = []
    for name in screen.NAMES:
        stats = screen.read(ROOT/'partitions'/name/'stats.json')['BFS']
        mean = summary[(summary.dataset == name) & (summary.noise == 'clean') & (summary.arm == 'BFS')].iloc[0]
        rows.append([name, stats['max_ball_size'], stats['num_balls'], stats['singleton_balls'],
            stats['num_components'], f"{100*mean.eligible_rate_mean:.2f}%", f"{100*mean.positive_clean_margin_fraction_mean:.2f}%",
            f"{stats['seconds']:.3f}s"])
    lines += [markdown_table(['图', '球容量', '实际球数', '单节点球', '原图组件', '其他票≥2覆盖', '正干净margin/全部训练', '构球时间'], rows), '',
        '三图原始孤立节点均为0。大量单节点球由移除已分配区域后产生，而不是简单等同于原图孤立点。BFS球内连通、覆盖互斥、残余度中心全部独立核验。只有一个确定性BFS分区，五协议seed覆盖不同划分/噪声；不能称五次独立构球或SSL。', '',
        '## 带噪目标Brier：均值±样本标准差，越低越好', '']
    rows = []
    for name in screen.NAMES:
        for noise in ['uniform', 'pair']:
            sub = target[(target.dataset == name) & (target.noise == noise)]
            rows.append([name, noise]+[summarize(sub[sub.arm == arm].target_brier)
                                       for arm in ['CE', 'OLD64', 'BFS', 'RANDOM', 'H_GEOM', 'X_GEOM', 'H_KNN']])
    lines += [markdown_table(['图', '条件', 'CE', '旧GB64', 'BFS', '随机', 'H几何', 'X几何', 'H-kNN'], rows), '',
        'Roman BFS相对CE改善并不等于拓扑证据有效：相同大小/训练支持的随机分组产生接近的Brier变化，预设独立增量门槛未过。低Brier与高语义正确性不能互相替代。', '',
        '## 共识质量与干净损害', '']
    rows = []
    for name in screen.NAMES:
        for arm in ['OLD64', 'BFS', 'H_GEOM', 'X_GEOM', 'H_KNN', 'RANDOM']:
            row = summary[(summary.dataset == name) & (summary.noise == 'clean') & (summary.arm == arm)].iloc[0]
            q = conditional[(conditional.dataset == name) & (conditional.noise == 'clean') & (conditional.arm == arm)].iloc[0]
            rows.append([name, arm, f'{100*row.eligible_rate_mean:.2f}%', f'{100*q.q_accuracy_eligible_mean:.2f}%',
                f'{100*row.positive_clean_margin_fraction_mean:.2f}%', f'{row.target_brier_mean:.6f}',
                f'{100*row.clean_harm_rate_mean:.2f}%'])
    lines += [markdown_table(['图', '方法', '支持覆盖', 'q准确率/合格节点', '正margin/全部训练', 'clean Brier', '干净目标argmax误改'], rows), '',
        '合格节点q准确率仍采用argmax并列约定。正margin严格要求root真类超过所有其他类，更贴近原理论。整体q_accuracy另在原CSV保留，但包括无支持均匀分布，不能把它单独解释为可靠共识。', '',
        '## 配对学生：Accuracy %均值±样本标准差', '']
    rows = []
    for name in gate['selected_datasets']:
        for noise in screen.NOISE:
            sub = frame[(frame.dataset == name) & (frame.noise == noise)]
            rows.append([name, noise]+[summarize(sub[sub.arm == arm].test_accuracy, True) for arm in screen.ARMS])
    lines += [markdown_table(['图', '条件']+screen.ARMS, rows), '',
        '五种子共享既有一个GRACE表示，BFS/H/X几何为固定seed0分区；学生初始化和噪声/划分按协议seed配对。不同图采用各自现存协议：CoraFull/CS约20/10/70，Roman官方50/25/25；CoraFull/CS整数split各有1节点未进既有集合，未静默修改。', '',
        '## BFS相对对照的配对增量：百分点', '']
    rows = [[r['dataset'], r['noise'], r['reference'], f"{r['mean_pp']:+.3f}±{r['sample_std_pp']:.3f}",
             f"{r['improved_seeds']}/5", f"{r['equal_seeds']}/5"] for r in paired_rows]
    lines += [markdown_table(['图', '条件', '参照', 'BFS−参照', '正增量seed', '相等seed'], rows), '',
        '## 预设分类简化判据', '']
    rows = [[r['dataset'], r['all_condition_mean_within_1pp_of_old'],
             r['both_noise_conditions_improve_ce_at_least4_seeds'], r['exploratory_simplification_supported']]
            for r in student_decisions]
    lines += [markdown_table(['图', '三条件均值距旧版不低于−1pp', '两带噪条件优于CE且≥4/5', '探索简化判据通过'], rows), '',
        '1pp容忍度和目标门槛在看结果前写入PLAN；不把它称正式非劣性检验，不依据test改变球容量、训练支持或其他参数。', '',
        '## 效率、验证与局限', '',
        'BFS构球本身约0.2秒量级；学生仍200轮、hidden64、2层含BN、原始X输入。完整315目标screen时间见screen_status.json，包含所有对照和重复loader，不是单个BFS方法的纯流水线计时。新学生时间/峰值allocated GPU显存见student_results.csv。未重跑GRACE预训练，不能把原稿CoraFull74.48分钟当每个学生的额外成本，也不据该数给出本轮端到端提速倍数。', '',
        '运行期检测到另一个prepare_greet.py实验进程，详见resource_context.json；没有停止或改动该进程。当前学生队列内部串行，但耗时是共享运行环境的测量值，不构成独占GPU下的性能比较；进程存在本身也不量化其GPU利用率。', '',
        f'15项BFS合同测试通过，含63组小图参考核对。3分区、15协议case、315目标和{len(records)}学生保存logits已审计；{len(history)}项历史baseline来源检查已保存。新学生全部重新训练，无缓存分数冒充新运行。', '',
        '保留一次joblib物理核数检测警告，scikit-learn自动回退逻辑核；实验显式四线程、正常完成，不关闭告警或隐藏失败。独立审查指出resume派生CSV可能滞后，本轮未受影响，check_gate.py逐键绑定原JSON并最终无条件重建汇总；旧frozen源码保留。', '',
        '## 最终决策与证据分层', '',
        '已支持：纯无标签拓扑规则可快速生成完整区域；CoraFull/CS相对随机分组有目标证据；Roman该独立增量未建立；去除GRACE与改变区域会同时影响支持覆盖、置信度和类别边际。学生正负结果以上表为准。', '',
        '合理解释：Roman上的目标改善含一般软化贡献；残余碎片化与投票支持不均可能解释部分退化。这些解释与数据相容，但尚未通过单变量因果消融证明。', '',
        '尚未验证：支持感知的拓扑区域合并、属性约束分裂、两跳关系、Amazon/Cora扩展及独立SSL重复；不能据本轮失败否定全部拓扑粒球。', '',
        '被削弱：仅用BFS初始化即可普遍替代GRACE、连通区域自然提供可靠同类监督、sqrt规模自然得到sqrt数量。', '',
        '决策为修改/暂缓：保留低成本BFS诊断实现，暂缓将纯BFS+原LCSS作为异配论文主线。下一讨论应围绕属性与拓扑证据的可验证互补，先研究Roman原X对照的强目标信号和区域支持；不立刻扫描球容量或堆叠新损失。', '',
        '## 文件与复现', '',
        'PLAN.md/config.json/source_manifest.json、screen_raw.log、partitions/*、runs/*、target_metrics/summary、gate.json、gate_integrity.json；student_source_manifest.json/student_raw.log/students/*/config,result,logits、student_results/summary、paired_student_summary、audit/*、decision.json。', '',
        '命令均使用D:/codex_temp/noisygl-rocm721/Scripts/python.exe -X utf8：test_topology.py unittest→run_screen.py→audit.py→analyze.py→check_gate.py→run_students.py→audit.py --students-only→finish_report.py。所有训练/源hash冻结，重新设计使用新版本目录。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    history_doc = ['# BFS拓扑粒球方向实验历史与经验', '', '2026-10-15。唯一正式轮次，完整字段见PLAN.md与实验结论.md。', '',
        '研究假设/角色/动机：纯无标签拓扑区域组织可靠局部软监督，省去表示预训练；不强制同球语义。',
        '代码/配置改动：新增BFS/随机支持匹配及独立脚本；原目标/学生固定，三图五配对seed三条件七臂。',
        '预期机制：拓扑关系收益依赖root正margin与投票支持，不由同配边比例或区域连通性保证。',
        '成功/失败标准：预先冻结CE与随机Brier至少改善.005且4/5；后续1pp探索容忍度及跨图/强对照全部报告。',
        f'实际结果：315目标记录；CoraFull/CS通过目标门槛，{len(records)}新学生；Roman目标未优于随机，未训练新学生。详表保留全部负结果。',
        '最终决策：修改/暂缓纯BFS主线，保留诊断基线；未实施新融合或参数搜索。',
        '经验：最终球数远高于sqrt；Brier降低可能含一般软化；目标改善未必达到旧分类水平；公平比较需支持匹配随机与同粒度/同票数语义邻域。',
        '证据边界：该固定残余度BFS规则与原GBGCL完整模型不同，结果不证明所有拓扑粒球无效；五协议seed不等于五SSL。']
    (ROOT/'02_experiment_history_and_lessons.md').write_text('\n'.join(history_doc)+'\n', encoding='utf-8')
    d.write(ROOT/'final_validation.json', dict(status='passed', target_rows=315,
        student_rows=len(records), historical_checks=len(history), completed_status=status['status'],
        report_hash=d.sha(ROOT/'实验结论.md'), source_hash=d.sha(__file__),
        audit_hash=d.sha(ROOT/'audit/report.json'), decision_hash=d.sha(ROOT/'decision.json')))
    print(json.dumps(dict(status='completed', target_rows=315, student_rows=len(records),
                         student_decisions=student_decisions), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
