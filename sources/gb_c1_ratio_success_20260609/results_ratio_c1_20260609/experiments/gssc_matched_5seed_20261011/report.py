from __future__ import annotations

import csv
import hashlib
import json

import torch

from run_grid import ROOT, BASE, DATASETS, location, load_data, final_metrics, audit, summarize, write_json


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    per_seed, summaries, audits = [], [], {}
    reused_count = 0
    for dataset in DATASETS:
        records = []
        for seed in range(5):
            folder = location(dataset, seed)
            record = json.loads((folder/'result.json').read_text())
            if (record['status'] != 'ok' or record['smoke'] or record['seed'] != seed
                    or record['dataset'] != dataset or record['noise_rate'] != .4):
                raise RuntimeError('Formal result identity mismatch')
            data, protocol = load_data(dataset, seed, .4)
            verified = audit(folder, data, record, protocol)
            saved = torch.load(folder/'probabilities.pt', map_location='cpu', weights_only=True)
            metrics = final_metrics(dataset, seed, saved)
            if metrics['test'] != record['test'] or metrics['valid'] != record['valid']:
                raise RuntimeError('Metric replay mismatch')
            reused = bool(record.get('reused_run'))
            if reused:
                origin = BASE/'runs'/dataset/'uniform_0.4'/'seed0'
                if seed != 0 or record['reused_run'] != str(origin):
                    raise RuntimeError('Unexpected seed0 reuse origin')
                if hashlib.sha256((origin/'result.json').read_bytes()).hexdigest() != record['original_result_sha256']:
                    raise RuntimeError('Original seed0 result changed')
                for name, digest in record['artifact_hashes'].items():
                    if hashlib.sha256((origin/name).read_bytes()).hexdigest() != digest:
                        raise RuntimeError('Reused seed0 artifacts differ from original')
                reused_count += 1
            audits[f'{dataset}/seed{seed}'] = verified | dict(seed0_reused=reused,
                historical_protocol_verified=True, metric_replay_verified=True)
            logs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            per_seed.append(dict(dataset=dataset, seed=seed, noise_type='uniform', noise_rate=.4,
                test_percent=100*record['test'], valid_percent=100*record['valid'],
                best_epoch=record['best_epoch'], best_phase=record['diagnostics']['best_phase'],
                final_valid_percent=100*logs[-1]['val'],
                inference_parameters=record['parameters'], reused_seed0=reused,
                wall_seconds=record['wall_seconds']))
            records.append(record)
            del data, saved
            torch.cuda.empty_cache()
        summary = dict(dataset=dataset, method='GSSC matched', noise_type='uniform', noise_rate=.4,
                       **summarize(records))
        summaries.append(summary)
    if reused_count != 5:
        raise RuntimeError('Expected five exact seed0 reuses, not retraining')
    write_csv(ROOT/'per_seed_results.csv', per_seed)
    write_csv(ROOT/'summary.csv', summaries)
    write_json(ROOT/'artifact_audit.json', audits)
    cells = [f"${r['mean_percent']:.2f}\\pm{r['std_percent']:.2f}$" for r in summaries]
    (ROOT/'table_row.tex').write_text('GSSC (matched) & '+' & '.join(cells)+' \\\\\n', encoding='utf-8')
    names = dict(cora='Cora',citeseer='CiteSeer',corafull='CoraFull',coauthorcs='Coauthor-CS',amazonpho='Amazon-Photo')
    warmup_count = sum(row['best_phase'] == 'classifier_pretrain' for row in per_seed)
    lines = ['# GSSC 适配协议五 seed 实验', '',
        '## 设置与代码边界', '',
        '五个主数据集，uniform 0.4，seed 0–4。数据划分、噪声生成和训练初始化均使用对应 seed；原始节点属性与图结构，20%训练、10%验证、70%测试。训练标签之外的优化标签屏蔽为 -1；干净验证标签仅用于选模，测试标签仅用于最终评价。', '',
        '与已确认单 seed 版本使用同一个冻结训练函数：共同两层 MLP、隐藏宽度64、ReLU–BN–dropout0.5；保留 GSSC 辅助头、边采样、虚拟节点混合、邻域对比、类别中心与同配性损失。100轮无标签边预训练，100轮分类器预热，100轮主训练；200轮监督阶段均参与选模，验证精度并列时保留首次最大值。Adam lr0.01、weight decay0.0005，mixup lr0.0001；作者预设和预声明的数据集映射不变。', '',
        '部署骨干与监督轮数对齐不代表总优化步数或训练算力相同；GSSC 仍保留自身的边批次更新和训练辅助模块。', '',
        '新增20次训练，核验并复用5组已有seed0；复用记录保留原始训练源代码哈希，新编排哈希单独存储。原单seed目录、其他方法结果与论文未改。', '',
        '## 结果', '', '| 数据集 | 五 seed 测试精度（%，均值 ± 样本标准差） |', '|---|---:|']
    for row in summaries:
        lines.append(f"| {names[row['dataset']]} | {row['mean_percent']:.2f} ± {row['std_percent']:.2f} |")
    lines += ['', '标准差使用ddof=1；未经舍入的每seed结果保存在per_seed_results.csv，汇总保存在summary.csv。', '',
        '## 验证与结论', '',
        '25组均完成200轮监督训练，划分及噪声指纹匹配历史主实验；配置与源文件、检查点及概率文件哈希通过复核。25组部署检查点重载预测均与保存概率完全一致（最大绝对差0），最终指标重新计算一致；5组seed0与原文件完全相同。', '',
        f'有{warmup_count}/25组的最佳验证点位于分类器预热阶段；这些组仍完整训练到主阶段结束，没有提前停止或按测试精度选点。全部seed保留，不按结果筛选。', '',
        '已被本轮支持：该适配实现能够在论文五个主数据集的五个配对seed上运行，并给出可追溯的均值和样本标准差。', '',
        '本轮不支持：未做配对统计检验，不能仅凭均值差宣称显著优于其他方法；未扩展uniform0.6、其他噪声或其他预设。数值属于共同骨干/预算适配版本，而非作者原始配置的直接复现。', '',
        '最终决策：本轮五seed实验完成，作为该适配版本在uniform0.4下的正式对照数据；table_row.tex可用于后续主表更新。本轮没有修改论文。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'All25 artifacts audited; seed0 reused={reused_count}; warmup selections={warmup_count}')


if __name__ == '__main__':
    main()
