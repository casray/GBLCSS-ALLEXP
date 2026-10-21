"""Verify saved artifacts and regenerate report; no training."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'new_graph_mlp_seed0_20261009'
sys.path.insert(0, str(OLD))
from common import DATASETS, final_metrics, load_data, write_json
from simmlp_adapter import build_model


def main():
    torch.set_num_threads(4)
    rows, audits = [], []
    for dataset in DATASETS:
        folder = ROOT / 'runs' / dataset / 'simmlp'
        result = json.loads((folder / 'result.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        hashes = json.loads((folder / 'source_hashes.json').read_text())
        for path, expected in hashes.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        data, protocol = load_data(dataset)
        assert protocol['split_hash'] == result['split_hash']
        assert protocol['noise_hash'] == result['noise_hash']
        ckpt = torch.load(folder / 'checkpoint.pt', map_location='cuda', weights_only=False)
        model, obj = build_model(data, cfg)
        model.load_state_dict(ckpt['model'])
        classifier = obj['LogisticRegression'](cfg['hidden_dim'], data.num_classes).cuda()
        classifier.load_state_dict(ckpt['classifier'])
        model.eval()
        classifier.eval()
        with torch.no_grad():
            replay = classifier(model.encode(data.x)).softmax(1).cpu()
        saved = torch.load(folder / 'probabilities.pt', map_location='cpu', weights_only=True)
        torch.testing.assert_close(replay, saved, atol=1e-6, rtol=1e-5)
        metrics = final_metrics(dataset, saved)
        assert abs(metrics['test'] - result['test']) < 1e-12
        assert abs(metrics['valid'] - result['valid']) < 1e-12
        epochs = [json.loads(x) for x in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert len([x for x in epochs if x['stage'] == 'pretrain']) == 200
        assert len([x for x in epochs if x['stage'] == 'probe']) == 200
        probes = [x for x in epochs if x['stage'] == 'probe']
        best = max(probes, key=lambda x: (x['valid'], x['epoch']))
        assert best['epoch'] == result['best_epoch']
        audits.append(dict(dataset=dataset, checkpoint_reloaded=True, hashes_match=True,
                           split_and_noise_match=True, selected_epoch_verified=True,
                           max_probability_difference=float((replay-saved).abs().max())))
        row = {key: result[key] for key in ['dataset', 'seed', 'test', 'valid', 'best_epoch',
                                             'wall_seconds', 'peak_allocated_gib', 'split_hash', 'noise_hash']}
        rows.append(row)
        del model, classifier, data, ckpt
    with (ROOT / 'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / 'artifact_audit.json', dict(status='passed', checks=audits,
               diagnostic_source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT / 'diagnose_tined.py', OLD / 'tined_adapter.py']}))
    diag = json.loads((ROOT / 'tined_diagnostics/summary.json').read_text())
    lines = ['# SimMLP 单 seed 结果与 TINED CoraFull 诊断', '',
             '## SimMLP', '',
             '五个主数据集，uniform 0.4，仅训练标签加噪，seed 0，GPU。原始输入特征、20/10/70 划分及噪声指纹与主实验一致。', '',
             '| 数据集 | 测试精度 (%) | 验证精度 (%) | 分类器选中轮次 | 训练耗时 (秒) |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {100*row['test']:.2f} | {100*row['valid']:.2f} | {row['best_epoch']} | {row['wall_seconds']:.2f} |")
    lines += ['', '实现复用 https://github.com/Zehong-Wang/SimMLP 的未修改核心定义。稀疏聚合替换不可用的二进制依赖，前向及梯度有小图等价检查。',
              '无监督预训练固定最后一轮，不按测试精度选择；分类器仅用噪声训练标签，按干净验证集选择。',
              '本轮统一宽度64、编码器dropout 0.5、学习率0.01、权重衰减0.0005；保留作者AdamW、调度器和各数据集的增强及损失参数。',
              'CoraFull无作者preset，预先指定使用Cora preset。预算为200轮预训练+200轮线性分类器，不是作者1000+5000轮原实验。',
              '以上是协议适配后的单seed筛查，不能解释成五seed均值或作者原文结果。所有结果原样保存，未挑选测试分数。', '',
              '## TINED：已经验证的原因', '',
              '既有CoraFull教师最佳验证61.14%，学生最佳验证18.70%（第4轮），测试19.19%。并非导出最后一轮或选错checkpoint；验证损失最小轮次也是第4轮。',
              '使用同一个保存教师、同一初始化和同一协议，独立目录只训练学生20轮；没有重新训练教师，没有读取测试标签。原配置重现原运行前缀。', '',
              '| 单因素诊断 | 20轮内最佳验证 (%) | 第20轮验证 (%) | 第20轮验证预测覆盖类别数 |',
              '|---|---:|---:|---:|']
    for variant in diag['variants']:
        lines.append(f"| {variant['variant']} | {100*variant['best']['val']:.2f} | {100*variant['final']['valid']:.2f} | {variant['final']['predicted_classes']} / 70 |")
    gradient = diag['gradient']
    lines += ['', '主因判断：将通用dropout 0.5套用到TINED的多层学生，是本次严重退化的关键因素。仅将学生dropout设为0，就明显恢复验证分类能力，并避免原配置第20轮在验证节点上只预测一个类别。作者Cora学生preset原本就是dropout=0。',
              '该因果证据限定于本次固定教师、20轮单seed诊断；并不证明dropout是唯一因素，也不构成新正式测试结果。',
              '能量约束也有影响：将其权重设为0有较小恢复，但仍明显低于初始化和dropout=0。零权重仍保留Adam step，不等价于完全移除这个优化阶段。',
              f"保存的第4轮学生在eval模式下，DE梯度范数={gradient['de_grad_norm']:.4f}，KL梯度范数={gradient['kd_grad_norm']:.4f}，两者夹角余弦={gradient['gradient_cosine']:.4f}。这支持目标冲突诊断；这是同一checkpoint的确定性梯度检查，不是所有训练轮次的梯度统计。",
              '初始化学生验证51.95%，第一轮零权重CE阶段后降至30.62%；lambda=1时官方仍执行零权重CE的Adam更新，权重衰减可以改变参数。这种设计和高学习率也是后续优化问题，不能将其误认成适配器独有bug。',
              'CoraFull有70类而学生隐藏宽度64，官方规则产生FT→GP→GP→FT；学生三处ReLU/dropout，教师一处。该层序符合官方规则，不是层序写反。',
              '核对未发现DE数学、KL定义或优化器分组的明确移植错误。完整DGL训练等价性未验证；全邻居教师替代采样、训练索引顺序等已知适配差异仍存在。', '',
              '## 最终决策', '',
              'SimMLP：本轮五图筛查完成，可以继续考虑正式对比，但不能直接写入五seed均值表。',
              'TINED：原19.19%保持不变，诊断结果分开保存。不要把该数值作为TINED方法本身失效的证据。正式重跑前应明确方法特定学生dropout政策，优先保留作者设定，而不是把两层MLP的通用dropout强加给四层学生。',
              '无论文文件改动，无原实验覆盖，无全量复现实验。报告核验只重载已保存模型，不训练。']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', rows=rows, tined=diag['variants']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
