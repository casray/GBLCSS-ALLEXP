"""Audit saved screening artifacts and generate the experiment report."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from common import DATASETS, ROOT, final_metrics, load_data, write_json
from run_screen import aggregate

NAMES = dict(cora="Cora", citeseer="CiteSeer", corafull="CoraFull",
             coauthorcs="Coauthor-CS", amazonpho="Amazon-Photo")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit_run(dataset, method):
    folder = ROOT / "runs" / dataset / method
    result = read(folder / "result.json")
    data, protocol = load_data(dataset, device="cpu")
    for key in ["split_hash", "noise_hash"]:
        assert result[key] == protocol[key]
    assert result["status"] == "ok" and not result["smoke"]
    assert result["seed"] == 0 and result["noise_rate"] == .4
    probabilities = torch.load(folder / "probabilities.pt", map_location="cpu", weights_only=False)
    metrics = final_metrics(dataset, probabilities)
    for key in ["test", "valid"]:
        assert abs(metrics[key] - result[key]) < 1e-12
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()]
    assert len(rows) == 400
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["val"]) for row in rows)
    if method == "gssc":
        eligible = [row for row in rows if row["checkpoint_eligible"]]
        assert len(eligible) == 300
        selected = max(eligible, key=lambda row: (row["val"], row["epoch"]))
        assert checkpoint["best_epoch"] == result["best_epoch"] == selected["epoch"]
        assert result["diagnostics"]["homophily_updates"] > 0
        assert result["diagnostics"]["sampler_updates"] > 100
        assert not result["diagnostics"]["cpu_training"]
        assert not result["diagnostics"]["validation_labels_in_optimization"]
    else:
        teachers = [row for row in rows if row["phase"] == "teacher"]
        eligible = [row for row in rows if row["phase"] == "student"]
        assert len(teachers) == len(eligible) == 200
        selected = max(eligible, key=lambda row: (row["val"], row["epoch"]))
        assert checkpoint["student_best"]["epoch"] == result["best_epoch"] == selected["epoch"]
        assert any(row["de"] > 0 for row in eligible)
        assert result["diagnostics"]["no_cpu_fallback"]
    assert selected["val"] == result["best_val"]
    assert not result["diagnostics"]["test_labels_read"]
    source = ROOT / f"{method}_adapter.py"
    provenance = folder / "adapter_provenance.json"
    if provenance.exists():
        assert read(provenance)[source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
        source_check = "launch SHA256 matches frozen source"
    else:
        # Earlier runs predate launch-hash logging; source was already frozen.
        assert source.stat().st_mtime <= (folder / "epochs.jsonl").stat().st_ctime
        source_check = "source freeze preceded log creation; no launch hash available"
    write_json(folder / "artifact_audit.json", dict(
        passed=True, source_check=source_check, epochs=len(rows),
        validation_only_selection=True, probabilities_rechecked=True,
        split_hash=protocol["split_hash"], noise_hash=protocol["noise_hash"],
        adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()))
    return result


def main():
    torch.set_num_threads(4)
    results = {(dataset, method): audit_run(dataset, method)
               for dataset in DATASETS for method in ["gssc", "tined"]}
    for dataset in DATASETS:
        blocked = read(ROOT / "runs" / dataset / "dgd" / "blocked.json")
        assert not blocked["trained"]
    vendor = {}
    for method in ["gssc", "tined", "dgd"]:
        directory = ROOT / "vendor" / method
        assert not subprocess.check_output(["git", "-C", str(directory), "status", "--short"], text=True).strip()
        vendor[method] = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    aggregate()
    table = ["| 数据集 | GSSC 测试准确率 | TINED 测试准确率 | DGD |",
             "|---|---:|---:|---|"]
    timing = ["| 数据集 | GSSC 秒 / GiB | TINED 秒 / GiB |",
              "|---|---:|---:|"]
    details = []
    for dataset in DATASETS:
        g, t = (results[dataset, method] for method in ["gssc", "tined"])
        table.append(f"| {NAMES[dataset]} | {100*g['test']:.2f}% | {100*t['test']:.2f}% | 实现阻塞，未训练 |")
        timing.append(f"| {NAMES[dataset]} | {g['wall_seconds']:.1f} / {g['peak_allocated_gib']:.3f} | {t['wall_seconds']:.1f} / {t['peak_allocated_gib']:.3f} |")
        details.append(f"- {NAMES[dataset]}：GSSC 验证准确率 {100*g['valid']:.2f}%，最佳阶段 `{g['diagnostics']['best_phase']}` 第 {g['diagnostics']['best_phase_epoch']} 轮；TINED 验证准确率 {100*t['valid']:.2f}%，最佳学生轮次 {t['best_epoch']}，教师最佳验证准确率 {100*t['diagnostics']['teacher_best_val']:.2f}%。")
    warmups = sum(results[dataset, "gssc"]["diagnostics"]["best_phase"] == "classifier_pretrain" for dataset in DATASETS)
    worst = max(DATASETS, key=lambda dataset: results[dataset, "tined"]["diagnostics"]["teacher_best_val"] - results[dataset, "tined"]["valid"])
    worst_result = results[worst, "tined"]
    gap = 100 * (worst_result["diagnostics"]["teacher_best_val"] - worst_result["valid"])
    report = f"""# 五图单 seed 基线试跑：uniform 0.4，seed 0

## 实验结论

GSSC、TINED 已完成全部五图 GPU 训练，共 10 项；DGD 的 5 项均在训练前阻塞，没有生成或填入准确率。所有完成项均通过保存预测的独立重算、历史划分与噪声指纹核验及验证集选模日志核验。

{chr(10).join(table)}

这些数字是单次试跑，不是五 seed 的均值或显著性结果。本轮没有修改论文表格，也没有重跑原论文主实验。两种方法的训练资源和网络结构不同，不能将这张表理解为等参数量、等总训练预算的排名。

## 协议与预算

- 五个主数据集，原始节点属性和图边；原有有序 20%/10%/70% 划分，seed 0。
- 仅训练标签加入 uniform 0.4 噪声；五图的划分和噪声 SHA256 均与主实验协议回放记录完全匹配。
- 训练接口中非训练节点标签为 -1；干净验证标签仅用于选模。保存最佳模型预测后才读取测试标签计算准确率；并列最佳验证准确率取最后一轮。
- 共用 hidden 64、dropout 0.5、Adam 学习率 0.01、weight decay 0.0005。方法特有系数采用预先固定的作者配置，不根据本轮测试结果调参。
- GSSC：边采样器预训练 100、分类器预训练 100、主训练 200，总阶段轮数 400，含标签监督的分类器轮数 300；保留作者的混合节点、双头、自对比和采样器外层更新。最佳模型允许来自分类器预训练或主阶段。
- TINED：SAGE-gcn 教师 200、学生 200；保留教师参数注入、4 个 FT/GP 全连接层、全节点响应蒸馏和跨层 Dirichlet-energy 蒸馏，CE、KL、DE 分阶段更新。学生部署只使用原始节点属性。
- TINED 复用官方学生序列类，并将 DGL 教师与能量计算移植为稀疏 PyTorch；教师采用全邻居聚合，不是作者采样邻居的相同随机轨迹。GSSC 使用等价的稀疏分块计算；ROCm 与旧作者运行环境不保证逐位一致。
- CoraFull 等作者没有预设的图采用事先声明的 Cora 配置映射；详细映射和公式核对见 `GSSC_NOTES.md`、`TINED_NOTES.md`。本轮不进行超参数搜索。

## 逐图观察

{chr(10).join(details)}

GSSC 有 {warmups}/5 个图的最佳验证模型来自分类器预训练。这说明在当前配置与训练标签噪声下，其后续主阶段没有在这些图上取得更高的验证准确率；不能据此断言该机制普遍无效。TINED 的所有学生训练日志均存在非零能量蒸馏项，GSSC 所有图均完成采样器和同质性外层更新；本轮没有用普通 MLP 或纯响应蒸馏替代核心机制。

TINED 的最大教师与学生验证准确率差出现在 {NAMES[worst]}，相差 {gap:.2f} 个百分点。这表明当前配置下学生未充分保持教师的预测能力；具体原因尚未确认，不能直接归因于方法本身。扩大到正式多 seed 对比前，应先核查学生端预设、目标权重与本协议的适配情况。本轮保留原分数，不进行按测试结果选参数或择优重跑。

## 时间与显存

{chr(10).join(timing)}

时间为本机本轮运行的阶段总墙钟时间，显存为 PyTorch 峰值分配量。GSSC 包含额外预训练、TINED 包含教师与学生训练，不是相同预算下的速度比较。硬件与软件版本见每项 `environment.json`。

## DGD 与 SimMLP

DGD 官方仓库缺少被调用的 `dtw_dsasm` 核心函数；另有融合、梯度生成反馈、参数索引和损失系数方面的未决问题。公开摘要不足以确认应该如何补齐，直接猜测会变成未经验证的变体。因此五项均保留 `blocked.json`，而不是填入分数。需要完整论文的方法公式或作者修正版；详见 `DGD_NOTES.md`、`dgd_preflight.json`。

SimMLP 的本地查找只找到文献和讨论材料，没有找到能追溯到配置、日志和原始结果的历史运行。此次不重跑，详见 `SIMMLP_HISTORY.md`。

## 改动、检查与最终决策

研究假设：用公开方法的核心实现检验其在当前噪声标签协议下的实际表现，不预设谁优于谁。粒球角色：本轮外部基线不使用粒球。代码改动仅在本实验目录：共用数据协议、GSSC/TINED 适配、运行与审计脚本；作者仓库未修改。预期机制分别是结构约束与混合节点训练，以及教师参数/能量信息迁移。

工程成功标准：GPU 完成训练、输出有限概率、选模和保存模型一致、数据指纹一致、核心模块实际更新。失败标准：泄漏测试标签、退回 CPU、非有限目标、核心实现缺失或 GPU 资源不足。前两种方法均满足工程成功标准；DGD 因核心实现证据缺失暂缓，不能作为性能失败。

冻结后快速测试：25 项中 23 通过、2 项可选 GPU 单元测试跳过；真实 Cora GPU 冒烟另行完成。TINED 初次冒烟在训练前因错误要求划分覆盖全部节点而失败，已修正为允许按比例取整的剩余节点作为无标签上下文；原划分、数据和训练公式均不变，失败日志保留。没有开展全量原论文复现实验。Cora 的首次 GSSC 运行与源文件最后修正重叠，因此原记录保留在 `excluded_runs/`，最终表使用冻结版补跑；其余图不重跑。

最终决策：保留当前结果作为新增基线的单 seed 筛选证据；未经进一步授权不扩大到多 seed、不写入论文均值。每项保留 config、原始 epoch/console 日志、checkpoint、probabilities、协议指纹、时间/显存及 `artifact_audit.json`。全部作者 Git 提交和本地代码哈希见 `screen_audit.json`。
"""
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(ROOT / "screen_audit.json", dict(passed=True, completed=10, blocked=5,
        author_commits=vendor, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.glob("*.py")}, gssc_warmup_selected=warmups))
    print("AUDIT PASSED: 10 completed GPU runs, 5 blocked; REPORT.md and results.csv saved")


if __name__ == "__main__":
    main()
