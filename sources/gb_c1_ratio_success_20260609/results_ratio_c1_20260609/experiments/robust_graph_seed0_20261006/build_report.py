from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "corafull": "CoraFull",
         "coauthorcs": "Coauthor-CS", "amazonpho": "Amazon-Photo"}
METHODS = ["gcn_ce", "gblcss", "tfr", "erase"]


def history_rows():
    directory = PROJECT / "external/NoisyGL/runs"
    rows = {}
    for path in [directory / f"mlp58_formal_lcss_teacher_20260813/formal_u0p{rate}_seeds0_4.csv" for rate in (4, 6)] + [
        directory / f"mlp59_final_extensions_20260814/teacher_u0p{rate}_seeds0_4.csv" for rate in (4, 6)]:
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["seed"] == "0" and row["method"] == "mlp_distill_formal_lcss_x":
                    rows[row["dataset"], float(row["noise_rate"])] = dict(row, source=str(path))
    return rows


def write_csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "runs").glob("*/*/*/result.json"))]
    write_csv(ROOT / "results.csv", rows)
    index = {(row["dataset"], row["noise_rate"], row["method"]): row for row in rows if row["status"] == "ok"}
    history = history_rows()
    paired, audits, historical = [], [], []
    for dataset in NAMES:
        for rate in (.4, .6):
            group = [index[dataset, rate, method] for method in METHODS if (dataset, rate, method) in index]
            audits.append(dict(dataset=dataset, rate=rate, completed=len(group), expected=4,
                               split_hashes=len({row["split_hash"] for row in group}),
                               noise_hashes=len({row["noise_hash"] for row in group}),
                               modified_validation=sum(row["modified_val_count"] for row in group),
                               modified_test=sum(row["modified_test_count"] for row in group)))
            reference = index.get((dataset, rate, "gblcss"))
            if reference is not None:
                old = history[dataset, rate]
                assert reference["split_hash"] == old["split_hash"]
                assert reference["noise_hash"] == old["noise_hash"]
                historical.append(dict(dataset=dataset, rate=rate, historical_student=100 * float(old["test"]),
                                       historical_teacher=100 * float(old["teacher_best_test"]),
                                       rerun_teacher=100 * reference["teacher_test"],
                                       teacher_difference_pp=100 * (reference["teacher_test"] - float(old["teacher_best_test"])),
                                       common_rng_student=100 * reference["student_test"], source=old["source"]))
                for method in ("tfr", "erase", "gcn_ce"):
                    row = index.get((dataset, rate, method))
                    if row:
                        paired.append(dict(dataset=dataset, rate=rate, method=method,
                                           teacher_difference_pp=100 * (row["teacher_test"] - reference["teacher_test"]),
                                           student_difference_pp=100 * (row["student_test"] - reference["student_test"]),
                                           student_minus_teacher_pp=100 * (row["student_test"] - row["teacher_test"])))
    write_csv(ROOT / "paired_deltas.csv", paired)
    write_csv(ROOT / "protocol_audit.csv", audits)
    write_csv(ROOT / "historical_reference.csv", historical)
    lines = ["# 图噪声鲁棒教师与蒸馏：seed 0 筛查", "",
             f"已完成 {len(index)}/40 组，失败 {sum(row['status'] == 'failed' for row in rows)} 组。", "",
             "## 1. 实验回答什么", "",
             "比较近期鲁棒图方法作为教师时的精度，以及这些教师接相同属性 MLP 后的效果。"
             "参考线为 GCN+CE 和 GB-LCSS 目标训练的 GCN。粒球只用于参考线，未加入 TFR、ERASE。", "",
             "五个数据集，训练标签 uniform 0.4/0.6，seed 0，20%/10%/70% 划分；"
             "验证和测试标签均未污染，检查点仅由验证精度选择。所有模型和逻辑回归探针在 RX 7700 XT 上训练。", "",
             "## 2. 蒸馏学生测试精度（%）", "",
             "| 数据集 | 噪声率 | GCN+CE→MLP | GB-LCSS→MLP | TFR→MLP | ERASE→MLP |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in NAMES:
        for rate in (.4, .6):
            values = [f"{100 * index[dataset, rate, method]['student_test']:.2f}" if (dataset, rate, method) in index else "未完成/失败"
                      for method in METHODS]
            lines.append(f"| {NAMES[dataset]} | {rate} | " + " | ".join(values) + " |")
    lines += ["", "## 3. 教师测试精度（%）", "",
              "| 数据集 | 噪声率 | GCN+CE | GB-LCSS GCN | TFR | ERASE+逻辑回归 |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in NAMES:
        for rate in (.4, .6):
            values = [f"{100 * index[dataset, rate, method]['teacher_test']:.2f}" if (dataset, rate, method) in index else "未完成/失败"
                      for method in METHODS]
            lines.append(f"| {NAMES[dataset]} | {rate} | " + " | ".join(values) + " |")
    lines += ["", "## 4. 配对差值与方向判断", ""]
    for method in ("tfr", "erase"):
        selected = [row for row in paired if row["method"] == method]
        if selected:
            teacher_positive = sum(row["teacher_difference_pp"] > 0 for row in selected)
            student_positive = sum(row["student_difference_pp"] > 0 for row in selected)
            average_teacher = sum(row["teacher_difference_pp"] for row in selected) / len(selected)
            average_student = sum(row["student_difference_pp"] for row in selected) / len(selected)
            lines.append(f"- {method.upper()}：教师在 {teacher_positive}/{len(selected)} 个设定高于 GB-LCSS GCN，"
                         f"平均差值 {average_teacher:+.2f} pp；学生在 {student_positive}/{len(selected)} 个设定更高，"
                         f"平均差值 {average_student:+.2f} pp。")
            for rate in (.4, .6):
                subset = [row for row in selected if row["rate"] == rate]
                if subset:
                    gain = sum(row["student_difference_pp"] for row in subset) / len(subset)
                    positive = sum(row["student_difference_pp"] > 0 for row in subset)
                    lines.append(f"  uniform {rate}：学生 {positive}/{len(subset)} 个设定更高，平均差值 {gain:+.2f} pp。")
            teacher_only = [row for row in selected if row["teacher_difference_pp"] > 0 and row["student_difference_pp"] <= 0]
            if teacher_only:
                cells = ", ".join(f"{NAMES[row['dataset']]} u{row['rate']}" for row in teacher_only)
                lines.append(f"  教师更强、学生未提升的设定：{cells}。蒸馏损失或属性 MLP 的压缩能力是待检查的解释，不是已证明原因。")
    if len(index) == 40:
        erase_rows = [row for row in paired if row["method"] == "erase"]
        weaker = [row for row in erase_rows if row["student_difference_pp"] < 0]
        weak_cells = ", ".join(f"{NAMES[row['dataset']]} u{row['rate']} ({row['student_difference_pp']:+.2f} pp)"
                               for row in weaker)
        reference_positive = sum(index[dataset, rate, "gblcss"]["student_test"] >
                                 index[dataset, rate, "gcn_ce"]["student_test"]
                                 for dataset in NAMES for rate in (.4, .6))
        high_noise = [row for row in erase_rows if row["rate"] == .6]
        high_positive = sum(row["student_difference_pp"] > 0 for row in high_noise)
        lines += ["", f"本轮证据：ERASE 在 uniform 0.6 下的学生在 {high_positive}/5 个数据集上高于 GB-LCSS；"
                  "较低噪声下并非全面占优。ERASE 学生低于 GB-LCSS 的设定为：" + (weak_cells or "无") + "。",
                  f"GB-LCSS 学生高于普通 GCN+CE 学生的设定为 {reference_positive}/10；"
                  "CoraFull 等负结果同时保留，不能只报告优势数据集。",
                  "解释边界：本轮支持筛选更强的教师路线，但尚不能确定差距来自鲁棒机制、"
                  "模型容量、预设参数、输出置信度还是这些因素的组合。"]
    lines += ["", "## 5. 可比性与限制", "",
              "- TFR 使用作者 DBLP 预设；作者未提供本轮五数据集的预设。ERASE 使用作者 Cora、CiteSeer、CoraFull 预设，"
              "Coauthor-CS 与 Amazon-Photo 固定迁移 Cora 预设。本轮没有调参，不代表各方法最优效果。",
              "- TFR 核心 GCN/解码器、ERASE GAT/损失类直接来自作者代码。稀疏传播、散射和原型计算做了 ROCm GPU 适配。"
              "ERASE 使用作者已有的等价多类压缩公式避免平方规模的成员矩阵；逻辑回归保持 C=1 多项式目标，GPU LBFGS 与 sklearn 合成测试通过。",
              "- TFR 按作者训练循环保留解码器在第一次验证后保持 eval 的行为。未擅自修正作者代码而改变方法。",
              "- TFR 第一批 8 次尝试提前添加了图级自环，与作者 DBLP 数据配置不符，已归档为 superseded_tfr_selfloops，"
              "不参与最终统计。修正后使用原始图，由 GCN 和解码器各自在内部添加自环，其他超参数未调整。",
              "- ERASE 作者对 Cora/CiteSeer 默认做特征行归一化；本轮为匹配主实验，统一使用原始特征。"
              "因此输入预处理也属于本轮协议适配，不能把本轮结果当作作者原协议的直接复现。",
              "- 参考教师使用主实验原始特征和已核验的粒球成员缓存。学生训练前对每个教师统一重置 RNG，"
              "因此本轮 GB-LCSS 学生不是论文原始 seed-0 学生的逐数值重现。历史值单独记录，未与本轮混算。",
              "- 教师预算按公开方法保留：TFR 500、ERASE 400（patience 150）、参考 GCN 200；学生均为 200。"
              "这里只统一评估及学生训练协议，尚未统一教师计算预算。",
              "- 教师架构和容量也按方法保留：参考 GCN 隐层 64，TFR 公开预设隐层 16，ERASE 隐层 256、表示维数 512。"
              "本轮比较完整路线，不是只替换损失的机制消融。",
              "- 记录的 GB-LCSS 运行时间不含已完成的 GRACE 预训练和粒球划分；不能与其他方法直接做端到端训练成本排名。",
              "- 单 seed 无法给出稳定排名、标准差或统计显著性。差值只是下一轮实验的筛选依据。", "",
              "## 6. 失败记录", ""]
    failed = [row for row in rows if row["status"] == "failed"]
    if failed:
        for row in failed:
            lines.append(f"- {row['dataset']} u{row['noise_rate']} {row['method']}：{row.get('error', '').strip().splitlines()[-1]}")
    else:
        lines.append("最终运行没有失败项。此前八组 TFR 适配错误的结果已归档并重跑，详见第 5 节。")
    lines += ["", "## 7. 决策", "",
              "继续：优先为 ERASE 与两条 GCN 对照补足 seeds 1-4，检验强噪声优势能否重复。"
              "TFR 优先跟进 CiteSeer uniform 0.6 的优势，其余数据集先核查迁移预设，不直接扩大调参。"
              "修改：增加与 ERASE 同容量的 GAT+CE 对照，隔离鲁棒机制与架构容量的贡献；"
              "随后再检查输出置信度和蒸馏损失。暂缓：不能用本轮迁移预设下的负结果否定某个已发表方法。", "",
              "完整配置、逐 epoch 日志、教师和学生检查点、全节点概率在 runs/。配对差值见 paired_deltas.csv，"
              "协议检查见 protocol_audit.csv，历史参考值见 historical_reference.csv，源代码与结果指纹见 sha256_manifest.csv。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = []
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.name != "sha256_manifest.csv":
            manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_csv(ROOT / "sha256_manifest.csv", manifest)
    print(f"completed={len(index)} failed={len(failed)}")


if __name__ == "__main__":
    main()
