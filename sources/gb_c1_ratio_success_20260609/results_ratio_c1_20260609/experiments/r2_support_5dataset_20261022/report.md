# R2 + SUPPORT + LCSS：五数据集结果

## 协议

覆盖 Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo；种子 0–2；噪声条件为 uniform 0.4、uniform 0.6、pair 0.4。没有运行 clean 训练条件，也没有加入 KNN 或 K-means 臂。R2 仍用 clean validation 选择首个最佳 epoch；test 仅在全部模型完成后评分。

共 45 个数据集—种子—噪声结果单元：复用已审计的前三个数据集 27 个 R2+SUPPORT 运行，新训练 Coauthor-CS 与 Amazon-Photo 18 个模型，并为新增数据集训练 6 个无标签 R2 对齐模型。各数据集的 GRACE H 固定复用，没有做独立 GRACE 重预训练。

## 各噪声条件

| Dataset | Condition | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) |
|---|---|---:|---:|
| cora | uniform0.4 | 77.04 ± 1.24 | 74.50 ± 1.90 |
| cora | uniform0.6 | 69.18 ± 4.48 | 64.22 ± 2.61 |
| cora | pair0.4 | 66.40 ± 2.15 | 62.03 ± 0.77 |
| citeseer | uniform0.4 | 68.97 ± 1.44 | 60.75 ± 2.55 |
| citeseer | uniform0.6 | 63.73 ± 3.36 | 56.97 ± 3.13 |
| citeseer | pair0.4 | 62.89 ± 3.50 | 54.43 ± 3.60 |
| corafull | uniform0.4 | 48.43 ± 0.45 | 31.32 ± 0.62 |
| corafull | uniform0.6 | 45.50 ± 1.37 | 29.11 ± 1.27 |
| corafull | pair0.4 | 42.43 ± 1.23 | 28.26 ± 0.45 |
| coauthorcs | uniform0.4 | 92.21 ± 0.52 | 88.35 ± 1.06 |
| coauthorcs | uniform0.6 | 90.89 ± 0.16 | 85.99 ± 0.81 |
| coauthorcs | pair0.4 | 87.55 ± 1.47 | 79.61 ± 3.44 |
| amazonpho | uniform0.4 | 83.68 ± 0.73 | 76.75 ± 1.59 |
| amazonpho | uniform0.6 | 81.80 ± 2.27 | 73.81 ± 2.46 |
| amazonpho | pair0.4 | 77.04 ± 2.76 | 69.77 ± 4.11 |

## 数据集平均

下表先在每个 seed 内对三种噪声条件取平均，再计算三个 seed 的均值和样本标准差。

| Dataset | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) |
|---|---:|---:|
| cora | 70.88 ± 2.13 | 66.92 ± 1.18 |
| citeseer | 65.20 ± 1.85 | 57.38 ± 1.51 |
| corafull | 45.45 ± 0.25 | 29.56 ± 0.57 |
| coauthorcs | 90.22 ± 0.64 | 84.65 ± 1.76 |
| amazonpho | 80.84 ± 1.76 | 73.44 ± 2.29 |

## 总体描述

按每个 seed 跨五个数据集和三种噪声条件的 15 个单元取宏平均，Accuracy 为 70.52 ± 1.18% ，Macro-F1 为 62.39 ± 1.26%。

这是单方法的绝对表现汇总，没有对照臂，不能据此估计 SUPPORT 或 LCSS 的增益，也不能推出它们优于其他分组方式。三个 seed 的 GRACE 表示并非独立预训练；结果主要反映固定 H 下 split/noise、R2 对齐初始化和分类训练的变化。

## 文件

- results.csv：45 个逐单元结果及来源。
- condition_summary.csv：每个数据集和噪声条件的 seed 均值与样本标准差。
- dataset_summary.csv、seed_summary.csv、overall_summary.json：跨噪声描述统计。
- new_alignments.csv、geometry.csv、protocol_checks.json、frozen_inputs.json：新增对齐模型、SUPPORT 分区、协议检查与输入哈希。
