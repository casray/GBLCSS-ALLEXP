# BNO-GB：Beneficial Noise Learning for Open Intent Classification via Granular-Ball Representation

## 基本信息

- **作者**：Yanhua Li, Xiaocao Ouyang, Jie Zhang, Chaofan Pan, Lingfei Ren, Xin Yang。
- **出处**：*Pattern Recognition* 177 (2026), 113283；DOI [10.1016/j.patcog.2026.113283](https://doi.org/10.1016/j.patcog.2026.113283)。[期刊页面](https://www.sciencedirect.com/science/article/pii/S0031320326002487)，[代码](https://github.com/Liyanhuaa/BNO_GB)。
- **原文核验**：本地全文抽取 `文献文件\粒球噪声学习\_BNO-GB_txt.txt`，据其摘要、方法、实验与结论整理。
- **阅读状态**：已读本地全文抽取中的方法、主要实验协议与结论。**相关度**：高（粒球 + 开放意图分类的标签噪声；非图对比学习）。

## 问题、机制与目标

研究开放意图识别中的两类噪声：已知类被错标为另一已知类（IND）与未知分布样本被错标为已知类（OOD）。先在特征上无监督聚类出粒球；球级 G-ODT（Granular-ball-level Outlier Distribution Tendency）衡量球相对同标签球/邻域的离群趋势，样本级 S-ICD（Sample-level Inlier Consistency Degree）度量样本与所属球的内部一致性。组合二者，将数据分为 clean、IND、OOD 与 uncertain。之后分别使用 OOD Dispersion Loss 扩展已知类与潜在未知空间间隔，以及 Prototype Softmax / Prototype Margin Loss 学习清洁、校正后的 IND 表示。目标是开放集意图分类表示和已知/未知决策，不是对比学习损失本身。

原文公式定义两个可靠性指标及三个训练损失；文本提取公式排版不稳定，本卡不转录数学式，避免符号失真。关键机制是球级与样本级证据分工、按噪声类型处理，而非将球成员统一设为硬正样本。

## 实验与限制

实验包括 SNIPS、StackOverflow、BANKING 等开放意图基准，并在 NoisyAG-NewsMed / NoisyAG-NewsHard 等真实噪声集上测试；覆盖不同已知类比例和 20%/40% 噪声设置，并分析组件、阈值和计算成本。论文指出高噪声下噪声识别更困难；球划分/推理阈值仍需人工设置，逐轮收集特征做粒球聚类也增加峰值内存。

## 迁移判断

可借鉴**球级与样本级证据分工、识别不确定样本、避免同球成员硬施同一关系**。但其对象是文本开放集分类，监督来自已知意图标签和类原型；迁移到图节点对比需重验同球关系是否在结构/标签噪声下可信。它未将对比关系作为核心机制，不能据此解释对比项自身的优化冲突。
