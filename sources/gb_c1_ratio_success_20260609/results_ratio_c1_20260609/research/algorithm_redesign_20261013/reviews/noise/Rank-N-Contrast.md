# Rank-N-Contrast：Learning Continuous Representations for Regression

## 基本信息

- **作者**：Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi。
- **出处**：NeurIPS 2023, Main Track；[会议论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)，[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/39e9c5913c970e3e49c2df629daff636-Paper-Conference.pdf)，[代码](https://github.com/kaiwenzha/Rank-N-Contrast)。

## 问题、方法与目标

普通回归端到端训练未必学出反映连续标签次序的表示。Rank-N-Contrast 依据锚点到各样本的目标距离排序：若样本 j 的目标比 k 更接近锚点，就要求表示空间里的相似/距离关系符合这一排序。论文用 listwise 排序式对比损失，将候选按目标距离排序，再对相对顺序构造概率目标；训练表示后冻结 encoder，再训练回归头。其理论声明是表示排序与目标排序相协调。

**目标变量是连续回归标签差异及其相对次序**。它不是分类标签噪声鲁棒损失；本身依赖目标标签构造样本关系，标签被错标时排名关系同样可能污染。这里不写损失全式，避免删去原文对距离、温度和候选集合的定义。

## 实验与迁移

作者报告在五个跨视觉、人机交互、医疗领域的真实回归数据集上测试，并分析数据效率、spurious targets、数据破坏与分布迁移。不能把其连续排序机制直接替代类标签噪声下的正负关系置信度。它启发的是“关系监督可编码有序/相对目标差异”，而当前节点分类的球间关系需由可验证语义/结构证据决定。**阅读状态**：已核 NeurIPS 官方论文页、方法与实验概述；未逐表复核统计细节。**相关度**：低（连续回归关系排序，不是分类标签噪声；注意勿与 RNCGLN 混淆）。


