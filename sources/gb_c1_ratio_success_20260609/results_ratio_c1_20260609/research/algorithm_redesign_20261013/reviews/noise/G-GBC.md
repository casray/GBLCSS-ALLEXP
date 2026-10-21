# G-GBC：A Gaussian Mixture-Based Granular Ball Computing Method for Robust CNN Classification under Noisy Labels

## 基本信息与阅读状态

- **作者**：Junxiao Yang, Qiang Liu, Xuemei Cao, Fei Li, Xin Yang。
- **出处**：*Knowledge-Based Systems* 341 (2026), 115754；DOI [10.1016/j.knosys.2026.115754](https://doi.org/10.1016/j.knosys.2026.115754)。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0950705126004855)。
- **全文来源**：本地完整抽取文本 `文献文件\_literature_review_work\extracted_text\粒球噪声学习\G-GBC--A-Gaussian-mixture-based-granular-ball-computing-me_2026_Knowledge-Ba.txt`；另有较短抽取版本。
- **阅读状态**：已读本地全文抽取的主要方法、实验设置、结论和限制；未逐项重核所有表格数值。**相关度**：高（CNN 分类标签噪声中的粒球机制；不是图或对比学习方法）。

## 问题与方法

方法在 CNN 表示上用 Mahalanobis 距离优化的 GMM 软分配生成/更新粒球，以类别纯度和球权重决定球对样本表征及分类的贡献。球的协方差用于表达非球形、多模态簇；权重层结合 ball purity 与到球中心的 Mahalanobis 距离，汇总后融合回特征表示。重点是概率化球分配、纯度加权和迭代粒度调整，不是用球成员构造图对比正负对。

本地抽取中公式编号与符号能见到，但跨页数学排版质量不足以在此准确复写完整目标函数；因此不抄一个看似完整、实则可能丢掉球权重/正则项的公式。

## 实验与边界

已核实验涉及 CIFAR-10、CIFAR-100、STL-10、Fashion-MNIST 等合成噪声设置（含对称噪声 0–40%、CIFAR 更高对称噪声、实例依赖噪声和 CIFAR-10 40% 非对称噪声），以及 CIFAR-10N、ANIMAL-10N、Clothing1M 等真实噪声基准。方法在 RN18/RN34/PreAct RN18 等视觉模型上比较。论文还报告球纯度/数量敏感性、计算开销和可解释性可视化。

作者在结论中明确指出：现有评估数据仍不足以代表更大、更复杂应用；粒球层带来额外计算开销，后续需提升效率并扩展到更丰富噪声类型。实验虽含多种合成与真实噪声，但不能据此推断图节点或图关系学习收益。对本项目可迁移的是 soft assignment、purity-weighted granularity 作为置信度信号；不可直接迁移的是其 CNN/GMM 生成机制或“球即标签”假设。具体提升应逐表核验后再引用。
