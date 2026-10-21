# SNSCL：Fine-Grained Classification With Noisy Labels

## 基本信息

- **作者与出处**：Qi Wei, Lei Feng, Haoliang Sun, Ren Wang, Chenhui Guo, Yilong Yin；CVPR 2023，pp. 11651–11660。
- **原始来源**：[CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html)、[官方 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.pdf)。
- **阅读状态**：已核官方论文页、PDF 摘要和方法描述；未逐表校对所有结果。**相关度**：中高（标签噪声下的 supervised contrastive 机制；任务为图像细粒度分类）。

## 问题与机制

论文研究细粒度分类的 noisy-label learning：类别间差异小，噪声标签更容易让分类器过拟合。SNSCL（stochastic noise-tolerated supervised contrastive learning）针对 SupCon 的两类污染源设计机制：以样本可靠性权重降低噪声 anchor 的影响，并据此调整当前批次中 anchor 的监督标签；只将被认为可信的样本加入/更新 momentum queue，减少错误标签进入后续 key 集合。另以随机特征采样模块从生成的特征分布中采样表示，代替完全手工设定强增强。

**优化目标**是噪声标签下细粒度图像的可分表示与分类表现。官方摘要没有给出足够稳定的符号定义供此处复写损失公式；卡片不拼凑公式。该方法仍依赖可靠性权重/标签调整来保护 SupCon 正对，不是图结构或粒球关系学习。

## 实验、限制与迁移

作者在多个细粒度图像基准上测试，官方 PDF 摘要报告四个细粒度 benchmark，并含合成噪声设置；其声称 SNSCL 可与既有 LNL 方法组合。详细数据集、各噪声比例、随机种子与逐项数值未在本卡中复核，引用时回到论文表格。方法迁移到节点图任务需重做“可靠 anchor / queue key”的定义：仅按模型置信度保留节点仍可能因邻居标签传播而形成相关错误；也不能由图像 benchmark 推断粒球关系安全。

**本项目用途**：作为“对比学习内部显式处理噪声 anchor 和正样本队列污染”的图外参照；可借鉴分开控制 anchor 与 key 污染的设计问题。
