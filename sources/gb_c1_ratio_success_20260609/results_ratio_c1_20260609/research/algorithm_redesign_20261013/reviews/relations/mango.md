# MANGO

## 元数据

Feiyang Chen, Ruiqiang Guo, Zhibin Gu. “Multi-Scale Diffusion-Guided Graph Learning with Power-Smoothing Random Walk Contrast for Multi-View Clustering.” ICLR 2026. [ICLR 官方论文页与摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html)；[OpenReview 作者稿入口](https://openreview.net/pdf?id=ynT6rqo4Lp)。

## 问题与任务

多视图特征上的无监督聚类。文章针对静态相似图难以表现跨视图全局关系、对比学习假负样本，以及视图一致性与视图特异性之间的冲突。这里的“图”是从多视图样本表示构建的关系图，不等于输入的带标签节点分类基准图；摘要没有提出标签噪声设置。

## 创新与流程

官方摘要明确三个模块：局部熵引导不同扩散步数的相似矩阵自适应融合；随机游走沿高概率语义路径修正对比关系并减少假负样本；对比目标中的 beta 幂变换重加权目标；共享结构嵌入做跨视图语义对齐，同时保留异构特征。其思想更接近从关系图生成软训练权重，而非仅按单步邻接二值决定正负样本。

## 目标及结构更新

由多尺度扩散形成和更新相似关系，随机游走关系进入对比目标，结构嵌入用于跨视图一致性。官方摘要没有给出足够信息核实完整损失、扩散更新频率、超参数和聚类头，故不写公式或复述二手解读中的细节。

## 实验

ICLR 摘要报告在 12 个基准数据集上评估多视图聚类并称整体表现优于已有方法；本次无法通过 OpenReview 验证挑战进入全文，也未能核对数据清单、指标和统计显著性。没有依据把摘要中的假负样本改进说成已验证的输入噪声/标签噪声鲁棒性。

## 对本项目的启发

粒球可以作为压缩后的关系图节点，粒球间路径概率可供软化负样本权重；但必须与样本级关系图对照，检验聚合是否保留了关键语义路径。可借鉴“关系可能程度连续变化”，不能仅把 MANGO 的关系节点替换成粒球就宣称新颖。

## 风险与边界

其任务是无监督多视图聚类，学习到的相似关系仍可能自我强化早期错误。论文摘要声称处理假负关系，不证明外部标签噪声识别有效。现阶段没有可核对正文，所有方法细节以摘要所述为上限。

## 阅读深度与证据

已读 ICLR 官方摘要；OpenReview PDF 在本次访问中进入浏览器验证页，未读全文方法与实验表，故标为摘要级。证据：[ICLR 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html)。
