# GBSVR：Granular Ball Support Vector Regression

## 基本信息

- **作者**：Reshma Rastogi, Ankush Bisht, Sanjay Kumar, Suresh Chandra。
- **状态**：arXiv:2503.10539 (2025)，[原文](https://arxiv.org/abs/2503.10539)。本次按 arXiv 版本记录，未确认后续正式出版状态。

## 问题与机制

经典 SVR 的二次规划在大样本上成本高，epsilon-insensitive 误差也会受离群点影响。GBSVR 先把连续回归目标分箱，用自顶向下划分形成有中心、半径、聚合响应值的 granular regression balls，再用球代替大量单点训练 SVR。损失和约束是 SVR 的 epsilon-tube / slack-variable 形式按球中心及半径改写；这里优化的是连续目标的预测误差，不是标签分类或对比关系。

可核实的概念式：球目标值可写作成员回归响应的均值 \(\hat y_B=|B|^{-1}\sum_{i\in B} y_i\)。完整软间隔约束需以原文公式为准；二手摘要中的简化公式没有包含球半径项，故不在此冒充原式。

## 实验、限制与迁移

作者在合成、UCI/基准及真实预测数据上报告回归效果，并讨论不同噪声类型；具体数据、表格与统计量须查 arXiv 全文。其方法目标首先是以球压缩训练点并提高回归可扩展性，不能据此推断能纠正分类标签噪声。分箱纯度、最小球规模和划分策略均可能影响结果。对本项目只适合作为“连续目标的粒球聚合”邻近文献；与 Rank-N-Contrast 一样，应与 noisy-label classification 分开。**阅读状态**：已核 arXiv 题录/摘要和方法概述；未逐项核实验表。**相关度**：低到中（连续目标回归，不是分类标签噪声）。


