# RAC-DMVC

## 元数据

Shihao Dong, Yue Liu, Xiaotong Zhou, Yuhui Zheng, Huiying Xu, Xinzhong Zhu. “RAC-DMVC: Reliability-Aware Contrastive Deep Multi-View Clustering Under Multi-Source Noise.” AAAI 2026, 40(25), pp. 20835–20843. [AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/39223)；[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/39223/43184)。

## 问题与任务

无监督多视图聚类中的多源输入噪声：视图缺失（missing noise）和可见视图特征受污染（observation noise）。论文不是训练标签翻转、图边噪声或节点分类任务；模型假设样本跨视图索引对应。

## 创新与机制

方法包含三条相互配合的路径：

1. **跨视图重构**：一个视图的编码表示通过另一视图的解码器重构数据，以吸收互补信息并减弱输入观测污染影响。
2. **可靠性加权对比**：从视图内/跨视图表示相似度构造连续样本关系权重，把关系权重作为正负对比贡献，而非仅按“同实例为正、其他实例为负”的固定二值划分。
3. **缺失表示补全与聚类蒸馏**：双注意力结合视图内和视图间信息补全缺失表示；对拼接视图做 K-means 得到软聚类目标，再以 KL 蒸馏约束各视图的聚类预测。

## 实验证据

使用 Scene-15、Caltech-101、LandUse-21、Reuters、NUS-WIDE 五个数据集，报告 ACC、NMI、ARI，并做组件消融和噪声比例分析。主表将 missing noise 与 observation noise 设为相同的比例；实现细节也以二者均为 0.5 作为默认值。此设计评估的是输入数据污染和缺失下的聚类表现，不是监督标签噪声鲁棒性。

## 对本项目的启发与迁移边界

最直接可借鉴的是把样本关系从二值正负对改为连续权重，并通过消融验证加权项是否真正改善学习。粒球要证明其成员、半径、局部一致性或跨尺度统计比直接相似度提供了额外的关系质量信息。

RAC-DMVC 的关系图由多视图特征表示产生，视图缺失时还需要对应的补全模块。它没有图神经网络、外部节点标签或标签噪声协议；其“reliability-aware”不能被写成已验证的训练标签去噪能力。关系相似度也可能随受污染表示一起偏移。

## 阅读深度与证据

已读 AAAI 官方 PDF 全文的方法、总损失、五数据集实验、噪声比例设置和消融（9 页）。证据：[官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/39223/43184)；发表信息和摘要见 [AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/39223)。阅读级别：全文方法与实验核读。
