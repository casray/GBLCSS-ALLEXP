# Foundations：SCE、Co-teaching 与 ELR

本页作为噪声学习的基础对照卡，聚焦三种不同抗噪路径：改损失、交叉选样、利用早期学习时间一致性。它们均源于一般分类/图像任务，不能直接视为图节点上的同条件基线。

## SCE：Symmetric Cross Entropy

- **出处**：Yisen Wang et al., *Symmetric Cross Entropy for Robust Learning With Noisy Labels*, ICCV 2019, pp. 322–330。原文：[CVF 官方页面](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html)、[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf)。
- **核心机制**：对标准 CE 加入 reverse cross entropy (RCE)，即 (L_{SCE}=\alpha L_{CE}+\beta L_{RCE})。RCE 交换预测与标签在交叉熵中的角色；对 one-hot 观测标签，原文对零标签概率作截断以避免 \(\log 0\)。目的在缓解 CE 的噪声过拟合并补偿类别学习不平衡。
- **边界**：RCE 对严重噪声鲁棒性与拟合能力间有权衡；原论文在图像 benchmark/真实噪声数据上评估，不提供粒球/图关系证据。复写代码时应遵循原文截断和权重设置，不能只照抄无截断数学式。

## Co-teaching

- **出处**：Bo Han et al., *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*, NeurIPS 2018。原文：[NeurIPS 官方页](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)、[PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/a19744e268754fb0148b017647355b7b-Paper.pdf)。
- **核心机制**：两个网络各自按小损失选 mini-batch 子集，再把自己选择的样本交给另一个网络更新；选择比例随训练日程逐渐变化。交叉选择意在降低单模型错误置信度自我强化。
- **边界**：方法仰赖“早期小损失样本较可能干净”及噪声率/forget rate 设置；两个网络若过早趋同，交叉纠错能力可能减弱。原文在 MNIST、CIFAR-10/100 噪声实验，非图节点协议。

## ELR：Early-Learning Regularization

- **出处**：Sheng Liu et al., *Early-Learning Regularization Prevents Memorization of Noisy Labels*, NeurIPS 2020。原文：[NeurIPS 官方页](https://proceedings.neurips.cc/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html)。
- **核心机制**：利用模型早期学习到的预测作为样本级 temporal target，通过半监督目标概率与正则项约束后续预测，减缓模型对错误标注的记忆。其关键变量是随训练更新的目标概率/预测轨迹，不是样本邻接或球置信度。
- **边界**：质量受早期 target 的可靠性、正则权重和训练日程影响；论文摘要报告合成与真实 benchmark 上有竞争力，具体数据/数值应回正文。不能由 ELR 的“early learning”机制推出图邻居/粒球标签必然早期正确。

## 放入当前图粒球研究时的定位

三者可作不同路线的背景：SCE 是 robust loss，Co-teaching 是 small-loss cross-selection，ELR 是 temporal regularization。当前项目的研究变量是粒球关系怎样作为 soft confidence/structural prior 影响节点对比关系；若采用上述策略作 baseline，需将数据划分、噪声率、训练轮数、增强和评价协议对齐，不能把图像结果直接横向比较。
