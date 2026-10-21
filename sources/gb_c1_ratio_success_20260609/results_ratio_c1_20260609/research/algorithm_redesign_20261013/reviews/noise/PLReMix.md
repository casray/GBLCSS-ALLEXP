# PLReMix：Combating Noisy Labels with Pseudo-Label Relaxed Contrastive Representation Learning

## 基本信息

- **作者与出处**：Xiaoyu Liu, Beitong Zhou, Zuogong Yue, Cheng Cheng；WACV 2025，pp. 6517–6527。arXiv:2402.17589。
- **原始来源**：[WACV 官方论文页](https://openaccess.thecvf.com/content/WACV2025/html/Liu_PLReMix_Combating_Noisy_Labels_with_Pseudo-Label_Relaxed_Contrastive_Representation_Learning_WACV_2025_paper.html)、[官方 PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Liu_PLReMix_Combating_Noisy_Labels_with_Pseudo-Label_Relaxed_Contrastive_Representation_Learning_WACV_2025_paper.pdf)、[arXiv](https://arxiv.org/abs/2402.17589)、[代码](https://github.com/lxysl/PLReMix)。
- **阅读状态**：已核官方全文中的问题动机、PLR 规则、方法流程和实验设置；未逐项重核所有数据集结果。**相关度**：高（端到端 LNL + contrastive 机制；非图/非粒球）。

## 问题、方法与目标

作者观察到：把现成的 contrastive representation learning (CRL) 损失直接与 noisy-label learning (LNL) 方法端到端相加，部分实验会退化；他们以实验分析优化冲突，并提出 PLReMix。该诊断是论文的**经验结果**，不是一条普遍适用的冲突定理。作者在 CIFAR-10、80% 标签噪声、epoch 30 的梯度余弦分析，是支撑这一现象的例子，不能推广成“所有对比项与 CE 必然冲突”。

框架以 CE warm-up 和 vanilla CRL 起步，双网络交叉选择样本；二维、双成分 GMM 使用两个明确的坐标区分 clean/noisy 样本：分类分支的 CE 值 \(CE(p_i,\tilde y_i)\)，以及表示与类别原型相似度经 softmax 后、相对于观测标签的 prototype-CE 值 \(CE(\operatorname{softmax}(\operatorname{sim}(z_i, C)),\tilde y_i)\)。之后以 PLR 建立可靠负样本集。对样本 i 和 j，只有其预测 top-κ 类别集合互不相交时，PLR 才把 j 保留为 i 的负样本；集合有交集则过滤该负对。早期规则还纳入观测标签，κ 随训练由 3 降到 2 再降到 1（原文切换 epoch 40、70）。该机制的目标是剔除可能语义相近的错误负对，缓解直接相加时观测到的优化干扰。

## 实验、限制与对本项目迁移

论文在多个分类噪声 benchmark 上验证，并报告 PLR 可插入其他 LNL 方法；它并非所有噪声/基线设置都优于 SCL 或 GMM。使用者应回表格逐协议比较，不能概括成普遍 SOTA。

本项目历史 C1 的 `negfree_bn` 设置在 phase 2 使用 CE + intra/inter 项，并未与 InfoNCE 同时优化；当前 `inter_ball_loss` 没有 class argument，是让所有高可靠球两两分离。此设置与 PLReMix 中“同一网络同时叠加分类损失与 vanilla CRL 产生不合适负对”的情形不相同，所以 PLReMix 不能直接解释历史 C1 失败。它能提供的检验思路是：显式定义哪些球对可能语义重叠、是否应从负集合删除；但需与 CGBC 的“同类拉近、异类 margin 推开”区分。PLR 不应被直接等同于无噪声节点图上的球间损失。


