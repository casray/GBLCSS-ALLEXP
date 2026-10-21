# CCA-SSG：OFA的理论与损失来源

核读：2026-10-14。Hengrui Zhang、Qitian Wu、Junchi Yan、David Wipf、Philip S. Yu；From Canonical Correlation Analysis to Self-supervised Graph Neural Networks；NeurIPS 2021，76–89。

来源：[官方全文](https://proceedings.neurips.cc/paper_files/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf)；本地PDF/提取文本../../sources/CCA_SSG_NeurIPS2021*，哈希见../../sources/OFA_reference_sources.json。

## 机制及理论边界

式4先对表示每维跨节点中心化，并缩放为标准差1/√N；式5约束两视图一致及各视图维度去相关，不依赖负样本。原编码器为GNN，并非属性MLP；HeterGCL将相关目标用于MLP。

第4节的互信息解释以表示边缘/条件分布高斯假设为前提。下游充分性还依赖任务相关信息在增强后保留。不能据此保证任意特征掩蔽、任意图或带噪分类目标都会改善。

## 对本项目的判断

实现应核归一化、损失尺度和掩蔽方式，不能直接将未归一化H的Gram与I相减称为同一方法。小图还需注意表示维度与矩阵秩。无负样本可减少负对构造问题，但不自动排除联合分类优化冲突。先作独立预训练/冻结分球消融更容易解释。

状态：原损失和信息论假设已核；OFA在本项目的效果尚未验证。
