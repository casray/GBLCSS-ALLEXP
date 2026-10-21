# 根号粒度与无标签几何分裂：本轮生成器依据

2026-10-14。用户允许修改粒球生成，优先解决跨数据集数量适配。

## 文献事实

- GBGCL全文p5式4将初始数量设为floor(sqrt(N))，式5的BFS初始球规模也取该经验量；正文将其描述为规模与数量的经验平衡。后续还有质量分裂，不能将经验式解释为任何任务的最优最终球数。其类别同质性质量不直接用于本项目无标签构球。
- Xu、Long、Meng，Clustering by Mining Density Distributions and Splitting Manifold Structure，AAAI2025：[官方全文](https://ojs.aaai.org/index.php/AAAI/article/download/35490/37645)。PDF及hash见../../sources/GBSC_AAAI2025*。该文p3回顾GBSC的中心平均距离/加权子球分布度与最远点分裂，并指出过度依赖紧致度会损失局部结构；论文自身的MDMSC利用密度与流形结构。不是本项目新SSE分裂的直接复现。
- Xia等，An Efficient and Adaptive Granular-Ball Generation Method in Classification Problem，TNNLS35(4)，2024，[DOI](https://doi.org/10.1109/TNNLS.2022.3203381)：公开摘要支持自适应生成和覆盖/重叠控制的研究动机。本轮仅摘要级核查，不声称移植其完整分类算法。

## 当前实验性设计

ceil(sqrt(N))只是明确预算先验；N是全图H的行数。将固定min_child32改为ceil(N/(4B))，以总SSE而非平均半径优先分裂；取消5%平均距离下降要求，接受正SSE改善。两均值约束失败时使用可行投影边界；重复几何不能为凑数硬切。

支持版用训练节点位置规定每球至少4个训练节点，避免原LCSS留一后不足两票。它没有类别输入；构球独立于人工标签翻转。支持上限与几何可行性仍不同，记录请求/上限/实际数量及拒绝原因。

这是预算分裂与投票支持的受控改造，不是已证明最优或完全无参数的新理论。数量增加可能改善类别局部性，也会减少抗噪样本量；应同时报告margin、Brier、干净误修正和准确率，并保留同数量KM/同支持KNN。

实验计划与输出：workspace experiments/adaptive_sqrt_ball_20261014。表示、原LCSS和MLP固定，原生成器作为历史可核对对照。实验结果写入该目录实验结论.md，不能以实现完成推断科研成功。

## 后续实测更新

第一轮支持4五协议目标已完成；第二轮固定支持16仅seed0也完成，均审计通过。16来自独立人工噪声原频率Var≤1/(4s)和预设条件std .13，要求其他票15。该界不保证正确语义多数、不约束q/alpha的非线性风险，也不证明16最优。

实际支持16球数Texas3/Wisconsin5/Roman151/CoraFull141/Coauthor-CS136；小图未强凑根号预算。CoraFull有目标与分类改善信号，小图和CS存在负结果，粒球没有统一超过KM/KNN。完整判读见workspace experiments/adaptive_support16_ball_20261014/综合结论与接续.md。当前使用新模块的支持16参数需显式传入min_train_per_ball=16，模块默认4保留为首轮研究版本，不作为已确认部署默认。
