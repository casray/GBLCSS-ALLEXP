# 粒球与深度学习结合：文献研究图谱

生成时间：2026-06-30

## 2026-10-14 最新增补：GREET与HALF独立验证

用户附件交流引出两篇分别实验后融合的需求。新增[建议与顺序](research/greet_half_review_20261014/00_建议与执行顺序.md)、[原文与代码核查](research/greet_half_review_20261014/01_两篇文献与代码核查.md)、[详细实验设计](research/greet_half_review_20261014/02_独立验证与融合实验设计.md)。GREET全文/作者实现已核，HALF仍有全文及官方实现缺口；本轮无训练。推荐先完整表示适配、再独立自标签可靠性、最后2×2；保留原split和普通近邻、均值收缩等强对照。GREET源码采样与论文/代码差别须在移植时明确。

## 2026-10-14 当前增补：异配节点分类强方法

用户当前要求先查SOTA和算法机制，再讨论实验。新增[强方法与clean GB-LCSS对照](research/heterophily_sota_20261014/00_异配图SOTA与GBLCSS干净对比.md)及[逐方法核读](research/heterophily_sota_20261014/01_强方法与机制核读.md)，含NT/Dir-Poly、GraphTARIF、Polynormer、TunedGNN、TS、ES-MLP、HiGCN、HPGNN、CNA、HarmonyGNNs及几何/稀疏/基础模型前沿边界。本轮为文献核查和已有结果重算，不是新模型实验。

其中HarmonyGNNs是冻结自监督表示+线性探针，ES-MLP是训练用图/推理无边，其余有图方法与GB-LCSS部署条件不同。WebKB、CoraFull不同标签比例，以及F1/accuracy、原始/替换特征、有向/无向结果均不能混排。该增补用于当前选方向；下方原图谱和原论文卡继续保留历史证据，涉及细节须回原文。

## 1. Current Research Goal

本项目关注 Granular Ball Computing 与深度学习，尤其是节点级图表示学习/图对比学习的结合方式。当前最有价值的研究问题不是把粒球当作硬标签，而是把粒球作为局部语义关系单元、置信度信号、soft structural constraint、weak view relation teacher 或 robust local structure prior。

完整逐篇总结位于 `C:\Users\Admin\Desktop\实验8去噪5.19-\文献文件`，其中每个子目录下均已生成 `论文详细总结.md`，根目录生成了 `论文横向对比与研究方向建议.md`。

## 2. High Level Literature Map

| 文献类别 | 核心问题 | 与本项目关系 | 下一步启发 | 限制 |
|---|---|---|---|---|
| 粒球 + 对比学习 | 多视图聚类、粒球模糊表示、对比关系构造 | 与 GRACE 风格图对比最接近 | 设计 soft relation / weak view，而不是硬正样本 | 多数不是节点级图对比，需要重构实验协议 |
| 深度学习 + 粒球 | GNN/GCN/GAE/RVFL/图像/开放意图识别中的粒球接口 | 最适合形成论文主线 | 图粗化、图构造、采样、边界、置信度 | 一些方法依赖监督纯度或预处理，存在标签泄漏风险 |
| 粒球基础生成/分类/聚类 | 自适应粒度、中心半径、纯度、密度与邻域关系 | 提供机制背景和可解释性 | 把中心、半径、密度转成 confidence | 多数偏传统机器学习，不能直接迁移到自监督 GCL |
| DPC/密度峰 | 局部密度、代表点、峰值发现 | 可作为局部可靠性和邻域筛选思想 | 用密度关系辅助边权重或 ball-ball relation | 高维嵌入空间中距离度量敏感 |
| 椭球/子空间粒 | 非球形局部几何和各向异性结构 | 可作为后续增强 | 在 embedding 空间中探索椭球粒球 | 实现复杂，第一版主线不宜优先 |
| 噪声学习/异常检测 | 标签噪声、样本分离、异常/离群处理 | 支持鲁棒 GCL 叙事 | 用粒球质量降权不可靠关系 | 多为监督噪声过滤，不可直接用标签纯度 |

## 3. How Granular Ball Can Connect to Contrastive Learning

| 接入位置 | 粒球角色 | 可能解决的问题 | 风险 | 优先级 |
|---|---|---|---|---|
| 数据增强/视图构造 | weak ball graph 或 node-ball bipartite view | 随机增强不稳定、边噪声 | 视图过强导致结构坍缩 | 高 |
| loss weighting | 半径、密度、成员距离、模糊隶属作为 confidence | 降低不可靠正负关系影响 | 权重调参化 | 高 |
| encoder hidden space | 在中间表示生成粒球 | 比 raw feature 更贴近语义 | 动态更新成本高 | 高 |
| projector 前后 | 比较表示空间对粒球稳定性的影响 | 识别最佳粒球生成空间 | projector 后语义可能过拟合 | 中 |
| positive construction | same-ball soft positive | 提供局部语义一致性 | hard same-ball positive 风险极高 | 中，必须 soft |
| negative filtering | 降权疑似假负样本 | 缓解 false negative | 单独做机制弱，已有实验不优先 | 低 |
| prototype/cluster | ball center / prototype | 多粒度语义摘要 | 过强 prototype pulling | 中低 |
| graph denoising | ball-ball relation 重估边 | 稳定拓扑视图 | 可能破坏原图信息 | 高 |
| scalability | 粒球采样/图粗化 | 降低 GCL 训练成本 | node-level 信息损失 | 中 |

## 4. Paper Notes

逐篇 paper notes 已拆分到各文献子目录的 `论文详细总结.md`。每篇包含任务、数据/数据集、粒球角色、深度模型接口、创新点、算法核心流程、迁移启发和风险。

本轮覆盖 145 个文件，其中 141 个已有可抽取文本，4 个不可完整读取：2 个 CAJ 文件和 2 个 `.pdf.downloading` 未完成下载文件。不可读条目已在文献总览中标注，不能作为论文 claim 依据。

## 5. Literature to Experiment Mapping

| 可执行方向 | 支撑文献类型 | 实验转化 |
|---|---|---|
| Soft relation | 粒球 + 对比学习、多视图粒球聚类、MGNR | node-ball / ball-ball soft weights |
| Confidence | 模糊粒球、W-GBC、噪声过滤、采样论文 | contrastive loss confidence weight |
| Graph denoising | GBGC、GNN graph coarsening、GB random walk anomaly | ball graph 作为弱拓扑视图 |
| Deep representation | GAE/GCN/RVFL/图像粒球表示 | 在 hidden embedding 中生成粒球 |
| Robust training | 标签噪声、异常检测、open intent | 降权不可靠关系或噪声边 |
| Multi-granularity | GBG++、Ball k-Means、多粒度融合 | 多尺度 ball relation 消融 |
| Warnings | 监督纯度分类、标签噪声过滤 | 不直接用标签纯度，不硬化 same-ball |

## 6. Recommended Research Directions

1. 粒球弱视图关系教师  
   核心假设：稳定表示空间中的粒球关系能提供比随机增强更鲁棒的局部语义约束。粒球只作为 soft teacher。最小实验是在 GRACE baseline 上比较 raw feature、encoder hidden、projector 前后的粒球关系权重。

2. 粒球置信度加权图对比损失  
   核心假设：半径、密度、成员距离和模糊隶属可估计局部结构可靠性。最小实验只改 loss weight，不改 split、evaluation 或 baseline protocol。

3. 粒球图构造/拓扑去噪视图  
   核心假设：ball graph 或 node-ball bipartite graph 可作为第三弱视图，缓解原图边噪声。必须避免替换 baseline evaluation。

4. 粒球采样/图粗化可扩展 GCL  
   核心假设：粒球粗化降低训练成本且保持节点表示质量。适合后续扩展，不建议作为第一主线，除非加入大图数据和效率指标。

5. 椭球/子空间粒球嵌入约束  
   核心假设：embedding 中局部结构可能各向异性。暂缓，适合作为第二阶段增强。

## 7. Directions to Avoid or Deprioritize

- 不要把 same-ball membership 直接当二值正样本标签。
- 不要把监督纯度生成的粒球直接迁移到自监督 GCL。
- 不要只做负样本过滤。
- 不要改变 train/validation/test split 或 baseline evaluation。
- 不要用单数据集、单 seed 或小于 std 的提升作为论文结论。
- 不要第一版就堆椭球、多子空间、量子核等复杂机制。

## 8. Open Questions

- 哪个空间生成粒球最稳定：raw feature、encoder hidden、projector 前、projector 后？
- 粒球关系是帮助 clean graph 更多，还是帮助 noisy graph 更多？
- node-ball relation 与 ball-ball relation 哪个更适合做 weak teacher？
- 粒球置信度应基于半径、密度、成员距离、模糊隶属，还是组合指标？
- 同球成员是否真的代表语义相似，还是只在部分数据集成立？

## 9. Key Takeaways for Future Experiments

五个重要启发：

- GRACE 类方法提供节点级图对比基线，粒球最好接入 view relation 或 loss weighting。
- GBGC/GNN 粗化论文说明粒球能进入深度图学习，但监督纯度必须规避。
- 模糊粒球和加权粒球文献支持把粒球作为 confidence，而不是 hard label。
- DPC/MGNR 提供密度和邻域可靠性思想，适合作为边权重或局部结构筛选。
- 噪声学习文献支持鲁棒性叙事，但不能照搬监督标签过滤。

五个必须避免的错误：

- 把同球关系硬化为正样本。
- 用标签纯度或测试信息生成粒球。
- 只在一个数据集上解释提升。
- 忽略 std，把波动写成显著改进。
- 用复杂机制掩盖缺少清晰消融的问题。

## 10. 精读论文总结索引

本轮进一步精读了与当前论文方向最相关的 16 篇/项核心文献，已生成：

- `C:\Users\Admin\Desktop\实验8去噪5.19-\文献文件\精读论文相关总结.md`

该文件重点覆盖：

- GRACE 与无增强简化 GCL 的 baseline 约束。
- 粒球 + 对比学习中的 MGBCC、GFSAF、CGBC、GBmooer。
- 粒球 + 深度图学习中的 GBGC、SGBGC、AW-GBGAE、GBCM-GCN、PGBSF、SCGNN。
- 可迁移辅助机制中的 MOGB、MGNR、Graph-based Image、GBRIP。

精读后的执行建议是：先做 GB confidence weighted GRACE，再做 node-ball / ball-ball soft relation，最后再做 GB graph weak view。仍然不建议从 hard same-ball positive、监督纯度粒球或单纯负样本过滤开始。

## 2026-10-14执行状态与OFA引用补充

上文为早期文献阶段建议，不可覆盖后续负结果。最新机制实验和三异配图原算法基线已经完成，见research/algorithm_redesign_20261013/06_OFA引用核查与先基线后改进.md及99_研究记忆与接续.md。ESS×JS可靠性收缩未通过推进门槛；原GB64在Texas/Wisconsin有带噪增量和干净退化并存，Roman Pair有五split平均+1.09pp，Uniform基本无增量。

GBGCL的OFA来自HeterGCL [42]，其属性MLP证据[64]来自HOG-GCN，CCA目标[65]来自CCA-SSG。三张核读卡在上述research目录reviews/core；尚未测试OFA。当前用户偏好是先直接测试原算法，再固定其他环节逐层改动，避免把多模块整合当作先验成功方案。

## 2026-10-14生成器改造与文献证据边界

用户随后授权先修改生成器。核读卡research/algorithm_redesign_20261013/reviews/core/granular_generation_sqrt_and_geometry.md记录GBGCL的sqrt初始经验、AAAI2025密度/流形方法及Xia2024摘要级证据。新SSE预算分裂不是这些完整算法的复现。

两轮已完成并审计，见experiments/adaptive_support16_ball_20261014/综合结论与接续.md。支持4五协议目标数量14/16/151/141/136达标，但只有Roman/CoraFull过带噪Brier筛查，小图学生退化；固定支持16单seed部分修复小图，CoraFull相对旧版三条件+3.10/+2.66/+1.02pp，其余图无统一收益。普通KM/KNN仍不能忽略。下一阶段先定位证据质量，不能把sqrt经验或支持数波动界写成最优分球理论，也不能主张新图对比算法已经验证。
