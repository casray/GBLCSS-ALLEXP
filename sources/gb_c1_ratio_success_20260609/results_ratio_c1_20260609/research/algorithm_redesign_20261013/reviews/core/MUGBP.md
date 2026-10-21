# MUGBP：Momentum-Updated Granular-Ball Prototypes: Stabilizing Cross-Modal Alignment for Multimodal Intent Recognition

## 文献信息与阅读层级
作者 Hao Leng, Xiaoliang Chen, Duoqian Miao, Hongyun Zhang, Xiaolin Qin, LinLin Yuan, Peng Lu。Information Sciences，DOI 10.1016/j.ins.2026.124207；2026-09-23 online，In Press/Journal Pre-proof。**阅读层级：官方长摘要/引言预览 + 官方公开代码 README 级，非全文级**。ScienceDirect 提供较长引言/贡献预览；GitHub README 描述实现架构。[ScienceDirect官方页面](https://www.sciencedirect.com/science/article/pii/S0020025526011382)；[代码库](https://github.com/lenghao200/MUGBP)；[DOI](https://doi.org/10.1016/j.ins.2026.124207)。

## 任务与方法
多模态意图识别（MIntRec、MIntRec2.0），输入文本、视频、音频，目标是跨模态分类和对齐，不是图任务。官方预览提出以文本语义为锚，为类别建立多个中心-半径局部区域；Adaptive Granular-Ball Splitting 拆分不纯区域；Online Momentum Update 在小批次中用 EMA 更新球中心与半径；球边界损失只惩罚落在同类球区域之外的视觉、声学和融合表征，而非将所有特征拉向单个中心。代码 README 给出的总目标为 L_total=L_cls+L_cons+λ_gb L_gb，且有 InfoNCE/SupCon 选项；此公式在本笔记中标为**由公开代码 README 核对**，待与论文公式逐项对齐。

实现使用 class-aware 球：README 记载 purity threshold、同类标签筛选与类别条件文本锚。主结果、跨数据集、长尾和零样本实验见论文第 6 节；本轮未逐表核录数值，故不报告具体百分点。

## 对噪声节点分类的迁移
边界区域、中心与半径共同描述类内变化、EMA 跟踪表征漂移，可能启发“软局部结构约束而非强中心拉近”。但它的锚明确依赖真值意图 token/类别标签；直接用噪声节点标签构建类条件球会将错误类别结构固化。任务不是图，标签鲁棒性不是该论文的证据。建议只抽取动态局部区域这一抽象思想，不移植类别监督、跨模态结构或多模态模型组件。

## 风险与待核实
全文和官方补充材料未获得。具体公式、纯度阈值/数据拆分、seed、长尾/零样本实验、结果表统计、标签可靠性假设需核对正文。代码 README 的总损失/实现机制需与论文版本对应；不可将它写成论文公式的已核实转录。


