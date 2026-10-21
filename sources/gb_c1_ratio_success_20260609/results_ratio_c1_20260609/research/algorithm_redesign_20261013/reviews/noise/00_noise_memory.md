# 噪声学习文献记忆：图对比、粒球与本项目边界

## 直接相关先例与 novelty 边界

1. **图节点标签噪声 + 对比学习已有先例**：CR-GNN（Neural Networks 2024）使用增强视图对比、同质性邻居对比及动态 CE；NR-GNN（Lu et al., Information Sciences 2024）用半监督对比约束推远噪声标注节点与未标注邻居；它不同于 NRGNN（Dai et al., KDD 2021），后者用特征相似连接未标注与可信标注节点并产生伪标签。卡片/报告应按标题、年份和 DOI/代码区分二者；RNCGLN（AAAI 2024）联合图噪声和标签噪声，以图对比局部学习、全局 self-attention、伪图与伪标签自改进。因此不能声称“首次在噪声节点分类中使用图对比学习”或“首次联合处理图噪声和标签噪声”。
2. **图结构 + 标签去噪先例**：ProCon（IJCAI 2025）以特征相似及邻接选 peer、原型伪标签一致性和 GMM 选 clean 节点；它不是以图对比损失为核心，但削弱“图拓扑与原型/peer 信息首次用于图标签噪声”的宽泛主张。
3. **粒球 + 对比/噪声相关候选不止一个**：AGPCS-OIC（PRAI 2025）目前只找到二手索引题录/摘要转载，机制待一手确认；转载线索提到球中心作 anchor 的开放意图对比机制，若后续官方全文证实，会构成相关先例，但当前不把它当作已确认技术事实。G-GBC（KBS 2026）和 BNO-GB（PR 2026）是分类/开放意图标签噪声中的粒球方法；GBRAIN（ICMR 2024）题名指向文本标签噪声鲁棒训练，但当前仅核题录/代码入口，算法仍待原文。GBC-GL 是图粗化/图表示学习，不等同于图标签噪声或图对比鲁棒性。MOGB 与 AGPCS-OIC 是不同论文，不能混写。
4. **SupCon/噪声对比的依据**：SupCon 以观测标签定义 positives，因此标签错误会污染正对。SNSCL 显式处理 noisy anchors 和 queue keys；SymNCE 给出在特定对称标签噪声及类别平衡条件下的风险理论；PLReMix 根据预测 top-κ 集合重叠过滤可能错误的负对。PLReMix 对“朴素同时加 CE+CRL 会造成优化冲突”的证据是经验分析，不是普适冲突定理；其视觉分类实证不能自动解释图节点粒球实验。
5. **概念勿混**：Rank-N-Contrast 与 GBSVR 面向连续回归目标/排序，不是分类标签噪声方法；RNCGLN 是图与标签噪声节点分类方法，不是回归 Rank-N-Contrast。

## 对当前 C1 方案的具体约束

当前研究偏好把粒球用于局部语义关系单元、置信度信号、soft structural constraint 或 weak relation teacher。不要把同球 membership 直接二值化为正样本，也不要只做负样本过滤或强 prototype pulling；球/关系的可信度应通过消融、噪声敏感性和跨数据集结果支持。

已知实验边界：历史 C1 的 `negfree_bn` 在 phase 2 采用 CE + intra/inter 项，不是 CE 与 InfoNCE 同时优化；因此 PLReMix 的优化冲突观察不能直接解释该负结果。当前 `inter_ball_loss` 没有 class argument，作用是让全部高可靠球两两分离；这与 CGBC 的同类收缩、异类 margin 推开不同，语义风险需独立判断。若作新比较，应对齐数据划分、噪声协议、baseline、训练过程及 seed，并报告 mean/std。

## 关键文献卡片索引

- 直接图噪声对比：`CR-GNN.md`、`NR-GNN.md`、`RNCGLN.md`。
- 图 peer/原型去噪：`ProCon.md`。
- 粒球与噪声/对比：`AGPCS-OIC.md`、`BNO-GB.md`、`G-GBC.md`、`GB_RAIN.md`。
- 对比噪声机制：`SupCon.md`、`SNSCL.md`、`PLReMix.md`、`SymNCE.md`。
- 一般 noisy-label foundations：`foundations.md`。
- 非分类噪声/邻近参考：`Rank-N-Contrast.md`、`GBSVR.md`、`GBS.md`、`VPGB.md`、`GBC-GL.md`。

## 证据状态提醒

卡片中的“阅读状态”说明实际核验深度，“相关度”说明与当前研究的主题接近程度，二者不可互相替代。全文不可得的 AGPCS-OIC、NR-GNN、CR-GNN 等条目，不应引用未核公式、逐表结果或未见的局限；GBRAIN 当前尤其不能据题名推断其方法。任何作者报告的性能结论都只在原论文的数据、噪声构造和训练协议内成立。



