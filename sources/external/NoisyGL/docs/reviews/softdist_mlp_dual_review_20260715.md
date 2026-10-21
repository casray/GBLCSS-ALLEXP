# SoftDist-MLP / LCSS 双视角审稿报告

日期：2026-07-15  
审稿对象：`docs/paper_final_zh_v12_residual_fixes_20260712.md`  
审稿框架：`peer-review`（期刊同行评审）+ `academic-paper-review`（CS/ML 方法评审）

## 1. 总体结论

**当前建议：Major Revision / 暂不投稿。**

稿件的优势是真实且可复核的：问题设定有现实意义；train-only 噪声协议、数值表和数据泄漏已经经过较严格的内部审计；随机化分组、kNN/kmeans 平替、失败数据集和 CoraFull 的负归因都没有被隐藏。这种证据纪律明显强于一般实验型稿件。

但当前仍有四个投稿级问题：

1. `24/25` 的核心胜率建立在按性能闸门筛选并扩展的数据集上，属于条件于结果的描述，不是无偏的总体成功率。
2. LCSS 使用完整图和 GRACE-H 构造监督，而主表鲁棒 MLP 基线只使用 X 与 noisy labels；已有 `[X||H]+Small-loss/JoCoR` 对照明显强于 train-only LCSS，但正文没有呈现这些数字。
3. 统计口径不足：5 个种子上做 339 次 paired t-test、无多重比较校正，并以“小于一倍标准差”定义相当，不能支持显著性或等效性结论。
4. 方法声称构造“去噪软监督”，但没有直接证明 target 质量提高；当前实现的球内投票还包含节点自己的 noisy label，缺少 leave-one-out 与普通 label smoothing 对照。

在修正这些问题后，稿件有希望成为一篇**范围明确、实验审计扎实的 KBS 类论文**。以当前算法新颖性与证据口径，Neural Networks 或 Pattern Recognition 风险明显更高。

---

## 2. Review A：期刊同行评审

### 2.1 工作概述

本文研究训练阶段可访问完整图、推理阶段只能使用节点特征，同时训练节点标签含噪的 transductive 节点分类问题。方法先以 label-free GRACE 获得冻结表示 H，再在 H 空间递归二分形成局部分组，由组内 noisy train labels 构造半硬 soft targets，最后训练 X-only MLP。扩展版本使用 GCN[X||H] 共识教师，将全节点预测蒸馏给 X-only MLP。

### 2.2 主要优点

1. **问题设定成立。** graph-free inference 与 training-label noise 的交叉问题具有清晰部署意义。
2. **协议卫生较好。** 正式行均为 train-only noise，验证和测试标签未污染，paired methods 的 split/noise hash 一致。
3. **负结果披露充分。** PubMed、Amazon-Computers、Attributed-Wiki 等边界数据集，以及球式分组被 kNN/kmeans 平替、CoraFull 教师归因失败，均在正文出现。
4. **归因意识强。** shuffled grouping、H-only、kNN/kmeans、学生容量、表征空间等对照比常见“只报主表”完整。
5. **部署端确实 X-only。** 主方法和蒸馏学生推理时不读取 A、H 或分组结构。

### 2.3 重大问题

#### Major 1：核心胜率存在结果驱动的数据集选择

稿件在 Highlights、摘要、贡献和结论中反复强调“24/25 设置最高均值”。但历史执行计划明确规定：先在 uniform 0.4 上做性能 gate，只有通过者才扩展完整噪声谱并进入主表。例如：

- `execution_plan_mlp21_dataset_replacement_20260710.md` 第 36–58 行规定，只有通过 LCSS-vs-best-baseline 门槛的数据集才补其余四种噪声设置。
- 同文档第 81 行明确写“Passed datasets can enter the main table”。
- Amazon-Photo 与 Coauthor-CS 的阶段报告也使用“过门并进入主表”的规则。

因此 `24/25` 是在成功数据集子集上的条件胜率。正文表 2 报告失败数据集是必要的，但不能把条件胜率转换成总体成功率。

**必须修改：**

- 将 `24/25` 从 Highlights 和摘要的中心位置移除，或明确写成“在五个经开发阶段选定并完成全谱的数据集上”。
- 将全部 11 个数据集的 uniform 0.4 结果作为首要广度结论。诚实口径是：真实分组相对 shuffled 在全部已测数据集上有贡献，但相对最强鲁棒策略只在部分数据集形成额外分类增益。
- 若仍需要无偏的 full-spectrum claim，应在方法和阈值全部冻结后，预先指定新的 holdout 数据集并完整运行，不能再用 seed0/performance gate 决定是否报告。

#### Major 2：主表不是同训练资源比较，且遗漏了关键 H-access 对照

LCSS 的学生虽然只输入 X，但训练 target 来自完整图上的 GRACE-H；CE、Small-loss、JoCoR 等主表方法只使用 X 和 noisy labels。因此“同骨干”成立，“同信息资源”不成立。

已有 MLP-6 公平性实验给出关键结果：

| 数据集 | LCSS | Small-loss [X||H] | JoCoR [X||H] |
|---|---:|---:|---:|
| Cora | 55.59 | 57.70 | 60.13 |
| CiteSeer | 53.83 | 56.30 | 58.60 |
| PubMed | 71.90 | 84.83 | 84.79 |

这些方法推理时需要 H，不属于相同部署类，因此不必放入主表 A；但它们必须出现在资源/精度权衡表或附录。当前正文只突出 `[X||H]+CE` 在小图上失败，会使读者误以为直接使用 H 普遍无效，这与已有鲁棒 H-input 结果不符。

**必须修改：**

- 每张比较表增加 `train graph / train H / unlabeled nodes / teacher / inference graph/H` 资源列。
- 在正文或附录恢复 MLP-6 H-access 结果，并明确：LCSS 的优势是部署 profile 与监督转换方式，不是对所有能访问 H 的分类器都更准。
- 将主表 claim 收窄为“在 X-only student、train-target-only 策略族中”，不要写成总体 graph-free SOTA。

#### Major 3：统计分析不能支持当前措辞

附录 B 对 339 组比较进行 paired t-test；n=5 时正态性无法可靠检查，Wilcoxon 两侧检验最小 p 值为 0.0625。当前又没有 family-wise/FDR 校正。`302/339` 个未校正 t-test p<0.05 不能当作稳健性证据，因为比较数量本身非常大。

“差异小于一倍标准差视为相当”也不是等效性检验。标准差与预设的实际等效边界不是同一概念。

**必须修改：**

- 预声明少量 primary comparisons，例如每个 dataset/noise setting 只比较 LCSS 与预先指定的 strongest baseline、CE、shuffled。
- 对 primary family 使用 Holm 校正，并报告 paired mean delta、95% CI 和 5/5 方向数。
- 对关键 Cora/CiteSeer/CoraFull 结果扩到至少 10 个 seeds，或明确所有统计仅为探索性。
- 若要写“相当”，预先定义实际等效区间（如 ±0.5pp 或 ±1pp），使用 TOST/ROPE；否则改为“数值接近，证据不足以区分”。

#### Major 4：缺少对“去噪 target”本身的直接验证

shuffled control 证明真实分组比随机分组更有用，但不能证明 soft target 更接近 clean label。方法公式和图 G.3 都在讲“噪声被修正”，正文却主动承认现有证据不是逐节点纠错证明。

此外，`utils/frozen_grace_targets.py` 的 `apply_ball_soft_targets()` 直接用球内全部训练标签计数，目标节点自己的 noisy label 同时进入 `p_j`，又作为外层 one-hot 参与混合。对约 10 个 vote 的小球，自包含影响不可忽略。

**必须补的最小消融：**

1. leave-one-out LCSS：为节点 v 构造 `p_{j\setminus v}`，避免自身标签参与纠正自身。
2. CE + uniform label smoothing 和 CE + empirical-class-prior smoothing。
3. target-quality 表：clean-class probability mass gain、Brier/KL 改善、noisy-node correction rate、clean-node damage rate、coverage；clean label 只用于事后诊断。
4. real/shuffled/kNN 的 target-quality 对照，而不只比较最终 accuracy。

若 LOO 与当前实现相当，可证明自包含不是主要收益来源；若明显下降，论文必须把机制改写为 group-aware smoothing，而不是独立邻域纠错。

#### Major 5：完整管线的不确定性和复现性不足

五个下游 seeds 共用同一份 H，因此误差条不包含 GRACE 预训练随机性。CoraFull、Coauthor-CS、Amazon-Photo 只有 H tensor、metadata 和 loss，没有可精确再生 H 的 encoder checkpoint。当前数值可追溯，但 clean checkout 不能完整重建全部管线。

**建议：**

- 至少在 Cora、CiteSeer、CoraFull 上使用 3 个 GRACE seeds，每个 H 配 3 个下游 seeds，采用分层方差报告。
- 保存并发布 encoder/projector checkpoint、预训练 config、数据哈希、H SHA256。
- 若算力不允许，所有“稳定”结论都应改成“conditional on a fixed frozen representation”。

#### Major 6：部署卖点缺少部署实验，且 transductive 边界需要更清楚

稿件用隐私、延迟和在线图访问作为核心动机，但没有推理 latency、吞吐、峰值显存/内存或邻居抓取成本表。当前只能证明“代码路径不使用图”，还没有量化部署收益。

另外，GRACE 在完整图和全部测试节点上预训练，当前实验是 transductive。它不能直接证明对训练阶段从未出现的新节点有效。

**建议：**

- 报告 CPU/GPU full-graph 与 single-node/batch inference latency、peak memory、model artifact size，对比 GCN、NRGNN、GLNN-style 和 LCSS student。
- 将现实场景写成“已知节点宇宙的离线图训练 + 无图在线 serving”，不要暗示已验证 unseen-node inductive deployment。
- 隐私措辞限于“推理服务不访问边”，不能写成隐私保护算法。

#### Major 7：算法新颖性有限，需调整论文定位

作者自己的实验表明：kNN/kmeans 可以平替球式分组，kNN 在若干设置略高；蒸馏扩展中 CoraFull 的共识 teacher 低于 CE、kNN 和 shuffled teacher。由此，技术贡献不是新的分组算法，也不是普遍更优的 teacher，而是：

> 一个把 label-free graph representation、局部标签聚合和 X-only student 组合到 noisy-label graph-free setting 的经验配方与边界研究。

这个贡献对 KBS/应用型期刊可以成立，但对强调算法创新的 Neural Networks 或 Pattern Recognition 偏弱。引言中的“第四条通道”应作为组织框架，而不是宣称此前不存在监督侧结构利用。

#### Major 8：只有合成噪声，现实动机尚未被直接验证

uniform、pair 和 instance-dependent 仍是合成污染。稿件用人工、众包、规则和 LLM 标注错误作为动机，但没有真实噪声标签或模型生成标签实验。干净验证集也比真实部署更理想化。

**建议：**

- 增加至少一个真实/弱标注噪声场景，例如规则标签、LLM 标签或多标注者分歧；若没有公开图基准，明确把这项作为关键限制。
- 增加 noisy-validation 或 validation-size sensitivity，说明方法选择对 10% clean validation 的依赖。

### 2.4 次要问题

1. 第 5.3 节把 H-only linear/MLP 称为“上界”，随后蒸馏学生又超过该“上界”；它不是数学上界，应改为 `direct-H reference` 或 `H-access baseline`。
2. 相关工作第 2.3 节写“Wu 等 [32]”，但参考文献 [32] 第一作者为 Yu，属于明确引用错误。
3. 第 5.4 节“监督信号会抹平类内局部簇”以及 `[X||H]+CE` “被噪声记忆吞噬”仍是机制猜测，应改为关联性解释，并列出其他可能原因（维度失衡、优化、正则不足）。
4. pair/instance 下 small-loss 双峰减弱的解释没有直接测量，应标为 hypothesis 或增加 loss-distribution 指标。
5. 主文表 1 只显示“最强 baseline”，建议至少在主文提供完整 uniform 0.4 方法表；否则读者无法观察方法族行为。
6. 增加标准数据集统计表：节点、边、特征、类别、同质性、训练节点数。
7. 明确哪些 baseline 使用已知合成噪声率设置 forget rate；这对真实未知噪声的可迁移性很重要。
8. β=0.5 来自初始 test-informed 探索，而非统一 validation 最优。现有措辞已经比旧稿诚实，但应将开发数据集与冻结后 holdout 数据集明确分开。
9. Elsevier Highlights 通常要求 3–5 条极短 bullet；当前每条过长，应压缩。
10. Fig. 1 的总体逻辑清晰，但 panel a–d 文本较拥挤；“Deployable model”图形看起来像小图网络，容易误解为仍有连接，建议画成标准两层 MLP。
11. Appendix G.2 中二维投影圆大量重叠，需在图内而非仅 caption 说明：高维 assignment 是互斥 partition，圆只是投影半径，不代表真实重叠球。

---

## 3. Review B：CS/ML 方法评审

### 3.1 贡献与定位

**可成立的贡献：**

- 把 graph-free inference 与 noisy training labels 作为统一任务研究。
- 提出简单、非迭代的 local-consensus target recipe。
- 系统展示真实分组、共识票数、表示局部可聚类性和学生容量的作用边界。
- 给出可选 denoised-teacher distillation extension。

**不能成立的贡献：**

- 粒球本身优于 kNN/kmeans。
- LCSS 是总体 graph-free SOTA。
- `24/25` 是对任意数据集的无偏成功率。
- 结构噪声鲁棒或隐私保护是学习得到的能力。
- H-only classifier 是上界。

### 3.2 评分

| 维度 | 评分（1–5） | 理由 |
|---|---:|---|
| 问题重要性 | 4.0 | 部署无图与标签噪声的交叉问题真实且被低估 |
| 技术新颖性 | 2.5 | 组件均已有，分组可平替；新意主要在问题组合和证据体系 |
| 方法可靠性 | 3.0 | 无泄漏、实现可复核，但 target 自包含与机制归因未闭环 |
| 实验广度 | 4.0 | 数据集、噪声谱、消融和边界都很丰富 |
| 实验设计 | 2.5 | 性能 gate 后扩全谱，且训练信息资源不对齐 |
| 统计有效性 | 2.0 | n=5、339 次未校正 t-test、无正式等效性分析 |
| 可复现性 | 2.5 | seed-level 资产完整，但部分 H 缺 encoder checkpoint，单一 H 低估方差 |
| 表达清晰度 | 3.5 | 边界披露好，但主线过长，部分因果解释和“上界”措辞不准 |
| 可扩展性证据 | 3.0 | 覆盖 34k 节点，但缺运行时间和更大图 |
| 总体 | 3.0 | 有价值的 scope-limited 工作，需重大修订后再投 |

审稿人置信度：**4/5**。

### 3.3 文献定位更新

当前 related work 需要补至少三类近期工作：

- [TINED, ICML 2025](https://proceedings.mlr.press/v267/zhou25a.html)：更强的 GNN-to-MLP 层级蒸馏，虽非标签噪声方法，但会抬高 graph-free distillation 对照的预期。
- [GNN-CFGD, Neural Networks 2025](https://doi.org/10.1016/j.neunet.2025.107338)：直接处理图标签噪声与标签稀缺，推理依赖图，适合强化部署边界切割。
- [SAMTD, 2026](https://www.sciencedirect.com/science/article/pii/S0306457326002682)：直接研究 noisy-node classification 中的多教师结构蒸馏。它未必满足严格 X-only 推理，但与本文第 6 节的 denoised teacher 定位高度邻近，必须讨论。

这三篇不必然推翻安全 gap，但会使“核验 13 篇、训练标签噪声从未进入该文献线”的绝对措辞过时。建议改为：

> To our knowledge, prior graph-free GNN-to-MLP studies have not systematically evaluated training-label noise as the primary problem under multi-noise protocols and noisy-label baselines; recent noisy-node distillation methods should be explicitly distinguished by whether their students remain graph-dependent at inference.

并在补充材料给出检索日期、数据库、关键词和纳入排除标准。

---

## 4. 两套评审的共同判定

### P0：投稿前必须处理

1. 重写 `24/25` headline，明确数据集经过性能 gate，或补真正冻结后的 holdout full-spectrum。
2. 恢复并披露 `[X||H]+Small-loss/JoCoR` 结果，建立训练/推理资源分层表。
3. 重做统计口径：primary comparisons、Holm、CI；删除“一倍 std = 相当”。
4. 增加 LOO target、普通 label smoothing 和 target-quality 诊断。
5. 将“H-only 上界”改为 H-access reference；修正 [32] 作者错误。
6. 补 GRACE seeds 或把稳定性 claim 限定为 fixed-H conditional results。

### P1：强烈建议

1. 增加推理 latency、吞吐、内存和模型大小表。
2. 增加数据集统计表和完整 uniform 0.4 主表。
3. 更新 TINED、GNN-CFGD、SAMTD 等 2025–2026 相关工作。
4. 加真实/弱标注噪声或 noisy-validation sensitivity。
5. 将因果解释统一降级为“与结果一致的解释”，除非有对应诊断。

### P2：写作与图表

1. 摘要压缩约 30%，只留任务、方法、核心结果、边界。
2. 将蒸馏扩展明确标为 optional extension；若篇幅紧，主文只保留一张表，其余放附录。
3. 重画 Fig. 1 的 MLP 图标，减少文字叠放；G.2 强调投影圆不是高维重叠球。
4. 将失败数据集表命名为 `breadth and applicability audit`，保持现在的诚实披露。

---

## 5. 投稿适配判断

| 期刊 | 当前建议 | 修订后前景 |
|---|---|---|
| Knowledge-Based Systems | Major revision / weak reject | 最匹配；问题设定、知识迁移和系统实验可成立 |
| Neural Networks | Reject in current form | 需要更强算法创新、真实噪声、完整统计与近期强基线 |
| Pattern Recognition | Reject in current form | 方法新颖性和领域适配不足，除非扩展为更一般的 noisy structured prediction 方法 |

**最稳的论文定位：**

> LCSS is a simple supervision-side recipe for training graph-free node classifiers from noisy labels. Its contribution lies in the deployment setting, controlled evidence, and applicability characterization rather than in a uniquely superior grouping algorithm.

这比“粒球方法击败现有方法”更准确，也更能承受审稿追问。

---

## 6. 审稿人最可能追问的十个问题

1. 为什么完整噪声谱只扩展通过 uniform 0.4 gate 的数据集？
2. 给 Small-loss/JoCoR 同样的 H 后会怎样？
3. 为什么节点自己的 noisy label 可以参与纠正自己？LOO 结果是什么？
4. 相比普通 label smoothing，局部共识到底增加了什么？
5. soft target 是否真的更接近 clean label？
6. 结果对 GRACE seed 是否稳定？
7. graph-free 部署到底节省多少 latency 和 memory？
8. transductive full-graph pretraining 如何对应在线新节点场景？
9. 为什么使用 339 次未校正 t-test，如何定义“相当”？
10. 在 SAMTD、GNN-CFGD、TINED 等近期工作出现后，first claim 还剩什么？

只要这十问能在正文或附录中被直接回答，稿件的可投稿性会明显提高。

---

## 7. 2026-07-15/16 修订处置记录

修订稿：`docs/paper_final_zh_v13_review_defense_20260715.md`。

| 审稿问题 | 处置 | 状态 |
|---|---|---|
| 24/25 条件胜率 | 摘要、Highlights、数据集、结果与结论均明确：先做 11 数据集 u0.4 筛选，24/25 只对应开发筛选后完成全谱的 5 数据集 | 已完成 |
| H-access 公平性 | 恢复 H-only、`[X||H]+CE/Small-loss/JoCoR`，新增资源分层表；claim 收窄到严格 X-only 类 | 已完成 |
| 统计口径 | 主比较限定为 LCSS vs CE/shuffled/JoCoR；生成逐设置 paired delta、95% CI、方向计数和 family-wise Holm；删除“一倍 std=相当” | 已完成 |
| 自票与普通平滑 | 六数据集 uniform0.4 五种子 LOO、LOO-shuffled、uniform/prior smoothing；新增 Brier/NCR/CDR 诊断 | 已完成 |
| LOO 终判 | LOO-current 平均 +0.60pp，4/6 均值非负；LOO 胜两种平滑及 LOO-shuffled 均为 6/6。作为机制消融，不替换默认算法 | 已完成 |
| H-only “上界” | 全文改为 H-access reference，并修正 [32] 为 Yu 等 | 已完成 |
| 延迟/显存 | Cora/CiteSeer/CoraFull seed0 实测；分离构球、下游墙钟与 full-batch inference；不把 GRACE 一次性预训练混入下游计时 | 已完成，GRACE 预训练时间仍需最终排版时从原始日志单列 |
| 近期文献 | 核验并加入 TINED、GNN-CFGD、SAMTD；SAMTD 与本文按是否严格 X-only 切割 | 已完成 |
| GRACE seed 方差 | 当前仍为 fixed-H conditional results，未扩 3×3；摘要和实验设置保留限制 | 未补实验，作为明确局限 |
| 真实噪声 | 未新增真实弱标注数据集 | 未补，保留局限 |

新增统计与效率产物位于 `runs/mlp36_review_defense_20260715/`。最关键的机制结论是：排除自身投票后，真实局部共识仍稳定优于随机分组和显式标签平滑，故收益不能仅解释为自标签回流；但 LOO 未在所有数据集统一优于默认目标，默认方法应描述为“外部局部共识 + 组感知平滑”的混合。
