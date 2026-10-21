# LCSS / SoftDist-MLP 二审 A：Academic Paper Review

## Paper Metadata

- **稿件**：`paper_final_zh_v14_second_review_ready_20260716.md`
- **目标领域**：图学习、标签噪声学习、GNN-to-MLP / graph-free inference
- **论文类型**：经验方法论文，包含机制消融与适用域分析
- **目标期刊口径**：Knowledge-Based Systems
- **审稿轮次**：内部二审；已知一审 P0 修改完成

## Executive Summary

稿件研究“训练期可访问图、推理期仅有节点特征，且训练标签含噪”的交叉问题。LCSS 使用 label-free GRACE 表征定义局部分组，以组内含噪训练标签产生半硬软监督，再训练 X-only MLP；扩展方案将共识目标用于 GCN teacher 并蒸馏给 X-only student。相较一审版本，v14 已实质修复结果驱动胜率、H-access 资源不对称、未校正多重检验、自票与标签平滑归因、完整 baseline 表和过强因果措辞等问题。

该工作的最强贡献不是一种不可替代的新聚类算法，而是一个定义清楚的 noisy-label graph-free 任务、一个简单的监督侧局部共识方案，以及异常完整的正负结果和适用域分析。以 KBS 的应用与知识迁移定位看，贡献达到中等水平；以强调算法新颖性的期刊口径看仍偏弱。

二审后我不再认为稿件存在明显的结果隐藏或统计误导，但仍有两个投稿前的核心风险：第一，方法效果高度依赖随机预训练得到的 H，而全部误差条只条件于单一 H，且三个核心数据集不能从 checkpoint 重建该 H；第二，beta 的开发使用了 test-informed pilot，稿件仍把开发数据与冻结后数据合并进 24/25 headline。真实噪声缺失和总训练成本未完整报告是次一级问题。综合建议为 **Major Revision / Borderline**，比一审明显提升，但尚不建议直接投稿。

## Claims And Evidence

| Claim | Evidence in v14 | Strength |
|---|---|---|
| LCSS 在固定 H 下优于多种 X-only train-target 抗噪策略 | 五个全谱数据集、五种设置、完整表 1 与附录，5 paired seeds | 强，但仅限 fixed-H 与开发筛选后的数据集 |
| 真实局部共识不只是标签平滑或自标签回流 | shuffled、LOO、LOO-shuffled、uniform/prior smoothing、target-quality | 强 |
| 推理端严格 X-only | 方法定义、资源表和推理代码路径 | 强 |
| 粒球是不可替代的精度来源 | kNN/kmeans 与 teacher 对照反驳该说法，稿件已不再提出 | 不主张，处理正确 |
| 蒸馏扩展能恢复绝对精度 | GLNN-style、球式/kNN teacher 与 GCN 参照 | 中等；主要收益经常来自 H 输入和全节点 teacher coverage |
| 方法适用于现实标签噪声 | 仅 uniform/pair/instance synthetic noise | 弱，尚无直接现实验证 |

## Strengths

### S1. 任务与资源边界现在定义准确

第 3.1 节明确训练可访问完整图、推理只接受节点自身特征，并将 transductive、X-only 与 H-access 资源类分开。第 5.3 节恢复 `[X||H]+Small-loss/JoCoR` 后，不再用弱 `[X||H]+CE` 代表整个 H-access 家族。这消除了此前最严重的公平性问题。

### S2. 机制归因明显强于一般经验论文

随机化分组、leave-one-out、两种普通标签平滑、Brier/NCR/CDR、kNN/kmeans 平替、票数、学生容量和表征替换形成了一条完整证据链。特别是默认目标被准确改写为“外部局部共识 + 组感知平滑”，没有继续包装成逐节点硬纠错器。

### S3. 负结果与适用域没有被隐藏

表 2 报告 PubMed、Amazon-Computers、Attributed-Wiki、CiteSeerFull 和 Coauthor-Physics 等边界；CoraFull 的 teacher 负归因、球式被 kNN 替代、较大图上固定球数触顶也被保留。这使论文更像机制审计而非只展示有利结果。

### S4. 统计口径已从“显著性堆叠”改为可辩护的探索性证据

主比较族冻结为 LCSS 对 CE、shuffled 和 JoCoR，按 family 使用 Holm 校正，并报告 paired delta、95% CI 和方向计数；稿件也不再把未拒绝差异写成统计等效。对 n=5 的限制说明是必要且诚实的。

### S5. 文献定位与技术身份更安全

稿件把 contribution 放在 noisy-label graph-free setting、监督侧结构利用与适用域，而不是宣称首创 MLP 蒸馏或粒球优于邻域方法。TINED、GNN-CFGD、SAMTD、KRD、VQGraph 等邻近文献已有明确切割。

## Weaknesses

### W1. 完整管线方差与表征资产仍不足以支持“稳定性”作为主要卖点

所有下游种子复用同一份 H，因此 5-seed 方差没有覆盖 GRACE 预训练随机性。LCSS 的核心几何、球划分、票数和 target 全部由 H 决定，这不是外围随机源。更严重的是 CoraFull、Coauthor-CS 与 Amazon-Photo 缺少 encoder/projector checkpoint，只保存 H、metadata 和 loss；读者可以复现下游，但不能验证这些关键 H 的生成过程。

**影响**：这不使下游数字无效，但会降低“稳定”“可复现”和“固定参数跨数据集泛化”的证据强度。

**建议**：投稿前至少补 Cora、CiteSeer、CoraFull 的 3 个 GRACE seeds，每个 H 配 3 个下游 seeds；若确实不补，应将摘要和 Highlights 中所有稳定性语言严格限定为“conditional on one archived frozen representation”，并把缺 checkpoint 的三个数据集从“端到端可复现”中单独列出。

### W2. beta 的 test-informed 开发仍使 aggregate headline 带有开发集偏差

第 3.2 节诚实说明 beta=0.5 来自 Cora、CiteSeer、PubMed 的 test-informed 五种子比较，这是正确披露；但摘要和结论的 24/25 仍把 Cora、CiteSeer 这些开发数据与后冻结数据合并计数。数据集筛选已经被条件化说明，超参数开发与最终确认集仍未完全分层。

**影响**：24/25 可以作为描述性汇总，但不能被读作独立泛化证据。

**建议**：在主结果中并列报告“开发数据结果”和“参数冻结后的确认数据结果”。至少单列 CoraFull、Coauthor-CS、Amazon-Photo 的 post-freeze 计数；24/25 可留在附录或次要句，不应继续承担摘要最醒目的效能 claim。

### W3. 实际最有部署价值的蒸馏扩展与主方法之间关系仍不够聚焦

train-only LCSS 的绝对精度在小图上远低于 direct-H 和蒸馏学生；蒸馏扩展解决了精度问题，但其主要增量常来自 H 输入和全节点 teacher coverage，而非球式共识。CoraFull 上 kNN/CE teacher 更强进一步说明这不是 LCSS 核心机制的普适延伸。

**影响**：读者可能不清楚论文到底主张一个 train-target 抗噪方法，还是一个 noisy GNN-to-MLP pipeline。

**建议**：保持 train-only LCSS 为主贡献，但在标题、摘要和 Fig. 1 中不要让 teacher 扩展与核心算法等权。第 6 节应明确为 optional deployment extension，并把“共识 target 的额外贡献具有数据集依赖性”放在节首而非节末。

### W4. 现实动机仍完全依赖合成噪声

uniform、pair 和 instance-dependent 都由 clean label 人工污染，且 Small-loss/Co-teaching/JoCoR 使用已知注入噪声率。现实中的众包、规则或 LLM 标签通常不知道真实污染率，错误也不一定服从这些模型。

**影响**：论文可以证明 synthetic noisy-label benchmark 下的有效性，但不能证明现实标注流程中的可用性。

**建议**：若没有真实图噪声基准，至少增加一个噪声率失配敏感性实验，或将现实案例严格保留为动机而非应用验证。当前局限段已经披露，但摘要应避免“现实鲁棒”类措辞。

### W5. 总训练成本比较不完整

第 3.3 节报告下游与构球时间，但主动排除了 GRACE 预训练。该成本虽然跨噪声设置复用，却是第一次部署到新数据集时不可避免的主要成本；与 MLP+CE 相比，完整训练资源并不相同。

**影响**：X-only 推理结论成立，但“简单/高效方法”的训练侧含义仍不完整。

**建议**：增加 GRACE 预训练墙钟、峰值显存和 H 存储，分别报告 amortized 与 one-shot 成本；不要把当前表解释为完整 pipeline efficiency。

## Methodology Assessment

| Criterion | Rating (1-5) | Assessment |
|---|:---:|---|
| Soundness | 4.0 | 方法定义和训练/推理边界清楚，无明显标签泄漏；target 自票已被 LOO 审计 |
| Novelty | 2.8 | 组件均已有，贡献主要是交叉问题、监督侧组合和系统边界刻画 |
| Reproducibility | 2.8 | 下游资产和哈希充分，但单一 H、三个数据集缺 checkpoint，匿名仓库未就绪 |
| Experimental design | 3.8 | 对照、消融、失败集和资源分层较完整；开发筛选和 test-informed beta 限制独立性 |
| Statistical rigor | 3.5 | Holm、CI、方向数显著改进；n=5 与单一 H 仍限制推断 |
| Scalability | 3.3 | 覆盖至 34k 节点并报告复杂度；未验证真正大图和端到端预训练成本 |

## Questions For The Authors

1. 在不同 GRACE seed 下，球数量、assignment、target-quality 和最终增量的方差分别是多少？
2. 若只统计 beta 冻结后新增的数据集和噪声设置，LCSS 相对七个正式 baseline 的胜出计数是多少？
3. 为什么 LOO 平均略优于当前目标却不作为默认方法？这是为了冻结算法，还是存在明确的数据集级退化？
4. 新数据集第一次训练时包含 GRACE 的完整墙钟与峰值显存是多少？复用多少个任务后一次性成本才被摊薄？
5. 在训练时假设噪声率与实际噪声率不一致时，Small-loss/JoCoR 与 LCSS 的相对排序是否保持？

## Minor Issues

- 最终投稿版需要给出正式标题、作者、匿名代码 URL、软件版本和完整命令；当前仍是研究稿形态。
- 表 1 现已完整，但宽表在 Elsevier 双栏格式中可能需要 landscape 或拆分；这属于排版问题，不影响证据。
- “较早研究之一”仍应在补充材料给出检索日期、数据库和关键词。
- 附录中 inspired proxy 与 faithful reimplementation 应使用一致的视觉标识，避免读者误将其纳入正式排名。

## Literature Positioning

文献线已经覆盖 noisy-label graph learning、graph-free GNN-to-MLP、可靠蒸馏和粒球计算。当前最安全的定位是：现有 graph-free 工作主要研究干净监督、特征扰动、缺失或标签稀缺，而本文系统研究训练标签噪声；现有 noisy-label GNN 方法通常保留图依赖推理，而本文要求严格 X-only student。不能声称所有既有工作都未触及噪声，也不能声称监督侧结构利用本身从未出现。

## Recommendation

**Overall assessment**：Major Revision / Borderline for KBS  
**Confidence**：High（4/5）；判断基于完整方法、逐种子结果口径、对照资产和一审修订记录。  
**Contribution level**：Moderate。

### Highest-priority actions

1. 补最小 3×3 GRACE/downstream 分层方差，或进一步降级稳定性 headline。
2. 将开发数据与 beta 冻结后的确认数据分开汇总，弱化 24/25 headline。
3. 补 GRACE 预训练成本与 artifact manifest；提交前填入匿名代码链接。
4. 若预算允许，优先做噪声率失配而非继续增加合成数据集。
