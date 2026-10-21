# LCSS / SoftDist-MLP 二审 B：Journal Peer Review

## Recommendation

**Decision: Major Revision**  
**Reviewer confidence: 4/5**

修订稿已经解决一审中最严重的选择性呈现、资源不对称和统计夸大问题。方法与实验现在基本自洽，且适合以 Knowledge-Based Systems 的应用型方法论文定位继续推进。当前不建议拒稿，但在录用前仍需要解决完整管线复现、开发/确认结果分层和总成本报告三项问题。真实噪声实验很有价值，但在明确收窄 claim 后不必作为唯一录用前提。

## 1. Summary Statement

本文提出 LCSS：先在完整图上以 GRACE 学习 label-free frozen representation，再以粒球式局部分组聚合 noisy train labels，生成半硬 soft targets，最后训练推理时只接收原始特征的 MLP。作者还提供以 GCN[X||H] 为 teacher 的蒸馏扩展。论文的技术复杂度不高，但实验覆盖较广，特别强调随机分组、LOO、标签平滑、分组替代性、H-access、公平资源分层和失败数据集。

v14 的主要结论可以被现有证据支持，但应严格限定为：固定归档 H、transductive、clean validation、train-only synthetic label noise、已知污染率 baseline 协议下的经验结果。稿件尚不能证明完整随机管线稳定，也不能证明现实未知噪声场景中的有效性。

## 2. Section-by-section Review

### 2.1 Abstract And Introduction

**优点**：现实问题被清楚拆为训练期图可用、推理期图不可用和训练标签含噪三项约束；没有把方法称为隐私保护算法。24/25 已明确是开发筛选后的条件计数。

**问题**：摘要仍以 24/25 为最醒目的效能数字，而 beta 由其中部分数据集的 test-informed pilot 决定。即使文字披露充分，读者仍容易把该数字理解为确认性结果。

**必要修改**：摘要同时给出 beta 冻结后数据的独立结果，或把 24/25 移到结果段，把摘要中心改为 11 数据集 uniform 0.4 全景、real-vs-shuffled 归因与 post-freeze 数据结果。

### 2.2 Related Work

**优点**：graph-free、noisy-label graph learning、reliable distillation 与 granular-ball 四条线的边界基本准确；近期方法已补齐。稿件没有把 NOSMOG 的 feature noise 误写为 label noise，也没有把 SA-MLP 当作严格 X-only。

**问题**：“较早研究之一”仍基于作者内部核验，不是可复核的系统综述。

**建议**：补充检索日期、数据库、关键词和纳入标准；该内容可放 Supplement，不需要扩大正文。

### 2.3 Methods

**优点**：符号、四阶段流程、固定球参数、自票、LOO 定义、复杂度与推理资源均清楚。label-blind grouping 与 train-label-only voting 的边界明确。

**问题 1**：beta 的选择使用开发期 test accuracy；这不是标签泄漏到单次训练，却属于超参数层面的 test-informed model development。

**问题 2**：instance-dependent noise 仅写为“依据样本特征决定翻转概率”，pair noise 仅写“预定相邻类别”，主文没有给出生成公式、映射、随机种子和实现版本。引用 NoisyGL 不能替代精确协议说明。

**问题 3**：效率表排除 GRACE 预训练，因此只能证明部署前向不依赖图，不能比较完整训练成本。

**必要修改**：

1. 把开发数据明确标为 development set，把冻结后的数据标为 confirmation set。
2. 在附录给出 pair/instance 噪声生成伪代码、映射规则和版本信息。
3. 报告每个核心数据集 GRACE 预训练墙钟、峰值显存、checkpoint/H 大小；分别给 one-shot 与跨噪声设置摊销成本。

### 2.4 Experimental Design

**优点**：统一 split、train-only pollution、clean val/test、paired seeds 和 hash 审计；表 1 已展开全部正式 baseline；H-access 对照单独分层；代理方法不参与正式排名。

**问题 1**：完整噪声谱数据集由 uniform 0.4 表现 gate 决定。当前披露已经避免虚假总体胜率，但 full-spectrum 表仍是条件样本，不能用于估计跨数据集平均性能。

**问题 2**：所有下游结果条件于单一 H，实际有效重复单位并不是完整方法运行。GRACE 是核心随机模块，不能只在局限中一笔带过。

**问题 3**：Small-loss、Co-teaching、JoCoR 知道真实注入噪声率，而 LCSS 不需要该值。这一设置对 LCSS 并不占便宜，但其现实解释必须保持为 benchmark comparison。

**必要修改**：至少在两个小图和一个中/大图上补多个 GRACE seeds；若算力确实不允许，标题、摘要、Highlights 和表注均应出现 fixed-H qualifier，而不仅在实验设置和局限中说明。

### 2.5 Results And Statistics

**优点**：均值±标准差、paired direction、CI 与 Holm correction 已给出；未把未显著差异写成等效；失败结果完整报告。LOO 和 smoothing 结果直接回答了自票问题。

**问题 1**：n=5 的 paired t interval 对尾部分布敏感，Holm 只能修正多重性，不能解决小样本分布假设。方向计数是合理补充，但不能称“显著稳定”。

**问题 2**：主检验对照固定为 JoCoR，而某些设置最强 baseline 是 Small-loss 或其他方法。该设计可避免事后挑选，但“最高均值 24/25”与“相对预声明 JoCoR 的推断”是两种不同证据，正文应保持分开。

**问题 3**：target-quality 使用 clean label 作事后诊断是合理的，但需要在表题和方法中反复标明它不参与 target 或超参数选择。

**建议**：保留当前 Holm 结果，但将“稳定”统一替换为“5/5 或 4/5 seeds 同向”；Supplement 提供全部 raw seeds 和 bootstrap/permutation sensitivity 作为稳健性检查，不必全量扩至 10 seeds。

### 2.6 Discussion And Limitations

**优点**：限制段非常完整，主动承认 transductive、kNN/kmeans 替代性、clean-label 代价、feature-noise 不鲁棒、真实噪声缺失和 single-H 方差。这是稿件的重要优点。

**问题**：限制披露不能完全替代核心复现资产。三个核心数据集没有 encoder checkpoint，意味着作者也无法从源码重建自己报告的完整流程。

**必要修改**：投稿前重新生成并归档这些 checkpoint，或把归档 H 明确作为数据资产发布并将方法复现定义为“两阶段：作者提供 H 后可复现下游”。后者可接受，但 artifact availability 必须在投稿时真实存在，不能只留占位符。

## 3. Major Concerns

### Major 1. Confirmation evidence is still mixed with development evidence

结果驱动的数据集 gate 已被诚实披露，但 beta 的 test-informed pilot 与 full-spectrum headline 仍混合。建议按时间冻结点重排结果，而不是继续只按数据集是否“成功”组织。

### Major 2. Full-pipeline stochasticity is not measured

单一 H 是当前最重要的科学不确定性。因为 grouping 与 targets 均由 H 决定，下游五种子不足以代表方法稳定性。至少需要小规模层次重复，或全篇在最醒目位置使用 fixed-H conditional qualifier。

### Major 3. Reproducibility is incomplete for three core datasets

归档 H 可以复现最终数字，但不能验证预训练和 H 选择。缺 checkpoint 与匿名代码链接占位必须在投稿前解决，否则不满足现代计算机期刊的基本 artifact 期望。

### Major 4. End-to-end cost and deployment evidence are only partially reported

当前表证明 MLP inference path 不读取图，但没有报告方法首次部署时的 GRACE 成本，也没有 CPU、online single-node 或 neighbor-fetch comparison。稿件已经正确收窄速度 claim，因此不要求完成真实服务 benchmark；但必须补完整训练资产成本，避免“简单方法”被误解为端到端廉价。

## 4. Minor Comments

1. 给 pair 和 instance-dependent noise 增加正式公式或算法，而不是只用一句描述。
2. 明确 split 是每 seed 重建还是固定 split 只改变噪声；在正文给出 hash 配对规则。
3. 对 `LCSS+LOO` 平均更高但不替换默认方法给出明确决策理由。
4. 在 Table 3 标题中把 `protocol port`、`common-backbone reimplementation` 和 `inspired proxy` 用脚注符号区分。
5. 数据、代码与结果资产节的匿名仓库 URL 必须在送审稿中真实存在。
6. 最终英文稿应控制摘要长度，并把 optional teacher extension 从主算法贡献中视觉分离。
7. 结构噪声免疫只能解释为推理不读取 A 的确定性结果，不应写成学习到的 robustness；v14 当前基本遵守这一点。

## 5. Statistical And Reporting Checklist

| Item | Status | Comment |
|---|---|---|
| Multiple seeds | Partial | 5 downstream seeds，但只有 1 个 GRACE seed |
| Paired design | Pass | split/noise hash 配对，逐 seed delta 可追溯 |
| Effect size | Pass | paired mean delta 已报告 |
| Confidence interval | Pass with caveat | 已报告，但 n=5 假设敏感 |
| Multiple-comparison correction | Pass | 三个 primary families 分别 Holm 校正 |
| Exploratory/confirmatory separation | Partial | 检验族已区分，开发数据与确认数据仍混合 |
| Negative results | Pass | 六个边界数据集和 teacher 负归因均保留 |
| Hyperparameter provenance | Partial | beta 来源已披露，但为 test-informed |
| Data leakage audit | Pass | noise 只污染 train，target 不读 val/test clean label |
| Full-pipeline reproducibility | Fail/Partial | 三个核心 H 无 checkpoint；匿名仓库未提供 |
| Real-world validation | Not met | 只有 synthetic noise |
| Compute reporting | Partial | 下游与推理充分，GRACE 预训练成本缺失 |

## 6. Required Revision List

### Required before submission

1. 将 development 与 post-freeze confirmation 结果分开报告，降低 24/25 的 headline 权重。
2. 补 pair/instance 噪声生成细节与 split/noise seed 规则。
3. 发布真实可访问的匿名代码、逐种子 CSV、H manifest；解决或明确三组缺 checkpoint 的 artifact 形式。
4. 补 GRACE 预训练成本。

### Strongly recommended

1. 在 Cora、CiteSeer 和 CoraFull 上补最小多 GRACE-seed 分层实验。
2. 增加噪声率误设敏感性；它比再补一个 synthetic dataset 更贴近现实问题。
3. 在 Supplement 增加非参数 permutation/bootstrap sensitivity，不必把全部设置扩到 10 seeds。

### Can remain as limitations

1. 没有真实噪声图基准。
2. 没有 unseen-node inductive 评测。
3. 没有在线 CPU/single-node latency。
4. 粒球不优于 kNN/kmeans。

## 7. Final Assessment

相较一审稿件，v14 已从“存在核心报告缺陷”提升为“方法与证据基本成立但复现和确认性不足”。对于 KBS，算法新颖性不强可以由问题设定、知识迁移视角、完整消融和适用域分析补偿。当前最可能导致再次 Major Revision 的不是精度，而是 single-H 完整管线方差、开发结果混入 headline 与 artifact 不完整。

完成 Required before submission 后，可进入 **Borderline / Weak Accept** 区间；若再补最小多 GRACE-seed 结果，稿件的最大剩余风险将转为中等技术新颖性，而不是实验可信度。
