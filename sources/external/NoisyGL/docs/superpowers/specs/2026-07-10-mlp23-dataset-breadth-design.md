# MLP-23 数据集广度扩展设计

## 目标

在不修改 LCSS 机制、超参数和噪声协议的前提下，按预注册顺序最多探测三个新数据集，至少找到一个能够稳定进入论文核心表的数据集。失败候选同样保留在附录适用域记录中，避免事后只报告有利结果。

## 候选顺序

1. `coraml`：PyG `CitationFull(name="cora_ml")`，2995 节点、7 类、边同质性 0.7886。规模小、类别覆盖充足，成功概率最高；局限是与 Cora 同族，因此论文中必须称为 Cora_ML，不能包装成跨域证据。
2. `attribwiki`：PyG `AttributedGraphDataset(name="Wiki")`，2405 节点、17 类、边同质性 0.7119。域和类别结构更有增量价值；风险是最小类别仅 9 个节点，20% split 后票数稀疏。GRACE 固定复用 Cora 配置，不做下游调参。
3. `coauthorph`：PyG `Coauthor(name="Physics")`，34493 节点、5 类、边同质性 0.9314。论文价值最高，可与 Coauthor-CS 构成同域对照；全矩阵 GRACE 会 OOM，因此仅把官方等价的 anchor-chunked contrastive loss（batch=1024）用于该大图，目标函数与超参数不变。若分块后仍不可行才转替代候选。

若 Coauthor-Physics 在 1-epoch GRACE smoke 即工程不可行，则以 `citeseerfull`（PyG `CitationFull(name="citeseer")`，4230 节点、6 类）作为替代候选。该数据集固定复用 CiteSeer GRACE 配置；若通过，论文必须明确其为 expanded citation benchmark，不将其描述为跨域证据。

候选顺序同时考虑成功概率、论文独立性与工程可行性。每个候选必须在前一候选完成正式判定后再开始。

## 固定协议

- split：percent 0.2/0.1/0.7
- noise scope：train only
- 主闸门：uniform 0.4，seeds 0-4
- LCSS：`max_balls=64`、`min_ball_size=32`、`split_improvement=0.05`、`beta=0.5`
- MLP：2 层、hidden 64、dropout 0.5
- GRACE：label-free；无官方配置时使用预注册 fallback，不依据下游结果调参
- 任何 clean label 只能用于事后诊断，不得进入构球、target 或模型选择

## 逐候选流程

1. loader 与节点顺序测试。
2. GRACE 1-epoch GPU smoke；通过后按固定配置完成预训练并导出逐行 L2 归一化的 `H.pt`。
3. H 行数/有限值/节点顺序检查与 label-blind 球几何检查。若 `num_balls < 8` 或 H 不可用，候选判工程/几何失败。
4. seed0 仅用于排除 NaN、OOM、单类坍缩和真实分组明显劣于 shuffled；不以 seed0 小胜直接纳入论文。
5. 正式 uniform 0.4：LCSS、shuffled、MLP+CE 与 10 个同骨干鲁棒策略，共 13 方法×5 seeds。

## 正式通过门

候选只有同时满足以下条件才进入核心主表：

- LCSS - shuffled 的均值至少 +1.0pp，且至少 4/5 paired seeds 为正；
- LCSS - 最强非 LCSS baseline 的均值至少 +1.0pp，且至少 4/5 paired seeds 为正；
- 65/65 行成功，验证/测试标签零污染，`modified_count == modified_train_count`；
- 无 NaN、OOM 或单类预测坍缩。

只满足真实分组门时记为 boundary；两个门都不过记为 fail。

### 次级全谱纳入规则（在其余噪声设置运行前冻结）

若某候选未达到上述 `+1.0pp` 主闸门，但在 uniform 0.4 下同时满足：LCSS 数值最高或与最高方法的绝对差不超过 0.25pp、LCSS 相对 shuffled 至少 +1.0pp 且 5/5 seeds 为正、运行与泄漏审计完整，则允许补跑完整噪声谱。该规则用于区分“稳定打平强 baseline”与“方法失效”，不把 uniform 0.4 的 boundary 事后改写为 pass。

补跑后，只有同时满足以下条件才可作为“全谱竞争数据集”加入核心实验表：

- 5 个噪声设置中至少 4 个，LCSS 为数值最高或与最高方法绝对差不超过 0.25pp；
- 5 个设置均满足 LCSS - shuffled 至少 +1.0pp，且各设置至少 4/5 paired seeds 为正；
- 任一设置均不得落后最强 baseline 超过 1.0pp；
- 全部 325 行成功并通过 train-only 噪声审计。

正文必须将这类数据集描述为“与最强同骨干策略相当且真实局部结构稳定贡献”，不得写成稳定领先；uniform 0.4 的原始 boundary 判定必须保留在适用域报告中。

## 过门后的全量扩展

对主闸门 pass 或符合次级全谱补跑资格的候选补齐 `uniform 0.2`、`uniform 0.6`、`pair 0.4`、`instance 0.4`，复用 uniform 0.4，共形成 5 噪声设置×13 方法×5 seeds。之后仅对最终满足主闸门或全谱纳入规则的数据集补 Table B；Table 3/4 只在候选成为正文核心数据集且运行资源允许时补齐。

## 论文处置

- pass：加入核心数据集数、主表、摘要与附录完整表，并可写稳定领先。
- spectrum-pass：加入核心实验表，但表述限定为“与最强策略相当且真实局部结构稳定贡献”。
- boundary/fail：不进入核心主表，但加入适用域表并报告失败原因。
- 最多尝试三个候选；一旦已有 1-3 个通过候选且继续尝试的论文收益低于成本，则收口。
