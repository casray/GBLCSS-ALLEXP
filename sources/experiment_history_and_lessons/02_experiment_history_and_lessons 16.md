# Phase43：椭球 SoftDist downstream 验证与停止决策（2026-06-29）

## 1. 研究问题

Phase41 的离线诊断显示，椭球化 SoftDist 在部分数据集上能改善 soft target 质量：Coauthor-CS、AmazonPhoto、AmazonCom 有默认椭球正信号；DBLP/PubMed 在自适应构球网格下也出现 offline-positive setting。本轮要回答更关键的问题：

> 这些 offline target 正信号能否转成 downstream accuracy 增益，并且真实椭球是否稳定优于 shuffled 椭球、current SoftDist 和 GCN[X‖H]？

本轮不是为了改主方法，而是为了决定椭球方向是否值得进入强 baseline 对比或正式下游实验。

## 2. 方法动机

椭球化 SoftDist 的动机是：普通 SoftDist 在 hard ball 内对 train nodes 等权投票，可能过度相信边界点或离群点。椭球版本保留 hard ball assignment，只在 ball 内用 frozen GRACE `H` 的协方差与 Mahalanobis distance 降低边界/离群 train node 的投票权重。

本轮仍遵守原协议：

- 不用 clean label 构造 target。
- 不改 split。
- 不改 baseline evaluation。
- 不引入模型预测反馈。
- shuffled control 必须与真实椭球同路径，只破坏几何对应。

## 3. 配置与产物

### Cora/CiteSeer 椭球几何补充

输出目录：

- `external/NoisyGL/runs/ellipsoid_geometry_small_20260626/`

关键文件：

- `geometry_report.md`
- `geometry_aggregate.csv`
- `geometry_raw.csv`

协议：Cora/CiteSeer，seeds `0..4`，uniform `0.4`，train noise，percent split `0.2/0.1/0.7`。

### 默认椭球 downstream seed0 smoke

输出目录：

- `external/NoisyGL/runs/ellipsoid_downstream_seed0_20260626/`

关键文件：

- `seed0_report.md`
- `seed0_summary.csv`
- `seed0_combined_raw.csv`

协议：

- 数据集：Coauthor-CS、AmazonPhoto、AmazonCom。
- seed：`0`。
- 噪声：uniform `0.4`，train scope。
- split：percent `0.2/0.1/0.7`。
- 方法：`gcn_xh_ce`、current SoftDist、current shuffled、ellipsoid、ellipsoid shuffled。

### selected 椭球 downstream sanity 与 DBLP 5-seed

输出目录：

- `external/NoisyGL/runs/ellipsoid_selected_downstream_seed0_20260629/`

关键文件：

- `seed0_sanity_report.md`
- `seed0_sanity_summary.csv`
- `dblp_5seed_report.md`
- `dblp_5seed_summary.csv`
- `dblp_5seed_paired_deltas.csv`

selected settings 来自 Phase41 离线网格：

| Dataset | max_balls | split_improvement |
|---|---:|---:|
| DBLP | 128 | 0.001 |
| PubMed | 512 | 0.01 |
| AmazonPhoto | 64 | 0.05 |
| AmazonCom | 64 | 0.001 |

## 4. 结果

### 4.1 Cora/CiteSeer 为什么不适合椭球

| Dataset | Balls | Root split gain | H-kNN agree | Lift | Eff. support | Anisotropy | Weighted true purity delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 52.4 | 0.0721 | 0.8146 | 4.54 | 0.9849 | 35.17 | +0.0049 |
| CiteSeer | 63.8 | 0.0914 | 0.6546 | 3.68 | 0.9829 | 96.72 | +0.0066 |

解释：

- Cora/CiteSeer 不是完全没有 H-space clustering。
- 关键问题是 `ellipsoid_effective_support_ratio_mean` 接近 `1`，说明 Mahalanobis weights 几乎等权。
- `weighted_true_train_purity_delta` 只有 `+0.0049/+0.0066`，球内加权几乎没有改善真实标签纯度。
- 因此椭球在 Cora/CiteSeer 上缺乏相对普通 SoftDist 的实际投票改变。

### 4.2 默认椭球 downstream seed0

| Dataset | GCN[X‖H] CE | SoftDist | SoftDist shuffled | Ellipsoid | Ellipsoid shuffled | E-Soft | E-Eshuf | E-CE | Smoke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Coauthor-CS | 0.9189 | 0.9208 | 0.9199 | 0.9235 | 0.9203 | +0.0026 | +0.0032 | +0.0046 | PASS |
| AmazonPhoto | 0.9109 | 0.8964 | 0.9001 | 0.8986 | 0.8949 | +0.0022 | +0.0037 | -0.0123 | FAIL |
| AmazonCom | 0.8564 | 0.8558 | 0.8578 | 0.8558 | 0.8616 | +0.0000 | -0.0058 | -0.0006 | FAIL |

解释：

- Coauthor-CS 是唯一 seed0 pass：ellipsoid 同时超过 current SoftDist、ellipsoid shuffled 和 GCN[X‖H]。
- AmazonPhoto 虽高于 current SoftDist 和 ellipsoid shuffled，但低于 GCN[X‖H] 与 ordinary shuffled。
- AmazonCom 直接失败：ellipsoid 与 current SoftDist 持平，低于 GCN[X‖H]，且低于 ellipsoid shuffled。

结论：默认椭球不应进入强 baseline 对比。最多只保留 Coauthor-CS 5-seed 作为低优先级确认候选。

### 4.3 selected 椭球 downstream sanity

seed0 sanity 结论：

- DBLP：selected ellipsoid 比 selected shuffled `+1.04pp`，但与 current SoftDist 基本持平，因此只作为 5-seed 候选。
- PubMed：selected shuffled 比 selected ellipsoid 高 `+0.43pp`，停止。
- AmazonPhoto：selected ellipsoid 高于 selected shuffled/current SoftDist，但低于 GCN[X‖H] 和 current shuffled，不推进。
- AmazonCom：selected ellipsoid 低于 selected shuffled 和所有 current controls，停止。

### 4.4 DBLP selected 椭球 5-seed

| Method | Mean | Std | ok seeds |
|---|---:|---:|---:|
| GCN[XH] | 0.8391 | 0.0032 | 5 |
| Current SoftDist | 0.8391 | 0.0036 | 5 |
| Current shuffled | 0.8391 | 0.0036 | 5 |
| Selected ellipsoid | 0.8392 | 0.0031 | 5 |
| Selected shuffled | 0.8374 | 0.0041 | 5 |

Paired deltas：

| Comparison | Mean delta | Positive seeds |
|---|---:|---:|
| selected minus selected shuffled | +0.19pp | 3/5 |
| selected minus current SoftDist | +0.01pp | 2/5 |
| selected minus current shuffled | +0.01pp | 2/5 |
| selected minus GCN[XH] | +0.01pp | 2/5 |
| current SoftDist minus current shuffled | +0.00pp | 0/5 |

最终判断：DBLP selected ellipsoid 5-seed STOP。seed0 的 `+1.04pp` real-vs-shuffled 是偶然种子现象，没有变成稳定 downstream 价值。

## 5. 配置修正记录

downstream smoke 过程中修正了一个路径配置错误：

- 文件：`external/NoisyGL/config/gcn_xh_ce/gcn_xh_ce_amazonpho.yaml`
- 修正：`embedding_path` 从小写 `amazonpho` 改为实际 checkpoint 目录 `AmazonPhoto`

这是 frozen H 路径修正，不是方法或协议改动。

## 6. Offline-to-downstream 断裂复盘

本轮最重要的反思不是某一步实验跑错，而是原来的 offline gate 主要衡量 "target 本身是否更像 clean label"，但 downstream 真正需要的是 "这个 target 是否能改变 GCN[X‖H] 的有效决策边界"。二者不是同一个问题。

### 6.1 `soft_clean_mass_delta` 不能再单独当推进门槛

`soft_clean_mass_delta` 是训练节点上的 clean-label 概率质量总增量，不是平均每个节点的增量。大图上总量看起来为正，除以 train 节点数后可能非常小：

| Dataset | mass_delta | 约 train nodes | 每节点平均增量 | seed0 selected-shuffled |
| --- | ---: | ---: | ---: | ---: |
| DBLP | +2.9623 | 3543 | +0.00084 | +1.04pp |
| PubMed | +1.8385 | 3943 | +0.00047 | -0.43pp |
| AmazonPhoto | +7.9166 | 1530 | +0.00517 | +0.37pp |
| AmazonCom | +10.9144 | 2750 | +0.00397 | -2.49pp |

因此 DBLP/PubMed 的 "offline-positive" 实际是很弱的 target 改善，并不足以稳定改变训练梯度。DBLP 更典型：soft clean mass 为正，但 hard pseudo 方向变坏：

| Dataset | cdr_delta | ncr_delta | hard_net 近似 |
| --- | ---: | ---: | ---: |
| DBLP | +0.0044 | -0.0102 | -0.0067 |
| PubMed | +0.0015 | +0.0141 | +0.0047 |
| AmazonPhoto | -0.0018 | -0.0000 | +0.0011 |
| AmazonCom | +0.0007 | +0.0007 | -0.0001 |

DBLP seed0 的 `+1.04pp` 更像 selected shuffled 偶然偏低，而不是 selected ellipsoid 真正有强下游信号。以后不能只看 clean mass 总量或 seed0 real-vs-shuffled 小胜就推进 5-seed/强 baseline。

### 6.2 H 同时用于构球和下游输入，粒球信号容易被吸收

当前球由 frozen GRACE `H` 构造，而 downstream 分类器也直接输入 `[X‖H]`。这意味着椭球/SoftDist target 很可能是在标签端重复强调模型输入中已经包含的 H 几何，而不是提供全新的条件信息源。

强旁证是 SoftDist-MLP(X)：下游不输入 H、也不用图卷积时，Cora/CiteSeer 上球软目标能带来约 `+7~9pp` 的大增益；但换成 GCN[X‖H] 后，增益经常只剩 `+0.x~1.xpp`，甚至为负。后续真正要问的变量不是单纯 `num_balls`、`split_improvement`，而是：

> 给了 downstream `[X‖H]` 以后，GB target 还剩多少条件增量。

### 6.3 Offline 指标缺少梯度杠杆信息

下游训练真正关心的是 soft target 改变了哪些节点的梯度：这些节点是否是 baseline GCN[X‖H] 会错的节点，是否处在低 margin / 高 loss 的决策边界附近，还是只是 easy nodes 上很小的概率微调。当前离线诊断只看 target 与 clean label 的关系，没有接入 baseline logits、margin、loss 或梯度方向。

这会导致几类误判：

- clean mass 增加发生在模型本来就能分对的 easy nodes 上，几乎没有下游杠杆。
- target 对少量 clean 节点的小损伤可能刚好发生在高杠杆边界节点上。
- shuffled 虽然语义差，但可能提供普通 label smoothing 正则化，因此 downstream 不一定差。
- `ellipsoid_effective_support_ratio_mean` 接近 `1` 时，本质几乎还是 uniform vote，不能期待强转化。

以后 offline-to-downstream 的桥接诊断至少应加入：

- `mass_delta_per_train = soft_clean_mass_delta / train_count`，不能再只看总量。
- `hard_net_delta = noise_rate * ncr_delta - (1 - noise_rate) * cdr_delta`；本轮 uniform `0.4` 下即 `0.4*ncr_delta - 0.6*cdr_delta`。
- `gradient_weighted_target_gain`：用 GCN[X‖H] baseline logits / loss / margin 加权 target 改动，检查增益是否集中在高 loss、低 margin 或 baseline 会错的节点。
- baseline headroom：如果 GCN[X‖H] 已接近饱和，例如 AmazonPhoto seed0 GCN[X‖H] `0.9109` 而 selected ellipsoid `0.8986`，就不应期待 target 微调带来稳定收益。
- real-vs-shuffled downstream gap 应作为主门槛；offline real > shuffled 只能作为预筛。
- effective support ratio 接近 `1` 时，椭球权重近似均匀投票，应降低推进优先级。

结论：现在最大问题不是继续调 `split_improvement/max_balls`，而是 offline gate 选错了。构球参数可以修复 "有没有球"、"球是否更纯"，但不能保证 "这个 target 对 GCN[X‖H] 的训练梯度有用"。下一次进入 downstream 前，必须先做上述桥接诊断；只有同时满足平均 target 改动足够大、hard-net 为正、相对 shuffled 有梯度优势、baseline 还有 headroom 的配置，才值得继续。

## 7. 结论分级

### 已被实验支持

- Cora/CiteSeer 上椭球权重接近 uniform，weighted true purity delta 很小，因此椭球加权不能有效改变球内投票。
- 默认椭球 downstream seed0 只有 Coauthor-CS 通过；AmazonPhoto/AmazonCom 不通过。
- selected 椭球参数不能把 DBLP 的离线正信号稳定转成 downstream accuracy。
- offline target diagnostic 正信号不能直接作为 downstream 性能 claim。
- `soft_clean_mass_delta` 总量门槛会误导；需要按 train 节点归一化并结合 hard-net 与梯度杠杆判断。

### 合理猜测

- 椭球 target 信号容易被强 GCN[X‖H] baseline、current SoftDist 或 shuffled control 吸收。
- Coauthor-CS 可能是唯一仍值得低优先级 5-seed 检查的椭球 downstream 数据集，但即使通过，也更像局部扩展而不是主方法替换。
- H 同时用于构球和 downstream 输入时，粒球 target 的边际贡献取决于它相对 `[X‖H]` 的条件增量，而不是球数或纯度本身。

### 尚未验证

- Coauthor-CS 默认椭球 seed0 正信号能否在 5-seed 下保持。
- 是否能设计不依赖 clean label 网格选择的 deployable dataset-adaptive ball policy。
- 使用 GCN[X‖H] baseline logits 的 `gradient_weighted_target_gain` 是否能更好预测 downstream 收益。

### 已被负结果削弱

- 椭球 SoftDist 是大图通用改进。
- selected 参数可以救 DBLP downstream。
- AmazonPhoto/AmazonCom 的 offline target 正信号能自然转成 accuracy。
- 现在就给椭球跑 NRGNN/R2LP 等强 baseline 是合理投入。
- 继续只调 `max_balls/split_improvement` 就能解决 offline-to-downstream 转化失败。

## 8. 最终决策

椭球 downstream 方向停止，不进入强 baseline 对比，不替代 SoftDist 主方法。

论文中可以保留的价值是边界分析：

> 椭球化球内投票能在部分数据集改善离线 soft target 质量，但 downstream accuracy 多被 GCN[X‖H]、current SoftDist 或 shuffled control 吸收。DBLP selected 参数 5-seed 也没有稳定收益，因此椭球是一个有解释价值的负结果/未来工作，而不是当前主方法升级。

下一步优先级：

1. 写论文正文与边界分析。
2. 标签率曲线图表。
3. 若时间充裕，仅可低优先级检查 Coauthor-CS 默认椭球 5-seed；不建议再扩展椭球强 baseline。
