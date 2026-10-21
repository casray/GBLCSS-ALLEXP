# Tie-safe `distribution_only` 完整网格实验报告

## 1. 实验结论

本实验达到预先固定的完整性和协议标准，可以进入论文主结果：

- 5 个数据集 × 5 个噪声设置 × 5 个 seed × 4 个方法，共 500/500 行完成，500/500 行 `status=ok`；
- 50/50 个命令完成，没有失败或无效输出；50 个命令 stderr 与两份编排器 stderr 均为空；
- 125/125 个 dataset × regime × seed 协议组通过 split、noise、污染计数和数据比例一致性检查；
- 250/250 个 SoftDist 行均使用 `tie_mode=distribution_only`，并闭合 config、冻结 embedding、构球 assignment 和 tie 暴露量；
- 相对 shuffled、MLP+CE 和 JoCoR，GB-LCSS 在 25 个设置上的平均增量分别为 7.8410、7.2517 和 2.9638 个百分点；
- 三个作者定义的 25 项 Holm 家族分别有 25、24 和 8 项拒绝零假设；把全部 75 项合并为单一 Holm 家族时，分别保留 22、20 和 8 项，共 50/75 项；
- 结果不支持“所有设置都优于所有对照”：Coauthor-CS uniform 0.2 相对 MLP+CE 的均值差为 −0.0655 个百分点；相对 JoCoR 的两个负向设置是 Coauthor-CS uniform 0.2（−0.4956）和 CoraFull uniform 0.2（−0.1184），三者均未通过 Holm 校正。

最终决策为：**继续**。论文主表和主要统计结论改用本轮 `distribution_only` 结果；历史 `min_index` 结果只保留为实现审计或历史敏感性证据，不得标成 tie-safe 主结果。

## 2. 研究假设

多数类并列时，历史实现通过最小类别索引选出 `argmax`，因此目标会依赖类别编号。训练前固定的主修复是：

\[
q_j =
\begin{cases}
(1-\beta)p_j+\beta e_{m_j}, & \text{球内多数类唯一},\\
p_j, & \text{球内多数类并列}.
\end{cases}
\]

修复只在并列球内令有效 \(\beta_j=0\)，保留球内经验分布 \(p_j\) 和原共识权重 \(\alpha_j\)。研究假设是：该修复可消除类别编号偏置，同时保留局部软监督的主要性能信号。

## 3. 粒球扮演的角色

粒球在本实验中是：

- 冻结图对比表征空间中的局部语义关系单元；
- 球内含噪训练标签分布 \(p_j\) 的估计域；
- 启发式共识权重 \(\alpha_j\) 的来源；
- 对 X-only 学生提供软结构约束的弱关系教师。

粒球不是 hard positive label、hard negative filter 或强制 prototype pulling 模块。并列修复也不改变构球、节点归属、数据划分或评估流程。

## 4. 方法动机

预跑阶段的历史 `min_index` 目标重建审计发现：

- 125/125 个历史重建单元存在至少一个 active tie；
- 受影响训练节点为 12,336/259,000，占 4.763%；
- 历史最小索引规则不满足类别置换等变性。

该预跑审计与正式 v2 网格使用的统计口径不同，不能替代正式行级诊断。v2 的真实球共有 911 个 tie balls，其中 910 个 active；123/125 个行级设置存在 active tie，受影响训练节点为 12,788/259,000，占 4.937%。因此，不能把旧 `min_index` 实现无条件视为类别语义无关。`distribution_only` 在数学上删除并列球内依赖类别编号的 one-hot 分量，同时不像 `alpha_zero` 那样完全放弃该球的软分布信息。

## 5. 代码改动

核心实现和审计改动如下：

- `external/NoisyGL/utils/frozen_grace_targets.py`
  - 增加 `min_index`、`alpha_zero` 和 `distribution_only` 三种 tie 模式；
  - 保持 `min_index` 向后兼容；
  - 为 safe 模式实现类别置换等变的目标。
- `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`
  - 读取 tie 模式；
  - 输出 tie 数量、受影响节点、受影响 \(\sum\alpha\) 和 assignment hash。
- `external/NoisyGL/scripts/run_ssl_swap_downstream.py`
  - 将 tie 诊断字段持久化到 seed-level CSV。
- `paper_pipeline/scripts/run_tie_safe_full_grid.py`
  - 固定 50 条训练命令和 500 个计划行；
  - 强制确定性线程环境；
  - 实现严格恢复、协议门禁、统计和证据快照。
- 测试：
  - `external/NoisyGL/tests/test_frozen_grace_targets.py`
  - `external/NoisyGL/tests/test_tie_safe_runner_diagnostics.py`
  - `tests/test_tie_safe_representative.py`
  - `tests/test_tie_safe_full_grid.py`

## 6. 配置改动

### 6.1 固定实验范围

- 数据集：Cora、CiteSeer、Coauthor-CS、CoraFull、Amazon-Photo；
- 噪声设置：uniform 0.2、uniform 0.4、uniform 0.6、pair 0.4、instance 0.4；
- seeds：0、1、2、3、4；
- 方法：
  1. `frozengrace_gb_mlp_softdist_beta05` + `distribution_only`；
  2. `frozengrace_gb_mlp_softdist_beta05_shuffled` + `distribution_only`；
  3. `mlp_ce`；
  4. `mlp_jocor`。

### 6.2 固定协议

- 噪声只注入训练集；
- split 为 20%/10%/70%；
- 同一 dataset × regime × seed 的四个方法共享 split、noise realization 和污染计数；
- 两个 SoftDist 方法共享同一数据集的冻结 GRACE embedding；
- 真实球与 shuffled 球各自记录 assignment hash；
- 设备固定为 `cuda:0`。

### 6.3 确定性修订

第一次代表性运行暴露多线程 KMeans/BLAS 归约导致的构球不确定性。正式网格在每个训练子进程启动前固定：

```text
PYTHONHASHSEED=0
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_DYNAMIC=FALSE
MKL_DYNAMIC=FALSE
```

该修订不改变数据集、噪声、seed、模型、阈值、统计量或完整网格范围。

## 7. 预期机制

- 非并列球：目标与历史实现逐元素一致；
- 并列球：删除最小索引 one-hot 分量，保留 \(p_j\) 和 \(\alpha_j\)；
- 类别置换后再逆置换：safe 模式的目标张量和 \(\alpha\) 恢复原值；
- 若真实性能信号来自局部成员关系，而非类别编号伪差，则真实球仍应优于随机分组；
- 不预设 `distribution_only` 与 `min_index` 等效，也不利用测试集选择 tie 模式。

## 8. 成功标准

- 数学单元测试与诊断持久化测试全部通过；
- 500/500 行完整且无 NaN/Inf；
- 125/125 个协议组通过；
- SoftDist 行的 tie mode、config、embedding path/hash、assignment hash 和 tie 诊断非空；
- 每个数据集的冻结 embedding 在全部 regime、seed 和两个 SoftDist 方法间一致；
- 真实球与 shuffled 球的 assignment 可追溯；
- 全部 stderr、失败行和缺失行不被隐藏；
- 只有上述条件全部满足时才生成聚合表和配对统计。

## 9. 失败标准

- 任一缺失、重复、失败或非有限指标；
- 同一比较组的 split、noise、污染计数或数据比例不一致；
- SoftDist 回退到非 `distribution_only`；
- 同一数据集的冻结 embedding 漂移；
- config、assignment 或 tie 诊断缺失；
- 自动用历史结果填补新网格；
- 使用本轮测试集结果事后改变候选方法、阈值、数据集或统计家族。

## 10. 实验结果

### 10.1 完整性与协议

| 检查项 | 结果 |
|---|---:|
| seed-level 计划行 | 500/500 |
| `status=ok` | 500/500 |
| 完成命令 | 50/50 |
| dataset × regime × seed 协议组 | 125/125 PASS |
| 聚合行 | 100/100 |
| 逐 seed 配对差 | 375/375 |
| 配对检验 | 75/75 |
| SoftDist `distribution_only` 行 | 250/250 |
| SoftDist assignment hash 非空 | 250/250 |
| 非空命令 stderr | 0/50 |
| 非空编排器 stderr | 0/2 |

每个数据集的 feature/ball embedding path 和 hash 均只有一个唯一值。125/125 个 dataset × regime × seed 单元中，真实球和 shuffled 球的 assignment hash 均不同；这符合两种分组定义不同，同时各自行级可追溯的预期。

### 10.2 主要比较

| 对照 | 25 设置平均差 (pp) | 正/负设置 | 至少 4/5 seeds 正向 | 25 项族内 Holm | 单一 75 项 Holm |
|---|---:|---:|---:|---:|---:|
| shuffled | +7.8410 | 25/0 | 25/25 | 25/25 | 22/25 |
| MLP+CE | +7.2517 | 24/1 | 24/25 | 24/25 | 20/25 |
| JoCoR | +2.9638 | 23/2 | 23/25 | 8/25 | 8/25 |

三个高噪声/结构化设置（uniform 0.6、pair 0.4、instance 0.4）跨 5 个数据集形成 15 个单元。在这 15 个单元中，GB-LCSS 相对三个对照均为 15/15 平均差正向，并且均有至少 4/5 seeds 正向。这个计数只描述已测单元，不证明噪声越强增益必然越大。

### 10.3 Uniform 0.4 主表子集

| 数据集 | MLP+CE | JoCoR | shuffled | GB-LCSS |
|---|---:|---:|---:|---:|
| Cora | 44.24±2.62 | 51.72±1.30 | 43.64±1.39 | 56.28±1.54 |
| CiteSeer | 43.50±3.02 | 49.91±3.37 | 42.90±2.33 | 53.52±2.71 |
| CoraFull | 31.98±0.98 | 35.70±0.32 | 32.15±0.76 | 37.30±0.57 |
| Coauthor-CS | 88.42±0.62 | 89.38±0.61 | 87.65±1.01 | 90.62±0.47 |
| Amazon-Photo | 73.90±1.80 | 79.57±0.53 | 72.71±0.48 | 81.33±0.70 |

数值为 5 个 seed 的 mean ± sample std，单位为百分比。

### 10.4 非正向单元

- 相对 MLP+CE：Coauthor-CS uniform 0.2，平均差 −0.0655 pp，3/5 seeds 正向，未校正 \(p=0.8555\)；
- 相对 JoCoR：
  - Coauthor-CS uniform 0.2，平均差 −0.4956 pp，0/5 seeds 正向，未校正 \(p=0.1121\)；
  - CoraFull uniform 0.2，平均差 −0.1184 pp，2/5 seeds 正向，未校正 \(p=0.7310\)。

这些差值均未通过 25 项家族内 Holm 校正。它们不能被描述成等效，也不能从正文中删除。

### 10.5 Tie 暴露量

下表只统计真实球 GB-LCSS 的 25 个行级设置：

| 数据集 | 平均 tie balls | 平均受影响训练节点 | 平均受影响训练节点比例 | 零暴露行 |
|---|---:|---:|---:|---:|
| Cora | 7.72 | 65.36 | 12.081% | 0/25 |
| CiteSeer | 11.68 | 96.04 | 14.442% | 0/25 |
| CoraFull | 8.20 | 241.08 | 6.091% | 0/25 |
| Coauthor-CS | 2.32 | 33.00 | 0.900% | 2/25 |
| Amazon-Photo | 6.52 | 76.04 | 4.970% | 0/25 |

Coauthor-CS 有 2 个设置没有 tie 暴露。其余数据集和设置的暴露量随 split、noise realization 和球内标签构成变化，因此不能用单个全局比例替代行级诊断。

真实球的 125 个行级设置合计包含 911 个 tie balls，其中 910 个 active；123/125 行存在 active tie。受影响训练节点合计 12,788/259,000，占 4.937%。按受影响节点对正 `alpha` 求和后计算的加权占比为 2.288%。随机分组出现更多 tie 或更大暴露量不代表其结构更有效，因其本身不保留真实成员关系。

### 10.6 独立统计复核

使用 SciPy `ttest_rel` 对 75 项配对检验独立重算：

- 最大绝对 \(t\) 差：\(1.42\times10^{-14}\)；
- 最大绝对 \(p\) 差：\(8.88\times10^{-16}\)；
- 超出容差的不一致：0/75；
- 独立 Holm 重算不一致：0/75。

训练运行时为 Python 3.12.10、PyTorch 2.9.1+rocm7.2.1，`cuda_available=true`，设备为 AMD Radeon RX 7700 XT。

## 11. Claim 分类

### 11.1 已被实验支持

- `distribution_only` 完整网格在冻结协议下可完整复现并通过行级证据门禁；
- 真实球在 25/25 个设置上高于 shuffled，且三个 25 项家族定义下均通过 Holm；
- 相对 MLP+CE 的改善覆盖 24/25 个设置，相对 JoCoR 的改善覆盖 23/25 个设置；
- 多重比较家族定义会改变显著计数，但不改变效应量和逐 seed 方向；
- tie 暴露不是纯理论边角情况，不同数据集的暴露程度差异明显。

### 11.2 仅为合理解释

- 真实球相对 shuffled 的优势与冻结表征中的局部成员关系包含有用标签信息一致；
- `distribution_only` 保留 \(p_j\) 可能是其相对 `alpha_zero` 更稳妥的原因；
- 高噪声设置中方向一致可能与局部共识减弱错误标签影响有关。

### 11.3 尚未验证

- 对未见节点的归纳泛化；
- 对真实世界、非合成标签噪声的稳健性；
- 对其他编码器、学生架构、图异配性范围和预训练 seed 的普遍稳健性；
- `distribution_only` 与历史 `min_index` 的等效性或非劣效性；
- 六个扩展数据集上的 tie-safe 全网格表现。

### 11.4 已被负结果削弱

- “GB-LCSS 在所有设置都优于 CE 或 JoCoR”；
- “未显著差异即可视为等效”；
- “粒球在任意数据集和噪声强度上都能提供有效 tie 软监督”；
- “历史 `min_index` 数值可以直接重标为 tie-safe 结果”。

## 12. 证据资产

运行目录：

`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_full_grid_20260727_v2_immutable`

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `raw_combined.csv` | 489,620 | `38EAA2536C5F76A561076CA035665A7A102138B5113FF18D74A7BA2D3239E2C2` |
| `completeness.csv` | 26,257 | `F511659324A56899A2B71BD43CBC3754E993A89281F570B207718499EAFCAD73` |
| `protocol_audit.csv` | 4,135 | `B6F9650887C5DB1C981CA0CCF61E2A00C7A01332627399138C2F07A8BC9C0133` |
| `aggregate.csv` | 11,074 | `4F15AB8EA637A4B808605213216ACBAFF25EBD7D2F67239E68D5BD4B98989597` |
| `paired_deltas.csv` | 93,723 | `AB8671EDC43EA82C80716B7B97208DEE51D78053979AC554732BC21678AE3852` |
| `paired_tests.csv` | 15,356 | `D5EFDF1901317AC7CA11150316356F0771BD1195F67741536B1F063A7F0C9ED4` |
| `command_manifest.csv` | 102,213 | `BD07F980BA2784461F30A4465D0A2557B5E8971F5508FAE2E7D49B87AFE7E2A6` |
| `immutable_raw_manifest.csv` | 13,326 | `6C659555EA8FC3A92DAC043DBB19BF105A2E24FB4D1EC69A5CA0B6A18E7D7EC1` |
| `input_snapshot_manifest.csv` | 9,744 | `2C8A3CDB773817E4F69C26589EF36F59F1BD9AA24E4D25608C7E9CA543F44B67` |
| `environment_fingerprint.json` | 1,340 | `B33370B2A283BB548901F57B5D6671F4E0BAEAE64E7B3253DFB1A1EC8D276A72` |
| `report.md` | 1,055 | `94696F5DF14F9C50F9240DB652F0274BB20279CF585320E439B976D075C8F5AD` |

冻结输入：

- protocol ledger SHA-256：`73CEC921C092EC398FC13EC4E1C4A85A43BC3792298F702930222E1524ADCB90`；
- 代表性实验计划 SHA-256：`FB9CCC612050FF8DD90322F62ACBB6A7439643D9CA45EE76D295CA69E432DAEF`；
- v2 确定性修订 SHA-256：`8A6D81B3AA71BB84E97EC6E90AFCC269F36E27A22EC6BA36616DC3D962A82E20`；
- v2 冻结完整网格 runner SHA-256：`DA7A25CFA251756780EE04670241988EE4AE8B4D9C7A46242E6DA74AE002C7EC`。

### 12.1 恢复运行谱系边界

旧版 runner 在恢复时会重写 `input_snapshots`、`input_snapshot_manifest.csv` 和根 `environment_fingerprint.json`。首次环境指纹已经从本任务持久化会话记录恢复：

- 首次采集：`2026-07-27T12:31:45.827246+00:00`；
- 首次 Git scoped-diff SHA-256：`21881DFCEAB9F6DD9C81D9290C49C7A1A8FB402B929FEA487EFDF814A9EB9BD2`；
- 恢复采集：`2026-07-27T15:02:59.211685+00:00`；
- 恢复 Git scoped-diff SHA-256：`F618673368FA2948E54BD7AA65F2C99152307B9EF968BE37732430EDE8852C03`。

根 `environment_fingerprint.json` 保留恢复运行内容，没有被静默替换。目录另存 `environment_fingerprint.initial_recovered.json`、`environment_fingerprint.resume_20260727_230258.json` 和 `lineage_amendment_20260727.md`。27 个当前输入源、快照和清单哈希三方一致；这些哈希与覆盖前会话记录一致。v1/v2 的 86 个确定性公共字段逐行比较为 0 差异，50 条命令去除输出路径后完全一致。

因此，可以声称 v2 的不可变原始结果、数值和统计通过交叉复核，且未发现恢复前后执行输入变化；不得声称整个目录的所有谱系元数据从首次启动后从未被覆盖。修复后的 runner 对既有快照只校验、不覆盖，发现任一源文件或快照漂移即中止，并把恢复指纹写入带时间戳的新文件。
