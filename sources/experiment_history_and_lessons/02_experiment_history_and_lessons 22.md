# MLP-25 中等票数粗球 Holdout Smoke（2026-07-12）

## 研究假设

MLP-13 与 kNN 敏感性提示小图默认球每球约 10 个 train votes，可能处于高方差区；把平均支持量推向约 20 votes，可能在不造成全局单球塌缩的情况下提高 noisy-label 共识稳定性。

## 粒球角色

粒球继续作为 frozen GRACE H 中 label-blind 的局部共识单元。构球阶段不读标签值；球构造完成后只统计 train noisy labels，生成 beta=0.5 soft target。clean labels 不参与 target 或 gate。

## 五臂设计

为同时回答“原则化票数规则是否有效”和“MLP-13 候选能否泛化”，使用五臂而非共享 shuffled：

1. fixed：`64/32/0.05`；
2. vote-target：`B=min(64,ceil(|V_L|/20))`，其余保持 `32/0.05`；
3. vote-target shuffled；
4. 冻结 MLP-13 候选：`128/64/0.01`；
5. candidate shuffled。

两个候选的球规模不同，必须分别使用自己的 size-matched shuffled。holdout 为 CiteSeerFull、Cora_ML、Attributed-Wiki；uniform 0.4、train-only、20/10/70、seed0、ROCm GPU。

## 预注册通过门

每个候选独立判定：至少 2/3 数据集相对 fixed 提升 ≥1pp；任一数据集不得下降超过 0.5pp；至少 2/3 数据集相对自己的 shuffled 提升 ≥1pp；无塌缩、target coverage 非零、协议/hash 完整。seed0 通过也只允许进入 5-seed，不直接替换默认。

## 结果

| 数据集 | fixed | vote-target | Δ | vote shuffled | candidate | Δ | candidate shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| CiteSeerFull | 59.81 | 60.08 | +0.27pp | 46.71 | 60.69 | +0.88pp | 47.42 |
| Cora_ML | 51.91 | 53.15 | +1.24pp | 46.66 | 52.67 | +0.76pp | 46.80 |
| Attributed-Wiki | 65.60 | 60.31 | −5.29pp | 59.60 | 63.64 | −1.96pp | 60.67 |

实际支持量：vote-target 在三个数据集分别约 19.67、19.97、19.24 votes/ball；candidate 分别约 22.26、29.95、21.86 votes/ball。规则确实把票数推到了目标区，不存在“公式没有触发”的解释。

Gate：

- vote-target：FAIL；只有 1/3 数据集达到 +1pp，出现 >0.5pp 回退，平均 `−1.26pp`。
- candidate：FAIL；0/3 数据集达到 +1pp，出现 >0.5pp 回退，平均 `−0.11pp`。
- 两个 real 候选在多数/全部数据集高于自己的 shuffled，说明真实局部结构仍存在；失败的是跨数据集可泛化的 accuracy 增量。

## 最终决策

- 不运行 5-seed；不替换 fixed `64/32/0.05`；不重跑论文矩阵。
- 粒度轴双向关闭：MLP-24 说明继续细分大图无稳定收益，MLP-25 说明把小图统一推到约 20 votes 也不能稳定泛化。
- MLP-13 的 Cora/CiteSeer 正结果保留为机制/敏感性证据，但不能升级成统一默认。
- 论文安全结论应为：中等支持量能改善部分数据集的投票方差，但有效性仍取决于 H 空间的语义一致性；票数不是单独充分条件。

## 证据分级

### 已被实验支持

- vote-target 规则成功产生约 19–20 votes/ball。
- Cora_ML 有 +1.24pp 局部正信号，CiteSeerFull 只有弱正；Attributed-Wiki 显著下降。
- 冻结 MLP-13 候选未在三个时间顺序 holdout 上达到预注册泛化门。

### 合理解释

- 增加票数降低投票方差，但 Attributed-Wiki 的 H 局部语义与类别不充分对齐，更大的球增加跨类混合偏差，抵消方差收益。

### 已被负结果削弱

- 约 20 votes 是跨数据集统一最优工作点。
- MLP-13 的 `128/64/0.01` 可直接作为统一新默认。
- 小图统一调粗可以稳定带来 1–2pp。

## 关键证据

- 计划：`external/NoisyGL/docs/execution_plan_mlp25_coarse_holdout_smoke_20260712.md`
- 结果：`external/NoisyGL/runs/mlp25_coarse_holdout_smoke_20260712/`
- 报告：`mlp25_report.md`
- Gate：`mlp25_gate_decisions.csv`
- Raw：`mlp25_raw.csv`（15/15）
