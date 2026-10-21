# Phase 40: 跨视角粒球冲突噪声检测 go/no-go 实验（2026-06-29）

## 研究假设

此前 Phase 39 及后续椭球、GEAF-style 实验反复说明：在 frozen GRACE 表征 `H` 上构造粒球，再让下游 GCN `[X||H]` 使用同源信息，粒球信号很容易退化为 H-space kNN 的粗粒化近似并被强 baseline 吸收。Phase 40 检验最后一个可能出口：粒球是否能捕捉 `X` 空间与 `H` 空间之间的跨视角局部矛盾，从而提供任一单空间 kNN 都拿不到的噪声信号。

## 粒球扮演的角色

本实验不训练 downstream，不构造 soft target。粒球只作为离线诊断器：

- 在原始特征 `X` 上构造一套 ordinary granular balls。
- 在 frozen GRACE `H` 上构造一套 ordinary granular balls。
- 对 train 节点分别计算两套球内 leave-one-out noisy-label 多数分布。
- 比较 `X` 球与 `H` 球的局部多数标签是否冲突。

clean label 只用于评测 AUROC/AUPRC/P@K，不进入任何 score 构造。

## 方法动机

如果粒球仍有图节点去噪价值，它必须提供“下游已有结构之外”的信息。单纯在 `H` 空间继续换球形状、调 alpha、做椭球或 GEAF-style hard correction，都没有改变同源吸收问题。跨视角冲突的价值在于：节点在 `X` 局部像 A 类、在 `H` 局部像 B 类，这类矛盾理论上不是单一 `H`-kNN 或 `X`-kNN 可以完整表达的。

## 代码改动

主要位于 `external/NoisyGL`：

- 新增脚本：`scripts/diagnose_cross_view_conflict.py`
- 新增测试：`tests/test_cross_view_conflict.py`
- 新增输出目录：`runs/cross_view_conflict_20260629/`
- 回填计划文档：`docs/execution_plan_cross_view_conflict_20260629.md`

脚本复用已有组件：

- `build_ordinary_granular_balls`：分别接收 dense `X` 与 frozen `H`。
- `_knn_suspicion_scores`：分别生成 `s_knn_X` 与 `s_knn_H`。
- 已有 label-noise diagnostic 的 AUROC/AUPRC/P@K、conditional AUC、safe metric 工具。

## 配置改动

无 downstream 配置改动。实验协议：

- 数据集 gate：`cora,citeseer,coauthorcs`
- 实际执行：`cora,citeseer` × seeds `0,1,2,3,4`
- noise：uniform `0.4`
- noise scope：train
- split：percent `0.2/0.1/0.7`
- kNN：`k=20`
- primary：`s_xview`, `s_xview_soft`
- 命门 baseline：`s_combo_2knn_max`, `s_combo_2knn_mean`

`coauthorcs` 未继续跑，因为 `cora` 与 `citeseer` 已经全部失败，而预注册 G1 要求至少 `2/3` 数据集通过；即使 `coauthorcs` 通过也无法改变全局 `STOP`。

## 预期机制

若假设成立，primary score 应满足：

- 在 `kNN-H` 认为干净的 50% train 子集里仍能抓真噪声。
- 在 `kNN-X` 认为干净的 50% train 子集里仍能抓真噪声。
- 在 `kNN-H ∩ kNN-X` 双干净交集里优于 `s_combo_2knn`。
- 冲突节点真噪声率应高于非冲突节点。

## 成功标准

G1 预注册标准：

- 某个 primary 在 `knnH_clean50` 与 `knnX_clean50` 上 mean 条件 AUROC 同时 `>=0.6`。
- 至少 `4/5` seeds 方向为正。
- 至少 `2/3` 数据集成立。
- 双干净交集里 `mean(primary - s_combo_2knn) >= 0.02`。

## 失败标准

- primary 在条件 AUROC 中接近随机或低于随机。
- primary 不能超过 `s_combo_2knn`。
- 冲突节点噪声率不高于非冲突节点。
- 低相关但无检测能力，即“非同源”不等于“有效”。

## 实验结果

测试：

```powershell
py -3 -m pytest tests/test_cross_view_conflict.py -q
```

结果：`5 passed`。

核心 gate 结果：

| Dataset | Detector | H-clean AUC | X-clean AUC | H∩X minus combo | H pass | X pass | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| Cora | `s_xview` | 0.4485 | 0.5179 | -0.2868 | 0/5 | 0/5 | FAIL |
| Cora | `s_xview_soft` | 0.4098 | 0.5266 | -0.2950 | 0/5 | 1/5 | FAIL |
| CiteSeer | `s_xview` | 0.4959 | 0.4900 | -0.3036 | 0/5 | 0/5 | FAIL |
| CiteSeer | `s_xview_soft` | 0.4878 | 0.4959 | -0.2788 | 0/5 | 0/5 | FAIL |

普通检测 AUROC 对比：

| Dataset | `s_knn_H` | `s_knn_X` | `s_combo_2knn_mean` | `s_xview` | `s_xview_soft` | `s_xview_disagree` |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.9361 | 0.8341 | 0.9408 | 0.4868 | 0.5090 | 0.6839 |
| CiteSeer | 0.8771 | 0.8058 | 0.8900 | 0.4916 | 0.4989 | 0.6796 |

冲突率与噪声率：

| Dataset | Conflict rate | Conflict noise | Non-conflict noise | Delta |
|---|---:|---:|---:|---:|
| Cora | 0.7242 | 0.3970 | 0.4287 | -0.0317 |
| CiteSeer | 0.7708 | 0.4036 | 0.4351 | -0.0314 |

## 最终决策

放弃。Phase 40 返回 `STOP`。

原因不是脚本失败，而是核心假设失败：`X/H` 粒球局部多数冲突没有抓到更多真实噪声，反而冲突节点噪声率低于非冲突节点；两个 primary 在条件 AUROC 上接近随机，并且在双干净交集里显著低于 `s_combo_2knn`。这说明跨视角粒球冲突虽然与单空间 kNN 低相关，但这种低相关没有转化为有效的噪声检测能力。

## 经验边界

本结果是粒球图节点去噪线的强封口证据之一：

- 只在 `H` 空间换球形状会被 `H`-kNN / GCN `[X||H]` 吸收。
- GEAF-style hard correction 有 offline 信号，但 downstream 不能稳定转化。
- 跨视角 `X/H` 冲突也没有提供可靠的非同源增量。

因此，后续不建议继续开新的粒球形状、粒球投票、椭球、GMM 或 X/H 硬冲突变体。更合理的论文定位是“图神经表示下粒球去噪的适用性边界”，把这一系列负结果作为可解释、可复现的边界刻画。
