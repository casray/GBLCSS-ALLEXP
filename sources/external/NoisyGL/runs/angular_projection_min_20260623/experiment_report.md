# Frozen GRACE + Ball-Specific Angular Projection 最小实验记录

## 研究假设

在不修改 GRACE encoder、不使用标签训练 projection、不改变 split/noise/eval 协议的前提下，使用一个轻量 angular projection `H -> H'` 可能改善建球空间，使 SoftDist 的粒球 target 比原始 GRACE 建球更可靠。

## 粒球扮演的角色

粒球只作为 weak view relation teacher / soft structural constraint。`H'` 仅用于建球与 SoftDist target；下游 `[X||H]` 继续使用原始 GRACE `H`，避免把下游特征增强和建球空间变化混在一起。

## 方法动机

原始 GRACE embedding 对下游分类有用，但不一定是最适合递归建球的几何空间。本实验只测试是否可以在 frozen-H 上学习一个轻量角度空间，让 kNN 局部关系更集中，同时通过 anchor loss 限制其偏离原始 GRACE。

## 代码改动

- 新增 `scripts/train_angular_ball_projection.py`：训练 identity 初始化线性 angular projection，保存 L2-normalized `H'`。
- 修改 `predictor/FrozenGraceGB_GCN_Predictor.py`：新增 `gb.ball_embedding_path`，区分 `[X||H]` 的 feature embedding 和建球/SoftDist 的 ball embedding。
- 修改 `scripts/run_ssl_swap_downstream.py`：新增 `--feature_ssl_method` 和 `--ball_embedding_root`，CSV 记录 feature/ball 两条 embedding path。
- 修改 `scripts/linear_probe_embeddings.py`：让 `grace` 能回退到旧的 `ratio_train0p2_val0p1_split0_pretrain0/H.pt` 布局。
- 修改 `scripts/diagnose_embedding_geometry.py`：geometry noise 默认 train-only，并按 `GRACE - 0.005` 的 purity tolerance 做门控。
- 新增/更新测试：`tests/test_angular_ball_projection.py`、`tests/test_xh_features.py`、`tests/test_embedding_geometry_diagnostic.py`、`tests/test_linear_probe_embeddings.py`。

## 配置改动

- projection：`epochs=100, lr=1e-3, tau=0.5, k=10, knn_weight=0.05, anchor_weight=1.0, drop1=0.1, drop2=0.2, seed=0`
- 数据集：Cora, CiteSeer
- 协议：percent split `0.2/0.1/0.7`，train-only uniform noise `0.4`
- 几何 seeds：`0,1,2`
- device：`cuda:0`

## 预期机制

- same-node two-view angular InfoNCE：保持同一节点在 dropout 视图下稳定。
- frozen-H cosine kNN soft relation：保留原始 GRACE 的弱局部关系。
- anchor loss：限制 `H'` 偏离原始 `H`，防止 projection 仅追求 uniformity 或局部聚集导致语义退化。

## 成功标准

几何门控：

- `num_balls_mean >= 0.5 * GRACE_num_balls_mean`
- `true_all_purity_weighted_mean >= GRACE - 0.005`
- probe 只记录，不作为通过条件。

seed0 downstream 门控：

- `SoftDist(H'-balls, X||H) > shuffled(H'-balls, X||H)`
- `SoftDist(H'-balls, X||H) >= SoftDist(H-balls, X||H) - 0.005`

## 失败标准

若 Cora 和 CiteSeer 都未通过几何门控，则不进入 seed0 downstream，记录为负结果并停止。

## 实验结果

Projection 输出：

| dataset | output shape | final loss | final anchor loss | saved norm check |
|---|---:|---:|---:|---:|
| Cora | `(2708, 128)` | 7.2322 | 0.2480 | mean norm 1.0, max deviation 1.19e-7 |
| CiteSeer | `(3327, 256)` | 7.1476 | 0.1808 | mean norm 1.0, max deviation 1.19e-7 |

Geometry gate：

| dataset | GRACE balls | H' balls | GRACE purity | H' purity | num_balls_ok | purity_ok | pass |
|---|---:|---:|---:|---:|---|---|---|
| Cora | 52.3333 | 47.3333 | 0.7871 | 0.7741 | true | false | false |
| CiteSeer | 64.0000 | 61.3333 | 0.6991 | 0.6881 | true | false | false |

Probe 仅记录：Cora clean probe 从 0.8294 到 0.8313，小幅上升；CiteSeer clean probe 从 0.7109 到 0.7002，下降。该指标不作为 gate 条件。

## 最终决策

暂缓 / 放弃当前 `grace_angproj_knn10_anchor1` 版本进入下游 seed0。两个数据集都满足球数约束，但都未满足 purity gate，因此按计划停止，不运行 downstream。

## 证据文件

- `runs/angular_projection_min_20260623/projection_summary.csv`
- `runs/angular_projection_min_20260623/geometry/geometry_raw.csv`
- `runs/angular_projection_min_20260623/geometry/geometry_summary.csv`
- `runs/angular_projection_min_20260623/geometry/geometry_gate_decisions.csv`
- `runs/angular_projection_min_20260623/geometry/geometry_report.md`
