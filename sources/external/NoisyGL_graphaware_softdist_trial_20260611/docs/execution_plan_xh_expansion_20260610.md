# 执行计划：[X||H] 分类器 + 球驱动无标签监督扩展

日期：2026-06-10
执行者：Codex
前置状态：E1（自适应粒度）已判负并关闭，见 `runs/adaptive_gb_seed0_rocm_20260610/adaptive_gb_e1_seed0_rocm_report_20260610.md`。本计划是新信息源路线：Phase 1 = 分类器吃 GRACE 表示，Phase 2/3 = 无标签节点监督扩展（先诊断后训练），Phase 4 = early-learning 双源裁决（暂缓，本文不实现）。

---

## 0. 环境与协议（必读，已踩过坑）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，device `cuda:0`
- **每个 single_exp 运行必须显式传 split 参数**，否则静默回退 Planetoid 标准 split（140 train，已经烧过一次）：
  ```
  --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7
  ```
- 统一协议：`--noise_type uniform --noise_rate 0.4 --seed 0`，seed0 smoke 通过后再谈扩 seeds
- 单元测试命令：`py -3 -m unittest discover -s tests`（当前 26/26 通过，新代码必须保持全绿）
- frozen GRACE embedding 路径（在 softdist config 里已有）：
  - Cora: `../../gb_graph_aware_minival_6_1/checkpoints/embeddings/Cora/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
  - CiteSeer: `../../gb_graph_aware_minival_6_1/checkpoints/embeddings/CiteSeer/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- 不要动 E1 的自适应建球代码（`ball_mode: adaptive`），本计划全部用 `geometric` 模式
- 注意 `gb:` config 块的键会通过 `_conf_value` 覆盖 predictor 类属性——每个对照必须在诊断 CSV 里核实控制字段真的不同

### ROCm seed0 基线（本计划的对照锚点）

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.7974 | 0.6435 |
| SoftDist static | 0.8042 | 0.6748 |
| SoftDist shuffled | 0.8047 | 0.6491 |
| NRGNN | 0.8311 | 0.7113 |

5-seed 参考：SoftDist Cora 0.7954±0.0094 / CiteSeer 0.6737±0.0113。seed0 差异 < 1pp 视为噪声内。

---

## Phase 1：[X || H] 分类器（先做，成本最低）

### 研究假设

GRACE 表示 H 目前只用于建球，分类器只吃原始特征 X。如果 H 对端到端分类本身有增量信息，拼接输入应该直接提升；这同时回答论文叙事的关键问题——对比表示是只适合建球，还是也能强化分类器。

### 实现

文件：`predictor/FrozenGraceGB_GCN_Predictor.py`

1. `_FrozenGraceGBGCNBase` 加类属性 `use_xh_features = False` 和 `skip_ball_targets = False`
2. `method_init` 中，在建 GCN **之前**：若 `use_xh_features`：
   - 用 `load_frozen_embeddings(embedding_path, expected_nodes)` 加载 H（embedding_path 从 `gb:` config 取）
   - H 逐行 L2 归一化（和建球时一致），`to(self.device)`
   - 先验证 `self.feats` 是 dense tensor（若是 sparse 需先 densify）；`self.feats = torch.cat([self.feats, H], dim=1)`
   - 用 `in_channels = self.feats.shape[1]` 建 GCN（不要用 conf.model['n_feat']）
3. `_make_corrected_labels` 中，若 `skip_ball_targets`：`self.corrected_label = self.noisy_label.clone()`，不建球、不产生 soft_target（即纯 noisy CE），diagnostics 记 `ball_mode: none`
4. 新增 predictor 类并在 `single_exp.py` 的 `PREDICTOR_MODULES` 和 argparse choices 注册：
   - `frozengrace_gb_gcn_softdist_xh`（use_soft_targets=True, use_xh_features=True）
   - `frozengrace_gb_gcn_softdist_xh_shuffled`（再加 random_control=True）
   - `gcn_xh_ce`（use_xh_features=True, skip_ball_targets=True）——归因对照：H 本身对纯 CE 的贡献
5. 配置：复制 `config/frozengrace_gb_gcn_softdist/` 改 method 名，三个方法 × cora/citeseer 共 6 个 yaml；`gcn_xh_ce` 的 yaml 也要保留 `gb.embedding_path`（加载 H 用）
6. 单元测试（新建 `tests/test_xh_features.py`）：
   - 拼接后特征维度 = X_dim + H_dim
   - `skip_ball_targets` 时 corrected_label 与 noisy_label 完全一致
   - H 节点数和图不匹配时报错

### 运行（6 个 seed0 run）

```
for ds in cora citeseer:
  for m in frozengrace_gb_gcn_softdist_xh frozengrace_gb_gcn_softdist_xh_shuffled gcn_xh_ce:
    single_exp.py --dataset $ds --method $m + 统一协议参数
```

日志放 `runs/xh_seed0_rocm_20260610/logs/`。

### 归因矩阵（报告必须包含）

| 对比 | 回答什么 |
|---|---|
| gcn_xh_ce vs GCN+CE (0.7974/0.6435) | H 对纯 CE 分类器的独立贡献 |
| softdist_xh vs softdist (0.8042/0.6748) | H 对球管线的贡献 |
| softdist_xh vs gcn_xh_ce | 在 XH 输入下球的增量是否仍在 |
| softdist_xh vs softdist_xh_shuffled | 球结构贡献的对照 |

### 通过门

- 保留 XH：`softdist_xh` 至少一个数据集比 `softdist` 高 ≥ 1pp 且另一个不降超过 0.5pp
- 失败处理：若两个数据集都无增量，记录"H 对 GCN 分类无增量信息"，Phase 3 不做 xh 组合，论文叙事收缩为"H 仅服务于球构造"

---

## Phase 2：无标签扩展 precision 诊断（不训练，必须先于 Phase 3）

### 目的

球多数票赋给无标签节点的伪监督，precision 够不够？不够就不训练，零成本止损。

### 实现

新脚本 `scripts/diagnose_ball_expansion.py`：

1. 按统一协议加载数据（percent split + uniform 0.4 + seed），加载 H，按 softdist 同参数建几何球：`build_ordinary_granular_balls(H, max_balls=64, min_ball_size=32, split_improvement=0.05, random_state=seed)`
2. 网格：`min_train_per_ball ∈ {5, 8, 10}` × `margin ∈ {0.5, 0.6, 0.7}`（margin = 球内 train noisy 票最大类占比）
3. 对每个网格点输出：
   - `coverage`：通过 gate 的球覆盖的无标签节点数 / 全部无标签节点数（无标签 = val+test）
   - `precision`：这些无标签节点中，球多数票 == 真实 clean label 的比例（**仅诊断用，clean label 不进任何训练**）
   - `shuffled_precision` / `shuffled_coverage`：同一网格点用 `build_random_size_matched_assignment` 的对照
4. seeds 0/1/2 都跑（纯诊断很便宜），输出 `runs/ball_expansion_diag_20260610/expansion_diag.csv` + 汇总 md

### 进入 Phase 3 的门槛（Codex 确认过的标准）

| 指标 | 门槛 |
|---|---|
| gated precision | 两个数据集都 ≥ 75%，≥ 80% 更好 |
| gated coverage | ≥ 10% 无标签节点，≥ 20% 更好 |
| real − shuffled precision | 差距明显（建议 ≥ 20pp），否则不是球结构贡献 |

任一不达标 → Phase 3 不做，写诊断报告说明哪个条件失败，停在这里等下一步决策。选 operating point 时在达标点里选 coverage 最大者，Cora/CiteSeer 可以各选各的，但报告里要写明。

---

## Phase 3：无标签扩展训练（仅当 Phase 2 过门）

### 实现

1. `utils/frozen_grace_targets.py` 新增：
   ```
   build_expansion_targets(noisy_label, assignment, train_mask, n_classes,
                           exp_min_train_per_ball, exp_margin)
     -> (expansion_mask: bool[N], expansion_target: float[N, C])
   ```
   - 票源只有 train noisy 标签；通过 gate 的球，其**无标签**成员进 expansion_mask，target = 球内 train 票归一化分布
   - train 节点永不进 expansion_mask（他们走 SoftDist target）
2. predictor 加 `use_expansion = False` 类属性；train loop 中：
   ```
   loss = soft CE(train, SoftDist目标)
        + lambda_u * soft_target_cross_entropy(output, expansion_target, mask=expansion_mask)
   ```
   lambda_u 从 `gb:` config 读（键 `lambda_u`），同时读 `exp_min_train_per_ball` / `exp_margin`（用 Phase 2 选的 operating point）
3. 新方法注册 + 配置：
   - `frozengrace_gb_gcn_softdist_exp` / `frozengrace_gb_gcn_softdist_exp_shuffled`
   - 若 Phase 1 通过：再加 `frozengrace_gb_gcn_softdist_xh_exp`（组合）
4. 单元测试（`tests/test_expansion_targets.py`）：
   - **防泄漏**：篡改 val/test 的 noisy/clean 标签，expansion_target 和 mask 必须不变（票源只能是 train）
   - gate 行为：低票/低 margin 球的成员不进 mask
   - train 节点不进 expansion_mask
   - lambda_u=0 时 loss 退化等于 SoftDist static
5. 诊断 CSV 每次运行记录：expansion_mask 节点数、coverage、训练用 operating point、（诊断）precision

### 运行（seed0）

`lambda_u ∈ {0.05, 0.1}` × {real, shuffled} × {cora, citeseer}，共 8 个 run；若 xh 通过再加 xh_exp 4 个 run。日志放 `runs/softdist_exp_seed0_rocm_20260610/logs/`。

### 通过门

- 两个数据集都 > SoftDist static（0.8042 / 0.6748），且 real > shuffled
- **只和 SoftDist static 比，这一阶段不和 NRGNN 比**
- 通过 → 下一轮再扩 5 seeds；失败 → 记录哪个 lambda/gate 失败，停下等决策

### 协议披露（报告必须写）

无标签扩展是 transductive 设定：val/test 节点以"球结构 + train-only 票"获得 soft target 参与训练，**它们的真实标签和 noisy 标签都不进 loss**。早停仍用 val noisy acc，机制不变。这一点论文里必须明示。

---

## Phase 1-3 结果备案（2026-06-10 已执行）

- Phase 1 过门：`gcn_xh_ce` Cora 0.8106 / CiteSeer 0.6890；`softdist_xh` 0.8058 / 0.6959；shuffled 0.7805 / 0.6753。**注意 Cora 上 softdist_xh < gcn_xh_ce（-0.5pp）**：分类器吃到 H 后，球的 train-target 校正在 Cora 无增量（信息冗余），CiteSeer 仍有 +0.7pp。
- Phase 2 过门：operating point `min_train=10, margin=0.5`，precision Cora 0.8754 / CiteSeer 0.8224。
- Phase 3 判负：expansion λ=0.05/0.10 全部低于 SoftDist static。失败原因是结构性的：球级 target 给同球全部成员相同分布，用 87% 精度的球恒定信号替换模型自身 ~81% 精度的逐节点分辨率预测，净亏。**无标签扩展永久关闭**（与 ball consistency 同病根，两次独立失败），不要再调 λ。

---

## Phase 5：5-seed 确认（先做，必须先于 Phase 4 任何机制工作）

### 目的

seed0 的 XH 差值都在噪声边缘（GCN+CE seed0 0.7974 vs 5-seed 0.7842）。必须先确认三件事：H 的增量稳不稳、Cora 上球在 H 之上是否真的无增量、real vs shuffled 是否保持。这直接决定论文主方法怎么写。

### 运行（30 个 run）

`{gcn_xh_ce, frozengrace_gb_gcn_softdist_xh, frozengrace_gb_gcn_softdist_xh_shuffled}` × `{cora, citeseer}` × `seeds 0-4`，统一协议参数。日志放 `runs/xh_5seed_rocm_20260611/logs/`。

### 已有 5-seed 锚点（ROCm，同协议 seeds 0-4，直接复用不重跑）

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| GCN+CE | 0.7842±0.0151 | 0.6530±0.0121 |
| SoftDist static | 0.7954±0.0094 | 0.6737±0.0113 |
| SoftDist shuffled | 0.7832±0.0167 | 0.6578±0.0116 |
| NRGNN | 0.8107±0.0230 | 0.7027±0.0108 |

### 汇总与报告要求

- mean±std + 逐 seed paired delta（同 seed 配对）：
  1. `gcn_xh_ce − GCN+CE`：H 对纯 CE 的稳定增量
  2. `softdist_xh − gcn_xh_ce`：H 之上球的增量（**关键问题**，Cora 预期可能 ≤ 0，如实记录）
  3. `softdist_xh − softdist`：H 对球管线的增量
  4. `softdist_xh − softdist_xh_shuffled`：球结构对照
- 输出 `xh_5seed_aggregate.csv` + 报告 md

### 判定

- H 增量确认：paired delta 1 在两个数据集 ≥ 4/5 seeds 为正 → XH 进主方法
- Cora 球增量判定：若 delta 2 在 Cora ≤ 0（多数 seeds）→ 备案"球的 target 校正价值是数据集条件性的"，这正是 Phase 4 要修的对象（CDR 拖累 Cora）；论文叙事写成"对比表示双重利用：H 进分类器 + 球做标签去噪"

---

## Phase 4：early-learning 双源裁决（Phase 5 之后做，先诊断后训练）

### 研究假设

SoftDist train target 的残余问题是 CDR ~0.22（球想翻标签时误伤干净标签），这正是 Cora 上球增量被 H 吸收后转负的最可能原因。early-learning 现象（DNN 先拟合干净标签、后记忆噪声）提供一个**与球投票独立**的噪声信号：warmup 期 GCN 对原标签的置信度。两源裁决：球想翻 + 早期模型也怀疑原标签 → 翻（保 NCR）；球想翻 + 早期模型强烈支持原标签 → 不翻（降 CDR）。

### 与已失败方向的本质区别（写进报告）

- vs dynalpha（失败）：早期信号在 warmup 一次性提取并冻结，之后丢弃 warmup 模型、**从零重训**主模型，无反馈回路
- vs NRGNN+BallAux（失败）：球 vs 图传播是冗余信号；球 vs 时间维度 early-learning 是独立信号，独立性是之前所有融合缺的东西

### Phase 4a：裁决诊断（不训练主模型）

新脚本 `scripts/diagnose_early_arbitration.py`：

1. 统一协议加载数据；建 warmup GCN（**特征用 Phase 5 确认的最佳配置**，XH 过门就用 XH），noisy CE 训练 `early_warmup_epochs=10`，收集 epochs 3-10 的 softmax 均值作为早期信号 `p_early`（窗口均值比单 epoch 稳；本协议 best epoch 在 3-6，记忆化很早，窗口不要超过 10）
2. 对 SoftDist 会施加校正的 train 节点（alpha>0 且 pseudo_label ≠ noisy_label），按保护规则分组：`p_early(y_noisy_i) ≥ tau_protect` → 保护（alpha_i 置 0）
3. 网格 `tau_protect ∈ {0.5, 0.6, 0.7, 0.8}`，对每个 tau 用 clean label（仅诊断）计算**预测的** NCR/CDR：
   - 被保护节点中实际是干净的比例（保护正确率）
   - 保护后剩余校正的 NCR / CDR（相对原 SoftDist 的变化）
4. seeds 0-2，两个数据集，输出 `runs/early_arbitration_diag_20260611/` CSV + md

### Phase 4a 进入训练的门槛

| 指标 | 门槛 |
|---|---|
| 预测 CDR 相对下降 | ≥ 30%（两个数据集都要） |
| 预测 NCR 相对损失 | ≤ 10% |
| 保护正确率 | ≥ 70%（被保护的节点确实多数是干净的） |

任一不达标 → 不训练，写诊断报告停下。选 operating tau：达标点中 CDR 降幅最大者。

### Phase 4b：裁决训练（仅当 4a 过门）

1. `utils/frozen_grace_targets.py`：`apply_ball_soft_targets` 加可选参数 `node_alpha_scale: Tensor | None = None`（逐节点 alpha 缩放，None 时行为完全不变——SoftDist anchor 不受影响，必须有回归测试）
2. predictor 加 `use_early_arbitration` 类属性：
   - `_warmup_early_signal()`：建同架构 fresh GCN，noisy CE 训 `early_warmup_epochs`，窗口均值 softmax → `p_early`，丢弃 warmup 模型
   - 构造 SoftDist target 时传 `node_alpha_scale`：gate 版 `scale_i = 0 if p_early_i(y_i) >= tau else 1`
   - 主模型**重新初始化**后用裁决后 target 训练（不沿用 warmup 权重）
   - config 键：`early_warmup_epochs`, `early_window_start`, `early_window_end`, `early_tau_protect`
3. 新方法注册：`frozengrace_gb_gcn_softdist_el` / `_el_shuffled`；若 XH 过门再加 `_xh_el` / `_xh_el_shuffled`
4. 单元测试（`tests/test_early_arbitration.py`）：
   - `node_alpha_scale=None` 时 target 与原 SoftDist 逐元素相等（anchor 回归）
   - 保护只作用于 train 节点
   - warmup 只用 noisy 标签（篡改 clean label 信号不变）
   - tau 单调性：更高 tau → 被保护节点数不增

### 运行与判定（seed0 smoke）

- `{el, el_shuffled}` ×（XH 过门则用 xh_el 系列）× 2 数据集，operating tau 用 4a 选定值
- 通过门：**Cora 上球增量转正**（`xh_el > gcn_xh_ce`）且 CiteSeer 不降；real > shuffled；诊断 CDR 实际下降与 4a 预测一致
- 通过 → 扩 5 seeds；失败 → 记录后停，论文走"数据集条件性"叙事

---

## Phase 5 / 4a 结果备案（2026-06-11 已执行）

- **Phase 5 过门，主线确认**：`softdist_xh` 5-seed Cora **0.8137±0.0067**（≥ NRGNN 0.8107±0.0230，std 小 3.4 倍），CiteSeer 0.6891±0.0159（NRGNN 0.7027，差 -1.36pp）。归因链全绿：H-on-CE +1.9/+2.3pp，ball-on-H +1.1/+1.3pp（Cora seed0 的负增量是误导，5-seed 为 4/5 正），real>shuffled 双数据集 5/5。**`softdist_xh` 是新的 anchor，后续所有实验不得改动其行为。**
- **Phase 4a 判负**：无 tau 同时过三门。Cora 保护节点太少（CDR 降幅最高 25%），CiteSeer 保护精度 48-68% < 70%。**已知设计缺陷**：warmup 用了 [X‖H]，早期信号通过 H 和球几何耦合，违反独立性假设——这正是 Phase 4a-2 要修的。

---

## Phase 4a-2：X-only warmup 裁决诊断（先做，10 分钟，只诊断不训练）

### 目的

对 Phase 4 核心假设的正确检验：早期信号必须和球几何**独立**。warmup 改用纯 X 特征（不拼 H），早期模型的置信度不再来自 H，和建在 H 上的球投票才是两个独立信号源。

### 实现

`scripts/diagnose_early_arbitration.py` 加参数 `--warmup_features {xh, x}`（默认 xh 保持现状）：
- `x` 模式：warmup GCN 输入用原始 `data.feats`，不加载 H、不拼接
- 其余协议完全不变：10 epochs noisy CE，epochs 3-10 softmax 均值，tau 网格 {0.5, 0.6, 0.7, 0.8}，seeds 0-2，两个数据集
- 注意：被诊断的 SoftDist 校正集合本身不变（球还是建在 H 上），只换早期信号的来源

### 门槛（与 4a 相同，不放宽）

CDR 相对降 ≥ 30%（双数据集）、NCR 损失 ≤ 10%、保护精度 ≥ 70%。

### 判定

- 过门 → 记录 operating tau，**但 Phase 4b 训练等 Phase 6 跑完再决定**（两者独立，先把论文必需的表拿到手）
- 不过门 → **Phase 4 永久关闭**，结论写法："early-learning 信号无论与球几何耦合（XH warmup）还是独立（X warmup）都不足以安全保护干净标签"，这是一个干净的机制边界负结果，可进论文
- 输出 `runs/early_arbitration_diag_xonly_20260611/` CSV + md

---

## Phase 6：noise grid 体制分析（本轮主体，论文必需表）

### 目的

uniform 0.4 是 NRGNN 的主场。高噪声下伪标签机制退化，粗粒度球投票恰好最抗噪；pair noise 下图传播易被同伴类污染。找出 `softdist_xh` 赢 NRGNN 的 regime——同时这些表本来就是论文必须有的。

### 一个写进报告的卖点

GRACE H 与噪声无关（label-free 预训练），所有噪声设置复用同一份 H 文件——表示阶段对标签噪声构造性免疫，这是和 NRGNN（全管线暴露在噪声下）的本质区别，regime 分析正是检验这一点。

### 网格（200 个 run）

- 噪声设置（4 个，uniform 0.4 已有不重跑）：`uniform 0.2`、`uniform 0.6`、`pair 0.2`、`pair 0.4`
- 方法（5 个）：`gcn`（GCN+CE）、`gcn_xh_ce`、`frozengrace_gb_gcn_softdist_xh`、`frozengrace_gb_gcn_softdist_xh_shuffled`、`nrgnn`
- 数据集 × seeds：{cora, citeseer} × {0-4}
- 统一参数照旧：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7 --device cuda:0`，只变 `--noise_type` / `--noise_rate`
- 日志 `runs/noise_grid_rocm_20260611/logs/`，文件名带 noise 设置：`{ds}_{method}_{ntype}{nrate}_seed{s}.log`

### 运行注意

- pair 0.4 接近 pair noise 可学习上限，预期所有方法都明显掉，这是正常现象，如实记录
- 每个 run 核对输出里的 `actual_noise_rate`，pair 模式下与名义值可能有偏差，写进 raw CSV
- NRGNN 比 GCN 慢得多，200 个 run 里它是时间主体；先跑完两个 GCN 系方法再跑 NRGNN，便于中途查看

### 汇总与报告要求

1. **主表**：每个噪声设置一个 mean±std 块（行=方法，列=数据集），含已有 uniform 0.4 锚点，共 5 个块
2. **paired delta**：`softdist_xh − nrgnn` 和 `softdist_xh − gcn_xh_ce` 逐 seed 配对，每个设置报 mean delta + positive seeds
3. **regime 胜负矩阵**：行=噪声设置，列=数据集，格子 = ours vs NRGNN 的 {win / tie / loss}（win 定义：mean delta > 0 且 ≥ 4/5 seeds 正；loss 对称；其余 tie）
4. **球增量随噪声的趋势**：`softdist_xh − gcn_xh_ce` 对噪声率作图数据（预期：噪声越高球增量越大，验证粗粒度投票的抗噪假设）
5. 输出 `noise_grid_aggregate.csv`、`noise_grid_paired_deltas.csv`、报告 md

### 判定

- 任一 regime 达成 win → 论文叙事升级为"分 regime 互有胜负，我们更简单、方差更小"，该 regime 进主表重点
- 全部 tie/loss → 论文叙事保持"uniform 0.4 下 Cora 持平且更稳、CiteSeer 略逊，方法显著更简单"，regime 表进分析章节
- 无论结果如何这些表都进论文，不存在白跑

### 执行顺序

1. Phase 4a-2（10 分钟）
2. Phase 6 的 GCN 系方法（gcn / gcn_xh_ce / softdist_xh / shuffled，~160 run 但单 run 几秒）
3. Phase 6 的 NRGNN（40 run，时间主体）
4. 汇总 + 报告
5. 若 4a-2 过门且 Phase 6 显示 CiteSeer 仍是短板 → 再决定是否做 Phase 4b

---

## 永久关闭清单（更新于 2026-06-10）

| 方向 | 证据 |
|---|---|
| 无标签扩展（球级伪监督） | Phase 3 双 λ 双数据集判负，球粒度分辨率不足 |
| ball consistency 正则 | seed0 无增量，同病根 |
| 动态 alpha（模型预测反馈） | 自强化，dynalpha/fast 双判负 |
| 自适应粒度（标签驱动分裂） | E1 三变体判负，20% 覆盖票密度不足 |
| NRGNN 杂交（BallAux） | 球与图传播信号冗余 |

---

## 报告要求（每个 Phase 一份 md）

按既有 research-record 格式：研究假设 / 粒球角色 / 代码与配置改动 / 预期机制 / 成功标准 / 失败标准 / 结果表 / 诊断表 / 决策。放对应 `runs/<phase>_*/` 目录。所有结论标注"seed0 smoke only"边界。

## 红线（不可违反）

1. clean label 只进诊断，永不进训练/gate/target
2. 票源永远只有 train_mask 内的 noisy 标签
3. 每个涉及标签的新函数必须配防泄漏单元测试
4. 对照（shuffled）与主方法唯一差异是 assignment，其余配置逐键相同
5. 不改 SoftDist static 的任何现有行为（它是对照锚点）
