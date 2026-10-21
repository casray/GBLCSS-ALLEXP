# Coverage-Aware Granular Ball Denoising for Graph Nodes

## 2026-06-22 Current Status Override

本节覆盖下方历史“下一步实验计划”：

- `C1_gated` 已完成并判负：soft gating 弱于 hard assignment，Phase B graph agreement 不再推进。
- 当前主方法已经迁移到 `external/NoisyGL` 中的 SoftDist `[X||H]` 管线。
- 2026-06-22 协议审计发现旧 NoisyGL standard/XH runner 实际污染全图标签并使用 noisy validation；由这些入口生成的历史对比结果全部降为 legacy。
- 最新工作是按 `0.2/0.1/0.7`、uniform 0.4、train-only noise 补充 TFR（CIKM 2025）、ProCon（IJCAI 2025 paper reimplementation）和 GD²（NeurIPS 2025 paper reimplementation）。
- CPU 与 ROCm smoke 均已通过；正式 12 配置 seed0 调参、冻结配置后的 5 seeds 主表和附录运行仍待执行。
- 当前执行入口：`external/NoisyGL/scripts/run_latest_baseline_experiments.py`。
- 当前输出目录：`external/NoisyGL/runs/latest_baselines_trainonly_20260622/`。

## 项目定位

本项目研究**图节点噪声标签鲁棒分类**。核心思路：GRACE 对比学习预训练得到高质量表示 → 在表示空间构建粒球 → 利用粒球的局部结构做噪声标签校正和正则化。

本仓库从 `gb_denoise_stage_b_compactness` 的 C1 ratio-split 成功实验快照而来。下一步目标是加入 coverage-aware gating 机制并与图去噪 baseline 对比。

---

## 核心算法：两阶段管线

```
Phase 1: GRACE SSL 预训练 (negfree_bn)
  - 输入: 图 (x, edge_index) + 噪声标签 (不参与 Phase 1)
  - 输出: encoder checkpoint + node embeddings H
  - 1000 epochs, lr=1e-3

Phase 2: GB 监督微调
  - 加载 Phase 1 encoder, unfreeze, 小 lr (1e-4)
  - 在 H 空间构建粒球 (gain_split)
  - 标签校正 + 结构正则化
  - 200 epochs, 每 20 epochs 刷新球
```

---

## 已验证的关键实验结果

### C1 ratio-split 5-seed 正式结果 (train_ratio=0.2, symm 0.4)

| 数据集 | C1 | C1_noball | C1_shuffled | C1_topk |
|--------|:---:|:---------:|:-----------:|:-------:|
| Cora   | **0.748±0.018** | 0.555±0.027 | 0.594±0.031 | 0.549±0.018 |
| CiteSeer | **0.571±0.026** | 0.424±0.011 | 0.447±0.018 | 0.411±0.019 |

Paired delta 全部 5/5 正，min_delta > 9pp。

### Coverage Sensitivity (Cora, seed 0)

| train_ratio | C1 | C1_noball | 增量 |
|:-----------:|:---:|:---------:|:----:|
| 0.05 (Planetoid) | ~0.589 | ~0.506 | +8pp (不稳定) |
| 0.1 | 0.634 | 0.507 | +12.7pp |
| 0.2 | 0.744 | 0.570 | +17.4pp |
| 0.6 | 0.814 | 0.615 | +19.9pp |

**结论**: 机制有效性随 train coverage 单调增长。Planetoid 标准 split (5.2%) 覆盖率不足。

---

## 已发现并修复的关键 Bug

### 1. 标签泄漏 (已修复)
- **问题**: `ball_majority_vote` / `ball_label_smoothing` / `knn_majority_vote` / `knn_label_smoothing` 统计全体球成员标签（含 val/test 干净标签）
- **影响**: C1 从 0.804 虚高降到 0.589
- **修复**: 四个函数加 `train_mask` 参数，只统计 train 节点标签
- **测试**: `test_ball_majority_vote_ignores_non_train_labels` 等 5 个防泄漏测试

### 2. L_inter 无梯度 (已修复)
- **问题**: `balls.centers` 在 `no_grad` + `detach` 下构造，L_inter 对 encoder 无梯度
- **修复**: 用 `scatter_mean` 从当前 `h` 动态计算可导中心
- **影响**: C1 从 0.589 升到 0.620

### 3. C1_topk 不公平 k (已修复)
- **问题**: kNN 的 k=10 固定，未匹配球粒度
- **修复**: 从参考球的平均 train 大小推算 `matched_k`

### 4. kNN 极小 train set 崩溃 (已修复)
- **问题**: `train_idx.numel() <= 1` 时 `actual_k=0`，topk 崩溃
- **修复**: `actual_k < 1` 时直接返回原标签

---

## 和 CGBC/GBmooer 的关系与差异化

### 当前 C1 和 CGBC/GBmooer 相似的部分
- 球内多数投票校正
- 球级 label smoothing
- intra compactness / inter separation
- encoder 适应粒球

### CGBC/GBmooer 成功但我们需要额外处理的原因
- CGBC/GBmooer 的数据集标签覆盖率 50-94% (TREC 91.6%, AGNEWS 94%, IMDB 50%)
- 所有样本都有标签，球内投票天然有效
- 图节点半监督场景下标签稀疏 (Planetoid 5.2%, ratio=0.2 也只有 20%)
- 需要显式的 coverage-aware 机制

### 差异化方向 (下一步实现)

**Coverage-Aware Gated Soft Correction**:
```
y_soft_i = (1 - α) · onehot(y_noisy_i) + α · p_B(i)
仅当 n_train(B_i) >= τ_cov 且 margin(B_i) >= τ_conf 时启用
```

**Graph-Supported Agreement Gate** (Phase B, 如果 Phase A 通过):
```
graph_prior_i = mean(onehot(y_noisy_j)) for j in 1-hop train neighbors
额外条件: argmax(graph_prior) == argmax(ball_vote) 才校正
```

---

## 代码结构

```
src/gb_stage_b/
  config.py      - StageBConfig dataclass, YAML 加载, 方法验证
  data.py        - load_graph (Planetoid/Synthetic), ratio split, noise injection
  models.py      - StageBModel (GCN encoder + projector + predictor)
  augment.py     - drop_edges, mask_features, make_two_views
  balls.py       - gain_split 粒球构造, reliable_ball_mask
  compactness.py - compactness_loss (L_intra)
  losses.py      - SSL losses, ball_majority_vote, ball_label_smoothing,
                   inter_ball_separation_loss, knn_*, SCE
  train.py       - run_experiment (Phase1 + Phase2), run_phase2_finetune
  eval.py        - linear_eval
  diagnostics.py - ball_diagnostics (purity, AUROC)
  report.py      - summarize_results
  cli.py         - CLI 入口, argparse
  utils.py       - 工具函数
```

---

## 方法命名约定

| 方法名 | 含义 |
|--------|------|
| `C1` | 真实球 + hard majority vote + label smoothing + intra/inter |
| `C1_noball` | 无球，只用 noisy CE 微调 |
| `C1_topk` | 用 kNN 替代球做 majority vote (matched k) |
| `C1_shuffled` | 随机打乱球分配 (结构对照) |
| `C1_gated` | (待实现) 真实球 + coverage gate + confidence gate + soft correction |
| `C1_graph_coverage` | (待实现) C1_gated + 图邻居 agreement gate |
| `C1_graph_only` | (待实现) 只用图邻居 soft prior，无球 |

---

## 运行环境

- **Python**: `/c/Users/Admin/AppData/Local/Programs/Python/Python312/python.exe`
  - 默认 `python` 是 Windows Store stub，不能用
- **运行方式**: `python -m gb_stage_b.cli run --config configs/xxx.yaml [options]`
  - 没有 `__main__.py`，必须用 `-m` 方式
- **工作目录**: 必须在项目根目录下
- **GPU**: 自动检测，无 CUDA 时 fallback CPU
- **测试**: `python -m pytest tests/ -q`

---

## 实验设置标准

| 参数 | 值 | 说明 |
|------|-----|------|
| split_mode | ratio | 不用 Planetoid 标准 split |
| train_ratio | 0.2 | 约 540 train nodes (Cora) |
| val_ratio | 0.1 | |
| noise_rate | 0.4 | symmetric noise on train only |
| seeds | 0, 1, 2, 3, 4 | 5-seed 正式实验 |
| Phase 1 epochs | 1000 | GRACE 预训练 |
| Phase 2 epochs | 200 | GB 监督微调 |
| Phase 2 lr | 1e-4 | 小学习率微调 |
| ball_refresh | 20 | 每 20 epochs 重建球 |

---

## 防泄漏红线

**绝对不能违反的规则**:

1. `ball_majority_vote` / `ball_label_smoothing` / 任何标签校正函数必须传入 `train_mask`，只统计 train 节点标签
2. `knn_majority_vote` / `knn_label_smoothing` 的候选邻居池必须限定为 train 节点
3. 新增的 `graph_neighbor_soft_prior` 也必须只看 train_mask 邻居
4. val/test 的干净标签只能用于 evaluation，永远不能进入 loss 计算或标签校正
5. 每新增一个涉及标签的函数，必须写对应的防泄漏单元测试

---

## 已否决的方向 (不要再尝试)

| 方向 | 否决原因 |
|------|---------|
| 伪标签扩充投票池 | precision 最高 75.8%，球会放大错误 |
| 纯 topology-only 球 | purity 高但 AUROC 接近随机 |
| hard same-ball positive / prototype pulling | 反复失败 |
| 多步图传播 (LP-style) | 40% noise 下多步传播会扩散噪声 |
| SCE 替代 CE 作为主 loss | 2x2 消融证明 SCE 无帮助 |
| Planetoid 标准 split 上的 SOTA claim | 5.2% 覆盖率下机制天然受限 |
| 纯 GAGB v1 graph-aware 建球 | fair controls 下没有稳定独立价值 |

---

## 下一步实验计划

详见 `next_experiments.html`。执行顺序：

1. **Phase A**: 实现 `C1_gated` + seed0 smoke → 验证 coverage-aware gating
2. **Phase B** (如果 A 通过): 实现 graph agreement gate + seed0 smoke
3. **Phase C**: NoisyGL baseline 对比 (NRGNN/RTGNN/CLNode)
4. **Phase D**: 5-seed 正式实验 + 论文表格

---

## 结果目录结构

```
results_ratio_formal_20260609/   - C1 5-seed 正式结果 (已完成)
results_ratio_c1_20260609/       - ratio coverage 诊断结果 (seed0, 3 个 train_ratio)
results_phase_a/                 - (待创建) Phase A smoke 结果
results_phase_b/                 - (待创建) Phase B smoke 结果
results_formal_final/            - (待创建) 最终正式实验结果
```

---

## 用户偏好

- **语言**: 所有回复必须用中文
- **风格**: 直接给结论，不要长篇解释
- **实验策略**: 先 seed0 smoke 验证，通过再扩 5 seeds
- **代码修改**: 最小改动原则，不要过度抽象
