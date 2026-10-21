# Stage 1b 执行任务 — Selection 诊断 + 双路线对比冒烟

## 背景

你要在一个图级噪声标签分类的实验项目中实现新的去噪转化机制。

项目位于 `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graphcl_noisy_graphcls`。
Python 路径：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`

### 已有成果
- GIN + GraphCL 对比学习训练好了 graph embedding（NCI1 seed0）
- 在 frozen embedding 上用 commit-style sample selection（hard drop 低信任样本）做去噪分类，跑了 9 个方法的冒烟
- **关键发现：所有 selection 方法全部输给 CE baseline（约 −2.7pp）**
- 原因已诊断：rho=0.6 扔掉 40% 训练样本后，线性分类器泛化崩溃（val_acc 在 ramp 阶段掉 5pp+），而 selection precision 只比随机高 8pp，不足以补偿样本量损失

### 你的任务
实现两条新的去噪转化路线，在 NCI1 seed0 上冒烟，看哪条能让去噪方法至少不输 CE：
- **路线 A（Reweight）**：不扔样本，用 reliability 引导的权重降低不可信样本的 loss 贡献
- **路线 B（Label Smoothing）**：不扔样本，用邻域投票分布平滑 one-hot noisy target

## 请先读取

1. `03_stage1b_execution_plan.html` — 完整执行计划，包含所有函数签名、算法细节、测试规格
2. `src/gb_graphcl/selection.py` — 已有的训练函数（特别是 `_train_loop`、`train_with_selection`、`_ramp_progress` 如果有的话、`_z_score`、`_result`）
3. `src/gb_graphcl/reliability.py` — 已有的 reliability 接口
4. `src/gb_graphcl/granular_ball.py` — GB 构建（`build_granular_balls`、`shuffle_group_ids`、`random_group_ids`）
5. `scripts/run_smoke.py` — 已有冒烟脚本结构（新脚本参照这个）
6. `scripts/run_selection_diagnostic.py` — 诊断脚本（参考 trace 的用法）
7. `tests/test_selection.py` — 已有测试风格
8. `configs/default.yaml` — 超参配置
9. `reports/smoke_NCI1_symmetric_0p4_seed0/results.csv` — Stage 1 结果（参考用）

## 执行步骤（严格按顺序）

### Step 1: 修改 `src/gb_graphcl/selection.py`

新增以下函数（不修改已有函数，除了可以复用 `_train_loop` 的内部逻辑）：

**`_ramp_progress(epoch, warmup, ramp) → float`**
- epoch < warmup: 0.0
- warmup ≤ epoch < warmup+ramp: (epoch - warmup) / ramp
- epoch ≥ warmup+ramp: 1.0

**`train_with_reweight(embeddings, noisy_labels, clean_labels, train_idx, val_idx, test_idx, reliability, num_classes, device, epochs=200, lr=0.01, wd=5e-4, patience=30, warmup=20, ramp=20, w_min=0.1, seed=0, trace=False)`**

核心逻辑（每 epoch）：
1. `logits = Linear(embeddings[train_idx])`
2. `per_sample_loss = CE(logits, noisy_labels[train_idx], reduction="none")`
3. `trust = _z_score(-per_sample_loss) + _z_score(reliability)`（如果 reliability 全零则只用 loss 项）
4. `w_raw = (trust - trust.min()) / (trust.max() - trust.min() + 1e-12)`
5. `w = w_min + (1 - w_min) * w_raw`（映射到 [w_min, 1.0]）
6. `progress = _ramp_progress(epoch, warmup, ramp)`
7. `actual_w = (1.0 - progress) * 1.0 + progress * w`（warmup 阶段全 1.0）
8. `loss = (actual_w * per_sample_loss).sum() / actual_w.sum()`
9. val early stopping，同 `_train_loop`

返回与 `_train_loop` 相同结构的 dict。trace 中记录与 `_train_loop` 相同的字段。
selection_precision 定义为：权重 > 0.5 的样本中 clean 的比例。
noise_exclusion_recall 定义为：noisy 样本中被降权（权重 < 0.5）的比例。

**注意**：不要复用 `_train_loop`，因为 reweight 的 loss 计算方式不同（加权 mean 而非 selection + mean）。建议写一个独立的训练循环，但复用 `_set_seed`、`_init_linear`、`_accuracy`、`_z_score` 等已有辅助函数。

**`train_with_smoothing(embeddings, noisy_labels, clean_labels, train_idx, val_idx, test_idx, soft_targets, num_classes, device, epochs=200, lr=0.01, wd=5e-4, patience=30, warmup=20, ramp=20, seed=0, trace=False)`**

核心逻辑（每 epoch）：
1. `logits = Linear(embeddings[train_idx])`
2. `one_hot = F.one_hot(noisy_labels[train_idx], num_classes).float()`
3. `progress = _ramp_progress(epoch, warmup, ramp)`
4. `target = (1.0 - progress) * one_hot + progress * soft_targets`
5. `target = target / target.sum(dim=-1, keepdim=True)`（确保归一化）
6. `log_prob = F.log_softmax(logits, dim=-1)`
7. `loss = -(target * log_prob).sum(dim=-1).mean()`
8. val early stopping，同 `_train_loop`

返回 dict，与其他训练函数结构一致。selection_precision 和 noise_exclusion_recall 置为 0.0（不适用）。
trace 中额外记录 `target_entropy`: `-(target * (target + 1e-12).log()).sum(dim=-1).mean().item()`。

### Step 2: 修改 `src/gb_graphcl/reliability.py`

新增函数：

**`compute_soft_targets(method, embeddings, noisy_labels, train_indices, num_classes, **kwargs) → Tensor[num_train, num_classes]`**

对每个 train 样本，统计其邻域内 noisy label 的投票分布（包含自身）。

- `method="topk"`: 对每个样本找 k 近邻（含自身，限 train set），统计 label 出现频率。kwargs: `k`
- `method="kmeans"`: 在 train embeddings 上做 kmeans，统计同 cluster 内 label 分布。kwargs: `n_clusters`
- `method="gb"`: 统计同球内 label 分布。kwargs: `group_ids`
- `method="shuffled"`: 用 `shuffle_group_ids(group_ids, seed)` 后统计。kwargs: `group_ids`, `seed`
- `method="random"`: 用 `random_group_ids(num_samples, group_sizes, seed)` 后统计。kwargs: `group_sizes`, `seed`
- `method="none"`: 返回 `F.one_hot(noisy_labels[train_indices], num_classes).float()`

实现提示：对 group-based 方法（gb/shuffled/random），遍历每个 group，统计组内 label 的 count → 归一化 → 广播给组内所有成员。对 topk，用 `torch.cdist` 或 `sklearn NearestNeighbors` 找 k 近邻后统计。对 kmeans，复用 `kmeans_consensus_reliability` 中已有的 kmeans 逻辑。

### Step 3: 写测试

**`tests/test_reweight.py`**
```python
# test_reweight_w_min_one_equals_ce:
#   w_min=1.0 时 reweight 应退化为 CE。
#   用 MUTAG 小数据集或随机 embedding (dim=16, 50 samples)，
#   比较 train_with_reweight(w_min=1.0, ...) 和 train_ce_baseline(...)
#   的 test_acc_at_best_val 是否一致（容差 0.02，因为浮点数精度差异）。
#   注意：seed 必须一致。

# test_reweight_warmup_is_uniform:
#   warmup 阶段所有权重 = 1.0。用 trace=True，
#   检查 epoch 0 的 selected_count == len(train_idx)。（selected_count 表示全样本参与）
```

**`tests/test_smoothing.py`**
```python
# test_soft_targets_sum_to_one:
#   用随机 embedding + 随机 labels 调 compute_soft_targets("gb", ...),
#   assert 每行 sum ≈ 1.0 (atol=1e-5)

# test_soft_targets_none_is_one_hot:
#   compute_soft_targets("none", ...) 的结果 == F.one_hot(noisy_labels[train], num_classes).float()

# test_smoothing_with_one_hot_equals_ce:
#   soft_targets = one_hot(noisy_labels) 时，train_with_smoothing ramp 后
#   target 仍然是 one_hot，所以应该和 CE 行为接近。
#   比较 test_acc 差 < 0.02。
```

测试风格参照已有的 `tests/test_selection.py`（不用 pytest，用函数 + run_unit_tests.py 调用）。在 `tests/run_unit_tests.py` 中注册新测试。

### Step 4: 写 `scripts/run_stage1b_smoke.py`

参照 `scripts/run_smoke.py` 的结构。主要差异：
- 多了路线 A（rw_*）和路线 B（sm_*）的方法
- 所有方法都用 trace=True
- trace CSV 保存到 `reports/stage1b_smoke_.../traces/` 子目录
- report 格式化时分三组报告：baselines、路线 A、路线 B
- w_min = 0.1 硬编码
- GB/kNN/kmeans 的配置直接从 configs/default.yaml 读取，不改配置文件

关键实现细节：
- reliabilities 复用 `compute_reliability()`（已有函数）
- soft_targets 用新的 `compute_soft_targets()`
- k = GB 平均球大小（matched），n_clusters = GB 球数（matched）——与 run_smoke.py 一致
- 同一份 split + noise + embeddings

方法列表（15 个）：
```
ce, sce, clean_oracle,
rw_topk, rw_kmeans, rw_gb, rw_shuffled, rw_random, rw_noball,
sm_topk, sm_kmeans, sm_gb, sm_shuffled, sm_random, sm_noball
```

报告内容：
- 15 个方法按 test_acc 排序
- headroom = clean_oracle - ce
- route_a_best vs ce delta
- route_b_best vs ce delta
- rw_gb vs rw_topk, rw_gb vs rw_kmeans
- sm_gb vs sm_topk, sm_gb vs sm_kmeans
- sm_gb vs sm_shuffled, sm_gb vs sm_random（结构有效性）
- rw_gb vs rw_shuffled, rw_gb vs rw_random（结构有效性）
- GB 统计（球数、平均大小、purity，同 run_smoke.py）

### Step 5: 跑测试

```
cd C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graphcl_noisy_graphcls
C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe tests/run_unit_tests.py
```

全绿才继续。如果测试有 import 问题，检查 sys.path 是否包含 `src/`。

### Step 6: 跑 Stage 1b 冒烟

```
C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe scripts/run_stage1b_smoke.py --dataset NCI1 --noise-type symmetric --noise-rate 0.4 --seed 0
```

预期输出在 `reports/stage1b_smoke_NCI1_symmetric_0p4_seed0/`。

### Step 7: 停在这里

输出报告后停止。不要自行扩 5 seed、跑 PROTEINS、或做 GB 碎片化 ablation。

## 硬约束

1. **Clean labels 绝不进入 reliability / soft_targets / weight / loss / training target。** 只用于 val/test 评估和事后诊断。
2. **不修改已有函数**。`_train_loop`、`train_with_selection`、`train_ce_baseline`、`train_sce_baseline`、`train_clean_oracle`、`compute_reliability` 等全部保持不变。新函数新增。
3. **不修改 configs/default.yaml**。新超参（w_min=0.1）在脚本中硬编码并注释。
4. **不做 hard same-ball positive、prototype pulling、hard inter-ball negative。**
5. **所有方法同 frozen embedding、同 split、同噪声实例、同 epochs/lr/patience。**
6. **test 一律 test_acc_at_best_val；数字来自 CSV。**
7. **新增代码必须补单测并保持全绿。**
