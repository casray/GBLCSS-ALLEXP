# GB Fragmentation Salvage Diagnostic — Codex 执行任务

## 背景

你要在一个图级噪声标签分类实验项目中执行一次 GB（粒球）碎片化修复诊断。

项目位于 `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graphcl_noisy_graphcls`。
Python 路径：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`

### 已有成果
- GIN + GraphCL 对比学习训练好了 NCI1 seed0 的 graph embedding（checkpoints/embeddings/NCI1/pretrain0/H.pt）
- split + noisy labels 已缓存（artifacts/NCI1/seed0_noise_symmetric_0p4/split.pt）
- Stage 1b 证明 reweight 有效（rw_noball = rw_kmeans = 0.6545 > CE = 0.6302）
- 但 rw_gb = rw_shuffled = 0.6448 < rw_noball，GB 结构当前零贡献
- 原因：GB 碎片化严重，780 个球平均大小 4.2，consensus 无统计意义

### 你的任务
用更大球的 GB 配置（gb_relaxed: purity_threshold=0.6, min_samples=10, max_size=200）重跑 reweight 的最小矩阵，看 GB 结构是否有被碎片化掩盖的信号。

**只做一次，只跑一个变体，失败就停。**

## 请先读取

1. `05_gb_fragmentation_diagnostic_plan.html` — 完整诊断计划，含预注册判定标准
2. `src/gb_graphcl/granular_ball.py` — `build_granular_balls()` 函数，参数 purity_threshold、min_samples、max_size
3. `src/gb_graphcl/reliability.py` — `compute_reliability()` 统一接口
4. `src/gb_graphcl/selection.py` — `train_with_reweight()` 函数（已实现）
5. `scripts/run_stage1b_smoke.py` — 参照结构
6. `configs/default.yaml` — 不修改此文件

## 执行步骤

### Step 1: 写 `scripts/run_gb_frag_diagnostic.py`

这是唯一需要新建的文件。脚本逻辑：

```
用法: python scripts/run_gb_frag_diagnostic.py --dataset NCI1 --noise-type symmetric --noise-rate 0.4 --seed 0

流程:
1. 加载 embeddings (H.pt) 和已缓存的 split + noisy labels (artifacts/)
2. 构建两种 GB:
   - gb_orig: purity_threshold=0.7, min_samples=5, max_size=100  (来自 configs/default.yaml)
   - gb_relaxed: purity_threshold=0.6, min_samples=10, max_size=200  (硬编码在脚本中)
3. 记录两种 GB 的统计: 球数、平均/最小/最大球大小、平均 purity
4. 计算 reliabilities（全部基于 gb_relaxed 的匹配参数）:
   - "none": 全零（rw_noball 用）
   - "kmeans": n_clusters = gb_relaxed 球数
   - "gb_orig": 用 gb_orig 的 group_ids
   - "gb_relaxed": 用 gb_relaxed 的 group_ids
   - "shuffled_relaxed": 用 shuffle_group_ids(gb_relaxed 的 group_ids, seed)
   - "random_relaxed": 用 random_group_ids(len(train_idx), gb_relaxed 的 sizes, seed)
5. 跑 6 个 rw_* 方法（全部 trace=True）:
   - rw_noball: reliability = none
   - rw_kmeans: reliability = kmeans
   - rw_gb_orig: reliability = gb_orig
   - rw_gb_relaxed: reliability = gb_relaxed
   - rw_shuffled_relaxed: reliability = shuffled_relaxed
   - rw_random_relaxed: reliability = random_relaxed
   全部使用 train_with_reweight()，参数:
     w_min=0.1, warmup=20, ramp=20, 其他从 configs/default.yaml 读取
6. 保存到 reports/gb_frag_diag_{dataset}_{noise}_{rate}_seed{seed}/:
   - results.csv (method, test_acc, selection_precision, noise_exclusion_recall)
   - gb_stats.json (包含 gb_orig 和 gb_relaxed 两组统计)
   - traces/trace_{method}.csv (每个方法)
   - report.md
7. 打印报告
```

### Step 1 的报告格式

report.md 内容：

```markdown
# GB Fragmentation Diagnostic: {dataset} {noise} {rate} seed{seed}

## GB Statistics
### gb_orig (purity=0.7, min=5, max=100)
- num_balls: ...
- mean_size: ...
- min_size: ...
- max_size: ...
- mean_purity: ...

### gb_relaxed (purity=0.6, min=10, max=200)
- num_balls: ...
- mean_size: ...
- min_size: ...
- max_size: ...
- mean_purity: ...

## Ranked Test Accuracy
- method: test_acc=..., precision=..., noise_exclusion=...
  (重复 6 行，从高到低排序)

## Pre-registered Deltas
- rw_gb_relaxed - rw_noball: ... (PASS if > 0, FAIL if ≤ 0)
- rw_gb_relaxed - rw_shuffled_relaxed: ... (PASS if > 0.005, FAIL if ≤ 0.005)
- rw_gb_relaxed - rw_random_relaxed: ... (PASS if > 0, FAIL if ≤ 0)
- rw_gb_relaxed - rw_kmeans: ... (PASS if ≥ 0, FAIL if < -0.005)

## Fragmentation Repair Delta
- rw_gb_relaxed - rw_gb_orig: ... (positive = repair helped)

## Verdict
PASS: all 4 pre-registered checks passed / FAIL: check X failed — [reason]
```

### Step 1 关键实现细节

对 reliability 计算：
- `compute_reliability("gb", ...)` 需要 `group_ids` kwarg — 传 gb_relaxed 的 group_ids
- `compute_reliability("shuffled", ...)` 需要 `group_ids` 和 `seed` kwargs
- `compute_reliability("random", ...)` 需要 `group_sizes` 和 `seed` kwargs
- `compute_reliability("kmeans", ...)` 需要 `n_clusters` kwarg — 设为 gb_relaxed 的球数
- `compute_reliability("none", ...)` 无额外 kwargs
- gb_orig 的 reliability 不通过 `compute_reliability("gb", ...)` 的第二次调用获得——直接用 gb_orig 的 group_ids 调 `gb_consensus_reliability(group_ids_orig, noisy_labels, train_idx)`（从 reliability.py 导入）

对 `train_with_reweight()` 调用：
- 已有函数在 `src/gb_graphcl/selection.py`，签名见文件
- 传入 reliability tensor（对应方法的 consensus score）和公共参数
- trace=True

gb_stats.json 格式：
```json
{
  "gb_orig": { "num_balls": ..., "mean_size": ..., "min_size": ..., "max_size": ..., "mean_purity": ... },
  "gb_relaxed": { "num_balls": ..., "mean_size": ..., "min_size": ..., "max_size": ..., "mean_purity": ... }
}
```

### Step 2: 跑脚本

```
cd C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graphcl_noisy_graphcls
C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe scripts/run_gb_frag_diagnostic.py --dataset NCI1 --noise-type symmetric --noise-rate 0.4 --seed 0
```

### Step 3: 停在这里

输出报告后停止。不要自行跑更多 GB 变体、更多 seed、或其他数据集。

## 硬约束

1. **只新建一个文件**：`scripts/run_gb_frag_diagnostic.py`。不修改任何已有源码。
2. **不修改 `configs/default.yaml`**。gb_relaxed 的参数（purity_threshold=0.6, min_samples=10, max_size=200）和 w_min=0.1 在脚本中硬编码并注释。
3. **Clean labels 只用于 val/test 评估和事后诊断**。不进入 reliability、loss、target。
4. **所有方法同 frozen embedding、同 split、同噪声实例**。
5. **不做参数搜索**。只跑 gb_relaxed 一个变体。
6. **数字来自 CSV，禁止手填。**
7. **如果 rw_gb_relaxed ≤ rw_noball 或 rw_gb_relaxed ≈ rw_shuffled_relaxed（差 < 0.5pp），verdict 必须写 FAIL，不能通过换指标或删对照挽救。**
