# Frozen Open-Intent Local-Boundary Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to reproduce this plan task-by-task. Subagent execution is not used in the current workspace.

**Goal:** 在不修改第一篇论文资产的前提下，复现并审计冻结 BERT 表示上的开放意图强基线、真实粒球局部边界、等球数 KMeans 与 shuffled controls，并依据预设必要性门槛作 Go/No-Go 决策。

**Architecture:** 训练阶段只产生固定的 train/validation/test features、logits 与预测；离线阶段从同一 NPZ 重建真实粒球、KMeans 和 shuffled controls。所有阈值只由 known validation scores 决定，test unknown labels 只进入最终评测。Pilot 采用顺序 kill-test：seed0 的结构必要性失败后，不扩展模型 seeds 或联合训练。

**Tech Stack:** Python、PyTorch、Transformers、scikit-learn、NumPy、Pandas、SciPy、pytest；PowerShell 负责命令与文件审计。

---

## 0. 文档性质与状态

本文档在 2026-07-14 统计审计之后依据原始代码、`config.json`、NPZ、CSV、JSON 和日志重建，**不是结果产生前的正式预注册文件**。为避免 HARKing：

- 已由原始产物证明完成的步骤标记为 `[x]`；
- 因 G2 失败而取消的扩展步骤明确标记为 `[ ] CANCELLED BY G2`；
- 任何后验统计不得反写成事前成功标准；
- 原始结果以 run 目录为准，`pilot_report.md` 只负责解释，不替代原始证据。

最终执行状态：**No-Go**。本计划的作用是使已经执行的 open-set Pilot 可核查、可复算，并说明为何没有继续到 4 datasets × 5 model seeds 或联合训练。

## 1. 文件结构与责任边界

### 1.1 只读实现与测试

| 路径 | 责任 |
|---|---|
| `开放意图粒球新方向0630/code/run_oi1_smoke.py` | BANKING77 数据、BERT 训练、冻结 features/logits、基线、GB controls 与报告 |
| `开放意图粒球新方向0630/code/run_oi1b_stackoverflow.py` | StackOverflow 本地数据读取、同协议训练与报告 |
| `开放意图粒球新方向0630/code/oi_gb/data.py` | known/unknown 类别划分与 train/val/test 数据组织 |
| `开放意图粒球新方向0630/code/oi_gb/bert_intent.py` | BERT 分类器、训练和批量编码 |
| `开放意图粒球新方向0630/code/oi_gb/baselines.py` | MSP、Energy、Prototype、kNN、Mahalanobis、GEN、ReAct、近似 OpenMax/ViM 等固定表示分数 |
| `开放意图粒球新方向0630/code/oi_gb/gb_scores.py` | ordinary GB、KMeans、shuffled control 与粒球分量 |
| `开放意图粒球新方向0630/code/oi_gb/metrics.py` | known-validation 阈值、AUROC/AUPR/FPR95 和 open classification 指标 |
| `开放意图粒球新方向0630/code/oi_gb/conditionals.py` | ALL-safe 等条件子集评测 |
| `开放意图粒球新方向0630/code/offline_probe_generic.py` | 单个 NPZ 的 real/KMeans/shuffled 通用复算 |
| `开放意图粒球新方向0630/code/offline_probe_B_hparam.py` | BANKING77 的 7 组 GB 超参 × 3 KMeans seeds 稳健性检查 |
| `开放意图粒球新方向0630/tests/` | 数据、基线、指标、条件评测、GB 与 post-hoc 单元测试 |

上述路径在本审计中只读，不允许为获得更好结果而修改。

### 1.2 权威输入与输出

BANKING77 完整 run：

```text
开放意图粒球新方向0630/runs/gb_score_conditional_gain_20260630/
  banking77_r25_seed0_20260630_172920/
```

StackOverflow 完整 run：

```text
开放意图粒球新方向0630/runs/gb_score_conditional_gain_20260630/
  stackoverflow_r25_seed0_20260630_201825/
```

每个 run 必须保留：`config.json`、`features_logits_seed0.npz`、`score_metrics.csv`、`conditional_gain.csv`、`correlations.json`、`seed0_report.md`。

独立统计审计：

```text
next_paper/runs/frozen_open_intent_pilot_audit_20260714/
  offline_probe_log.md
  statistics.json
```

文档输出：

```text
next_paper/docs/experiment_plan.md
next_paper/docs/pilot_report.md
```

## 2. 固定实验协议

| 项目 | 固定设置 |
|---|---|
| encoder | `bert-base-uncased` |
| fine-tuning | 最后一层，3 epochs |
| max length | 64 |
| batch / eval batch | 64 / 128 |
| known ratio | 0.25 |
| model/data split seed | 0 |
| GB default | max balls/class 16；min ball size 8；split improvement 0.05 |
| 原始 KMeans controls | seeds 13、14、15 |
| 结构随机性审计 | real GB seeds 0–4；配对 KMeans seeds 13–17；配对 shuffle seeds 1013–1017 |
| threshold | known validation score 上目标 5% FPR 的分位数 |
| test labels | unknown 标记只用于生成 score 后的最终评测 |
| primary structural score | `dist+margin` |
| conditional score | MSP、Prototype、kNN 均判为相对安全的 ALL-safe 子集 |

### 2.1 预设 Gate

- G1 条件信息：ALL-safe 中真实 GB 必须仍有实质识别力，组合不能只靠退化成最近中心距离。
- G2 结构必要性：real GB 必须在 `dist+margin` 和 ALL-safe 两个口径**同时严格高于**逐配置等球数 KMeans。
- G3 扩展资格：只有 G2 通过，才运行 4 datasets × model seeds `0..4`；否则立即停止。

G2 是本 Pilot 的决定性 kill gate。KMeans mean±std 衡量控制聚类初始化，不代表 encoder/model seed 方差。

## 3. Task 1：环境与核心单元测试

**Files:**

- Test: `开放意图粒球新方向0630/tests/test_data.py`
- Test: `开放意图粒球新方向0630/tests/test_baselines.py`
- Test: `开放意图粒球新方向0630/tests/test_metrics.py`
- Test: `开放意图粒球新方向0630/tests/test_conditionals.py`
- Test: `开放意图粒球新方向0630/tests/test_gb_scores.py`
- Test: `开放意图粒球新方向0630/tests/test_posthoc.py`

- [x] **Step 1: 固定离线审计解释器**

```powershell
py -3.12 -c "import sys, numpy, sklearn, torch; print(sys.executable)"
```

Expected: 输出 `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`，核心离线依赖 import 成功。

- [x] **Step 2: 运行不依赖 Transformers 的核心测试**

```powershell
py -3.12 -m pytest "开放意图粒球新方向0630\tests" -q --ignore="开放意图粒球新方向0630\tests\test_stackoverflow_runner.py"
```

Expected: `13 passed, 3 warnings`。

- [x] **Step 3: 记录完整 collection 的依赖限制**

```powershell
py -3.12 -c "import importlib.util; print(importlib.util.find_spec('transformers'))"
```

Expected: `None`。因此 `test_stackoverflow_runner.py` 在当前 Python 3.12 环境会于 import 阶段因缺少可选 `transformers` 停止；不能把未执行的断言报告为通过。

## 4. Task 2：生成冻结表示与基础开放集结果

**Files:**

- Run: `开放意图粒球新方向0630/code/run_oi1_smoke.py`
- Run: `开放意图粒球新方向0630/code/run_oi1b_stackoverflow.py`
- Output: 两个权威 run 目录

- [x] **Step 1: 执行 BANKING77 seed0 训练与导出**

历史调用由 `config.json` 和 CLI 参数重建：

```powershell
python "开放意图粒球新方向0630\code\run_oi1_smoke.py" --seed 0 --known-ratio 0.25 --epochs 3 --batch-size 64 --eval-batch-size 128 --max-length 64 --model-name bert-base-uncased
```

Expected historical output: `banking77_r25_seed0_20260630_172920/`，包含 2,107 train-known、235 val-known、3,080 test 样本以及 19 个 known classes。

- [x] **Step 2: 执行 StackOverflow seed0 训练与导出**

```powershell
python "开放意图粒球新方向0630\code\run_oi1b_stackoverflow.py" --seed 0 --known-ratio 0.25 --epochs 3 --batch-size 64 --eval-batch-size 128 --max-length 64 --model-name bert-base-uncased
```

Expected historical output: `stackoverflow_r25_seed0_20260630_201825/`，包含 3,000 train-known、500 val-known、6,000 test 样本以及 5 个 known classes。

这两条训练命令在 2026-07-14 审计时没有重跑：当前可发现的 Python 环境均缺少 `transformers`/`datasets`，且 G2 已失败，不允许为了补依赖而重新训练并替换权威 seed0 产物。

- [x] **Step 3: 检查 run 完整性**

```powershell
$runs = @(
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920",
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825"
)
foreach ($run in $runs) {
  Get-ChildItem -LiteralPath $run -File | Select-Object Name, Length
}
```

Expected: 每个目录恰有前述 6 类权威产物，NPZ 大小分别为 15,860,992 和 27,276,009 bytes。

## 5. Task 3：强基线与公平阈值审计

**Files:**

- Inspect: `开放意图粒球新方向0630/code/oi_gb/baselines.py`
- Inspect: `开放意图粒球新方向0630/code/oi_gb/metrics.py`
- Inspect: `开放意图粒球新方向0630/code/run_oi1_smoke.py`
- Inspect: `开放意图粒球新方向0630/code/run_oi1b_stackoverflow.py`
- Evidence: 两个 run 的 `score_metrics.csv` 与 `conditional_gain.csv`

- [x] **Step 1: 核对基础分数覆盖**

```powershell
Import-Csv "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\score_metrics.csv" | Select-Object score, auroc, aupr, fpr95
Import-Csv "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\score_metrics.csv" | Select-Object score, auroc, aupr, fpr95
```

Expected: 包含 MSP、ODIN-temperature approximation、Energy、Prototype、kNN、Mahalanobis、GEN、ReAct+Energy、OpenMax-tail approximation、ViM-residual approximation、random、GB、GB_shuffled 和 GB_kmeans_proto。ODIN/OpenMax/ViM 必须标为 fixed-feature approximation，不得声称忠实端到端复现。

- [x] **Step 2: 核对 threshold 无 test-label 泄漏**

```powershell
rg -n "threshold_for_known_fpr|val_scores|test_open_labels|fit_granular_balls" "开放意图粒球新方向0630\code\run_oi1_smoke.py" "开放意图粒球新方向0630\code\run_oi1b_stackoverflow.py"
```

Expected data flow:

1. threshold 只由 known validation scores 计算；
2. target known FPR 固定为 0.05；
3. test unknown labels 在 scores 生成后才进入 metrics；
4. 粒球拟合不接收 test labels。

- [x] **Step 3: 区分正式拒识阈值与离线条件子集**

```powershell
Get-Content -Raw "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\conditional_gain.csv"
Get-Content -Raw "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\conditional_gain.csv"
```

Expected: 正式 open-set rejection threshold 仍只由 known validation scores 决定。`offline_probe_generic.py` 与 Task 5A 的 ALL-safe 是另一种 label-free conditional diagnostic：在当前 test score 分布内分别取 MSP、Prototype、kNN 的中位数安全侧再求交集。它不读取 test unknown labels 选子集，但使用 test score distribution，属于 transductive diagnostic，不得写成 validation-derived threshold，也不参与最终 open classification 决策。

## 6. Task 4：真实粒球、KMeans 与 shuffled controls

**Files:**

- Inspect: `开放意图粒球新方向0630/code/oi_gb/gb_scores.py`
- Run: `开放意图粒球新方向0630/code/offline_probe_generic.py`
- Run: `开放意图粒球新方向0630/code/offline_probe_B_hparam.py`

- [x] **Step 1: 固定默认粒球与 control 协议**

Default real GB：每类最多 16 球、最小球 8、split improvement 0.05。KMeans 对每一类和每一组 GB 超参使用与 real GB 相同的球数；shuffled control 保留分组规模结构但随机分配节点。

- [x] **Step 2: 复算 BANKING77 通用 probe**

```powershell
py -3.12 "开放意图粒球新方向0630\code\offline_probe_generic.py" "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\features_logits_seed0.npz" 0
```

Expected: 150 个 real balls；`dist+margin` real=0.7741、KMeans mean=0.7867；ALL-safe real=0.7525、KMeans mean=0.7882；G2 FAIL。

- [x] **Step 3: 复算 StackOverflow 通用 probe**

```powershell
py -3.12 "开放意图粒球新方向0630\code\offline_probe_generic.py" "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\features_logits_seed0.npz" 0
```

Expected: 6 个 real balls；`dist+margin` real=KMeans=0.85704637；ALL-safe real=KMeans=0.85926631；G2 FAIL。

- [x] **Step 4: 运行 BANKING77 7×3 超参稳健性**

```powershell
py -3.12 "开放意图粒球新方向0630\code\offline_probe_B_hparam.py"
```

Expected: 7 组 GB 配置，每组 KMeans seeds 13/14/15；real wins 为 `dist+margin 0/7`、`distance-only 3/7`、`ALL-safe 0/7`。distance-only 的 3 次优势不超过约 0.002 AUROC，且不满足联合 gate。

## 7. Task 5：多随机控制与统计推断

**Files:**

- Read: `next_paper/runs/frozen_open_intent_pilot_audit_20260714/statistics.json`
- Read: `next_paper/runs/frozen_open_intent_pilot_audit_20260714/offline_probe_log.md`

- [x] **Step 1: 固定随机性层级**

KMeans 使用 seeds 13/14/15，报告 sample mean±sample std。encoder/model 与 known-class split 只有 seed0；不得用 KMeans 的 std 替代模型 seed 方差。

- [x] **Step 2: 执行 paired stratified bootstrap**

统计协议：在同一 test set 内分别对 known/unknown 样本分层有放回抽样，比较 real GB AUROC 与三个 KMeans AUROC 均值，3,000 replicates；full seed=20260714，ALL-safe seed=20260715。

Expected BANKING77：

- full Δ=-0.01258，95% CI `[-0.02077,-0.00411]`，two-sided p=0.002；
- ALL-safe Δ=-0.03570，95% CI `[-0.05136,-0.02047]`，0/3000 bootstrap deltas ≥0，报告 `p<0.001`。

Expected StackOverflow：

- full 和 ALL-safe Δ=0；95% CI 在浮点精度内为 `[≈0,≈0]`；p=1.0。

- [x] **Step 3: 核对审计 JSON**

```powershell
Get-Content -Raw "next_paper\runs\frozen_open_intent_pilot_audit_20260714\statistics.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

Expected: 包含 bootstrap 类型、replicates、两个 bootstrap seeds、每个数据集样本量、球数、KMeans 三 seed 原始 AUROC、mean/std、delta、CI、p 和测试状态。

Bootstrap CI 仅衡量当前冻结 seed0 test samples 上的 paired score difference，不支持跨模型随机初始化的总体结论。

## 7A. Task 5A：五个结构随机 seed 审计

该任务补足最小 Pilot 的“五随机种子”要求，但随机层级严格限定为**冻结表示上的局部结构构造**。encoder/model、known-class split 和 NPZ 仍只有 seed0；结构 seed 的 mean±std 不得冒充模型 seed 稳定性。

**Files:**

- Create: `next_paper/open_intent_pilot/structural_seed_audit.py`
- Create: `next_paper/open_intent_pilot/configs/structural_seed_audit.json`
- Test: `next_paper/open_intent_pilot/tests/test_structural_seed_audit.py`
- Output: `next_paper/runs/frozen_open_intent_structural_seeds_20260714/`

- [x] **Step 1: 写失败测试，固定汇总和统计接口**

```python
def test_summarize_metric_reports_sample_std_and_positive_count():
    rows = [
        {"dataset": "toy", "full_delta": -0.02},
        {"dataset": "toy", "full_delta": -0.01},
        {"dataset": "toy", "full_delta": 0.01},
    ]
    result = summarize_metric(rows, "full_delta")
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(-0.0066666667)
    assert result["sample_std"] == pytest.approx(0.0152752523)
    assert result["positive_count"] == 1


def test_exact_sign_flip_p_uses_all_sign_assignments():
    assert exact_sign_flip_p(np.array([-1.0, -1.0, -1.0])) == pytest.approx(0.25)


def test_validate_config_requires_five_distinct_seeds():
    with pytest.raises(ValueError, match="five distinct structural seeds"):
        validate_config({"seeds": [0, 1, 2]})
```

- [x] **Step 2: 运行 RED，确认因模块尚不存在而失败**

```powershell
py -3.12 -m pytest "next_paper\open_intent_pilot\tests\test_structural_seed_audit.py" -q
```

Expected: FAIL，失败原因是 `next_paper.open_intent_pilot.structural_seed_audit` 尚不存在，而不是测试语法错误。

- [x] **Step 3: 实现最小纯统计函数**

```python
def validate_config(config: dict) -> None:
    seeds = [int(seed) for seed in config.get("seeds", [])]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("config must define five distinct structural seeds")


def summarize_metric(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "positive_count": int(np.sum(values > 0.0)),
    }


def exact_sign_flip_p(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=np.float64)
    observed = abs(float(deltas.mean()))
    means = []
    for mask in range(1 << deltas.size):
        signs = np.asarray([1.0 if mask & (1 << idx) else -1.0 for idx in range(deltas.size)])
        means.append(abs(float(np.mean(signs * deltas))))
    return float(np.mean(np.asarray(means) >= observed - 1e-15))
```

- [x] **Step 4: 实现数据集审计与不可变输出**

实现必须：

1. 从 `--source-workspace` 下的权威 NPZ 读取 `train_features/train_labels/train_pred/test_features/test_open_labels/test_logits`；
2. 固定 MSP/Prototype/kNN 在当前 test score 分布上的 label-free median-safe mask，并标注为 transductive diagnostic；
3. 对 real seeds `0..4` 分别运行 `fit_gb_model`；
4. 对 seed `s` 运行 KMeans seed `13+s` 和 shuffle seed `1013+s`，两者均由该 real model 的实际球数/球大小生成；
5. 对每个模型独立 z-normalize `s_dist` 与 `s_margin` 后取均值；
6. 保存 `raw_seed_metrics.csv`，字段至少包括 dataset、structural_seed、control seeds、n_balls、full/ALL-safe real、KMeans、shuffle 与 paired delta；
7. 保存 `summary.csv`、`statistics.json`、`config.resolved.json` 和 `run.log`；
8. seed-level CI 使用 10,000 次 paired bootstrap，bootstrap seed base=20260716，并按数据集顺序加 index（BANKING77=20260716，StackOverflow=20260717）；p-value 使用全部 \(2^5\) 个 sign flips；
9. 若输出目录已存在且非空则报错，不覆盖既有结果。

- [x] **Step 5: 运行 GREEN 与完整核心回归测试**

```powershell
py -3.12 -m pytest "next_paper\open_intent_pilot\tests\test_structural_seed_audit.py" -q
py -3.12 -m pytest "开放意图粒球新方向0630\tests" -q --ignore="开放意图粒球新方向0630\tests\test_stackoverflow_runner.py"
```

Expected: 新测试全部 PASS；旧核心测试仍为 `13 passed, 3 warnings`。

- [x] **Step 6: 运行两数据集五 seed 审计**

```powershell
py -3.12 "next_paper\open_intent_pilot\structural_seed_audit.py" `
  --source-workspace "C:\Users\Admin\Desktop\实验8去噪5.19-" `
  --config "next_paper\open_intent_pilot\configs\structural_seed_audit.json" `
  --output-dir "next_paper\runs\frozen_open_intent_structural_seeds_20260714"
```

Expected: 两数据集 × 5 seeds = 10 条 raw rows；每个数据集分别产生 full 和 ALL-safe 的 real/KMeans/shuffle mean±sample std、paired delta、bootstrap CI、sign-flip p 与 positive count。

- [x] **Step 7: 应用结构稳定性 Gate**

只有以下条件全部成立才允许推翻当前 G2：BANKING77 和 StackOverflow 必须在 full 与 ALL-safe 四个数据集—口径组合中 `mean(real−KMeans)>0`，每个组合至少 4/5 positive seeds，且真实 GB 的均值高于 shuffle。任一失败即保持 No-Go，不启动 BERT model seeds 或联合训练。

实际结果：BANKING77 full Δ=`−0.01101±0.02152`、2/5 positive，ALL-safe Δ=`−0.04023±0.04458`、0/5 positive；StackOverflow 两个口径均为 Δ=`0.00000±0.00000`、0/5 positive，且 shuffle 均值略高。两个数据集均未通过 Gate，最终保持 No-Go。

## 8. Task 6：产物指纹与不可变性

**Files:**

- Verify: 两个 `config.json`
- Verify: 两个 `features_logits_seed0.npz`
- Verify: `statistics.json`

- [x] **Step 1: 计算 SHA-256**

```powershell
$files = @(
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\config.json",
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\features_logits_seed0.npz",
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\config.json",
  "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\features_logits_seed0.npz",
  "next_paper\runs\frozen_open_intent_pilot_audit_20260714\statistics.json"
)
$files | ForEach-Object { Get-FileHash -Algorithm SHA256 -LiteralPath $_ }
```

Expected hashes：

| 文件 | SHA-256 |
|---|---|
| BANKING77 `config.json` | `6D322978CCB6CD52364C4EF5FDBF39CA109F68D828E7AE1C68A35245C1D9C285` |
| BANKING77 NPZ | `F63DFE8709F50760A341B7FA24CEDC95018C4868B2DA303E223460705BAE675C` |
| StackOverflow `config.json` | `C426170A59926F3C3098A7B5BCEE1454DF051D727D99F3F2B994E3A90033FBCB` |
| StackOverflow NPZ | `1C496BEF29BF2A2D3966089322757DBD768AD14DB227CCEFFFA79A5CF9D8A33D` |
| `statistics.json` | `313F45A82E3C73777C1F847B401317ACD271D46095F0008E3537C0EB84D45088` |

若未来任一 hash 改变，必须视为新 artifact，不能继续引用本报告中的统计值。

## 9. Task 7：Gate 判定与详细实验结论

**Files:**

- Create/Update: `next_paper/docs/pilot_report.md`
- Evidence: 两个 run 与 audit 目录

- [x] **Step 1: 判定 G1**

BANKING77 的单一 distance component 有较高 AUROC，但联合 score 与 ALL-safe 均不能稳定超过 KMeans；StackOverflow 的强 logits baselines 高于 GB。G1 不支持“粒球组合提供额外条件信息”。

- [x] **Step 2: 判定 G2**

BANKING77 在两个主口径显著低于 KMeans；StackOverflow 输出逐样本等价。G2 明确失败。

- [x] **Step 3: 执行停止规则**

不运行联合 BERT–GB training，不新增同类 `dist/margin/density/purity/overlap` 权重搜索，不以单个 distance component 的偶然优势覆盖联合 gate。

- [x] **Step 4: 写出 claim 分层**

`pilot_report.md` 必须分别列出：实验支持、合理解释、尚未验证、负结果削弱四类内容，并明确 No-Go 只针对“当前冻结表示 + post-hoc 粒球局部边界/条件分数”。

## 10. Task 8：因 G2 失败而取消的扩展

- [ ] **CANCELLED BY G2: 运行 BANKING77、StackOverflow、CLINC150、SNIPS × model seeds 0..4。**

取消原因：必要性 gate 在两个 seed0 数据集均失败，其中 BANKING77 real GB 显著更差，StackOverflow 与 KMeans 完全等价。扩 model seeds 不能修复“当前真实结构未超过普通聚类”的核心反证。

- [ ] **CANCELLED BY G2: 联合训练 BERT 与粒球边界。**

取消原因：目标明确规定只有 frozen Pilot 显示稳定独立价值后才进入联合训练；当前证据相反。

- [ ] **CANCELLED BY G2: 大规模超参搜索或改变 test protocol。**

取消原因：7×3 控制已经覆盖粒度和 split improvement；继续按结果选权重、known ratio 或阈值会破坏公平性。

这些取消项是预设 stop rule 的正确执行，不是可以在本 Pilot 内补跑的缺失任务。

## 11. 与后续主方向的接口

open-set frozen Pilot 的 No-Go 不等于整个粒球研究终止。方向排序已将后续主方向切换为 Stable-GB Reliability，方法规格见 `next_paper/docs/method_design.md`。它提出的是新研究问题：跨视图粒球稳定性能否预测局部标签校正建议的可靠性。

Stable-GB 不得复用本 Pilot 的 open-set 结论作为正证据，也不得把人工 feature-masking views 写成真实 GRACE augmentation。其 A0/A1 若实施，必须使用独立目录、独立 config、独立原始输出和新的 Go/No-Go 门。

## 12. 最终交付检查

- [x] 两个完整 seed0 run 保存 config、NPZ、CSV、JSON 和 Markdown 报告；
- [x] 强 logits/距离/局部原型基线齐全，并标注 approximation 边界；
- [x] threshold 只来自 known validation，test labels 只用于最终评测；
- [x] real GB、逐配置等球数 KMeans、shuffled controls 齐全；
- [x] KMeans seeds 13/14/15 的 mean±sample std 齐全；
- [x] real GB seeds 0–4、配对 KMeans/shuffle seeds 与结构级 mean±std 已完成；
- [x] 7 组 GB 超参消融齐全；
- [x] paired bootstrap CI 与 p-value 齐全；
- [x] model-seed0 限制已显式披露；
- [x] 核心离线测试为 13 passed，缺少 Transformers 的 collection 限制已披露；
- [x] 详细 No-Go、claim 边界与停止逻辑已写入 `pilot_report.md`；
- [x] 第一篇论文代码、日志和结果未被覆盖。

**最终决策：No-Go。停止当前 frozen open-set 局部边界方向，不进入联合训练或大规模实验。**
