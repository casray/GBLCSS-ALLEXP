# Tie-safe 代表性实验协议修订 01：确定性构球

## 1. 修订状态

- 状态：`FROZEN_BEFORE_VALID_RERUN`
- 日期：2026-07-27（Asia/Shanghai）
- 原冻结设计 SHA-256：`D64A836F421D58AF0CE295DF86014D2FDBE79A864B528432145DBC92CF63D7B7`
- 修订范围：只固定训练子进程的线程环境；不改变 tie 模式、数据集、噪声、seed、模型、阈值、统计量、方法角色或完整网格计划。

## 2. 触发失败

第一次代表性运行生成了 30/30 个 `status=ok` 的训练行，但汇总器按冻结计划拒绝结果：

```text
protocol drift:
dataset=corafull seed=1 mode=alpha_zero
field=ball_assignment_hash
reference=b1cacc6c56ee9d46ab75def7719e43688bf84da7205702342182aca4c57a856c
observed=a7b0f90c0c83417c78c72466dbce2390a467cd1fac8db693543fab205fa83206
```

因此该批次结论是 `INVALID_PROTOCOL`。不得解释或引用其准确率差异。

## 3. 根因实验

### 3.1 排除球 ID 任意重编号

对 CoraFull seed 1 使用相同 embedding、split、uniform 0.6 noise、config 和 `random_state=1`，在三个独立进程中只重建粒球。

默认多线程环境得到：

| run | raw assignment hash | canonical partition hash | balls |
|---:|---|---|---:|
| 1 | `b1cacc6c56ee9d46ab75def7719e43688bf84da7205702342182aca4c57a856c` | `a76fda59c6c29f4f20e9cceb387a544a55d68fabdf90ea06d949be4a4a38d356` | 64 |
| 2 | `a7b0f90c0c83417c78c72466dbce2390a467cd1fac8db693543fab205fa83206` | `bda9eea7455c898459702ec70bc33f395f512ec37a45e21bad1b01135c115aef` | 64 |
| 3 | `a7b0f90c0c83417c78c72466dbce2390a467cd1fac8db693543fab205fa83206` | `bda9eea7455c898459702ec70bc33f395f512ec37a45e21bad1b01135c115aef` | 64 |

canonical hash 也不同，说明不是球 ID 重编号，而是实际成员分区不同。

### 3.2 单线程验证

在每个独立进程启动前固定：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OMP_DYNAMIC=FALSE
MKL_DYNAMIC=FALSE
PYTHONHASHSEED=0
```

三次独立重建均得到：

```text
raw=9bfef5c5332489438ecd9980931ec770b831ab7b3e79c22c315a4713430f63d3
canonical=2ecfd798303dc118af68b490e2e6d3d1b935a8fb95eaa9a3582ff27945badc04
balls=64
```

操作性根因由此定位为 scikit-learn KMeans/底层 BLAS 或 OpenMP 多线程浮点归约造成的跨进程构球非确定性。固定 `random_state` 本身不足以保证当前环境中的 partition 可复现。

## 4. 研究假设

在不改变算法参数和数据实例的情况下，将构球相关数值库固定为单线程，可使同一 dataset/seed 的粒球 partition 在独立进程间完全一致，从而恢复三个 tie 模式的配对可比性。

## 5. 粒球角色与方法动机

粒球角色不变，仍为局部关系和软置信度来源。本修订不改变任何球内目标公式；它只消除“同 seed 却得到不同局部关系单元”的实现噪声。

## 6. 代码与配置改动

- `paper_pipeline/scripts/run_tie_safe_representative.py`
  - 为所有训练子进程显式传入上述确定性环境；
  - environment fingerprint 记录实际传入值。
- 不修改 YAML 数值配置。
- 不修改构球参数、KMeans `n_init` 或 `random_state`。
- 不修改 frozen experiment design 第 1–12 节。

## 7. 预期机制

环境变量在 Python/scikit-learn/BLAS 初始化前进入子进程，使 KMeans 距离计算和归约顺序固定；相同 input bytes 与 seed 应得到相同 raw assignment hash。

## 8. 成功标准

- 独立构球预检 3/3 raw assignment hash 完全一致；
- 代表性 30 行中，同 dataset/seed 的三个 tie 模式：
  - split/noise/modified counts/embedding path 完全一致；
  - `ball_assignment_hash` 完全一致；
  - 非并列机制诊断一致；
- 30/30 行 `status=ok`，无静默缺行；
- 汇总器不再报 protocol drift。

## 9. 失败标准

- 任一独立预检 hash 不同；
- 任一 dataset/seed 跨模式 assignment hash 不同；
- 线程环境未出现在 environment fingerprint；
- 为追求历史准确率而改 seed、KMeans 参数或容差。

## 10. 历史数值边界

旧主表来自未显式固定线程数的历史运行。确定性重跑的 `min_index` 不保证在 `1e-6` 内复现旧准确率；若不复现，必须报告为环境/partition lineage 断裂，不能据此否定同轮新结果。

新论文主表将使用完整网格中同轮运行的四个方法，不把旧多线程 baseline 与新单线程方法直接拼接。只有同轮 fingerprint 闭合的行进入新主表。

## 11. 实验结果

- 独立单线程构球预检：`3/3 PASS`。
- 有效 30 行重跑：`NOT_RUN`。

## 12. 最终决策

`修改`

保留第一次 30 行为无效协议审计资产；使用确定性子进程环境从空输出目录完整重跑。

