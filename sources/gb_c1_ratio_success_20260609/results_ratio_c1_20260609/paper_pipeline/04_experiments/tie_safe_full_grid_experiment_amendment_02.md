# Tie-safe 完整网格协议修订 02：不可变原始输出与派生统计闭环

## 1. 修订状态

- 修订时间：2026-07-27；
- 修订触发点：第一次完整网格训练结束后的独立证据审计；
- 修订性质：证据工程硬化，不改变研究问题、方法或性能选择；
- 当前第一次完整网格目录：
  - `external/NoisyGL/runs/tie_safe_full_grid_20260727`；
- 固定的第二次运行目录：
  - `external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable`。

在看到第二次运行的任何结果前，本修订固定执行第二次完整网格。第一次运行及其日志、CSV 和审计结论全部保留，不删除、不覆盖。

## 2. 审计发现

第一次完整网格已达到 500/500 行完成、125/125 协议组 PASS 和 50/50 命令成功，但独立审计发现三项证据工程问题：

1. `run_tie_safe_full_grid.py` 在下游脚本写完 CSV 后，把 dataset、regime、noise/split 参数和 tie mode 写回工作 CSV。命令、源码快照、split/noise hash 和污染计数可以闭合这些值，但这种实现会让工作 CSV 不再是未经处理的 immutable raw；
2. 冻结计划规定逐 seed 差的正/负/零容差为 `1e-12`，runner 使用了 `1e-15`。第一次结果中只有一个差值恰为零，次小绝对差为 `3.116964×10^-4`，因此已有计数未受影响，但代码仍应与计划一致；
3. `paired_tests.csv` 保存了均值、sample std、t 和 p，却没有显式保存计划要求的 nominal 95% CI，导致 CI 需要临时推导。

上述问题不构成根据测试集性能改变方法的理由，也不改变第一次运行的数值含义。第二次运行的目的仅是形成更严格的原始—规范化—统计 lineage。

## 3. 固定修订

### 3.1 Immutable raw

每个下游命令成功返回后：

1. 先把下游脚本原样写出的 CSV bytes 保存到 `raw_immutable/`；
2. immutable 文件不得加入 runner 专属的 `regime` 或 embedding hash 派生列；
3. immutable 文件的 path、bytes、SHA-256、行数和完成状态写入 `immutable_raw_manifest.csv`；
4. 后续规范化只作用于 `raw/` 工作副本。

### 3.2 原始字段验证

对下游原始 CSV 中已经存在的字段只验证、不覆写：

- dataset；
- method、base method、SSL method；
- noise type/rate/scope；
- split type 和 train/validation/test 比例；
- `tie_mode`；
- split/noise hash、污染计数、config 和指标。

runner 只允许补充下游原始 CSV 不提供的字段：

- 固定网格的 `regime` 键；
- feature/ball embedding 的 SHA-256。

任何原始字段与命令预期不一致，当前命令标记为 invalid，完整网格不得进入主表。

### 3.3 统计字段

- 逐 seed 差的正/负/零容差固定为 `1e-12`；
- 每个配对检验显式保存：
  - `paired_delta_ci95_low`；
  - `paired_delta_ci95_high`；
- \(n=5\)、df=4 的临界值固定为 `2.7764451051977987`；
- Holm family、双侧配对 t 检验、sample std 和比较对象不变。

## 4. 明确不改变的内容

- 主修复仍为 `distribution_only`；
- 数据集仍为 5 个；
- 每个数据集仍为 5 个噪声设置；
- seeds 仍为 0–4；
- 方法仍为 GB-LCSS、shuffled、MLP+CE 和 JoCoR；
- split、noise scope、噪声率、训练比例和评估流程不变；
- GRACE embedding 不变；
- 构球配置不变；
- 确定性线程环境不变；
- 三个 comparator-specific 25-test Holm families 不变；
- 单一 75-test Holm 仍只作为 family 定义敏感性分析；
- 第二次结果不得用于选择方法、删数据集、改阈值或修改统计家族。

## 5. 成功标准

第二次运行只有同时满足以下条件才可替代第一次运行成为论文主证据：

- 50/50 个 immutable raw 文件存在，每个恰好 10 行且通过原始字段验证；
- 50/50 个 enriched raw 文件存在，每个恰好 10 行；
- 500/500 行 `status=ok`；
- 125/125 个四方法协议组 PASS；
- 250/250 个 SoftDist 行使用 `distribution_only`；
- 250/250 个 SoftDist 行闭合 embedding 和 assignment hash；
- 75/75 个配对检验显式保存 95% CI；
- 50 个 stderr 均为空；
- 与第一次运行重叠的 500 个键逐行比较：
  - test/validation 指标完全一致；
  - split/noise hash 完全一致；
  - assignment hash 完全一致；
  - tie diagnostics 完全一致；
- 若只出现文本编码或派生列顺序差异，必须单独记录，不得描述为训练结果漂移。

## 6. 失败标准

- 任一 immutable raw 缺失、被规范化字段污染或不能解码；
- 原始 protocol 字段与命令不一致；
- 任一 seed-level 训练指标或关键 fingerprint 相对第一次运行漂移；
- 任一 NaN/Inf、失败、重复或缺行；
- 通过第二次测试结果事后改变方法或论文比较范围。

发生失败时，第一次运行继续作为诊断资产，但论文主表不得以“v2 immutable 闭环”名义发布。

## 7. 预先固定的最终决策规则

- 全部成功标准满足：论文主证据切换到 v2；v1 降为独立重复与审计资产；
- v2 数值与 v1 完全一致：可以报告一次完整网格的确定性重复成功，但不能把两次相同 seed 当成 10 个独立重复；
- v2 发生漂移：停止论文数字替换，定位漂移来源，不挑选两次中更有利的结果。

