# Phase41：Ellipsoid SoftDist 自适应构球离线参数网格（2026-06-28/29）

## 1. 总判断

本轮只做 offline target diagnostic，不跑 downstream accuracy，不改变主方法 SoftDist `[X‖H]`。实验回答两个构球问题：

- `split_improvement` 是否需要按数据集自适应，以避免 DBLP 这类单球坍塌。
- 固定或受限的 `max_balls` 是否压制 PubMed、AmazonPhoto、AmazonCom、DBLP 的粒度质量。

结论：

- **DBLP：collapse-fixed。** 默认 `split_improvement=0.05` 导致单球坍塌；降低 split 后恢复为 64/128/256/338 左右的球数并通过 gate。最佳 offline setting 为 `max_balls=128, split_improvement=0.001`，`mass_delta=+2.9623`，positive seeds `5/5`。
- **PubMed：ball-count-sensitive。** 默认 64 球不是最佳；`max_balls=512, split_improvement=0.01` 平均约 `376.2` 个球并通过 gate，`mass_delta=+1.8385`，positive seeds `4/5`。
- **AmazonPhoto / AmazonCom：不是 64/256 上限过低。** 两者最佳 offline setting 都使用 `max_balls=64`，更高上限 clean-mass delta 变弱或饱和。因此不能把 Amazon 系旧 downstream 边界简单归因于球数上限太小。
- 本轮结果只能作为“是否值得进入 downstream”的候选证据；不能写成 downstream 性能结论或论文主 claim。

核心证据文件：

- 报告：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/adaptive_candidate_report.md`
- Gate 表：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/diagnostic_grid_gate.csv`
- Raw 表：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/diagnostic_grid_raw.csv`
- 聚合表：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/diagnostic_grid_aggregate.csv`
- 配对 delta：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/diagnostic_grid_paired_deltas.csv`
- 运行日志：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/run.log`
- 新脚本：`external/NoisyGL/scripts/diagnose_ellipsoid_param_grid.py`
- 新测试：`external/NoisyGL/tests/test_ellipsoid_param_grid.py`

## 2. 研究假设

`ellipsoid_softdist_large_20260626` 和 `ellipsoid_softdist_dblp_split001_20260626` 已提示：

- DBLP 默认失败很可能不是椭球权重本身失败，而是 `split_improvement=0.05` 卡住了第一个 root split，导致 `num_balls=1`。
- PubMed 默认 64 球下 target signal 弱，可能与球数上限或粒度不足有关。
- AmazonPhoto / AmazonCom 旧 downstream 边界可能与固定球数有关，但也可能是强 H+CE baseline 吸收了 target 层收益。

本轮假设：如果构球参数是真瓶颈，则降低 `split_improvement` 或提高 `max_balls` 应使 offline gate 从 fail 变 pass，并且 real ellipsoid 应优于 shuffled ellipsoid。

## 3. 粒球扮演的角色

粒球仍是 SoftDist `[X‖H]` 的局部 noisy-label vote 单元；椭球只在每个 hard ball 内用 Mahalanobis 权重调整 train noisy-label vote。clean labels 只用于诊断指标和 gate，不参与构球、不参与可部署 policy。

本轮没有把 same-ball membership 当 hard positive label，没有引入 prototype pulling，也没有修改下游 evaluation。

## 4. 方法动机

固定参数下的失败不能直接解释为“椭球机制理论无效”。本轮用离线参数网格把三类现象分开：

- **collapse-fixed**：降低 split 后从单球或少量球恢复，并通过 gate。
- **ball-count-sensitive**：不同 `max_balls` 明显改变 gate 或 clean mass。
- **no-offline-value**：所有 setting 都不稳定或不优于 shuffled。

## 5. 代码改动

新增：

- `external/NoisyGL/scripts/diagnose_ellipsoid_param_grid.py`
- `external/NoisyGL/tests/test_ellipsoid_param_grid.py`

脚本特性：

- 复用 `diagnose_ellipsoid_softdist.py` 的 target diagnostic 逻辑和方法常量。
- 在内存中覆盖 `gb.max_balls`、`gb.min_ball_size`、`gb.split_improvement`，不改 YAML。
- 支持 `--resume`，可从已有 `diagnostic_grid_raw.csv` 跳过已完成组合。
- 按同一 `dataset/seed/max_balls/split_improvement` 批处理四个 methods，避免重复构球。

输出文件：

- `diagnostic_grid_raw.csv`
- `diagnostic_grid_aggregate.csv`
- `diagnostic_grid_paired_deltas.csv`
- `diagnostic_grid_gate.csv`
- `adaptive_candidate_report.md`

## 6. 配置改动

无 YAML 改动。CLI 参数：

```powershell
cd C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  scripts\diagnose_ellipsoid_param_grid.py `
  --datasets dblp,pubmed,amazonpho,amazoncom `
  --seeds 0,1,2,3,4 `
  --max_balls 64,128,256,512 `
  --split_improvements 0.05,0.02,0.01,0.005,0.001 `
  --min_ball_size 32 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --device cpu `
  --output_dir runs\ellipsoid_param_grid_20260628
```

Grid 规模：4 datasets × 5 seeds × 4 max_balls × 5 split thresholds × 4 methods = 1600 raw rows。

四个 methods：

- `frozengrace_gb_gcn_softdist_xh`
- `frozengrace_gb_gcn_softdist_xh_shuffled`
- `frozengrace_gb_gcn_softdist_xh_ellipsoid`
- `frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled`

## 7. 成功标准

沿用 ellipsoid diagnostic gate：

- `positive_seed_count >= 4`
- `beats_shuffled = true`
- `fallback_ratio_mean <= 0.05`
- `mass_delta_mean >= 0.01` 或 `cdr_delta_mean <= -0.02 且 ncr_delta_mean >= -0.02`

分类标准：

- `collapse-fixed`：降低 split 后从 1/少量球恢复，并通过 gate。
- `ball-count-sensitive`：不同 `max_balls` 明显改变 gate 或 clean mass。
- `no-offline-value`：所有 grid 都不稳定或不优于 shuffled。

## 8. 实验结果

验收：

- `diagnostic_grid_raw.csv` 共 1600 行。
- `status=ok` 共 1600 行，错误行 0。
- 输出目录：`external/NoisyGL/runs/ellipsoid_param_grid_20260628/`

按数据集汇总：

| Dataset | Classification | Passed settings | Best setting | mass_delta | CDR delta | NCR delta | positive seeds | balls | fallback |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| AmazonCom | ball-count-sensitive | 20/20 | max_balls=64, split=0.001 | +10.9144 | +0.0007 | +0.0007 | 5/5 | 64.0 | 0.0000 |
| AmazonPhoto | ball-count-sensitive | 20/20 | max_balls=64, split=0.05 | +7.9166 | -0.0018 | -0.0000 | 5/5 | 64.0 | 0.0000 |
| DBLP | collapse-fixed | 10/20 | max_balls=128, split=0.001 | +2.9623 | +0.0044 | -0.0102 | 5/5 | 128.0 | 0.0000 |
| PubMed | ball-count-sensitive | 7/20 | max_balls=512, split=0.01 | +1.8385 | +0.0015 | +0.0141 | 4/5 | 376.2 | 0.0000 |

Direct answers from report:

- DBLP split adaptation：yes。Lower `split_improvement` recovers balls and passes the gate。
- PubMed max_balls impact：yes。Gate and clean-mass delta change with `max_balls`。
- AmazonPhoto 64/256 cap：no evidence that cap is too low；best offline setting uses `max_balls=64` and higher caps are weaker。
- AmazonCom 64/256 cap：same as AmazonPhoto；best offline setting uses `max_balls=64`。

## 9. 最终决策

**继续，但只作为 downstream 候选筛选，不改主方法。**

推荐进入后续 downstream sanity check 的最小候选：

- DBLP：`max_balls=128, split_improvement=0.001`，只作为 split-adaptive 反事实，不混入默认主表。
- PubMed：`max_balls=512, split_improvement=0.01`，验证是否 offline target signal 能转化为 accuracy。
- AmazonPhoto：`max_balls=64, split_improvement=0.05`，验证强 offline mass 是否被 downstream H+CE baseline 吸收。
- AmazonCom：`max_balls=64, split_improvement=0.001` 或任一等价 64-ball split，注意 CDR delta 仍小幅变差。

停止或降级：

- 不要把 AmazonPhoto/AmazonCom 继续加大 `max_balls` 当作主方向；当前证据显示更高上限弱化或饱和。
- 不要把 PubMed 默认 64 球失败写成“PubMed 不适合椭球”；本轮显示它是 ball-count-sensitive。
- 不要把 DBLP 默认失败写成“DBLP 不适合椭球”；本轮显示它是 split-collapse。

## 10. 证据分类

### 已被实验支持的结论

- DBLP 默认 `split_improvement=0.05` 的失败是构球坍塌型失败；降低 split 后可恢复可用 offline target signal。证据：`adaptive_candidate_report.md`、`diagnostic_grid_gate.csv`。
- PubMed offline target signal 对 `max_balls` 敏感，512 上限下出现通过 gate 的 setting。证据：`diagnostic_grid_gate.csv`。
- AmazonPhoto/AmazonCom 当前没有“64/256 上限过低”的证据；最佳 offline setting 均为 `max_balls=64`。证据：`adaptive_candidate_report.md` Direct answers。
- 本轮 raw 覆盖完整，1600 行全部 `ok`。证据：`diagnostic_grid_raw.csv` 与本轮验收命令输出。

### 合理猜测

- DBLP 需要更低 split threshold 是因为 root split gain 接近默认阈值，默认几何分裂过严。
- PubMed 需要更高球数可能因为 64 球粒度不足，椭球权重在更细分区内才有可用 target signal。
- AmazonPhoto/AmazonCom 的 downstream 边界更可能来自强 H+CE baseline 吸收或 target-to-accuracy 转化失败，而不是球数上限太低。

### 尚未验证的假设

- DBLP `128/0.001`、PubMed `512/0.01`、AmazonPhoto `64/0.05`、AmazonCom `64/0.001` 是否能提升 downstream accuracy。
- 能否设计 deployable 的 dataset-adaptive split/max_balls policy；本轮只验证网格，不提供无 clean-label 的选择规则。
- offline clean-mass gate 与 downstream accuracy 的相关性是否在这四个数据集上成立。

### 已被负结果削弱的方向

- 固定默认 `split_improvement=0.05` 可以普适用于大图椭球 SoftDist。
- 简单提高 `max_balls` 一定能改善 AmazonPhoto/AmazonCom。
- 仅凭 offline target 通过 gate 就断言下游提升。

## 11. 下一步建议

若继续，应写一个预注册 downstream sanity check，而不是直接改主表：

- 数据集/setting：DBLP `128/0.001`、PubMed `512/0.01`、AmazonPhoto `64/0.05`、AmazonCom `64/0.001`。
- 对照：`gcn_xh_ce`、current SoftDist、current shuffled、ellipsoid selected setting、ellipsoid selected shuffled。
- 协议：same split、same seeds 0..4、same noise、same GRACE H、same evaluation。
- 成功标准：每个数据集必须 real > shuffled；相对 current SoftDist 的 mean accuracy 提升应超过可解释 std 噪声，否则只写为 offline-positive / downstream-not-confirmed。

论文 claim 边界：这一方向最多写成“adaptive construction diagnostic reveals why fixed GB parameters fail on some datasets”，不能在 downstream 前写成性能提升。
