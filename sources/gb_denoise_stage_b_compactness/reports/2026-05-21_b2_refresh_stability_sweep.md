# B2 动态刷新粒球稳定性验证报告

## 研究假设

本轮实验验证 B1 不稳定是否主要来自“warmup 后只固定生成一次粒球 assignment”。如果 encoder 的表示空间在后续训练中继续变化，那么固定粒球可能逐渐过时，导致粒球 compactness loss 约束到不再可靠的局部结构。B2 的想法是在训练过程中定期刷新粒球，让粒球结构跟随当前表示空间变化，从而缓解 B1 的 seed 不稳定。

## 粒球扮演的角色

本轮仍然把粒球作为 soft structural constraint，而不是 hard label 或 hard positive。粒球只提供局部 compactness 约束，用来反向塑造 encoder 表示空间。标签不参与训练，只用于 `label_purity_diagnostic` 和 `linear_eval_acc` 的后验诊断。

## 方法动机

B1 的静态粒球约束已经证明 compactness 可以被真实激活，但小权重仍然不稳定。一个合理解释是：warmup 结束时生成的粒球只反映当时的表示空间，后面 encoder 继续更新后，旧粒球 assignment 可能变成错误约束。B2 因此只改一个变量：定期重新生成粒球 assignment，检查动态粒球是否比静态 B1 更稳定。

本轮不修改 encoder、BN、EMA、距离度量、增强强度或粒球生成算法，避免变量混杂。

## 实验设置

- 固定代码版本：`fc62447`
- 运行目录：`C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs\gb_denoise_stage_b_compactness`
- 数据集：Cora、CiteSeer
- seeds：0、1、2
- baseline：复用同版本 B0；复用同 lambda 的静态 B1 sweep 结果
- 新跑 B2：24 个 run
- 训练协议：`epochs=1000`，`warmup_epochs=300`，`ssl_objective=negfree`，`ball_space=H`
- 粒球约束权重：`lambda_gb=0.001`、`0.0005`
- 刷新间隔：`ball_refresh_interval=50`、`100`
- 输出目录：`results_b2_refresh_sweep/`

完整性检查：

| 检查项 | 结果 |
|---|---:|
| summary 行数 | 72 |
| B2 新 run 数 | 24 |
| B2 vs B0 成对差值行数 | 24 |
| B2 vs B1 成对差值行数 | 24 |
| B2 `compact_nonzero_epochs` 最小值 | 700 |
| B2 `covered_nodes_max` 最小值 | 1838 |

这说明 B2 的 compactness loss 确实被激活，不是因为约束没有生效导致的结果。

## B2 相对 B0 的结果

表中 `acc_ge_b0` 表示 6 个 dataset-seed pair 中，B2 线性分类准确率不低于对应 B0 的个数。

| lambda_gb | refresh_interval | 平均 accuracy 差值 | std | acc_ge_b0 | 平均 effective rank 差值 | 平均 AD 差值 | 平均 radius 差值 | 平均 purity 差值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 50 | -0.0353 | 0.1294 | 3/6 | +1.7359 | +0.0086 | +0.0312 | -0.0030 |
| 0.0005 | 100 | -0.0443 | 0.1046 | 2/6 | -0.4800 | -0.0048 | -0.0112 | -0.0251 |
| 0.001 | 50 | -0.1400 | 0.1180 | 0/6 | -0.6675 | -0.0072 | -0.0114 | -0.0466 |
| 0.001 | 100 | -0.1602 | 0.1235 | 0/6 | -0.6715 | -0.0062 | -0.0108 | -0.0419 |

结果显示，`lambda_gb=0.001` 在动态刷新后仍然明显差于 B0，6 个 pair 全部低于 B0。`lambda_gb=0.0005` 有所缓和，但最好也只是 3/6 不低于 B0，没有达到“多数不低于 B0”的成功标准。

## B2 相对静态 B1 的结果

表中 `acc_ge_b1` 表示 6 个 dataset-seed pair 中，B2 线性分类准确率不低于同 lambda 静态 B1 的个数。

| lambda_gb | refresh_interval | 平均 accuracy 差值 | std | acc_ge_b1 | 平均 effective rank 差值 | 平均 AD 差值 | 平均 radius 差值 | 平均 purity 差值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 50 | -0.0338 | 0.1131 | 2/6 | +1.6044 | +0.0127 | +0.0457 | -0.0056 |
| 0.0005 | 100 | -0.0428 | 0.0722 | 2/6 | -0.6115 | -0.0006 | +0.0033 | -0.0276 |
| 0.001 | 50 | -0.1542 | 0.1513 | 1/6 | +0.2363 | +0.0012 | -0.0017 | -0.0236 |
| 0.001 | 100 | -0.1743 | 0.1264 | 0/6 | +0.2324 | +0.0022 | -0.0012 | -0.0189 |

动态刷新没有整体优于静态 B1。对于 `lambda_gb=0.001`，B2 比 B1 更差；对于 `lambda_gb=0.0005`，B2 只在少数 seed 上改善，平均 accuracy 仍然下降。

## 重点 seed 检查

本轮重点关注之前不稳定的 Cora seed1、Cora seed2、CiteSeer seed2。

| setting | dataset | seed | B2 acc | B1 acc | acc 差值 | B2 eff_rank | B1 eff_rank | eff_rank 差值 | AD 差值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lambda_0p0005_refresh_100 | CiteSeer | 2 | 0.463 | 0.535 | -0.072 | 4.8126 | 5.0265 | -0.2139 | -0.0017 |
| lambda_0p0005_refresh_100 | Cora | 1 | 0.334 | 0.283 | +0.051 | 2.0531 | 1.1503 | +0.9028 | +0.0148 |
| lambda_0p0005_refresh_100 | Cora | 2 | 0.402 | 0.525 | -0.123 | 7.6453 | 7.0750 | +0.5702 | +0.0110 |
| lambda_0p0005_refresh_50 | CiteSeer | 2 | 0.471 | 0.535 | -0.064 | 4.8953 | 5.0265 | -0.1312 | -0.0008 |
| lambda_0p0005_refresh_50 | Cora | 1 | 0.413 | 0.283 | +0.130 | 4.1635 | 1.1503 | +3.0131 | +0.0485 |
| lambda_0p0005_refresh_50 | Cora | 2 | 0.356 | 0.525 | -0.169 | 3.4298 | 7.0750 | -3.6452 | -0.0249 |
| lambda_0p001_refresh_100 | CiteSeer | 2 | 0.369 | 0.504 | -0.135 | 3.4733 | 3.7702 | -0.2969 | -0.0026 |
| lambda_0p001_refresh_100 | Cora | 1 | 0.221 | 0.226 | -0.005 | 1.6033 | 1.4689 | +0.1344 | +0.0001 |
| lambda_0p001_refresh_100 | Cora | 2 | 0.215 | 0.541 | -0.326 | 5.2849 | 5.3947 | -0.1098 | -0.0037 |
| lambda_0p001_refresh_50 | CiteSeer | 2 | 0.386 | 0.504 | -0.118 | 3.4959 | 3.7702 | -0.2743 | -0.0029 |
| lambda_0p001_refresh_50 | Cora | 1 | 0.281 | 0.226 | +0.055 | 1.8023 | 1.4689 | +0.3334 | +0.0011 |
| lambda_0p001_refresh_50 | Cora | 2 | 0.254 | 0.541 | -0.287 | 5.2396 | 5.3947 | -0.1552 | -0.0053 |

重点 seed 的结果说明：动态刷新可以在 Cora seed1 上改善静态 B1，但没有解决 Cora seed2 和 CiteSeer seed2 的下降。尤其是 Cora seed2，在所有 B2 设置下仍然明显低于 B1 和 B0。

## 已被实验支持的结论

1. 动态刷新粒球结构本身不能解决当前 B1 的不稳定问题。
2. `lambda_gb=0.001` 对当前 negative-free baseline 来说仍然过强，即使加入刷新也会明显损伤 accuracy。
3. `lambda_gb=0.0005` 比 `0.001` 稳定一些，但仍未达到多数 seed-dataset pair 不低于 B0 的成功标准。
4. compactness 被真实激活，但 AD/radius 的改善不能稳定转化为 accuracy、purity 或 effective rank 的改善。
5. 目前的结果继续支持一个判断：单纯优化局部粒球 compactness 容易与全局语义可分性发生冲突。

## 合理猜测

1. B1/B2 的主要问题可能不是“粒球是否过时”这一单点问题，而是 compactness loss 本身缺少可靠性门控、anti-collapse 或全局分离约束。
2. 动态刷新可能会把 encoder 的短期波动直接反馈给粒球 assignment，使约束目标变得更不稳定。这个解释符合结果，但本轮没有直接测量 assignment churn，所以仍属于猜测。
3. 当前 raw negative-free encoder 表示空间本身偏压缩，粒球约束继续拉近局部结构时，容易进一步牺牲语义可分性。

## 尚未验证的假设

1. 如果换成 BN-stabilized negative-free encoder，B2 是否会更稳定。
2. 如果换成 InfoNCE baseline，粒球 compactness 是否会从“过度压缩”变成“适度局部约束”。
3. 如果加入球可靠性权重、purity-free confidence、assignment EMA 或低 churn 约束，动态刷新是否会变得可用。
4. 角距离和余弦距离的差异尚未在本轮验证，仍留给后续猜想 3。

## 被负结果削弱的方向

1. “只要定期刷新粒球，就能解决 B1 不稳定”的方向被削弱。
2. “继续单纯调小 lambda 或调刷新间隔”作为主线不值得继续投入太多。
3. “用 raw negative-free + 粒球 compactness 直接构成方法贡献”的方向风险很高。

## 下一步建议

下一步不建议继续在 raw negative-free 上微调刷新间隔。更值得做的是换一个更稳定的 encoder baseline，再验证粒球约束：

1. 使用 BN-stabilized negative-free 或 InfoNCE 作为稳定 encoder。
2. 保持粒球约束很弱，只作为 soft local prior。
3. 加入明确的防塌缩指标门槛，例如 effective rank、feature std、linear accuracy 不允许明显下降。
4. 如果继续做动态刷新，需要增加 assignment churn 诊断，记录相邻刷新之间节点所属球变化比例，否则无法判断刷新是在修正旧球，还是引入新噪声。

## 证据文件

- `results_b2_refresh_sweep/stage_b_summary.csv`
- `results_b2_refresh_sweep/stage_b_paired_deltas_vs_b0.csv`
- `results_b2_refresh_sweep/stage_b_paired_deltas_vs_b1.csv`
- `results_b2_refresh_sweep/configs/`
- `C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs\gb_denoise_stage_b_compactness`
