# Phase A Smoke Report - Coverage-Aware Gated Soft Correction

## 研究假设

在 C1 的真实粒球基础上加入 coverage gate 与 confidence gate，只在球内 train 标签覆盖和投票置信度足够时启用 soft correction，可以减少低覆盖球带来的错误校正，并保持粒球去噪收益。

## 粒球扮演的角色

粒球作为局部语义关系单元和 train-label coverage 统计单元。新增方法不把 same-ball membership 当 hard positive，而是用球内 train noisy label 分布构造 gated soft target。

## 方法动机

CGBC/GBmooer 默认标签覆盖率高，球内多数投票天然更可靠；图节点半监督设置中 train label 稀疏，因此需要显式 coverage-aware gating。

## 代码改动

- `src/gb_stage_b/losses.py`: 新增 `gated_ball_soft_correction`。
- `src/gb_stage_b/config.py`: 新增 `C1_gated`、`C1_gated_noball`、`C1_gated_shuffled` 与 `gate_tau_cov`、`gate_tau_conf`、`gate_alpha`。
- `src/gb_stage_b/cli.py`: 新增 `--gate-tau-cov`、`--gate-tau-conf`、`--gate-alpha`。
- `src/gb_stage_b/train.py`: `C1_gated*` 接入 gated soft target，原 `C1` 逻辑保持不变。
- `tests/`: 新增 gate 行为、防泄漏、配置与 CLI 测试。

## 配置改动

基础设置：Cora, ratio split, train_ratio=0.2, val_ratio=0.1, symmetric noise_rate=0.4, seed=0, Phase1=1000 epochs, Phase2=200 epochs。

调参轮次：

| method | tau_cov | tau_conf | alpha | classifier_test_acc |
|---|---:|---:|---:|---:|
| C1 baseline | - | - | - | 0.7443 |
| C1_gated | 3 | 0.6 | 0.3 | 0.6389 |
| C1_gated | 3 | 0.6 | 0.7 | 0.6531 |
| C1_gated | 3 | 0.6 | 1.0 | 0.6552 |
| C1_gated | 2 | 0.5 | 1.0 | 0.7143 |
| C1_gated_noball | 3 | 0.6 | 0.3 | 0.5704 |
| C1_gated_shuffled | 2 | 0.5 | 1.0 | 0.5756 |

## 预期机制

通过 `n_train(B_i) >= tau_cov` 避免低覆盖球参与校正；通过 `margin(B_i) >= tau_conf` 避免 train 标签投票不一致的球参与校正；通过 `alpha` 控制 noisy one-hot 与球内 train-label 分布的混合强度。

## 成功标准

- GA1: `C1_gated >= C1`
- GA2: `C1_gated > C1_gated_shuffled + 5pp`
- GA3: `C1_gated > C1_gated_noball + 10pp`

## 失败标准

最多 3 轮调参后仍不能达到 GA1，则不进入 Phase B，暂缓 graph agreement gate。

## 实验结果

最佳 `C1_gated` 为 `tau_cov=2, tau_conf=0.5, alpha=1.0`，test_acc=0.7143。

Gate 判定：

| Gate | 结果 | 依据 |
|---|---|---|
| GA1 | FAIL | 0.7143 < C1 0.7443，低 3.00pp |
| GA2 | PASS | 0.7143 > C1_gated_shuffled 0.5756，高 13.86pp |
| GA3 | PASS | 0.7143 > C1_gated_noball 0.5704，高 14.39pp |

## 最终决策

暂缓 Phase B。Coverage-aware gated soft correction 保留了真实球相对 noball/shuffled 的贡献，但未超过现有 hard majority C1。当前证据不支持把 `C1_gated` 升级为主方法，也不支持继续实现 graph agreement gate。下一步应先分析为什么 soft distribution target 弱于 C1 hard majority target，再决定是否改成 hard correction + coverage/confidence gate 的更小变体。

