# 75 个 p 值的单一 Holm 敏感性重算报告

## 范围与输入

本分析严格限于已有配对检验的统计敏感性重算：未训练模型，未改变任何 seed 结果、效应量、置信区间或论文正文，也未修改原始统计表。

- 75-test 输入：`..\..\external\NoisyGL\runs\mlp36_review_defense_20260715\primary_statistics_holm.csv`；SHA-256 `7AE75564FC1C9FF98EDB3CC0FCC401C8AB711FA401A0C3EAE484F4F9F30C8C9C`。
- seed-level 输入：`..\..\external\NoisyGL\runs\mlp36_review_defense_20260715\primary_statistics_raw.csv`；SHA-256 `42B7968222CF2340E7E761DE5FF2C9CC77AB2D969D301619BF34F4ECBA045D27`；500 行、25 个 setting、500 个唯一五列键。
- 输出 CSV：`paper_pipeline\00_evidence\holm_75_sensitivity.csv`；SHA-256 `988B7D7E801769FC7B18D04EA38DF27582093C6ED0403016FD4A7CBACCA6CB12`。

## 实际列与唯一键

- `primary_statistics_raw.csv` 的实际列为 `dataset, noise_type, noise_rate, seed, method, test`。它提供 25 settings × 5 seeds × 4 methods 的 500 个 seed-level 观测；其唯一键为 `(dataset, noise_type, noise_rate, seed, method)`。该表不含 p 值列。
- `primary_statistics_holm.csv` 的实际列为 `dataset, noise_type, noise_rate, comparison, n, mean_delta, std_delta, ci95_low, ci95_high, positive_seeds, negative_seeds, paired_t_p, holm_p`。
- 本次 75 个检验的唯一键是 `(dataset, noise_type, noise_rate, comparison)`；校验得到 75/75 个唯一键，无重复。
- 原始 p 值只从现有列 `paired_t_p` 读取；现有列 `holm_p` 映射为各 comparator 内 25-test Holm 校正值 `holm_p_25_family`。没有从列名、行顺序或效应量猜测 p 值。
- 三个 comparison 分别为 `lcss_vs_ce`、`lcss_vs_jocor`、`lcss_vs_shuffled`，每组各 25 项。

## Holm step-down 公式

令全部检验数 \(m=75\)，将原始 p 值升序排列为
\(p_{(1)}\le\cdots\le p_{(m)}\)。第 \(i\) 项的未单调化候选值为

\[
s_{(i)}=\min\{1,(m-i+1)p_{(i)}\}.
\]

Holm 调整值采用累计最大值保证单调性：

\[
\tilde p_{(i)}=\max_{1\le j\le i}s_{(j)}.
\]

最后按四列唯一键映射回原行。CSV 中 `holm_rank_75` 是升序位置，
`holm_factor_75=75-i+1`，`holm_scaled_p_75=s_{(i)}`，
`holm_p_75_family=\tilde p_{(i)}`。并列 p 值仅以四列键作确定性排序，
不改变并列项的最终调整值。

## 数量与校验

- 输入检验数：75；原始 `paired_t_p < 0.05`：69。
- 最小原始 p 值：8.3505533151580126e-08。
- 现有三个 25-test 家族的 Holm 独立复算最大绝对差：
  0。
- 按 `holm_rank_75` 检查的调整值单调性违反数：0。
- `holm_p_75_family < paired_t_p` 的违反数：0。
- `holm_p_75_family < holm_p_25_family` 的违反数：0。

| comparison | 25-test 家族显著数 | 单一 75-test 家族显著数 | 减少 |
|---|---:|---:|---:|
| `lcss_vs_ce` | 24 | 19 | 5 |
| `lcss_vs_jocor` | 8 | 6 | 2 |
| `lcss_vs_shuffled` | 25 | 17 | 8 |
| **合计** | **57** | **42** | **15** |

在 \(\alpha=0.05\) 下，单一 75-test 家族保留 42/75 项，
而三个分开的 25-test 家族合计为 57/75 项；有
15 项从“族内显著”变为“75-test 不显著”，没有反向变化。

## 显著性状态改变的项目

| dataset | noise_type | noise_rate | comparison | raw p | Holm-25 p | Holm-75 p |
|---|---|---:|---|---:|---:|---:|
| `coauthorcs` | `uniform` | `0.6` | `lcss_vs_ce` | 0.00160970662 | 0.00965823971 | 0.0531203184 |
| `cora` | `pair` | `0.4` | `lcss_vs_ce` | 0.00176354189 | 0.00965823971 | 0.0564333405 |
| `corafull` | `instance` | `0.4` | `lcss_vs_shuffled` | 0.00186849204 | 0.0150334209 | 0.0579232532 |
| `cora` | `uniform` | `0.6` | `lcss_vs_shuffled` | 0.00214311097 | 0.0150334209 | 0.0642933292 |
| `coauthorcs` | `uniform` | `0.4` | `lcss_vs_shuffled` | 0.0023302514 | 0.0150334209 | 0.0675772907 |
| `corafull` | `pair` | `0.4` | `lcss_vs_jocor` | 0.00233840933 | 0.0444297773 | 0.0675772907 |
| `corafull` | `uniform` | `0.6` | `lcss_vs_jocor` | 0.00240495624 | 0.0444297773 | 0.0675772907 |
| `coauthorcs` | `instance` | `0.4` | `lcss_vs_shuffled` | 0.00271928917 | 0.0150334209 | 0.0707015184 |
| `amazonpho` | `uniform` | `0.4` | `lcss_vs_shuffled` | 0.00382978544 | 0.0153191418 | 0.083200295 |
| `coauthorcs` | `uniform` | `0.2` | `lcss_vs_shuffled` | 0.00488379153 | 0.0153191418 | 0.0976758306 |
| `citeseer` | `uniform` | `0.6` | `lcss_vs_ce` | 0.00569865029 | 0.0227946012 | 0.108274356 |
| `cora` | `uniform` | `0.6` | `lcss_vs_ce` | 0.00588335039 | 0.0227946012 | 0.108274356 |
| `citeseer` | `pair` | `0.4` | `lcss_vs_ce` | 0.00595554192 | 0.0227946012 | 0.108274356 |
| `corafull` | `uniform` | `0.2` | `lcss_vs_shuffled` | 0.00672737941 | 0.0153191418 | 0.108274356 |
| `citeseer` | `pair` | `0.4` | `lcss_vs_shuffled` | 0.0103340756 | 0.0153191418 | 0.144677059 |

## 结论边界

### 已被本次重算支持

1. 对这 75 个冻结的 `paired_t_p` 一次性控制 Holm FWER 后，显著项数为
   42/75；按 CE、JoCoR、shuffled 分别为
   19/6/17。
2. 现有三个作者定义的 25-test 家族算术可复算，显著项数为
   24/8/25；
   单一 75-test 敏感性分析更保守，减少 15 项。
3. 本次调整不改变 `mean_delta`、`std_delta`、95% CI、正/负方向 seed 计数或
   任一原始 p 值。

### 仅为合理解释

显著项减少来自检验族从 25 扩大到 75 后阈值变严；它说明统计结论对 family
定义敏感，但不说明效应量本身变小，也不构成方法失效的证据。

### 尚未由本次分析验证

本次没有检验 n=5 配对差的正态性、异常值稳健性，也没有建立比较族在观察结果前
已外部预注册的时间证据。75-test Holm 只回答给定 75 个 p 值下的 FWER
敏感性，不替代效应量、95% CI、方向计数或稳健统计分析。

### 已被敏感性结果削弱的表述

不能把三个分别校正的 25-test 家族结果表述为“全论文 75 个确认性检验的统一
FWER 保证”，也不能用 24/8/25 概括单一 75-test 家族；对应数字应明确标注
family 定义。

## 最终决策

保留三个 comparator-specific 25-test 家族作为作者定义、非外部预注册的原报告，
同时增加本文件的单一 75-test Holm 作为敏感性分析。论文层面的显著性措辞应限定到
具体 family 与 setting；本次不产生新的模型性能 claim。
