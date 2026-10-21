# 验证标签噪声敏感性实验报告

## 完整性

- noisy-validation 正式行：90/90。
- 测试标签污染：0。
- 旧 GSDN 身份：provisional adapted implementation。

## 预注册判定

- GB-LCSS vs shuffled 三数据集：True。
- GB-LCSS vs MLP+CE 至少二数据集：True。
- shuffled 下界：True。
- 主要机制结论：通过。

## 配对结果

| 协议 | 数据集 | 对照 | 均值差(pp) | 正向 seeds |
|---|---|---|---:|---:|
| clean_validation | cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 12.64 | 5/5 |
| clean_validation | cora | mlp_ce | 12.04 | 5/5 |
| clean_validation | cora | postel_mlp_noipl_rm | 9.19 | 5/5 |
| clean_validation | cora | gsdn | 0.32 | 3/5 |
| clean_validation | citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 10.63 | 5/5 |
| clean_validation | citeseer | mlp_ce | 10.03 | 5/5 |
| clean_validation | citeseer | postel_mlp_noipl_rm | 7.68 | 5/5 |
| clean_validation | citeseer | gsdn | 3.16 | 4/5 |
| clean_validation | corafull | frozengrace_gb_mlp_softdist_beta05_shuffled | 5.16 | 5/5 |
| clean_validation | corafull | mlp_ce | 5.32 | 5/5 |
| clean_validation | corafull | postel_mlp_noipl_rm | 0.79 | 5/5 |
| clean_validation | corafull | gsdn | 1.50 | 5/5 |
| noisy_validation | cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 12.45 | 5/5 |
| noisy_validation | cora | mlp_ce | 12.64 | 5/5 |
| noisy_validation | cora | postel_mlp_noipl_rm | 9.14 | 5/5 |
| noisy_validation | cora | gsdn | 1.14 | 3/5 |
| noisy_validation | citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 11.45 | 5/5 |
| noisy_validation | citeseer | mlp_ce | 11.41 | 5/5 |
| noisy_validation | citeseer | postel_mlp_noipl_rm | 8.20 | 5/5 |
| noisy_validation | citeseer | gsdn | 3.87 | 5/5 |
| noisy_validation | corafull | frozengrace_gb_mlp_softdist_beta05_shuffled | 4.96 | 5/5 |
| noisy_validation | corafull | mlp_ce | 5.05 | 5/5 |
| noisy_validation | corafull | postel_mlp_noipl_rm | 1.30 | 5/5 |
| noisy_validation | corafull | gsdn | 0.91 | 4/5 |

## 排序敏感性

| 数据集 | Kendall tau | 前二重合数 |
|---|---:|---:|
| cora | 0.867 | 2/2 |
| citeseer | 0.867 | 1/2 |
| corafull | 0.867 | 1/2 |

## 解释边界

本实验检验相对排序对验证标签污染的敏感性，不把绝对精度下降直接解释为方法失效。旧 `gsdn` 只作临时 adapted 参照，不能支持对 GSDN 的最终强弱结论。
