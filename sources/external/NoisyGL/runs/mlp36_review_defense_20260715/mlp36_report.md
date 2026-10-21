# MLP-36 LOO 与显式平滑审稿防御报告

## 协议与完整性

- Cora/CiteSeer/CoraFull/Coauthor-CS/Amazon-Photo/PubMed。
- uniform 0.4、train-only、percent 0.2/0.1/0.7、seeds 0-4。
- 每个 dataset/method 均为 5 个成功运行；逐 seed split/noise hash 完全配对；val/test 修改数均为 0。

## 五种子精度

| Dataset | Current | LOO | LOO shuffled | Uniform LS | Prior LS | LOO-Current (pp) |
|---|---:|---:|---:|---:|---:|---:|
| cora | 55.59 | 57.15 | 42.79 | 44.69 | 44.89 | +1.56 |
| citeseer | 53.83 | 55.21 | 41.62 | 43.69 | 43.43 | +1.37 |
| corafull | 37.33 | 37.18 | 32.18 | 32.59 | 32.63 | -0.15 |
| coauthorcs | 90.62 | 90.61 | 87.46 | 88.25 | 88.04 | -0.01 |
| amazonpho | 81.59 | 82.07 | 72.83 | 74.18 | 74.72 | +0.47 |
| pubmed | 71.69 | 72.02 | 63.38 | 63.67 | 64.25 | +0.32 |

## 终判

- LOO 相对 current 的六数据集平均 paired delta：+0.60pp。
- LOO 对 current 非负：4/6；达到 4/5 seeds 正向：3/6。
- LOO 同时超过 uniform/prior smoothing：6/6。
- LOO 超过 LOO-shuffled：6/6，其中 4/5 seeds 正向：6/6。
- 机制结论：`external_local_consensus_supported`。

LOO 在不使用当前节点自身投票的情况下保持或提高总体表现，因此现有收益不能仅归因于自标签回流。同时，LOO 并非在每个数据集都显著优于 current，论文应将其作为机制消融，避免把结果写成统一算法升级。

## 统计口径

逐数据集 paired t-test 只用于预声明 comparison family，Holm 校正在每个六数据集 family 内完成。由于 n=5，正文以 paired delta、95% CI 与方向计数为主，不把未拒绝零假设写成等效。
