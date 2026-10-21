# MLP-24 忠实 ANL-CE / GMM+CT 基线补跑

## 研究假设

此前表中的 `mlp_anl` 与 `mlp_ct` 是机制代理，不能直接以已发表方法名进入论文排名。若在统一 MLP 骨干与 train-only 标签噪声协议下实现论文核心算法，能够得到可审计的公平对照；若 GMM+CT 的精确 Mann-Kendall 更新在大图上不可承受，则明确报告可扩展性边界，不以近似实现冒充原方法。

## 方法角色

- `mlp_anl_ce`：不使用粒球；使用官方 ANL-CE 的 NCE+NNCE+L1 目标。
- `mlp_gmm_ct`：不使用粒球；使用 noisy-label confidence gap 的 Mann-Kendall 趋势与低损失 GMM 选择并集。

## 固定协议

- 主 smoke：Cora / CiteSeer，uniform 0.4，train-only，percent split 0.2/0.1/0.7，seed 0。
- 骨干：2-layer MLP，hidden 64，dropout 0.5，Adam，200 epochs。
- ANL-CE：alpha=5，beta=5，delta=5e-5，min_prob=1e-7。
- GMM+CT：warmup=30，alpha=0.01，GMM threshold=0.5，reg_covar=5e-4。
- 方法称为 MLP-adapted，不声称复现原图像训练栈。

## 成功标准

1. 四行均 `status=ok`，test/valid 有限；
2. GMM+CT 最终选择集非空且未全程坍缩为单样本；
3. Cora/CiteSeer 单 run 时间可接受；
4. 只有通过内存和时间投影后才扩到大图与 5 seeds。

## 失败标准

- NaN、OOM、单类输出坍缩；
- CT 精确历史超过 2 GiB 或预计单 run 超过 30 分钟；
- 为跑通而改成 EMA/top-k 等非官方核心统计。

## 输出

- `runs/mlp24_faithful_anl_ct_20260712/smoke_seed0.csv`
- 本文档追加实验结果与最终决策。

## Seed-0 结果

| 数据集 | ANL-CE | GMM+CT | GMM+CT 时间 | CT 历史内存 |
|---|---:|---:|---:|---:|
| Cora | 38.10 | 41.74 | 4.47 s | 2.89 MiB |
| CiteSeer | 48.02 | 46.09 | 3.03 s | 3.04 MiB |
| Coauthor-CS | 85.52 | 88.48 | 12.10 s | 41.97 MiB |
| Amazon-Photo | 72.27 | 74.49 | 3.64 s | 9.34 MiB |
| CoraFull | 29.80 | 32.00 | 19.95 s | 211.43 MiB |

五个数据集均 `status=ok`，无 NaN/OOM。CoraFull 的精确 CT 历史低于 2 GiB 且单 run 远低于 30 分钟，故通过工程可行性门，继续补 seeds 1--4。需注意最终 GMM+CT 选择比例约为 0.98--1.00，说明在该 full-batch MLP 协议下 GMM 分量趋于把几乎全部样本选回；这是方法迁移后的实测行为，不通过改阈值追点。

## 当前决策

- `mlp_anl_ce` 与 `mlp_gmm_ct` 均扩到五个核心数据集、5 seeds。
- 旧 `mlp_anl` / `mlp_ct` 继续保留为历史 proxy，不覆盖旧结果。
- 论文只使用新 ID 的五种子结果作为 ANL-CE / GMM+CT 已发表机制对照。

## 五种子终判

| 数据集 | ANL-CE | GMM+CT |
|---|---:|---:|
| Cora | 43.35+-3.77 | 43.53+-1.41 |
| CiteSeer | 44.27+-4.28 | 43.93+-2.72 |
| CoraFull | 31.09+-1.46 | 31.98+-0.98 |
| Coauthor-CS | 84.46+-0.91 | 88.42+-0.62 |
| Amazon-Photo | 67.27+-3.40 | 73.90+-1.80 |

完整性为 50/50 行、全部 `status=ok`。GMM+CT 在后三个数据集与 MLP+CE 锚点逐项相同，结合最终 0.98--1.00 的选择比例，表明 GMM 选择器在该 full-batch MLP 协议下基本退化为保留全体训练样本。ANL-CE 也未形成稳定强基线。该结果用于忠实度修复，不因结果不利而删除。
