# Phase 41: 免K粒球图节点聚类可行性最小验证（2026-06-30）

## 研究假设

SoftDist 分类/检测线已经多次证明：在 frozen GRACE `H` 上构造粒球，再让强下游模型使用同源表示，粒球信号容易被 `H`-kNN、GCN `[X||H]` 或其它强 baseline 吸收。因此本轮尝试彻底换任务：从 noisy-label node classification 转向 graph node clustering。

核心假设是：聚类任务没有下游分类器吸收问题，粒球可以作为聚类引擎本身，尤其利用“粒球数由 split_improvement 自适应停止”形成免K能力。若进一步把几十到几百个粒球凝聚成簇，并自动得到 `K_auto`，则可能形成 “免K + 误设K鲁棒” 的论文卖点。

## 粒球扮演的角色

粒球不再作为 soft target teacher、噪声检测器或下游辅助信号，而是作为聚类的中间粒度单元：

1. 在 frozen GRACE `H` 上构造 ordinary granular balls。
2. 以每个 ball 为节点构造球图。
3. 在球图上用 modularity greedy agglomeration 自动凝聚成簇。
4. 将球簇标签广播回节点，得到 node clustering。

真实类别只用于评测 ACC/NMI/ARI，不进入构球、球图构造或凝聚决策。

## 方法动机

图聚类和分类的关键差异是：聚类评价承认“免K”和“误设K鲁棒”是独立价值，而分类任务不承认“轻量但不强”的方法优势。k-means、谱聚类和很多深度聚类方法通常需要真实簇数 `K`；若粒球凝聚能在不输入 `K` 的情况下逼近 true-K kmeans，并且在 kmeans 被喂错 `K` 时更稳，就可能形成新的主线。

## 代码改动

主要位于 `external/NoisyGL`：

- 新增脚本：`scripts/diagnose_gb_clustering.py`
- 新增测试：`tests/test_gb_clustering.py`
- 新增结果目录：`runs/gb_clustering_feasibility_20260630/`
- 回填计划文档：`docs/execution_plan_gb_clustering_feasibility_20260630.md`

脚本实现内容：

- Hungarian ACC
- NMI/ARI
- ordinary granular balls on `H`
- ball graph construction using ball-center cosine top-k graph
- NetworkX greedy modularity communities as no-K ball agglomeration
- `kmeans_H_trueK`
- `spectral_H_trueK`
- `kmeans_H_autoK`
- wrong-K kmeans scan
- G1-G4 gate report

## 配置改动

无模型训练配置改动。实验协议：

- datasets：`cora,citeseer,pubmed`
- seeds：`0,1,2,3,4`
- split improvements：`0.02,0.05,0.1`
- main split：`0.05`
- clustering uses all nodes
- no label noise
- labels evaluation-only
- ball graph k：`10`
- edge density weight：`0.0`

## 预期机制

若假设成立，应看到：

- `gb_cluster` 在免K下的 NMI 至少达到 true-K kmeans 的 90%。
- `gb_cluster` ACC 与 true-K kmeans 差距不超过 5pp。
- `K_auto` 与真类数同量级。
- kmeans 被喂错 `K` 时显著下降，而 `gb_cluster` 因免K保持稳定。
- `gb_cluster` 明显强于 `kmeans_H_autoK`，证明球图凝聚不只是自动定K。

## 成功标准

G1 决定性标准：

- `gb_cluster` NMI >= `0.9 * kmeans_H_trueK`
- `gb_cluster` ACC 与 `kmeans_H_trueK` 差距 >= `-5pp`
- 至少 `2/3` 数据集通过

G2 决定性标准：

- wrong-K kmeans 设置下，`gb_cluster` 平均 ACC 优势 >= `+5pp`
- 至少 `2/3` 数据集通过

G3/G4 为诊断标准：

- K_auto/trueK in `[0.5, 2.0]`
- `gb_cluster` 明显优于 `kmeans_H_autoK`

## 失败标准

- G1 不过：免K精度掉太多，图聚类主线关闭。
- G2 不过：免K没有实际鲁棒收益，核心卖点空心。
- G3 不过：凝聚机制 K_auto 退化。
- G4 不过：球凝聚没有独立贡献，方法退化为“自动定K + kmeans”甚至更弱。

## 实验结果

测试：

```powershell
py -3 -m pytest tests/test_gb_clustering.py -q
```

结果：`4 passed`。

主 split `0.05` gate：

| Dataset | true K | K_auto | balls | GB ACC | kmeans trueK ACC | ACC gap | GB NMI | kmeans trueK NMI | NMI ratio | wrong-K ACC adv | G1 | G2 | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 7 | 3.00 | 52.40 | 0.4823 | 0.6317 | -0.1493 | 0.3404 | 0.5193 | 0.6555 | -0.1208 | 0 | 0 | 0 | 0 |
| CiteSeer | 6 | 4.40 | 63.80 | 0.5342 | 0.6395 | -0.1053 | 0.2942 | 0.3782 | 0.7779 | +0.0240 | 0 | 0 | 1 | 0 |
| PubMed | 3 | 3.40 | 64.00 | 0.5248 | 0.6766 | -0.1518 | 0.1918 | 0.3107 | 0.6172 | +0.0096 | 0 | 0 | 1 | 0 |

完整结果文件：

- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/gb_clustering_report.md`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/metric_raw.csv`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/metric_summary.csv`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/wrong_k_raw.csv`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/wrong_k_summary.csv`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/gate_summary.csv`

## 最终决策

放弃。Phase 41 返回 **NO-GO**。

原因不是脚本或评测失败，而是核心卖点失败：

- G1 在三个数据集全失败。`gb_cluster` 与 true-K kmeans 的 ACC 差距约 `-10.5pp` 到 `-15.2pp`，NMI ratio 只有 `0.62~0.78`。
- G2 在三个数据集全失败。Cora 中本方法比 wrong-K kmeans 还低 `-12.08pp`；CiteSeer/PubMed 只有 `+2.40pp/+0.96pp`，没有达到 `+5pp`。
- G4 全失败。`gb_cluster` 不优于 `kmeans_H_autoK`，说明当前球图 modularity 凝聚没有独立贡献。

## 经验边界

本结果说明：把粒球转成图聚类引擎，并不能自动摆脱“GRACE H 上粒球只是局部粗粒化结构”的问题。即使分类任务中的下游吸收问题被移除，球→簇凝聚仍然没有比 kmeans 更有效，且免K鲁棒性没有形成。

后续不建议继续投入“GRACE H + ordinary balls + 无参凝聚”的图聚类新主线。若必须继续图聚类，需要完全新的机制证据，例如不同于 kmeans 的 density/connectedness objective、可证明的 K selection criterion、或明确超过 `kmeans_H_autoK` 的球图凝聚贡献；否则会变成又一轮调参循环。

当前最稳的收口路线仍是回到 SoftDist 分类论文：不再追求新 SOTA，而写成受控粒球增量、适用性边界和系统性负结果。
