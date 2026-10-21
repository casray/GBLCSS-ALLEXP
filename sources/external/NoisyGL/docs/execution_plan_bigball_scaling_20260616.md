# 执行计划：大图球构造参数适配诊断（Phase 32）

日期：2026-06-16
执行者：Codex
前置状态：Phase 28 DBLP no-go——DBLP GRACE H 正常（17716×256，对齐，row-norm 1.0）但球构造塌缩到 `num_balls=1`，SoftDist==shuffled。系统性模式：球只在 Cora(2.7k)/CiteSeer(3.3k) 两个最小引文图有效，所有更大图（PubMed 19.7k / Amazon-Photo 7.6k / DBLP 17.7k）全部失败。

**本计划要回答一个尚未排除的关键问题**：大图球塌缩是 **(A) 本质限制**（大图 GRACE 几何就是不可分球，方法只适用小图）还是 **(B) 参数失配**（球构造参数 `max_balls=64/min_ball_size=32/split_improvement=0.05` 是在 2-3k 节点小图上定的，搬到 17-19k 节点大图后分裂条件永不触发）？

`num_balls=1` 强烈提示 (B)：一个球都没分出来，更像"分裂准则没满足"而非"分了但没语义"。若是 (B)，重调参数可能解锁 DBLP/PubMed/Computers，把数据集从 2 个救到 5 个——当前 ROI 最高的一步。**本计划是纯诊断,不改主方法、不重训 H、不跑完整下游。**

---

## 0. 环境与红线

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H（不重新预训练）：
  - DBLP: `gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/DBLP/grace/H.pt`（17716×256）
  - PubMed: `checkpoints/embeddings_v2/PubMed/grace/H.pt`（19717×256）
  - Amazon-Photo: `checkpoints/embeddings_v2/AmazonPhoto/grace/H.pt`（7650×?）
  - Cora/CiteSeer: 现有锚点 H（作为"已知可聚类"的对照基准）
- **红线**：
  1. anchor 机制（`frozengrace_gb_gcn_softdist_xh` 默认 max_balls=64/min_ball_size=32/split_improvement=0.05）**绝不改动**——参数扫描只在诊断脚本里 override，不碰 anchor config 与 predictor 默认值
  2. clean label 只进诊断（true_purity）；球构造与投票只用几何 H + train noisy 标签
  3. 纯诊断,不跑下游训练（除非诊断过门,见 Phase 32c）
  4. research-record 报告；测试全绿

---

## Phase 32a：球构造参数扫描诊断（核心,几分钟,不训练）

### 目的

在**已有的大图 H 上**，扫球构造参数，看 num_balls 和 true_purity 能否恢复到 Cora/CiteSeer 水平。直接区分 (A) vs (B)。

### 实现

新脚本 `scripts/diagnose_ball_construction_scaling.py`（复用现有 `build_ordinary_granular_balls` + `partition_purity_diagnostics`，不写新建球逻辑）：

1. 对每个数据集 H ∈ {Cora, CiteSeer（对照基准）, DBLP, PubMed, AmazonPhoto}，加载 H + 同协议 split（percent 0.2/0.1/0.7）+ uniform 0.4 noisy 标签，seeds 0,1,2
2. 参数网格（笛卡尔积）：
   - `min_ball_size ∈ {32, 64, 128, 256, 512}`（关键怀疑变量——大图需要更大的最小球）
   - `split_improvement ∈ {0.05, 0.01, 0.005, 0.001}`（关键怀疑变量——大图簇内方差结构不同，0.05 阈值可能太严）
   - `max_balls ∈ {64, 128, 256}`（允许更多球）
3. 对每个 (数据集 × 参数点 × seed) 计算并记录：
   - `num_balls`（是否还塌缩）
   - `mean_ball_size`、`singleton_ratio`
   - `true_all_purity_weighted` 与 `noisy_train_purity_weighted`（球是否捕捉到真实类结构）
   - `mean_train_per_ball`（覆盖是否够投票）
4. 输出 `runs/ball_scaling_diag_20260616/`：raw CSV（全网格）、summary、报告 md

### 关键对照逻辑（报告必须给）

- **Cora/CiteSeer 在默认参数下 true_purity ≈ 0.79/0.70**——这是"可聚类"的基准线
- 问：DBLP/PubMed/AmazonPhoto 在**任何**参数点能否达到接近的 true_purity 且 num_balls ≥ 8？
- 同时报告每个数据集"达到非塌缩（num_balls≥8）所需的最小 min_ball_size"——若大图需要的 min_ball_size 随节点数大致成比例，强烈支持 (B)

### 判定（决定 32b/32c 是否做）

| 诊断结果 | 结论 | 后续 |
|---|---|---|
| 某参数点 DBLP/PubMed 等 num_balls≥8 **且** true_purity 接近 Cora/CiteSeer 水平 | **(B) 参数失配** | → Phase 32b：用该参数点跑下游验证球增量是否恢复 |
| 任何参数点都塌缩 **或** 球数起来但 true_purity 仍低（~0.45，接近随机） | **(A) 本质限制** | → Phase 32c：记录为最强边界证据，叙事收缩到小图，不再扩数据集 |
| 介于之间（球数起来、purity 部分恢复但不及小图） | 部分参数效应 | → 32b 小规模验证一个最优点，按结果定 |

---

## Phase 32b：下游验证（仅当 32a 判 (B) 或部分效应）

### 目的

参数诊断只证明"球能分出来且纯度够"，但纯度高 ≠ 下游有增量（参考 Phase 9 教训：probe 高 ≠ 下游好）。必须用下游确认。

### 实现

1. 用 32a 找到的最优参数点，在 NoisyGL 里通过 `gb:` config override（**不改 anchor 默认**，新建一个 `*_bigball` config 变体或用 CLI override）跑：
   - DBLP（首选,Phase 28 主角）+ 若 32a 显示 PubMed 也救活则加 PubMed
   - 方法：`softdist_xh`、`softdist_xh_shuffled`、`gcn_xh_ce`、`nrgnn` × 5 seeds × uniform 0.4
2. 必报：球诊断（num_balls/purity/NCR/CDR）+ `softdist_xh − gcn_xh_ce`（球增量）+ `softdist_xh − shuffled`
3. 输出 `runs/bigball_downstream_20260616/`

### 判定

| 下游结果 | 论文影响 |
|---|---|
| `softdist_xh − gcn_xh_ce` ≥ 0 **且** real > shuffled（DBLP 或 PubMed） | **数据集广度有救！** 球构造需随图规模缩放是一个有价值的方法论发现（不是 bug 是 insight）。重跑该数据集进主表,叙事从"2 数据集"升级 |
| 球分出来了但下游仍 real≈shuffled | 球的几何分裂在大图上即使非塌缩也无语义增量 → 回到 (A)，收缩叙事 |

---

## Phase 32c：边界证据固化（仅当 32a 判 (A)）

若是本质限制：把这个系统性诊断写成论文 §6 的**最强边界证据**——

- 一张图：true_purity vs 数据集规模/参数，展示小图可聚类、大图无论怎么调参都不可聚类
- 论点升级：SoftDist 的适用条件不是工程巧合，而是"GRACE 表示的局部可聚类性随图规模/密度退化"的可测现象。这把"只适用小图"从弱点转为**有机制刻画的适用边界**（C3 的最强实证）
- 诚实定位：方法是 focused contribution（小-中规模引文图上的简单高效去噪 + 系统的适用性分析），适合 KBS/Neural Networks

---

## 执行顺序

```
Phase 32a（参数扫描诊断,几分钟,纯几何）
   ├─判 (B)/部分 ─► Phase 32b（下游验证,可能救活数据集广度）
   └─判 (A) ──────► Phase 32c（固化边界证据,收缩叙事）
```

- 32a 是闸门,先出诊断再决定 32b/32c
- 无论 (A)/(B),结果都有论文价值：(B) 救数据集广度,(A) 给最强边界证据
- 不改 anchor,不重训 H,32a 零下游成本

## 完成后

回填 `SoftDist_XH_论文初稿雏形.html`（§6 或 §8）与经验文档、记忆。若 32b 救活数据集，更新主表与数据集广度表。

---

## 执行记录（2026-06-16）

### Phase 32a 结果：构造层面判为参数失配

已新增并运行 `scripts/diagnose_ball_construction_scaling.py`，输出：

- `runs/ball_scaling_diag_20260616/ball_scaling_raw.csv`
- `runs/ball_scaling_diag_20260616/ball_scaling_summary.csv`
- `runs/ball_scaling_diag_20260616/ball_scaling_report.md`

关键结论：大图不是完全不可分球。DBLP/PubMed 在降低 `split_improvement` 并允许更多球后，都能达到甚至超过 Cora/CiteSeer 默认球纯度。

| 数据集 | 最优参数 `max_balls / min_ball_size / split_improvement` | num_balls | true_all_purity | mean_train_per_ball |
|---|---:|---:|---:|---:|
| Cora 默认参考 | 64 / 32 / 0.05 | 52.33 | 0.7871 | 10.34 |
| CiteSeer 默认参考 | 64 / 32 / 0.05 | 64.00 | 0.6991 | 10.39 |
| DBLP | 256 / 32 / 0.001 | 256.00 | 0.8377 | 13.84 |
| PubMed | 256 / 32 / 0.005 | 256.00 | 0.8290 | 15.40 |
| Amazon-Photo | 256 / 32 / 0.001 | 129.00 | 0.8698 | 11.87 |

解释：Phase 28 的 DBLP `num_balls=1` 主要是构造参数不适配，不应再写成“DBLP 的 H 完全不可聚类”。更准确的说法是：默认小图参数会让 DBLP/PubMed 球构造塌缩；参数放宽后几何球可以恢复。

### Phase 32b 结果：下游层面未救活数据集广度

为避免只凭纯度下结论，已用 CLI runtime override 跑下游验证，不改 anchor yaml。为此仅给 `scripts/run_ssl_swap_downstream.py` 增加了 `--gb_max_balls`、`--gb_min_ball_size`、`--gb_split_improvement` 三个进程内 override 参数。

输出：

- `runs/bigball_downstream_20260616/dblp_bigball_seed0.csv`
- `runs/bigball_downstream_20260616/pubmed_bigball_seed0.csv`
- `runs/bigball_downstream_20260616/amazonpho_bigball_5seed.csv`
- `runs/bigball_downstream_20260616/bigball_phase32_report.md`

| 数据集 | 参数 | Seeds | SoftDist [X\|\|H] | GCN [X\|\|H] | Shuffled | Real - GCN | Real - Shuffled | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DBLP | 256 / 32 / 0.001 | 0 | 0.8300 | 0.8300 | 0.8383 | +0.00pp | -0.83pp | 失败 |
| PubMed | 256 / 32 / 0.005 | 0 | 0.8213 | 0.8208 | 0.8250 | +0.05pp | -0.37pp | 失败 |
| Amazon-Photo | 256 / 32 / 0.001 | 0-4 | 0.9085 ± 0.0123 | 0.9049 ± 0.0145 | 0.9089 ± 0.0140 | +0.35pp | -0.04pp | 不构成独立球证据 |

最终判定：Phase 32 把边界细化为“两层”：

- 构造层面：大图默认塌缩可由参数失配解释，不能再说“大图 H 完全不可分球”。
- 下游层面：即使调参后球纯度很高，DBLP/PubMed/Amazon-Photo 仍没有稳定 real > shuffled 的标签校正增量。

论文处理：这些数据集仍不能加入主成功表；应写入适用边界分析。更准确的 claim 是：`H` 可以作为分类特征，也可以形成高纯度几何划分，但在更大图或共购图上，球诱导的 noisy-label target 对下游分类器的独立贡献可能被 H 特征本身或训练随机性吸收。
