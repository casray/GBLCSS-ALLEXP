# 执行计划：Label-Free 可聚类性增强对比预训练（Phase 20）

日期：2026-06-12
执行者：Codex
前置状态：SoftDist [X||H] 主线确认，论文初稿成型。Phase 8 已证明换 SSL 方法（BGRL/GraphMAE/CCA-SSG/可学习τ）全部失败，且 §6 发现「probe 可分性 ≠ 下游可聚类性」——CCA-SSG probe 更高却使球塌缩。本计划不换 SSL 方法，而是**在 GRACE 上加一个纯几何、完全 label-free 的可聚类性正则**，验证「比 GRACE 更强地优化球可分性的表示」能否提升下游。

---

## 0. 不可动摇的前提（本计划存在的理由）

> **预训练必须保持完全 label-free。** 这是整条技术路线的核心 claim（C3：表示阶段对标签噪声构造性免疫，同一份 H 跨所有噪声/split 复用）。任何把 noisy 标签、train_mask、球的标签投票引入预训练损失的做法，都会摧毁这个卖点，**一律禁止**。本计划的正则项是纯几何的（只依赖表示向量的分布），不触碰任何标签信号。

### 历史不利证据（必须正视，写进报告动机）

- 跨多个项目，所有预训练侧改动均失败（view generation、目标替换、可学习 τ）——见 `03_cross_direction_meta_review.md`
- probe 高 ≠ 下游好（Phase 9 CCA-SSG 教训）：判定必须用**下游球质量 + real>shuffled**，绝不能用 probe acc
- CiteSeer 1.4pp 差距在五个维度都补不上，无证据表明它源于表示质量

因此本计划是**高风险、诊断先行、预注册止损**的探索，不是必胜方向。

---

## 1. 环境与红线

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- GRACE 预训练代码在 `gb_graph_aware_minival_6_1/`（产出 H 的上游仓库），新损失改在该仓库的 GRACE 实现里，不进 NoisyGL
- 下游 NoisyGL 必须显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- GRACE H 锚点：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- **红线**：
  1. 预训练 label-free（见 §0），新损失只读表示向量，不读 y/train_mask/球标签
  2. `frozengrace_gb_gcn_softdist_xh` 下游 anchor 机制不动，只换 `gb.embedding_path` 指向新 H'
  3. GRACE 原有超参（epochs/lr/drop rate/hidden dim）保持锚点配置不变，只新增正则项及其权重
  4. research-record 报告
- 测试全绿：`py -3 -m unittest discover -s tests`

---

## 2. 方法：GRACE + Uniformity 正则

### 动机（可写入论文）

§6 发现 GRACE 之所以适合粒球是因为 InfoNCE 的 uniformity——表示在超球面均匀展开，天然可被距离聚类。自然的假设：**显式强化 uniformity 会产生更可分的球**。Uniformity 是纯几何量（Wang & Isola, ICML 2020），完全不依赖标签。

### 损失

GRACE 原损失 `L_GRACE`（InfoNCE over两视图）。新增 uniformity 正则：
```
L_unif = log E_{i,j} [ exp(-t · ||z_i - z_j||^2) ]      # z 为 L2 归一化表示，t=2 (Wang&Isola 默认)
L_total = L_GRACE + λ_u · L_unif
```
- `λ_u ∈ {0.1, 0.5, 1.0}` 网格（小范围，不过度搜参）
- batch 内估计 E（节点采样），与 GRACE 现有 minibatch 一致
- t 固定 2.0，不调（避免引入额外自由度）

### 实现

1. 在上游 GRACE 训练脚本（`gb_graph_aware_minival_6_1` 的 GRACE 实现，参考 `scripts/pretrain_ssl_candidates.py` 的 grace 分支）加一个 `--uniformity_weight` 参数与 `uniformity_loss(z)` 函数
2. 每个 λ_u × {Cora, CiteSeer} 预训练（label-free，seed 0），导出 H' 到 `checkpoints/embeddings_v2/{dataset}/grace_unif{λ_u}/H.pt`，格式与现有 H 一致（N×d，对齐 NoisyGL 节点序）
3. 维度、epochs、lr、drop rate 与锚点 GRACE 完全一致——唯一变量是 λ_u
4. 单元测试：`uniformity_loss` 对随机输入有限可导；λ_u=0 时 L_total 与原 GRACE 逐值相等（回归）

---

## Phase 20a：表示几何诊断（先做，不下游，廉价）

复用 `scripts/linear_probe_embeddings.py` 思路，新写或扩展 `scripts/diagnose_embedding_geometry.py`：

对每个 H'（及 GRACE 锚点 H 作对照），用下游同协议建**几何**球（max_balls=64, min_ball_size=32, 同 random_state），报告：
- **num_balls**（是否塌缩——CCA-SSG 塌缩到 1 的教训）
- **uniformity 指标值**（量化正则是否真的生效）
- **球的 true-purity**（clean label 仅诊断，weighted）：更可分的球是否真的更纯
- **probe acc**（clean + noisy）：仅作参考，**不作判据**

### 进入 20b 的门槛

- num_balls 不塌缩（≥ 锚点 GRACE 的一半，即 ≥26 左右）
- 球 true-purity ≥ GRACE 锚点（这才是「更可聚类」的真信号，比 probe 可靠）
- 任一 λ_u 满足即带该 λ_u 进 20b；全不满足 → **整条定制 SSL 线关闭**（理由：最安全的 uniformity 正则都无法产出更好的球，说明 GRACE uniformity 已够用）

---

## Phase 20b：下游验证（仅当 20a 过门）

1. 过门的 H'：复制 `softdist_xh` / `gcn_xh_ce` / `softdist_xh_shuffled` 配置，仅改 `gb.embedding_path`，seed0 smoke × {Cora, CiteSeer}
2. seed0 `softdist_xh(H') ≥ softdist_xh(GRACE) − 0.5pp` 且 real>shuffled → 扩 5-seed
3. 5-seed 主判据：**CiteSeer mean > 0.7027 (NRGNN)** 或双数据集稳定 > GRACE 版

### 必报诊断

- 球质量链：num_balls / mean_train_per_ball / NCR / CDR（H' vs GRACE）
- probe acc vs 下游 acc 的对应（继续积累「可分性≠可聚类性」的证据）

### 判定与止损

| 结果 | 处置 |
|---|---|
| CiteSeer 翻盘 NRGNN | grace_unif 成主配置，GRACE 降为消融行；C3 分析升级为「可主动优化可聚类性」 |
| 有增量未翻盘 | grace_unif 作为「表示可改进轴」的正向证据进论文 |
| 不如 GRACE / 20a 未过门 | **定制 SSL 线永久关闭**，结论写入 §6：GRACE uniformity 已是该任务局部最优，强化「为什么是 GRACE」的论证 |

---

## 3. 为什么只做方向 1（uniformity 正则），不做方向 2/3

- 方向 2（粒球感知对比）：预训练中途建球当软正负样本——撞我们历史上 GB-view-generation 反复失败的坑（早期表示上球不稳、view drift），风险最高，放最后
- 方向 3（原型/簇对比，SwAV/PCL 式）：成本高，需重写预训练框架
- 方向 1 是**最小改动、最安全、最可能成功**的一刀：纯几何正则、改动量一个 loss 项、直接强化已被 §6 证明有用的性质。**它若失败，方向 2/3 的先验成功率更低，整条线应关闭**——这是预注册的止损逻辑，不是做完方向 1 再看心情。

## 4. 执行顺序

```
实现 uniformity_loss + 回归测试
  └─► Phase 20a（几何诊断，3 个 λ_u + GRACE 对照，廉价）
        └─过门─► Phase 20b（下游 seed0 → 5-seed）
        └─不过门─► 关闭定制 SSL 线，结论进 §6
```

完成后无论正负，把结论回填进 `SoftDist_XH_论文初稿雏形.html` §6（表示分析）与记忆文件。
