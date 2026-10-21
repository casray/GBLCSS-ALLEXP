# 执行计划：Coauthor-CS 进主表 + 边界图回填（Phase 36）

日期：2026-06-17
执行者：Codex
前置状态：Phase 35 突破——Coauthor-CS(18.3k,15类) 球有效（SoftDist −GCN[X||H] +0.53pp 5/5、−shuffled +0.55pp 5/5；GBSelect 命门也过），推翻"大图必败"，证明**类数（非规模）是关键调节变量**。Amazon-Computers(13.7k,10类) 球信号真实(real>shuffled +1.8pp 5/5)但被强基线吸收(SoftDist −0.70pp 输全样本)。成功数据集从 2 → 3（Cora/CiteSeer/Coauthor-CS）。

**本计划坐实 Coauthor-CS 进主表**：它目前只有内部对照（SoftDist/shuffled/gcn_xh_ce），缺外部 baseline。主表第三个数据集必须有与 Cora/CiteSeer 同样的外部对比，否则审稿人问"CS 上比 NRGNN 如何"答不上。**不改主方法机制。**

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H（不重训）：`embeddings_v2/CoauthorCS/grace/H.pt`(18333×256)、`embeddings_v2/amazoncom/grace/H.pt`(13752×256)
- NoisyGL 显式 percent split 0.2/0.1/0.7；anchor 机制不动；clean 只进诊断；研报告；测试全绿

### Coauthor-CS / Amazon-Computers 当前已有数据（Phase 35,uniform 0.4, 5 seeds）

| 数据集 | SoftDist[X‖H] | GCN[X‖H] | shuffled | SoftDist−GCN[X‖H] |
|---|---:|---:|---:|---:|
| Coauthor-CS | 0.9243±0.0016 | 0.9190±0.0009 | 0.9188±0.0012 | +0.53pp 5/5 ✅ |
| Amazon-Computers | 0.8625±0.0056 | 0.8695±0.0082 | 0.8443±0.0176 | −0.70pp 0/5（被吸收）|

**缺**：两数据集的 GCN+CE（无H基线）、NRGNN、NRGNN[X‖H]。

---

## Phase 36a：补外部 baseline（核心）

### 运行

- 数据集：Coauthor-CS（主目标,进主表）+ Amazon-Computers（进边界表）
- 补的方法：`gcn`(CE无H基线)、`nrgnn`、`nrgnn_xh` × 5 seeds × uniform 0.4，percent split
- 输出 `runs/coauthorcs_baseline_20260617/`、`runs/amazoncom_baseline_20260617/`

### ★风险预案：NRGNN 在 18k 节点 + 6805 维特征上可能 OOM/超慢

NRGNN 有基于特征的边预测器,大图 + 高维特征下显存和时间风险高（参考 TSS 在 CiteSeer 就 OOM）。预案：
1. 先单 seed smoke 跑 `nrgnn` on Coauthor-CS,确认能否跑通、单 seed 耗时
2. 跑通 → 5 seeds 正常做
3. OOM → 先试调小 NRGNN 自身的 edge-predictor 采样/batch（仅其内部参数,记录,不改算法）;仍 OOM → 用 CPU（慢但 NRGNN 不是计时对象）;CPU 也不行 → 主表该格标注"NRGNN: OOM on 18k+high-dim under ROCm",诚实留空+脚注（已有 TSS CiteSeer 先例）
4. `nrgnn_xh` 同理（[X‖H] 拼接后特征更宽,风险更高,先 smoke）

### 判定（决定 CS 在主表的位置）

| Coauthor-CS 上 SoftDist vs NRGNN[X‖H] | 论文表述 |
|---|---|
| SoftDist ≥ NRGNN[X‖H] | CS 上简单方法持平/超过强基线,主表强证据 |
| SoftDist < NRGNN[X‖H] | 与 Cora/CiteSeer 一致——H 是共同资产,球有受控增量但不全面超 NRGNN[X‖H]。如实写,不影响"球有受控增量"的核心 claim（那个由 vs gcn_xh_ce + vs shuffled 支撑,已 5/5 成立）|

---

## Phase 36b：论文初稿回填

### 主表 §3.1：加 Coauthor-CS 第三列/第三组

- 行：GCN+CE / NRGNN / NRGNN[X‖H] / SoftDist[X‖H]（+ shuffled 在消融）
- 三数据集（Cora/CiteSeer/Coauthor-CS）横向并列,Coauthor-CS 是"大图(18k)+多类(15)成功"的关键新证据

### 消融 §4.1：归因链加 Coauthor-CS 行

- H 贡献 / 球贡献(SoftDist−gcn_xh_ce) / 真实-随机(SoftDist−shuffled),CS 上 +0.53/+0.55pp 5/5,与小图一致

### ★新增 §8 二维边界图（本轮最有价值的论文素材）

- 散点图：x=节点数(log), y=类别数, 每个数据集一个点,**点色/标记 = 球增量正负**（绿✅=SoftDist>gcn_xh_ce 且 real>shuffled；红❌=否；黄=部分如 Amazon-Computers real>shuffled 但<全样本）
- 数据点：Cora(2.7k,7)✅ / CiteSeer(3.3k,6)✅ / Coauthor-CS(18.3k,15)✅ / Amazon-Photo(7.6k,8)❌ / Amazon-Computers(13.7k,10)黄 / DBLP(17.7k,4)❌ / PubMed(19.7k,3)❌
- 配套二维边界表 + 论述：**规模不是死因（CS 18k 成功）,类别复杂度是关键调节变量（DBLP/PubMed 类少失败 vs CS 类多成功）,但强同质+强基线 regime 仍会吸收（Amazon 系）**
- 脚本 `scripts/plot_size_class_boundary.py`,出图到 `runs/figures_20260617/`

### 边界分析论述（可写入论文）

把 Phase 32/34/35 串成完整故事：
- 球在大图的 soft-target(32)和 selection(34)用法在**类少**大图上都失败 → 一度误判"只适用小图"
- 类数判别实验(35)推翻误判：Coauthor-CS(大+多类)成功 → **真变量是类别复杂度/标签噪声可分性,非图规模**
- 机制解释：类少时 GCN 单视角已能识别噪声（类边界简单）,球无正交边际;类多时任务难,球的局部共识提供 GCN 缺的信息
- Amazon 系给出第二维边界：即便类够多,若同质性强+基线已高,噪声伤害轻,去噪收益被吸收

---

## 执行顺序

```
36a 补 baseline（先 NRGNN smoke 验可行性 → 5 seeds）
   └─► 36b 回填主表 + 消融 + 二维边界图 + 边界论述
```

- 36a 先做,NRGNN 大图可行性是唯一不确定项,先 smoke
- 36b 纯回填,数据已基本齐（只等 36a 的 baseline 数字）

## 完成后

论文实验部分达到：**3 个成功数据集（含 18k 大图）+ 完整外部对比 + 二维适用域边界图 + 32/34/35 串起的边界机制分析**。这是比"2 个小数据集 focused"强一个档次的格局。更新经验文档与记忆。此后若无新轴,进入正文写作。

---

## 执行结果备案（Codex, 2026-06-17）

执行环境：ROCm Python `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`。协议保持 percent split 0.2/0.1/0.7，uniform 0.4，seeds 0-4。

### Phase 36a 外部 baseline

输出：

- `runs/coauthorcs_baseline_20260617/coauthorcs_baselines_5seed.csv`
- `runs/amazoncom_baseline_20260617/amazoncom_baselines_5seed.csv`
- `runs/figures_20260617/phase36_aggregate.csv`
- `runs/figures_20260617/phase36_paired_deltas.csv`

Coauthor-CS：

| 方法 | Test acc |
|---|---:|
| GCN + CE | 0.9196 ± 0.0017 |
| GCN + CE [X\|\|H] | 0.9190 ± 0.0010 |
| NRGNN | 0.9356 ± 0.0022 |
| NRGNN [X\|\|H] | **0.9366 ± 0.0043** |
| SoftDist [X\|\|H] | 0.9243 ± 0.0018 |
| SoftDist shuffled | 0.9188 ± 0.0013 |

关键配对：

- SoftDist − GCN [X\|\|H]：+0.53pp，5/5。
- SoftDist − shuffled：+0.55pp，5/5。
- SoftDist − NRGNN [X\|\|H]：-1.23pp，0/5。
- NRGNN [X\|\|H] − NRGNN：+0.10pp，4/5。

结论：Coauthor-CS 可以进主表作为第三个成功数据集，但它支持的是“球有受控增量且适用域扩展到大图多类”，不是“超过 NRGNN”。主表必须诚实显示 NRGNN [X\|\|H] 更强。

Amazon-Computers：

| 方法 | Test acc |
|---|---:|
| GCN + CE | 0.8682 ± 0.0061 |
| GCN + CE [X\|\|H] | **0.8695 ± 0.0092** |
| NRGNN | 0.7923 ± 0.0614 |
| NRGNN [X\|\|H] | 0.8230 ± 0.0150 |
| SoftDist [X\|\|H] | 0.8625 ± 0.0063 |
| SoftDist shuffled | 0.8443 ± 0.0196 |

关键配对：

- SoftDist − GCN [X\|\|H]：-0.70pp，0/5。
- SoftDist − shuffled：+1.82pp，5/5。
- SoftDist − NRGNN [X\|\|H]：+3.95pp，5/5。

结论：Amazon-Computers 是边界点。真实球显著强于 shuffled，也强于 NRGNN [X\|\|H]，但仍被全样本 GCN [X\|\|H] 吸收，不能算主成功数据集。

### Phase 36b 回填

已更新：

- `C:\Users\Admin\Desktop\实验8去噪5.19-\SoftDist_XH_论文初稿雏形.html`
  - 主表增加 Coauthor-CS。
  - 归因链增加 Coauthor-CS。
  - 公平性对照增加 Coauthor-CS。
  - 规模 × 类数边界段增加 Phase 36 外部 baseline。
  - 嵌入二维边界图。
- 新增脚本：`scripts/plot_size_class_boundary.py`
- 新增图与报告：
  - `runs/figures_20260617/size_class_boundary.png`
  - `runs/figures_20260617/phase36_coauthorcs_maintable_report.md`

最终写作边界：SoftDist 在 Cora/CiteSeer/Coauthor-CS 上有稳定球增量；Cora 持平 NRGNN [X\|\|H]；CiteSeer 与 Coauthor-CS 输给 NRGNN [X\|\|H]。论文应强调简单性、非迭代、受控增量、适用域边界，而不是全面 SOTA。
