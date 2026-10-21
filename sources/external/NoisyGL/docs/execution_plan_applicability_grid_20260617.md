# 执行计划：SoftDist / GBSelect 适用域多变量边界刻画（Phase 37）

日期：2026-06-17
执行者：Codex
前置状态：已知球增量出现在 Cora/CiteSeer/Coauthor-CS（3 成功），不出现在 PubMed/DBLP/Amazon-Photo/Amazon-Computers（4 边界/失败）。Phase 35 证明"规模"不是死因（Coauthor-CS 18k 成功），"类数"是调节变量之一，但 Amazon 系（类够多仍失败）说明**还有第三变量**——疑似"GCN 自己能不能搞定噪声"（强同质+强基线则噪声伤害轻，球无边际）。

**本计划目标**：把零散、协议不齐的现有数据点，整理成一张**统一协议、多变量**的适用域网格，用数据说话指出**球增量到底由哪个变量决定**（不是事后挑规模/类数两个表面变量）。产出直接回答：(a) SoftDist/GBSelect 各自的适用边界；(b) 球增量是否与某个 **label-free 可测信号**强相关——这是后续"自适应层"（想法2）能否成立的前提判据。**不改主方法机制。**

---

## 0. 环境与红线（沿用）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 GRACE H：`embeddings_v2/{Cora,CiteSeer,PubMed,AmazonPhoto,DBLP,CoauthorCS,amazoncom}/grace/H.pt`
- NoisyGL 显式 percent split 0.2/0.1/0.7，uniform 0.4，seeds 0-4；anchor 机制不动；clean 只进诊断；SSL label-free；研报告；测试全绿
- loader 已接好待补数据集：`coauthorph`→Coauthor('Physics')、`wikics`→WikiCS（无需改代码，只加 config + 预训练 H）

---

## Phase 37a：补两个填空白象限的数据集

现有 7 点在"规模×类数"上的分布仍有洞，且同质性/noise-harm 维度未铺开。补两个**判别性**点：

| 数据集 | 节点 | 类 | 同质性 | 填的空白 / 判别作用 |
|---|---:|---:|---:|---|
| Coauthor-Physics | 34.5k | 5 | ~0.92 极高 | 大规模 + 类少 + 极强同质——与 Coauthor-CS(大,多类) 对照,进一步隔离"类数"vs"同质性" |
| WikiCS | 11.7k | 10 | ~0.66 中等 | 中规模 + 类多 + **中等同质**——Amazon 系是强同质失败,WikiCS 检验"类多 + 同质不那么强"球会不会成功 |

步骤（每个数据集）：
1. 触发下载（PyG 自动）
2. GRACE label-free 预训练 → `embeddings_v2/<Name>/grace/H.pt`，**预注册**：Physics(34.5k) 用 PubMed/DBLP 同档大图 GRACE 配置、WikiCS(11.7k) 用 Amazon-Photo 同档配置，执行前写进报告；监控显存（Physics 34.5k 是目前最大，可能慢/紧）
3. H-图对齐 sanity check（同 Phase 28）
4. 跑通则纳入 37b；Physics OOM → 降配/CPU/记录留空（同 NRGNN 大图预案）

---

## Phase 37b：统一协议重整全部数据集的主网格

### 关键：协议统一

现有 7 个数据点是多轮应急跑的，球构造参数、是否用大图适配参数不完全一致。**本步用完全统一的协议重新测量所有 9 个数据集**（7 旧 + 2 新），保证可比：
- 球构造：默认 `max_balls=64/min_ball_size=32/split_improvement=0.05`，**外加** Phase 32a 大图适配参数（大图额外测一组），两组都报，避免"大图用默认塌缩"混入边界判断
- 协议：percent 0.2/0.1/0.7，uniform 0.4，seeds 0-4
- 方法：`gcn`(CE)、`gcn_xh_ce`、`softdist_xh`、`softdist_xh_shuffled`、`gbselect`、`gbselect_gcn_only`、`gbselect_shuffled`、`nrgnn_xh`

### ★每个数据集测齐 5 个候选解释变量（核心）

不是只看规模/类数，把所有候选变量都量出来，让数据指出哪个决定球增量：

| 变量 | 怎么测 | 假设 |
|---|---|---|
| 规模 N | 节点数 | Phase35 已证非死因 |
| 类数 C | 类别数 | Phase35 疑似调节变量 |
| 同质性 h | edge homophily ratio（标准公式，clean label 仅诊断算 h，h 是数据集固有属性不进训练） | Amazon 系强同质失败 |
| **noise harm** | `GCN+CE(clean) − GCN+CE(noisy)`（两者都跑，clean 版仅作诊断上界） | **疑似真变量：噪声没伤到→球无边际** |
| **球条件正交 AUC** | Phase 33 脚本 `ball_auc_in_gcn_clean50` | label-free 可测,自适应层的候选触发信号 |

### 两个被解释量（每个数据集 × 每个算法）

- SoftDist 增量：`softdist_xh − gcn_xh_ce`（5-seed 配对 + 正 seed 数）+ `softdist_xh − shuffled`
- GBSelect 增量：`gbselect − gcn_only`（命门）+ `gbselect − 全样本gcn_xh_ce`

### 输出 `runs/applicability_grid_20260617/`：每数据集一行的总表（9 数据集 × 5 变量 × 2 算法增量）

---

## Phase 37c：相关性分析 —— 球增量到底由哪个变量决定

### 分析（这是本计划的科学核心）

用 37b 的 9 行总表：
1. **球增量 vs 各变量的相关性**：SoftDist 增量分别对 N / C / h / noise_harm / 正交AUC 做相关（Spearman，9 点样本小，报相关方向+散点不强行 p 值）
2. **关键判别**：哪个变量最能分开"成功(Cora/CiteSeer/Coauthor-CS)"和"失败(PubMed/DBLP/Amazon系)"?
   - 若 noise_harm 或正交AUC 分得最干净 → 真变量找到,且是 label-free 可测的（正交AUC）→ 自适应层有触发信号
   - 若只有类数 C 勉强分 → 边界是类数,但 Amazon 反例说明不充分
3. **SoftDist vs GBSelect 适用域对比**：两个算法的成功集是否相同?GBSelect 在哪些数据集命门过而 SoftDist 不过(或反之)?→ 定清两个算法各自的边界

### ★自适应层可行性判据（直接为想法2 出结论）

| 37c 发现 | 想法2（自适应层）判定 |
|---|---|
| 存在一个 label-free 信号（如正交AUC/球纯度）能在**不看 test** 的情况下分开成功/失败数据集 | **自适应层可行** → 可设计 λ=f(信号) 自动调球权重,下一步做 |
| 只有 noise_harm（需 clean 才能精确算）或 test 表现能分开,label-free 信号分不开 | **自适应层不可行**（会变成 cherry-picking）→ 老实写适用边界,不做自适应,想法2 关闭 |

### 输出

- `runs/applicability_grid_20260617/correlation_analysis.md` + 报告
- 二维/多变量边界图：扩展现有 size×class 散点,点色=球增量,**并新增一张 noise_harm（或正交AUC）vs 球增量的散点**,看哪个轴分得最干净

---

## 执行顺序

```
37a 补 Coauthor-Physics + WikiCS（先预训练 H,Physics 先 smoke 验显存）
   └─► 37b 统一协议重整 9 数据集 × 5 变量 × 多算法
         └─► 37c 相关性分析 + 自适应层可行性判据
```

- 37a 的 Physics 是最大图,先 smoke 验可行,OOM 则留空不阻塞
- 37b 是算力主体（9 数据集 × 多方法 × 5 seeds）,大图慢,按数据集分批
- 37c 是纯分析,数据齐后几分钟

## 完成后

- 论文 §8 升级为完整适用域刻画：多变量网格表 + "球增量由 X 变量决定"的相关性结论 + SoftDist/GBSelect 各自边界图
- §6/§8 论述：从"只在某些数据集有效"升级为"球增量由 [某 label-free 可测信号] 决定,我们刻画了完整适用域"
- 明确给出想法2 是否可做的结论,更新经验文档与记忆
- 此结果若显示自适应层可行 → 下一计划做自适应层；若不可行 → 实验收口,进入正文写作
