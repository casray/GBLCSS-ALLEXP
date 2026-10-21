# 执行计划：SoftDist-MLP 双重噪声鲁棒 —— 补全实验矩阵成篇（Phase MLP-2 / 论文方案层）

日期：2026-07-01
执行者：Codex
定位：**普通 SCI 完整实验**。MLP-1 已 GO（同弱MLP backbone下 SoftDist 软标签在多数中低/结构噪声上赢 CE/small-loss/co-teaching/ELR/JoCoR/shuffled）。本计划把这个正结果扩成一篇论文的完整证据链。**不再做 kill-test（已过闸），做的是"把论文该有的表和图补齐"。**

## 0. 论文定位（已定，写在最前防跑偏）

**卖点：graph-less（推理不用图传播）节点分类的双重噪声鲁棒 —— 标签噪声 + 结构噪声。**
- 现实图数据同时有标签噪声(标注错) + 结构噪声(边错/缺/新节点无连边)；GNN 对两者都脆弱。
- 已有 graph-less MLP 方法(Graph-MLP/GLNN/LinkDist/ES-MLP)证明"推理不用邻接→对结构噪声鲁棒"，但**它们不处理标签噪声**。
- **本方法**：训练时 GRACE 图对比表示吸收图结构 + 粒球软标签抗标签噪声；推理时轻量 MLP → 对结构噪声天然免疫。**首次在 graph-less 节点分类中同时做标签+结构双重鲁棒。**
- 粒球软标签的**必要性**：MLP 本身表达弱（文献共识 "mere MLPs 表达差"），需要粒球软标签这种额外监督补短板 —— 球不是可有可无。

## 0.1 红线（沿用）

1. **同backbone公平性**：标签噪声主表里所有抗噪方法用同一 X-only MLP（MLP-1 已保证，沿用 `MLPNoiseBaselines_Predictor`）。
2. clean标签只进评测；噪声只加train（标签噪声）；结构噪声只加**测试推理时的图**（见§2）。
3. seed0先smoke，通过再5 seeds；多seed配对(mean±std)进结论。
4. Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`（ROCm）。NoisyGL percent split 0.2/0.1/0.7 显式传参。

## 0.2 复用与新增

- **复用（MLP-1 已建）**：`MLPNoiseBaselines_Predictor`(CE/small-loss/co-teaching/JoCoR/ELR/SCE，X-only MLP)、`frozengrace_gb_mlp_softdist_Predictor`(本方法)、`shuffled` 版；Cora/CiteSeer config。
- **GNN对照（结构噪声主表用）**：现成 `GCN_Predictor` / `GIN_Predictor`（GAT 若无可跳过）。
- **需新增**：
  1. Coauthor-CS 的 MLP-noise + SoftDist-MLP config（仿 cora/citeseer，用 CoauthorCS 的 GRACE H）；
  2. **结构噪声注入**（新写，很简单，见§2）；
  3. instance 噪声（`label_process` 已支持 `noise_type=instance`，直接用）。

## 1. 主表一：标签噪声鲁棒（同 MLP backbone 比抗噪策略）—— 扩展 MLP-1

在 MLP-1(Cora/CiteSeer × uniform{0.2,0.4,0.6}+pair0.4)基础上**补全**：
- **加数据集**：Coauthor-CS（第三个 SoftDist 成功数据集，增强多数据集说服力）。
- **加噪声类型**：instance 0.4（MgCNL 也报 instance/asymmetric，增强噪声覆盖）。
- 方法：CE / small-loss / co-teaching / JoCoR / ELR / SCE / **SoftDist** / shuffled（全 X-only MLP，同配置）。
- 指标：test acc（mean±std over 5 seeds），SoftDist vs 各抗噪策略配对 delta。

## 2. 主表二：结构噪声鲁棒（本方法的第二卖点，MLP-1 没做，必须补）

**这是双重鲁棒故事的核心证据，也是"为什么用MLP不用GCN"的硬支撑。**

- **训练**：正常（干净图训练，或叠加标签噪声——先做干净标签+结构噪声，隔离结构噪声效应）。
- **测试推理时注入结构噪声**（三种，选做，删边最标准）：
  - `edge_drop`：随机删掉测试图 p% 的边（p=25/50/75%）；
  - `edge_add`：随机加 p% 的假边；
  - `isolated`：模拟"新节点无连边"——把一部分测试节点的所有边删光。
- **对照**：
  - GNN 方（`GCN`/`GIN`）：推理用**被扰动的图** → 精度应随结构噪声下降；
  - 本方法（SoftDist-MLP）：推理**不用图**（MLP 只吃 X）→ 精度**不受结构噪声影响**（水平线）；
  - Graph-MLP（若能跑，最强对照）：同样推理不用图，但**不抗标签噪声** → 见主表三。
- **主图**：x=结构噪声强度，y=test acc。GCN/GIN 曲线下降，SoftDist-MLP 水平不降。**这一张图直接证明"推理graph-free对结构噪声免疫"，是双卖点里"结构鲁棒"的铁证。**

> 实现极简：结构噪声只在**评测阶段对 GNN 的 adj 做扰动**；MLP 推理本就不碰 adj，天然是水平线。新写一个 `perturb_test_adj(adj, mode, p, seed)` 即可。

## 3. 消融（MgCNL Table VI 式，证明每个部件必要）

同 backbone、固定一个噪声设定（如 uniform0.4），逐个去掉：
- `w/o GRACE`（球建在原始X而非GRACE H）→ 验证"好表示"的必要（预期掉，呼应NIDS的0.52教训）；
- `w/o 粒球软标签`（= 裸MLP+CE）→ 验证球的必要；
- `球软标签→硬多数票`→ 验证"软"的必要（Phase41教训：硬改标签有害）；
- `SoftDist → shuffled球`→ 真结构必要（已有）。

## 4. 竞品切割（related work + 至少一格实验）

| 竞品 | 它做什么 | 与你的切割 |
|---|---|---|
| Graph-MLP / GLNN / LinkDist / ES-MLP | graph-less 节点分类，推理不用图，对**结构**噪声鲁棒 | **它们不处理标签噪声** → 在你的标签噪声setting下，它们应崩/无优势；你两者都做。至少让 Graph-MLP（若可跑）进标签噪声主表一格 |
| MgCNL / MgBIF | 对比表示+粒球+弱下游，抗**标签**噪声 | **它们在时序/表格，不做图、不做结构噪声**；你在图 + 双重鲁棒 |
| NRGNN/RTGNN 等图噪声方法 | 强GNN抗标签噪声 | 它们是强下游（会吸收球、且推理仍需图→不抗结构噪声）；你走轻量graph-less路线，不同定位（不在同一backbone，作参考不作主对照）|

## 5. go/no-go（此阶段是"论文能不能成篇"，非机制生死）

| 检查 | 判据 | 结论 |
|---|---|---|
| **C1 标签鲁棒扩展** | Coauthor-CS + instance0.4 上，SoftDist 仍 ≥ 多数抗噪策略（延续MLP-1趋势，≥2/3新设定） | 标签鲁棒主表成篇 |
| **C2 结构鲁棒成立** | 结构噪声下 SoftDist-MLP 精度基本不降，而 GCN/GIN 明显降（曲线分离清晰） | **第二卖点成立**（这个几乎必过，因MLP推理不用图是结构性的）|
| **C3 消融清楚** | GRACE/软标签/软vs硬 各去掉都掉，可归因 | 方法各部件必要，可写 |
| 成篇判定 | C1 + C2 + C3 基本成立 | **进入写作**（普通SCI）：双重鲁棒framing + 主表一/二 + 消融 + 切割 |
| 退路 | C1 大面积不过（Coauthor-CS/instance 上 SoftDist 塌） | 缩回 Cora/CiteSeer + uniform/pair 的 focused claim，或补域 |

> 注意：C2 几乎必过（MLP 推理不用图=水平线是结构性的，不是运气），所以**双卖点里"结构鲁棒"这半篇基本稳**。真正的变量是 C1（标签鲁棒能否扩到第三数据集/instance）。

## 6. 执行顺序

```
1. 新增 Coauthor-CS 的 MLP-noise + SoftDist-MLP config（仿 cora/citeseer，用CoauthorCS GRACE H）
2. 新增 perturb_test_adj(adj, mode∈{drop,add,isolated}, p, seed)（评测时扰动GNN的adj；MLP不受影响）
3. 主表一扩展：Cora/CiteSeer/CoauthorCS × {uniform0.2/0.4/0.6, pair0.4, instance0.4} × 8方法 × seeds0-4
4. 主表二：干净标签训练 → 测试edge_drop{25,50,75}/isolated → GCN/GIN vs SoftDist-MLP × seeds0-4
5. 消融：uniform0.4 下 w/o GRACE / w/o软标签 / 硬多数票 / shuffled × seeds0-4
6. （可选）Graph-MLP 复现一格，放标签噪声主表证明它不抗标签噪声
7. 汇总：主表一、主表二、结构噪声曲线图、消融表；按§5判 C1/C2/C3
   单测：结构噪声不影响MLP推理(MLP结果在adj扰动前后完全一致)；clean不进train
```

- 输出目录：`external/NoisyGL/runs/mlp_dualrobust_20260701/`
- 报告：主表一(标签鲁棒,数据集×噪声×方法) + 主表二(结构鲁棒,GNN vs MLP曲线) + 消融表 + C1/C2/C3判定 + 竞品切割说明。

## 7. 完成后

- 更新记忆（MLP-2 结果 + 双重鲁棒是否成篇）。
- **成篇**：进入论文写作——Claude 起草骨架(Intro双重鲁棒motivation + 方法 + 主表一/二 + 消融 + related work切割Graph-MLP/MgCNL) + 导师话术。目标普通SCI。
- **C1 塌**：缩 focused claim 或评估补域，重新定范围。

---

## Phase MLP-2 执行结果备案（已完成，2026-07-01）

结果目录：`external/NoisyGL/runs/mlp_dualrobust_20260701/`

关键产物：
- `mlp_dualrobust_report.md`
- `mlp_dualrobust_label_aggregate.csv`
- `mlp_dualrobust_label_paired_deltas.csv`
- `mlp_dualrobust_c1_gate.csv`
- `mlp_dualrobust_structure_aggregate.csv`
- `mlp_dualrobust_c2_gate.csv`
- `mlp_dualrobust_ablation_paired_deltas.csv`

完整性与验证：
- 标签噪声主表：600/600 完整，缺失 0。
- 结构噪声表：315/315 完整，缺失 0。
- 消融表：75/75 完整，缺失 0。
- CSV `status` 异常数：0。
- 日志扫描：无 Traceback / OOM / FAILED。
- 单测：`python -m unittest tests.test_mlp_dualrobust tests.test_mlp_noise_baselines tests.test_mlp_noiserobust_summary tests.test_standard_runner_baselines -v`，27/27 passed。

### C1：标签噪声鲁棒扩展

预注册新设定 7 个：Cora/CiteSeer instance0.4 + Coauthor-CS 五个噪声设定。SoftDist-MLP 通过 5/7：

| Dataset | Noise | SoftDist mean±std | 判定 |
|---|---:|---:|---|
| Cora | instance0.4 | 52.96±2.49 | pass，赢 5/5 robust MLP baselines |
| CiteSeer | instance0.4 | 51.81±1.61 | pass，赢 5/5 robust MLP baselines |
| Coauthor-CS | uniform0.2 | 91.46±0.49 | fail，只赢 2/5 robust MLP baselines |
| Coauthor-CS | uniform0.4 | 89.95±0.70 | pass，赢 5/5 robust MLP baselines |
| Coauthor-CS | uniform0.6 | 84.89±0.96 | pass，赢 5/5 robust MLP baselines |
| Coauthor-CS | pair0.4 | 79.43±1.57 | fail，只赢 2/5 robust MLP baselines |
| Coauthor-CS | instance0.4 | 89.70±0.82 | pass，赢 5/5 robust MLP baselines |

结论：C1 按“≥2/3 新设定通过”成立（5/7），标签鲁棒主表可写。但不能写“全面优于所有抗噪 MLP 策略”。边界很明确：Coauthor-CS 的低噪声 uniform0.2 和 pair0.4 上，small-loss / co-teaching / JoCoR / SCE 等部分策略能追平或超过 SoftDist。

### C2：结构噪声鲁棒

结构噪声下 SoftDist-MLP 对所有数据集、所有扰动强度完全水平；GCN/GIN 随扰动明显下降。p=0.75 关键数字：

| Dataset | Mode | SoftDist drop | Worst GNN drop | 判定 |
|---|---|---:|---:|---|
| Cora | edge_drop | 0.00pp | -11.49pp | pass |
| Cora | isolated | 0.00pp | -15.89pp | pass |
| CiteSeer | edge_drop | 0.00pp | -5.23pp | pass |
| CiteSeer | isolated | 0.00pp | -7.23pp | pass |
| Coauthor-CS | edge_drop | 0.00pp | -11.46pp | pass |
| Coauthor-CS | isolated | 0.00pp | -23.26pp | pass |

结论：C2 强成立。这个结果支持“推理 graph-free 带来结构噪声免疫”的论文卖点。注意这只证明结构扰动下的稳定性，不代表 clean graph 上 SoftDist-MLP 的绝对精度高于 GCN；Cora/CiteSeer clean 下 GCN 仍明显更高。

### C3：消融

uniform0.4 下 SoftDist 相对组件消融的 paired delta：

| Dataset | 对照 | Delta | Seedwise |
|---|---|---:|---:|
| Cora | vs MLP+CE | +7.60pp | 5/5 |
| Cora | vs w/o GRACE (X balls) | +7.25pp | 5/5 |
| Cora | vs shuffled | +7.74pp | 5/5 |
| Cora | vs hard majority | -7.20pp | 0/5 |
| CiteSeer | vs MLP+CE | +8.12pp | 5/5 |
| CiteSeer | vs w/o GRACE (X balls) | +8.87pp | 5/5 |
| CiteSeer | vs shuffled | +8.14pp | 5/5 |
| CiteSeer | vs hard majority | -5.11pp | 0/5 |
| Coauthor-CS | vs MLP+CE | +1.53pp | 5/5 |
| Coauthor-CS | vs w/o GRACE (X balls) | +0.56pp | 5/5 |
| Coauthor-CS | vs shuffled | +2.23pp | 5/5 |
| Coauthor-CS | vs hard majority | -1.84pp | 0/5 |

结论：C3 部分成立。GRACE 表示、真实球结构、球监督相对裸 MLP 都有稳定贡献；但“软标签优于硬多数票”被三数据集 5/5 负结果否定。论文不能再把“soft target 必要性”作为 MLP 线的核心 claim。更准确的表述是：粒球监督必要，soft distribution 是保守稳定版本，但在弱 MLP + uniform0.4 下 hard majority 更强。

### 总体决策

MLP-2 支持进入普通 SCI 写作，但论文 claim 必须收窄：
- 可以写：graph-less MLP 推理带来结构噪声免疫；GRACE-H 粒球监督在多数标签噪声设定中优于同 backbone MLP+CE、shuffled、以及多数抗噪 MLP 策略。
- 必须写边界：SoftDist-MLP 不是所有噪声设定全胜；Coauthor-CS uniform0.2 / pair0.4 失败；hard majority 在 uniform0.4 消融中稳定超过 soft distribution。
- 不应写：soft label 一定优于 hard correction；SoftDist-MLP 在 clean graph 绝对性能上优于 GCN；方法是通用 SOTA。
