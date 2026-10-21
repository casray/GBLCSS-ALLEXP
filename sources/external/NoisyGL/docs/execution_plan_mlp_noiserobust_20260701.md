# 执行计划：SoftDist-MLP 噪声鲁棒范式对照 —— 用对的指标重新审判被埋没的正结果（Phase MLP-1）

日期：2026-07-01
执行者：Codex
定位：普通 SCI 可行性验证。**不换域(仍用图 Cora/CiteSeer + GRACE H)，改的是"评价范式"——从"绝对精度赢NRGNN"改为"同一个弱MLP backbone下，我的抗噪策略 vs 其他抗噪策略"(MgCNL/MgBIF式)。**

## 0. 为什么做这个（关键认知修正）

- 读 MgCNL/MgBIF 原文后确认：**噪声鲁棒学习的SOTA定义 = 固定同一个弱backbone，比各抗噪策略谁强，尤其看高噪声下降得慢；不是绝对精度、不是打败最强baseline。**
- **用户之前的评价指标错了**：一直用"SoftDist绝对精度赢不赢NRGNN[X‖H]"(分类SOTA尺子)判断→判失败。但换噪声鲁棒尺子，用户手里有个**被埋没的正结果**：`SoftDist-MLP(弱X-only MLP下游)` 在 2026-06-18 记录里 **+6~9pp over CE、6/6噪声条件稳定正、都赢shuffled、uniform-0.6还+6.8pp**。当时被"赢不赢NRGNN"否掉了。
- **但它离SCI差一个关键对照**：SoftDist-MLP 只跟"裸MLP+CE"比过，**从没跟co-teaching/small-loss/ELR等真抗噪策略在同一个弱MLP上比**。MgCNL式主表要求赢的是抗噪策略，不是裸baseline。**本实验就补这个对照——它决定"SoftDist能不能不换域直接重新定位成噪声鲁棒SCI"。**

## 0.1 唯一要回答的问题

> 在**同一个弱MLP backbone**下、同噪声率下，`SoftDist-MLP软标签` 的test精度，是否 **> 裸MLP+CE、> co-teaching、> small-loss、> ELR、> shuffled**，在多数据集×多噪声率上稳定？
> - 是 → SoftDist可重新定位为噪声鲁棒方法(弱下游+粒球软标签)，不换域即有SCI雏形，按MgCNL式主表展开。
> - 否 → +6~9pp只是赢裸baseline、赢不了真抗噪策略，不够格 → 回NIDS/换域。

## 0.2 红线

1. **同backbone公平性(命门)**：所有抗噪方法(SoftDist / co-teaching / small-loss / ELR / JoCoR / SCE)**必须用同一个弱MLP backbone(X-only MLP，不输入H、不图传播)**，同隐层维度、同优化器、同epoch。否则不是"比抗噪策略"，是"比backbone"，整个实验作废。
2. clean标签只进评测(算test acc)；噪声只加train。
3. seed0先smoke，通过再5 seeds；多seed配对进结论。
4. Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`(ROCm，SoftDist-MLP原环境)。NoisyGL percent split 0.2/0.1/0.7 显式传参。

## 0.3 环境与复用（几乎全现成，核心是"把抗噪baseline改成MLP backbone"）

- `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py` 已有：
  - `frozengrace_gb_mlp_softdist_Predictor`（**本方法，MLP backbone**，已注册）
  - `mlp_ce_Predictor`（裸MLP+CE基线）
  - `frozengrace_gb_mlp_softdist_shuffled_Predictor`（shuffled对照）
- 抗噪baseline已有predictor：`Coteaching_Predictor.py` / `JoCoR_Predictor.py` / `SCE_Predictor.py`——**但默认GCN backbone**。**本实验核心工程量 = 让这些抗噪方法跑在同一个弱MLP backbone上**（见§2）。
- small-loss / ELR：若无现成MLP版，用最小实现（small-loss=按样本loss选低loss子集训MLP；ELR=early-learning正则项加到MLP的CE上）。

## 1. 对照方法（全部 X-only 弱MLP backbone，同配置）

| 方法 | 说明 | 角色 |
|---|---|---|
| `MLP+CE` | 裸MLP+交叉熵，无抗噪 | **基线(要打败)** |
| `MLP+small-loss` | 按样本loss选低loss样本训 | 经典抗噪 |
| `MLP+co-teaching` | 双MLP互选小loss样本 | 强抗噪对照 |
| `MLP+ELR` | early-learning正则 | 强抗噪对照 |
| `MLP+SCE`（可选） | 对称交叉熵损失 | 鲁棒损失对照 |
| **`MLP+SoftDist`（本方法）** | GRACE H球软标签 → 训X-only MLP | **本方法** |
| `MLP+SoftDist-shuffled` | 打乱球软标签 | **真结构对照** |

> 关键：SoftDist的球建在GRACE H上（表示层，这部分不变），但**下游MLP只吃X**（弱下游）。co-teaching/ELR等也用同一个X-only MLP，保证"表示信息量"对所有方法一致——它们不用H，SoftDist的H只体现在软标签里。（若要更严格，可加一个"MLP+SoftDist但软标签用shuffled"已覆盖。）

## 2. 核心工程：抗噪baseline统一到弱MLP backbone

- Coteaching/JoCoR/SCE 现有predictor把backbone从GCN改为MLP(X-only)：改其内部encoder构造，或复用 `MLP_Predictor.py` 的MLP模块 + 各自的抗噪loss/选样逻辑。
- 若改动大，最小可行：**自写一个统一的 `mlp_noise_baselines.py`**——一个X-only MLP + 可切换抗噪策略(none/small-loss/co-teaching/ELR)，保证所有策略共享完全相同的MLP结构和训练循环，只有抗噪逻辑不同。**这样公平性最强，且比改5个predictor省事。**
- SoftDist-MLP 用现有 `frozengrace_gb_mlp_softdist_Predictor`，但要确认它的MLP结构和上面统一MLP一致（同隐层维度）。

## 3. 数据与噪声

- 数据：Cora、CiteSeer（SoftDist-MLP已验证有+6~9pp的两个数据集）。**可选加Coauthor-CS**（第三个成功数据集，增强多数据集说服力）。
- 噪声：uniform 0.2 / 0.4 / 0.6 + pair 0.4（覆盖低中高，MgCNL式多噪声率）。
- seed0先smoke(Cora uniform0.4)，过了再 seeds 0-4。

## 4. ★判据（MgCNL式，非绝对SOTA）

**主判据**：在同一弱MLP backbone下，`MLP+SoftDist` 的test精度：
- **> MLP+CE**（清洗有用，已知+6~9pp，预期稳过）
- **且 > MLP+small-loss、> MLP+co-teaching、> MLP+ELR**（关键：赢真抗噪策略，这是从没测过的）
- **且 > shuffled**（真结构）
- 在 **≥2/3 噪声率、≥3/5 seeds** 成立，**≥2 数据集**。

**加分诊断（MgCNL式卖点）**：画"精度 vs 噪声率"曲线——SoftDist是否在**高噪声(0.6)下降得比其他抗噪策略慢**（MgCNL的核心卖点就是高噪声不崩）。

## 5. go/no-go 闸门

| 闸门 | 判据 | 结论 |
|---|---|---|
| **G1（决定性）** | `MLP+SoftDist` > co-teaching 且 > small-loss 且 > ELR（同MLP backbone），≥2/3噪声率、≥2数据集、≥3/5 seeds | **GO**：SoftDist可不换域重新定位为噪声鲁棒方法。展开完整SCI（+对比表示消融、+Coauthor-CS、+高噪声曲线、MgCNL式主表、justify"为何弱MLP下游"=轻量/无图传播叙事）|
| G2（卖点强度） | 高噪声(0.6)下SoftDist相对其他抗噪策略优势更大 | 强卖点，写进主图 |
| **G-fail** | SoftDist 打不过 co-teaching/ELR（即只赢裸CE，赢不了真抗噪策略），≥2/3噪声率 | **不换域这条不成立**：+6~9pp只是赢裸baseline → 回NIDS/换域(fault-diagnosis/HAR) |

## 6. 执行顺序

```
1. 建统一弱MLP + 可切换抗噪策略(none/small-loss/co-teaching/ELR/SCE) —— code: mlp_noise_baselines.py
   确认与 frozengrace_gb_mlp_softdist_Predictor 的MLP结构一致(同隐层维度/优化器/epoch)
2. 复用 SoftDist-MLP(球建在GRACE H、下游X-only MLP)
3. seed0 smoke(Cora uniform0.4)：所有方法能训、出test acc、SoftDist>CE复现(应≈+6~9pp)、
   clean不进train(单测)
   └─ 异常 → 停，修
4. Cora+CiteSeer(+可选CoauthorCS) × uniform{0.2,0.4,0.6}+pair0.4 × seeds0-4 → 按§5判
5. 报告：同MLP backbone下各抗噪策略精度表(数据集×噪声率) + 高噪声曲线 + G1/G2/G-fail判定
```

- 输出目录：`external/NoisyGL/runs/mlp_noiserobust_20260701/`
- 报告：每方法×数据集×噪声率的test acc(mean±std over 5 seeds) + SoftDist vs 各抗噪策略配对delta + 高噪声曲线 + 明确go/no-go。

## 7. 完成后

- 更新记忆 `gb-direction-turning-point-2026-06.md`（MLP-1结果 + "SoftDist按噪声鲁棒指标能否赢抗噪策略"的定论）。
- **G1过**：起草SoftDist噪声鲁棒SCI计划（重新定位、MgCNL式主表、对比表示消融、弱下游justify、目标普通SCI）。**这将是不换域、用现有资产的最快SCI路径。**
- **G-fail**：不换域路线关闭，SoftDist的+6~9pp仅相对裸baseline有效，转NIDS/换域(fault-diagnosis noisy-label，MgCNL/MgBIF已证球有效)。

---

## Phase MLP-1 执行结果备案（2026-07-01）

结果文件：`external/NoisyGL/runs/mlp_noiserobust_20260701/`

代码与配置改动：
- 新增统一 X-only MLP 抗噪 predictor：`predictor/MLPNoiseBaselines_Predictor.py`。
- 新增方法：`mlp_small_loss`、`mlp_coteaching`、`mlp_jocor`、`mlp_elr`、`mlp_sce`。
- 注册到 `single_exp.py`、`scripts/run_standard_downstream.py`、`scripts/run_ssl_swap_downstream.py`。
- 新增 Cora/CiteSeer config，均与 `mlp_ce` / `frozengrace_gb_mlp_softdist` 对齐：2-layer MLP、hidden=64、dropout=0.5、lr=1e-2、weight_decay=5e-4、epoch=200。
- 新增批处理与汇总：`scripts/run_mlp_noiserobust_20260701.ps1`、`scripts/summarize_mlp_noiserobust.py`。

验证：
- `py_compile` 通过。
- 新增测试 `tests.test_mlp_noise_baselines` 与 `tests.test_mlp_noiserobust_summary`：8/8 pass。
- ROCm 环境：torch `2.9.1+rocm7.2.1`，`cuda_available=True`，设备 `AMD Radeon RX 7700 XT`。
- 主批次：Cora/CiteSeer × uniform {0.2,0.4,0.6} + pair 0.4 × 8 methods × seeds 0-4，共 320/320 行，无缺失，无 Traceback/OOM。

核心结果（mean±std，test acc）：

| Dataset | Noise | Rate | CE | Small-loss | Co-teaching | JoCoR | ELR | SCE | SoftDist | Shuffled | G1 | Strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| CiteSeer | pair | 0.4 | 0.451±0.017 | 0.482±0.026 | 0.463±0.023 | 0.486±0.024 | 0.427±0.024 | 0.422±0.016 | **0.500±0.022** | 0.451±0.022 | pass | pass |
| CiteSeer | uniform | 0.2 | 0.551±0.012 | 0.593±0.019 | 0.571±0.008 | 0.594±0.014 | 0.536±0.015 | 0.521±0.033 | **0.623±0.008** | 0.566±0.011 | pass | pass |
| CiteSeer | uniform | 0.4 | 0.435±0.030 | 0.497±0.031 | 0.462±0.029 | 0.499±0.034 | 0.432±0.040 | 0.421±0.035 | **0.516±0.027** | 0.435±0.024 | pass | pass |
| CiteSeer | uniform | 0.6 | 0.334±0.014 | **0.377±0.029** | 0.342±0.029 | **0.391±0.025** | 0.308±0.027 | 0.306±0.028 | 0.373±0.024 | 0.316±0.018 | fail | fail |
| Cora | pair | 0.4 | 0.428±0.022 | 0.470±0.028 | 0.443±0.023 | 0.459±0.024 | 0.418±0.017 | 0.414±0.020 | **0.479±0.021** | 0.411±0.015 | pass | pass |
| Cora | uniform | 0.2 | 0.550±0.022 | 0.598±0.013 | 0.586±0.020 | 0.601±0.023 | 0.529±0.024 | 0.536±0.022 | **0.614±0.012** | 0.551±0.019 | pass | pass |
| Cora | uniform | 0.4 | 0.442±0.026 | **0.519±0.020** | 0.468±0.018 | 0.517±0.013 | 0.443±0.014 | 0.436±0.019 | 0.518±0.014 | 0.441±0.019 | fail | fail |
| Cora | uniform | 0.6 | 0.325±0.034 | 0.367±0.011 | 0.333±0.035 | 0.369±0.021 | 0.328±0.035 | 0.327±0.038 | **0.371±0.029** | 0.330±0.027 | pass | fail |

预注册判定：
- G1（CE、small-loss、co-teaching、ELR、shuffled 全部过，且 ≥3/5 seeds）：Cora uniform 2/3，CiteSeer uniform 2/3；pair 0.4 两数据集均通过。
- 因此 **Phase MLP-1 按预注册弱 MLP 噪声鲁棒判据为 GO**。
- Strict 口径（额外要求赢 JoCoR 和 SCE）：CiteSeer 3/4 通过，Cora 2/4 通过；Cora uniform0.4 和 uniform0.6 对 small-loss/JoCoR 边际或不稳定，不能写成“全面碾压所有抗噪方法”。

最终决策：
- **继续**：SoftDist-MLP 可以作为“不换域、弱下游噪声鲁棒”SCI 路线的可行起点。
- 论文 claim 应写为：在同一 X-only MLP backbone 下，SoftDist 软标签在 Cora/CiteSeer 的多数中低噪声/结构噪声设置中优于 CE、small-loss、co-teaching、ELR 和 shuffled；高噪声 uniform0.6 与 JoCoR/small-loss 存在边界。
- 下一步不应继续调本轮 baseline；应进入论文方案层：补 Coauthor-CS 或 instance0.4 抗噪 baseline 表、画噪声率曲线、整理 MgCNL式主表与公平性说明。
