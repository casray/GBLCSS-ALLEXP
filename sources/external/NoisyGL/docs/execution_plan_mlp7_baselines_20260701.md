# 执行计划：最新方法对比 —— 分两层补，图方法进"定位表"、通用LNL进"主表"（Phase MLP-7）

日期：2026-07-01
执行者：Codex
定位：**满足"比过最新方法"的诉求，但用对的方式——不重蹈"绝对精度错标尺"。两层：(7a) 通用 LNL 最新策略(CT/DSS)移植到同弱 MLP，进【同 backbone 主表】；(7b) 图原生 LNL(TFR/ProCon/ICGNN…)各自原生 GNN 设置跑，进【定位对比表 + 过 C2 结构扰动/效率】，绝不并入主表。**
前置：**MLP-6(H-input 公平性门)先过**。若 MLP-6 Part A 显示主要增益来自 H-access（β0.5 < H-input baseline），先按新 claim reframe，再决定 7b 是否值得砸算力。

## 0. 铁律（这是本计划存在的唯一理由，违反就是重犯 45 天的坑）

1. **主表 = 同一 X-only 弱 MLP + 只比抗噪策略**(MgCNL 范式)。只有能移植到该 backbone 的策略级方法可进。
2. **图原生 GNN 方法(TFR/ProCon/ICGNN/GD2/DND-NET)绝不进主表**：它们用邻接+强 GCN 下游，绝对精度天然高 ~15-20pp（TFR Cora≈0.77 vs 本线 MLP≈0.56），且机制本质用图、无法同 backbone 移植。放主表 = 用绝对精度自残 + 把审稿人拉回错标尺。
3. 图方法只进**独立"定位对比表"**(各自原生设置)，且**必须同时过 C2 结构扰动 + 效率测量**——我们在 profile 轴(结构鲁棒/轻量)赢，不在绝对精度轴claim赢。
4. 所有移植方法(FINE/DivideMix-lite/CT/DSS)**诚实标 "MLP-adapted，非原版"**。clean 只进评测。多 seed 配对 + ≥2/3 数据集一致性。ROCm venv，percent split 显式传参。

## 7a. 通用 LNL 最新策略 → 进同 backbone 主表（低成本、必做）

- **加进 `MLPNoiseBaselines_Predictor` 的 strategy 分发**（同 FINE/DivideMix-lite 的做法，天然同 backbone）：
  - **CT / Confidence Tracking（AAAI 2025，优先，最易移植）**：核心 = 跨 epoch 追踪每样本预测置信度轨迹做样本选择(比 small-loss 单点更稳的 plug-and-play selection)。加 `strategy="ct"`：维护每样本置信度 EMA/历史，选高置信轨迹样本训练。
  - **DSS / Debiased Sample Selection（CVPR 2026，可选，small-loss 最新升级版）**：核心 = 修正 small-loss 选择偏差(按类/难度去偏)。加 `strategy="dss"`。若工程超预算，7a 只做 CT。
- **单测**：CT/DSS 的 clean 标签不进选择决策(篡改 clean 不变)；同 MLP 结构。
- **矩阵**（并入 MLP-5 主表，新增列）：
  ```
  Cora/CiteSeer/PubMed × {uniform0.2/0.4/0.6, pair0.4, instance0.4} × {CT, (可选)DSS} × seeds0-4
  ```
  β0.5 与其余 baseline 复用 MLP-5，不重跑。产出 = β0.5 vs CT/DSS 的配对 delta，补进主表"vs 最新抗噪策略"。
- **诚实预期**：CT/DSS 是比 small-loss/JoCoR 更强的样本选择 → PubMed 上大概率也强(像 small-loss)，Cora/CiteSeer 上 β0.5 应仍占优。结果照实写。

## 7b. 图原生 LNL 最新方法 → 独立"定位对比表"（分级、带 fallback，勿卡实验）

- **协议**：各方法**原生 GNN 设置**，NoisyGL 框架，**同 percent split 0.2/0.1/0.7**、同数据(Cora/CiteSeer/PubMed)、同噪声谱起步(uniform0.4；行有余力加 pair0.4/instance0.4)、seeds0-4。
- **分级(按可运行性 + 相关性)**：
  | 级 | 方法 | 动作 | Fallback |
  |---|---|---|---|
  | 必跑 | **TFR**(CIKM2025, 官方基于 NoisyGL) | 对齐官方 repo 跑同协议 | — |
  | 尝试 | **ProCon**(IJCAI2025, prototype+GMM+peer 一致性，与"球/局部一致性"最近) | 有官方码则跑；否则 related-work 重点讨论 | related-work |
  | 尝试 | **ICGNN**(arXiv2026, 图扩散 ICS+GMM 软纠错) | 码可跑则跑，作 2026 对比 | related-work |
  | 讨论 | **GD2**(NeurIPS2025 双视角判别) / **DA-GNN**(依赖噪声) / **DND-NET**(KDD2024 simple-model) | 码现成且低成本才跑；否则 related-work | related-work |
- **工程护栏**：每个外部 repo 设**安装/跑通时间预算**(如 ≤ 半天)；超预算即转 related-work，**不许卡住主线**。记录哪些跑了、哪些转讨论。
- **★关键——图方法必须过 profile 轴对比(这才是我们的武器)**：
  1. **过 C2 结构扰动**：把 TFR/ProCon(跑通的)也施加 test 邻接 edge-drop/isolated p∈{0.25,0.5,0.75}，测精度下降。预期 GNN 掉 5~23pp，本线 MLP 掉 0.00pp。
  2. **过效率表**：参数量 / 单次训练墙钟 / 是否迭代。预期图方法重且迭代。
- **报告口径(铁律3)**：定位表明确写"图方法绝对精度更高，但(a)推理依赖图→结构噪声下垮、(b)重/迭代；本方法不 claim 赢其绝对精度，卖点是组合 profile(双鲁棒+轻量graph-less+低标签率)"。**这些数字进定位/positioning 表，不进主表抗噪策略对比。**

> ★MLP-6 已 reframe（β0.5 输 small-loss[X‖H]/JoCoR[X‖H]）→ 7b 的角色变了：不再拿绝对精度硬碰图方法；而"message-passing GNN 推理依赖图、结构噪声下垮"这句 **MLP-2 C2 已铁证（GCN/GIN 掉 5~23pp）**，不 depend on 外部 repo。故 **7b 整体降级为"待定/可选"**：普通 SCI 里 TFR/ProCon 可 related-work 讨论+引用即可；只有在 7a/7c 出来后确实想要"一个 2025 图方法数字"，才补 TFR(NoisyGL-native)。**不阻塞主线。**

## 7c. ★H-input baseline 结构鲁棒对照（新增，用户 catch，优先于外部 repo）★

- **动机**：reframe 后想把"**结构鲁棒压倒 H-input baseline**"作为强卖点——即"small-loss[X‖H]/JoCoR[X‖H] 精度赢我，但结构扰动下垮，我(X-only)不垮"。**但这没跑前不能写死**，因为结局真未知（见下）。
- **命门协议（必须这样测，否则测试无意义）**：结构扰动扰的是**邻接**、不动 X。若 [X‖H] baseline 用**冻结的 clean-H**，扰动边不改变其输入 → 它纹丝不动 → 测试作废。**唯一有效测法 = 在扰动后的图上重算 GRACE-H 作为 baseline 的输入。**
- **⚠️ 工程前提（第一步先确认）**：重算 H 需要 **GRACE 编码器 checkpoint**，不只是存好的 embedding。
  - encoder 在 → 正常做：对每个扰动图跑 GRACE encoder → 得扰动 H → 喂 [X‖H] baseline。
  - encoder 不在 → 要么恢复/重训 GRACE encoder；要么**放弃本对照**，结构卖点退回"message-passing GNN 会掉"(MLP-2 C2 已有)，**不硬凑**。
- **矩阵**：
  ```
  Datasets: Cora, CiteSeer (, PubMed 可选)
  扰动: test 邻接 edge-drop / isolated, p ∈ {0.25, 0.5, 0.75}   (复用 MLP-2 C2 harness)
  Methods: SoftDist-MLP β0.5 (X-only, 预期掉 0.00pp) 
           small-loss[X‖H] / JoCoR[X‖H] (H 在扰动图上重算)
  Noise: uniform 0.4 (label noise 固定，看结构扰动单独效应)  Seeds: 0-4 (先 seed0 看方向)
  ```
- **★预注册三结局（决定卖点强弱）**：
  | seed0→5seed 观察 | 卖点定性 |
  |---|---|
  | [X‖H] baseline 随 p 明显掉(如 >5pp)，β0.5 掉≈0 | **强卖点**：结构轴上碾压那个在精度上赢我的 baseline，写进 headline |
  | [X‖H] baseline 掉一点(1~5pp) | 权衡叙事：H-input 用结构鲁棒换精度 |
  | [X‖H] baseline 基本不掉(GRACE 抗 edge-drop) | **放弃"对 H-baseline"这句**；结构卖点退回"GNN 会掉"(MLP-2 C2)。诚实记录，不写死 |

## 7c-ext. ★冷启动 p=1.0 探针（用户批准 A，写作前唯一待跑）★

- **动机**：7c seed0 显示中等扰动(p≤0.75)下 H-input baseline 只小掉(GRACE 抗 edge-drop)。但 graph-free 优势的**真正的家 = 完全孤立的冷启动节点**(test 节点零边)——这也是最锋利攻击"为什么不用 [X‖H]+small-loss"的正面回答。
- **设置（在 7c 的 isolated 扫描上加 p=1.0）**：
  ```
  Datasets: Cora, CiteSeer
  扰动: isolated p=1.0  (test 节点与所有节点断开所有边；train/val 图不变)
  Methods: SoftDist-MLP β0.5 (X-only, 预期 0.00pp)
           small-loss[X‖H] / JoCoR[X‖H]  (在"test节点全孤立"的图上重算 GRACE-H)
  Noise: uniform 0.4   Seeds: 先 seed0，明显信号再补 5-seed
  ```
- **机制预期（供解读）**：GRACE encoder 是 GCN，test 节点零边时其 H 退化为"仅自身特征的变换"(无邻域聚合)→ 相对训练期 H 分布 OOD → [X‖H] 分类可能明显掉；β0.5 X-only 不碰 H/邻接 → 0.00pp。**但 GRACE 仍能给出 self-based H,不一定崩,故必须实测。**
- **★预注册两结局**：
  | seed0 观察 | 结论 |
  |---|---|
  | p=1.0 下 small-loss[X‖H]/JoCoR[X‖H] 明显掉(如 >5pp)、β0.5≈0 | **结构卖点复活**：graph-free 在冷启动/零邻接部署下真赢 H-input baseline → 补 5-seed，写进论文(正面答"为何不用[X‖H]") |
  | p=1.0 下仍只小掉 | 结构刀彻底砍不动 → 诚实不写结构 headline，论文靠脊梁三条(球必要性 + X-only 全胜 + 低标签率) |
- 跑完无论哪个结局 → **进论文写作**(脊梁不依赖它)。

## 8. 执行顺序（按 reframe 重排：内部先、外部后）

```
== 现在跑（内部、便宜、直接服务 reframe 后的论文）==
1. 7a: 加 CT(+可选DSS) 到 MLPNoiseBaselines，单测；跑 3数据集×5噪声×{CT,(DSS)}×seeds0-4；并入 X-only 主表
2. 7c-step0: 确认 GRACE encoder checkpoint 是否存在（决定 7c 能否做）
3. 7c: 若 encoder 在 → Cora/CiteSeer × {edge-drop,isolated}×p{0.25,0.5,0.75} × {β0.5, small-loss[X‖H], JoCoR[X‖H](H重算)} × seeds(先0再补5)；按预注册三结局判卖点
== 待定/可选（外部 repo，7a/7c 出来后再决定，不阻塞）==
4. 7b(可选): 若确需 2025 图方法数字 → TFR(NoisyGL-native) 同协议跑；ProCon/ICGNN/GD2/DND-NET 默认 related-work 讨论
5. 汇总: X-only 主表新增 CT/DSS 列 + 7c 结构鲁棒对照(含三结局判定) + (可选)图方法定位表 + related-work 覆盖清单
6. 报告 + 分开陈述：X-only 策略主表 / 结构鲁棒 profile / (可选)图方法定位，claim 已按 MLP-6 收窄
```

- 输出目录：`external/NoisyGL/runs/mlp7_baselines_20260701/`
- 报告：7a X-only 主表 CT/DSS 配对 delta；7c H-input baseline 结构鲁棒对照(三结局判定 + encoder 前提记录)；7b(若跑)图方法定位表 + fallback 清单；结论分开、claim 收窄后陈述。

## 9. 完成后

- 更新记忆(MLP-7：CT/DSS 进主表位置 + 图方法定位表结局 + 哪些转 related-work)。
- **主表(经典+最新策略齐) + 图方法定位表 + C2 + 效率** → "比过最新方法"诉求以正确方式满足 → 进论文写作。
- related-work 切割：Graph-MLP/GLNN(不做标签噪声)、MgCNL/DivideMix/FINE/CT(不做图/结构噪声)、TFR/ProCon/ICGNN(图原生但推理依赖图、重、不做结构鲁棒) —— 逐一划清与本方法的差异。

---

## Phase MLP-7 执行结果备案
结果文件：`external/NoisyGL/runs/mlp7_baselines_20260701/`

完整性：
- 7a CT 新增 run：75/75，missing=0。
- 合并 MLP-5 主表 anchor 后：825/825，missing=0。
- 7c seed0 H-input 结构鲁棒对照：Cora/CiteSeer × 3 methods × clean+6 perturb settings = 42 rows。
- 7c-ext 冷启动 isolated p=1.0 探针：Cora/CiteSeer × 3 methods × clean+1 perturb setting = 12 rows，输出 `hinput_structure_coldstart_p1_seed0.csv`。
- 单测：`tests/test_mlp_noise_baselines.py tests/test_mlp7_baselines.py`，11 passed。

7a：CT-MLP adapted baseline 结论 = **Cora/CiteSeer 核心数据集上 β0.5 全胜 CT；PubMed 仍是边界**。

CT 是 MLP-adapted 策略，不是官方 CT 原版复现；DSS 是预注册 optional，本轮未实现。

核心配对结果（β0.5 - CT，5 seeds）：
- Cora：uniform0.2 +3.07pp；uniform0.4 +4.14pp；uniform0.6 +6.56pp；pair0.4 +4.60pp；instance0.4 +5.25pp；全部 5/5 positive。
- CiteSeer：uniform0.2 +3.77pp；uniform0.4 +4.38pp；uniform0.6 +2.89pp；pair0.4 +1.27pp；instance0.4 +4.71pp；全部 5/5 positive。
- PubMed：uniform0.2 +0.91pp（5/5 positive），但 uniform0.4 -5.33pp、uniform0.6 -5.79pp、pair0.4 -2.48pp、instance0.4 -4.24pp，四个 regime 都 0/5 positive。

解释：
- 对 Cora/CiteSeer，CT 没有推翻 X-only 主表优势；β0.5 对最新风格的 confidence-tracking sample selection 仍有稳定增量。
- 对 PubMed，CT 与 small-loss/JoCoR 一致，再次说明 PubMed 是“样本选择强、球软监督弱”的边界数据集，不能把 β0.5 claim 推广为全数据集全胜。

7c-step0：GRACE encoder checkpoint 检查 = **可用**。
- conservative scan 找到 112 个候选 checkpoint。
- 关键路径包括 `gb_graph_aware_minival_6_1/checkpoints/grace/Cora/ratio_train0p2_val0p1_split0_pretrain0/checkpoint.pt` 和 CiteSeer 对应 checkpoint，可用于扰动图上重抽 H。

7c seed0：H-input baseline 结构鲁棒对照结论 = **weak_structure_tradeoff_signal_seed0**。

协议：训练阶段使用 clean-H；评估阶段对 small-loss[X‖H]/JoCoR[X‖H] 在扰动后的图上重算 GRACE-H，再拼接 `[X‖H_perturbed]` 测 test；β0.5 为 X-only，H_recomputed=0。

结果要点：
- β0.5 在 Cora/CiteSeer 的 edge_drop/isolated 全部 0.00pp 掉点，符合 graph-free 推理。
- Cora small-loss[X‖H] 最坏掉点 -2.48pp（isolated p=0.75）；JoCoR[X‖H] 最坏 -1.85pp。
- CiteSeer small-loss[X‖H] 最坏 -0.73pp；JoCoR[X‖H] 最坏 -0.73pp。

解释：
- 这不是强结构卖点。H-input baseline 在扰动 H 下只小幅下降，没有出现 >5pp 崩塌。
- 论文不能写“small-loss[X‖H]/JoCoR[X‖H] 结构扰动下垮，所以我们结构轴碾压”。最多写成：**X-only β0.5 对推理结构扰动完全不敏感；H-input baselines 对重抽 H 有轻微敏感性，但 seed0 证据只支持弱 trade-off，不支持强 headline。**

7c-ext cold-start isolated p=1.0：结论 = **coldstart_structure_headline_not_supported_seed0**。

协议：test 节点在评估时完全孤立；small-loss[X‖H]/JoCoR[X‖H] 在孤立图上重算 GRACE-H；β0.5 仍为 X-only 推理。

结果要点：
- Cora β0.5：0.5546 → 0.5546，delta +0.00pp。
- Cora small-loss[X‖H]：0.5799 → 0.5515，delta -2.85pp。
- Cora JoCoR[X‖H]：0.5979 → 0.5710，delta -2.69pp。
- CiteSeer β0.5：0.5533 → 0.5533，delta +0.00pp。
- CiteSeer small-loss[X‖H]：0.5253 → 0.5185，delta -0.69pp。
- CiteSeer JoCoR[X‖H]：0.5558 → 0.5468，delta -0.90pp。

解释：
- p=1.0 冷启动确实比 p<=0.75 更伤 Cora 的 H-input baseline，但仍低于预注册的 >5pp 强卖点门槛。
- CiteSeer 几乎不受影响，说明 GRACE-H 在零邻接下并未出现足够大的推理崩溃。
- 因此 7c-ext 不补 5 seeds；结构轴只能作为“X-only 推理不依赖图，H-input 对 H 漂移有轻微敏感性”的弱 profile 证据，不能作为 headline。

7b：图原生外部方法本轮未跑。
- 按 reframe 后计划，TFR/ProCon/ICGNN/GD2/DND-NET 暂时转 related-work / optional positioning table，不阻塞主线。

最终决策：
- 主表可新增 CT-MLP adapted，作为“最新风格 MLP sample-selection baseline”的补充；Cora/CiteSeer 支持 β0.5，PubMed 诚实作为边界。
- 7c/7c-ext 不建议扩 5 seeds，除非论文特别需要 H-input 结构扰动附录；seed0 已显示它不是强卖点，继续扩的 ROI 较低。
- 写作时继续遵守 MLP-6 reframe：不 claim 同等 H-access 下赢 small-loss/JoCoR；强调 X-only、低标签率、球必要性和结构不敏感 profile。
