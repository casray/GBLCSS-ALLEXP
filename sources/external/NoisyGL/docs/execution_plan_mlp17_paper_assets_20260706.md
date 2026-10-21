# 执行计划：论文附录表格与图表生成（Phase MLP-17）

日期：2026-07-06
执行者：Codex（可调用绘图相关 skill）
定位：**纯产出物生成，零新实验。** 为 `paper_manuscript_v1_zh_20260706.md` 的附录清单 A-H 生成正式表格（markdown + LaTeX 双格式）与图（PDF+PNG），全部数据来自已有 run 目录。另含两项数据核验（H1/H2）。

## 0. 总红线

1. **零新训练**：所有数字从已有 CSV/报告读取；任何缺数按"缺失"标注，不补跑（需补跑先报告，等指令）。
2. **数字对拍**：每张表生成后与正文/主表底稿对拍关键格（脚本断言，不靠目测）；参照行一律按 `train_percent==0.2` 过滤（MLP-14 教训）。
3. **命名统一**：全部产出物中方法名用 **LCSS**（train-only）/ **LCSS-distill**（球式 teacher T3）/ **kNN-consensus distill**（T4，必须与 LCSS-distill 分开标注，不得合并——正文 v1.1 勘误的根源）/ shuffled / 各 baseline 原名。绝不出现 SoftDist-MLP 字样。
4. 图表风格：单栏宽度适配 Elsevier 双栏（图宽 ~8.5cm 或通栏 ~17.5cm）；无标题内嵌（caption 由正文管理）；色盲友好配色；字号 ≥8pt；误差棒 = ±1 std。
5. 输出目录：`paper_assets_20260706/`（仓库内新建），结构：`tables/`（.md + .tex 各一份）、`figures/`（.pdf + .png 各一份）、`asset_manifest.md`（每个产出物 → 数据来源文件 → 对拍结果）。

## A. 完整主表（5 数据集 × 5 设置 × 13 方法）

- 数据：`runs/mlp10_baseline_refresh_20260702/mlp10_aggregate.csv` + `runs/mlp12_amazonphoto_20260702/mlp12_aggregate.csv`。
- 格式：按数据集分块（5 块），行=方法（年份括号），列=5 噪声设置，mean±std；**每列数值最高加粗（诚实加粗——baseline 最高就粗 baseline）**；shuffled 行放最后加横线分隔。
- 参考模板：`docs/main_table_for_advisor_20260703.md`（已有对拍基准，直接校验一致性）。
- 显著性标记：结合 P0b（`runs/mlp15_review_defense_20260704/p0b_paired_significance.csv`），LCSS 相对最强 baseline 的格加 * (paired t-test p<0.05)；表脚注注明 n=5。

## B. 显著性检验表

- 数据：`p0b_paired_significance.csv`。
- 内容：LCSS vs 每 baseline 的逐设置 paired t-test p 值 + Wilcoxon p 值 + 逐 seed 方向计数；表脚注写明 "n=5 paired seeds; Wilcoxon lower bound 0.0625"。
- 呈现：一张长表（附录可跨页），按数据集×设置分组。

## C. Graph-dependent 参考表

- 数据：Phase 24/25/36 的 NRGNN / NRGNN[X‖H] / TFR 5-seed 行（`runs/` 下对应目录；先做 split/noise hash 与 MLP 线一致性核对——**此核对是本计划附带的数据核验 H1**，结果写进 manifest；不一致的行不进表并注明）。
- 表头必须含声明行："inference requires the graph — different deployment setting; numbers are same-protocol references, not paired comparisons"。

## D. 构造参数敏感性 + β 审计（两张）

- D1 敏感性网格：`runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageA_grid_summary.csv`——27 配置 × seed0 delta，塌缩配置（num_balls=1）单独标注区；附 Stage B 三配置 5-seed 表。
- D2 β 审计表：`runs/mlp15_review_defense_20260704/p0a_beta_validation_audit.csv`——6 试点设置的 val-best/test-best β 与 β=0.5 差距。

## E. 特征噪声完整曲线（图）

- 数据：`runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_aggregate.csv`。
- 图：5 子图（每数据集一个），x=α∈{0,...,0.9}，y=test acc，5 方法曲线 + 误差带；已有 `mlp16_feature_noise_curves.png` 可作底本重绘为出版质量。

## F. Cut Value 表（附坍缩污染说明）

- 数据：`mlp16_cut_value_aggregate.csv` + `mlp16_cut_value_raw.csv`。
- 表：6 臂 × 5 数据集 mean±std；**必须附脚注**："CE/shuffled 的高 CV 部分来自预测坍缩（per-seed std 0.15-0.28 vs distill arms 0.003-0.01），CV 在标签噪声下非干净结构度量"；正文只引用 distill vs train-only 的差值行（+0.246/+0.246/+0.139）。

## G. t-SNE 三着色图（唯一需要新计算的图，非实验）

- 数据：现成 GRACE H（`embeddings_v2/<Dataset>/grace/H.pt`）+ 球归属（球构造函数以主表 config 重算一次归属即可，确定性、秒级）+ 真类/噪声标签（seed0 噪声注入复现）。
- 图：Cora（或 Cora+CiteSeer 双图），同一 t-SNE 坐标三份着色：真实类别 / 含噪标签 / 局部组归属；固定 random_state 保证可复现；caption 备注"插图性质，非证据"。

## H. 低标签率完整曲线（图）

- 数据：`runs/mlp6_fairness_labelrate_20260701/`（Part B 6 点 5-seed）。
- 图：Cora/CiteSeer 双子图，x=train ratio {5,10,15,20,25,65}%（按实际点位），y=acc，5 方法 + 误差带。**顺带数据核验 H2**：逐点核对 LCSS−最强选择族的 delta 是否 6 点严格单调（正文 v1.1 已改为"总体扩大"，若实际严格单调可在 caption 注明，若非单调 manifest 里写明哪点回折）。

## 附加产出：表 B（正文 §5.2）与扩展节表（正文 §6）的正式版

- 表 B：`runs/mlp14_graphfree_family_20260703/mlp14_aggregate.csv` + MLP-9 重组行，**LCSS-distill(T3) 与 kNN-consensus distill(T4) 必须分两行**（v1.1 勘误落实），资源标注列保留；对拍正文 §5.2 数字。
- 扩展节表：`runs/mlp9_denoised_teacher_20260702/mlp9_aggregate.csv` 的五臂 × u0.4/u0.6 + teacher acc + 三层分解（D1/D2）；对拍正文 §6 数字。

## 交付清单与验收

- `asset_manifest.md`：每个产出物一行——文件名 / 数据来源 / 对拍断言结果（pass/fail）/ H1、H2 核验结论。
- 全部 .tex 表可直接 `\input`（booktabs 风格，无 vertical rules）。
- 验收标准：manifest 全 pass；正文引用的每个数字都能在某张表/图中找到；两处核验（H1 hash 一致性、H2 单调性）有明确结论。

---

## 执行结果备案（Codex 完成后追加于此）
