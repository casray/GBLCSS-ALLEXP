# S2a 执行方案 — 对比表示上的 reliable mutual-kNN 粒球去噪（2026-06-07）

> 本文件是「转折点后第一项关键实验」的固定执行方案，目的是**防止实现时跑偏**。
> 上游结论见 `02_experiment_history_and_lessons.md` §12、`external/NoisyGL/runs/reliablegb_s1_20260607/s1_report.md`。
> 这是 plan / spec，不是结果报告；跑出来的数写到对应 runs 报告里，不要回填到本文件。

## 0. 一句话目标

验证「在 **frozen 对比表示（GRACE）** 上构同一套 reliable mutual-kNN 粒球去噪」是否优于「在**监督 GCN hidden** 上构球」以及优于 baseline / 对照——即检验 S1 的工作假设「瓶颈在监督表示本身」。

成功 → 进入 S2b（联合训练）/ S3（表示 × 球质量系统研究）。
失败（B 持平 A/baseline，或球质量提升不转化为下游增益）→ reliable-GB 去噪降级为 diagnostic，论文主线改为「表示空间 × 球质量」研究。

## 1. 不变量（每条都不许动）

- 构球：**唯一**用 `build_dcgb_balls(mode="emb_only")`（= reciprocal/mutual-kNN reliable 构造）或 `utils/reliable_gb.build_reliable_granular_balls`。A、B 两个表示用**完全相同**的构球与去噪超参，**只有表示不同**。
- clean labels 仅用于 diagnostics / eval：**禁止**进入构球、suspicion、gate、权重、teacher、target。
- 去噪只能是**保守重加权**或 boundary-safe qsafe；**禁止** hard same-ball positive / prototype pulling。
- 不改 train/val/test split、评估协议、训练协议。
- **不做 threshold-shopping**：top_ratio / min_ball_conf / beta 不作为「新方法」去调。
- 每个 claim 必须带 **no-ball + random-size-matched** 对照，且在**同一 gate size**下比较。
- 不从单 seed、不从 within-std 的差判定成功；不只看 Cora。
- 异配图不做（用户已定方向）。

## 2. 两条 protocol lane（**严禁混报**）

S1 用 NoisyGL（seeds 3000-3004、uniform/pair）；minival 用 planetoid split（seeds 0-4、symmetric）。
代码已核对：`scripts/run_gate_experiment.py` 只支持 `--mode frozen`、datasets `Cora/CiteSeer`、`--noise-type symmetric`、`--noise-rates`、`--seeds`（默认 0-4）。

| lane | 用途 | split | 噪声 | seeds | 表示来源 |
|---|---|---|---|---|---|
| **minival lane** (`gb_graph_aware_minival_6_1`) | first signal，快 | planetoid | **symmetric only** | **0–4** | frozen GRACE（`scripts/extract_embeddings.py`） |
| **NoisyGL lane** (`external/NoisyGL`) | main table，可比基线 | NoisyGL split | uniform **+ pair** | **3000–3004** | 新增 `frozengb` predictor 载入 GRACE H |

> ⚠️ GPT 草案里的「uniform/pair × seeds 3000-3004」属于 **NoisyGL lane**，**不是** minival。minival 当前 runner 没有 pair 噪声、seed 也不是 3000 段。两条 lane 的数各自成表，论文里分开陈述。

## 3. Phase 0 — 决定「表示 A（GCN hidden）」怎么来（先决问题）

minival 的 `extract_embeddings.py` 只产 GRACE H，**不**原生产 GCN-hidden，所以 A-vs-B 同协议对比需要先决定：

- **路线 (a) 同协议、强 claim（推荐若可行）**：给 minival 加一个 GCN-hidden 嵌入源（训一个普通 GCN，dump 倒数第二层 hidden 存成与 GRACE H 同形状的 `.pt`），喂进**同一个** gate。A、B 在同一 planetoid/symmetric/seed 协议下逐点可比。
- **路线 (b) 跨协议、快 first signal**：直接用 S1 的 NoisyGL GCN-hidden reliable-GB 结果当「表示 A 参照」，与 minival frozen-GRACE 结果对比，**明确标注为两套协议**（不可逐点相减，只看趋势）。

决策记录在本文件 Phase 0 小节（执行时补），不要默默选一个。

## 4. Phase 1 — minival first-signal smoke（最先跑）

**先跑一格：Cora, symmetric, noise-rate 0.4, seed 0。**

入口（示意，实际以 `--help` 为准）：
```
python scripts/run_vanilla_grace.py   --dataset Cora --pretrain-seed <fixed>      # 若无现成 ckpt
python scripts/extract_embeddings.py  --dataset Cora --split-seed 0 --pretrain-seed <fixed>
python scripts/run_gate_experiment.py --mode frozen --datasets Cora \
       --noise-type symmetric --noise-rates 0.4 --seeds 0 \
       --methods <见下>
python scripts/summarize_gate_results.py <out-dir>
```

**methods（同一 gate size 下）**：
- `dcgb_emb_only` — 表示 B 的 reliable 球（主角）
- `gb` — ordinary GB（弱构造对照）
- no-ball 对照、random-size-matched 对照、topk-train-matched 对照（已注册：`kmeans_dcgb_matched` / `topk_dcgb_train_matched` / `dcgb_shuffled_alpha_matched` 等；执行时核对 method key 仍存在）
- baseline：GCN / 无去噪

**Phase 1 通过条件**：在该 smoke 上 `dcgb_emb_only` 的 gated precision / 下游 test 至少**明显**优于 no-ball 与 random（不是 within-noise）。否则不扩 seed，直接判 first-signal 失败并记录。

## 5. Phase 2 — minival 表示对比 + 多 seed（Phase 1 通过才做）

- 按 Phase 0 选定路线得到表示 A，与 B 在**同一 gate**跑 A vs B。
- 扩到 **Cora/CiteSeer × symmetric × {0.2, 0.4} × seeds 0–4**（保留 0.0 作 clean 参照）。
- 成功 = B 的球质量更高（purity/AUROC）**且**下游 test 超过 A **且**超过 GCN baseline，跨 seed 均值差 > std。

## 6. Phase 3 — NoisyGL main-table（Phase 2 通过才做）

- 新增 `frozengb` predictor：载入对应 split 的 frozen GRACE H（**必须与 NoisyGL 的 `add_self_loop`/split/seeds 完全一致**，否则报为单独协议），一次性构 reliable 球，训 GCN 分类头 + 保守重加权。
- 注册进 `single_exp.py` 与 `total_exp.py`（`PREDICTOR_MODULES` + `--method/--methods`）；config 路径 `config/frozengb_*/frozengb_*_<dataset>.yaml`。
- 跑 **Cora/CiteSeer × uniform+pair × {0.2,0.4} × seeds 3000–3004**，对照 GCN（floor）/ RNCGLN（强基线）。
- **对照 bug 复查**：no-ball/random 控制字段不能只靠类属性——`_conf_value` 会用 `gb:` 块覆盖类属性。跑完务必在 diagnostics CSV 里确认 `suspicion_mode` / `random_control` 真正不同（S1 踩过这个坑）。

## 7. 统一记录的指标

每个 method / 每格记录：`test acc`、`best_phase`、ball `purity`、suspicion `AUROC`、`gated precision`、`NCR`（noise correction rate）、`CDR`（clean damage rate），外加 gate size（保证对照同 size）。

## 8. 各 Phase 产物落点

- Phase 1/2：`gb_graph_aware_minival_6_1/reports/s2a_*_<date>/`（report.md + summary csv）。
- Phase 3：`external/NoisyGL/runs/frozengb_s2a_<date>/`（report.md + diagnostics csv）。
- 关键结论回写 `02_experiment_history_and_lessons.md` §12 与 `external/NoisyGL/runs/orgingb_noisygl_experiment_history_20260604.md`。

## 9. 验证 / 回归

- 改任何 minival 代码后：`python -m unittest discover -s tests`（当前 77 pass 应保持）。
- 改任何 NoisyGL 代码后：`python -m unittest discover -s tests`（当前 11 pass 应保持）。
- 先单格 smoke 通过再扩 seed；任何「截断/采样/不重试」都要 `log` 出来，不许静默。
