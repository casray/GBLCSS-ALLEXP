# S2a Phase 1 — first-signal smoke 结果（**single-cell, seed 0, NO-SIGNAL**）

> 这是 **single-cell smoke**（Cora, symmetric 0.4, seed 0, frozen GRACE），不是正式多 seed 结果。
> 按 `s2a_execution_plan.md` 的 Phase 1 通过条件判定：**未通过 → 不扩 seed、不进 NoisyGL lane**。

## 运行配置

- 入口：`scripts/run_noisy_classifier.py`（直接调用，带 in-CSV 诊断）
- 数据/split：**Cora, `--split-mode ratio --train-ratio 0.2 --val-ratio 0.1`, split0, pretrain0**
  （= §12 DCGB v2.8 诊断所用 split；复用已验证的 `checkpoints/embeddings/Cora/ratio_train0p2_val0p1_split0_pretrain0/H.pt`）
- 噪声：symmetric 0.4，noise-seed 0，clf-seed 0
- 表示：**frozen GRACE（representation B）**——本格**不**含 GCN-hidden（representation A），故**不**是 "GCN hidden vs GRACE" 结论
- alpha：sweep 0.3/0.5，primary 0.3
- 结果：`results/gate_runs/Cora/symmetric_0p4/ratio_train0p2_val0p1_split0_pretrain0_noise0_clf0/result_rows.csv`
- 偏离 GPT 草案：GPT 写 `planetoid`，但本仓库**无 planetoid checkpoint**，且 §12 证据全部建立在 `ratio_train0.2` 上；
  用 planetoid 既要重新预训练 GRACE，又会与 §12 证据不可比，故改用 `ratio_train0.2`（同 split，复用验证资产）。

## 结果（primary α=0.3，test_acc_at_best_val）

| method | test acc | coverage | clean_purity | noise AUROC | NCR | CDR |
|---|---|---|---|---|---|---|
| ce | 0.7878 | 0.000 | – | 0.500 | 0.000 | 0.000 |
| gb（ordinary GB） | **0.7962** | 0.627 | 0.899 | 0.654 | 0.000 | 0.000 |
| dcgb_emb_only（reliable 互惠 kNN） | 0.7941 | 0.780 | 0.889 | 0.739 | 0.000 | 0.000 |
| dcgb_qsafe_boundary_alpha09_q50 | 0.7914 | 0.026 | – | – | **0.032** | 0.000 |
| topk_dcgb_train_matched（no-ball matched） | 0.7951 | 1.000 | 0.889 | **0.904** | 0.000 | 0.000 |
| dcgb_shuffled_alpha_matched（破坏球结构对照） | 0.7941 | 0.780 | 0.495 | 0.489 | 0.000 | 0.000 |

## GPT 三问的回答

1. **dcgb_emb_only / qsafe 是否超过 gb 和 ce？** —— **否。** 仅比 ce 高 +0.6pp（单 seed 噪声内），且**低于** ordinary `gb`（0.7941/0.7914 vs 0.7962）。reliable 球没有跑赢最弱的 ordinary GB。
2. **是否超过 topk matched 或至少更低 clean damage？** —— **否。** test 上不如 `topk_dcgb_train_matched`（0.7941 < 0.7951）；clean_damage_rate **所有方法都是 0.000**（无差异，无优势）。
3. **shuffled alpha matched 是否被明显击败？** —— **否，精确打平**（0.7941 = 0.7941）。尽管真实球 purity 0.889 / AUROC 0.739 **远高于** shuffled 的 0.495 / 0.489，下游 test acc 完全相同。

## 机制解读（为什么"没信号"）

- **primary α=0.3 的方法（gb / dcgb_emb_only / topk / shuffled）NCR=0**：在该 alpha 下软目标修复**没翻转任何训练标签的 argmax**，"去噪"在标签层面 inert，这些方法相对 ce 的 +0.6~0.8pp 来自 **label-smoothing 正则**，不是纠错。
- **但不是简单"把 gate 放强"就能解决**：`dcgb_qsafe_boundary_alpha09_q50`（α=0.9）确实有 **NCR=0.0323、CDR=0.0000**（干净地纠了 **14** 个训练标签，argmax-change-rate 0.0129），可 test=0.7914 **反而低于** gb/dcgb/topk——**少量正确纠错没有转化为 accuracy gain**。无标签结构的 `topk@α=0.5` 同样 NCR=0.0230 / test 0.7878，重现"干净小纠错、下游不受益"。
- **球质量不转化为下游增益**：dcgb_emb_only 的 purity/AUROC 明显优于 shuffled，但 test 打平 → 球质量是**诊断信号**，不是下游驱动力（正中 `s2a_execution_plan.md` 的 Failure 分支）。
- **no-ball 的 topk 反而 AUROC 最高（0.904）**：噪声可分性主要来自表示+覆盖，不来自球结构本身。
- §12 结论 #2（互惠 kNN 构球在 raw 上恢复 purity）在 GRACE 上仍成立（purity 0.889 vs shuffled 0.495），但**仅止于诊断层**。

## Headroom 判生死（clean-oracle 诊断，2026-06-07）

诊断脚本 `scripts/diagnose_clean_oracle_headroom.py`（**clean labels 仅作 oracle 上界测量，绝不进方法选择/训练**），同一 frozen GRACE / 同 split / 同超参：

| row | test | NCR | CDR |
|---|---|---|---|
| noisy CE | 0.7878 | 0 | 0 |
| **clean-oracle CE（上界）** | **0.8221** | – | – |
| gb | 0.7962 | 0 | 0 |
| dcgb_emb_only | 0.7941 | 0 | 0 |
| dcgb_qsafe_α09 | 0.7914 | 0.032 | 0 |
| topk matched | 0.7951 | 0 | 0 |
| shuffled matched | 0.7941 | 0 | 0 |

- **HEADROOM = oracle − noisy CE = 0.8221 − 0.7878 = 3.43 pp**（>1pp，**不是 no-headroom 分支**）。
- clean-oracle 0.8221 ≈ 该 GRACE 嵌入的 clean 线性上界 → **冻结嵌入上唯一能动的杠杆是标签处理**（嵌入质量本身固定）。
- 当前最好的 `gb` 只吃到 0.84pp / 3.43pp（≈25%）；`dcgb_emb_only` 0.63pp 且与 shuffled 打平 → **球的成员对应内容吃到 ~0 的可恢复 headroom**。
- 落入 GPT 第二分支：**有 headroom，但"球→下游"转化失败**；规则 = 转 **S3 或 S2-redesign**，**不是**调 alpha/gate。

## 判定与下一步（按 plan 的判定规则）

- **Phase 1 first-signal：未通过（no-signal）。** 按用户/GPT 规则：**不扩 seed、不进 Phase 2、不进 NoisyGL lane。**
- **协议边界（必须始终明确）**：这是 **ratio_train0.2 split 的 first-signal smoke，不是 planetoid，也不是 NoisyGL protocol**；Cora / symmetric 0.4 / seed 0 / frozen-GRACE 单格。记为"当前 smoke 无首信号"，**不**升级为"reliable-GB 去噪普遍无效"的普适结论。
- **S2 未降级（headroom 存在 3.43pp）。** Phase 1 否定的是**"frozen-head + soft-target/reweight"这一条转化链路**，不是整个 S2。诊断把瓶颈精确定位到**转化**：可恢复 3.43pp，但球的对应内容吃到 ~0。
- **开放分叉（待用户拍板，未自动执行）**：
  - **S2-redesign（换转化机制，非调 alpha）**：把球/可疑度信号从"软目标平滑"换成**能提交决策的形态**——ball-guided **sample selection / co-teaching-style 小损失选择 / loss correction**。直接攻击诊断出的转化失败（soft-target 不提交、selection 提交）。目标：吃到 >gb 的 0.84pp，且稳超 no-ball/shuffled。属于机制重设计，**区别于** threshold/formula shopping。
  - **S3（系统研究）**：把"高质量球为何不转化"做成论文问题（表示空间 × 球质量 × 转化失败机制）。
- **仍然否决**：继续调 `alpha/gate/top_ratio/软目标公式`（formula shopping，先验已多次失败；qsafe 已证明 NCR>0 也不转化）。
- **协议提醒**：pair / high-noise headroom（baseline 更弱、headroom 可能更大）是**单独 NoisyGL lane**，不与本 ratio-split symmetric 表混报。
