# S3 Phase 2 — 最小 encoder-coupling 冒烟：球质量未改善 → 早停（2026-06-07）

> seed0 冒烟（Cora / symmetric 0.4 / `ratio_train0.2_val0.1` / GRACE init pretrain0）。
> **NOT planetoid, NOT NoisyGL。** 给"粒球作主角"在正确旋钮（encoder）上的最后一次机会。
> 证据 `reports/s3_phase2_coupling_20260607/ball_quality_diag.json`、`.../finetune.log`、
> `scripts/finetune_ball_coupled_encoder.py`、`scripts/diagnose_coupled_ball_quality.py`、
> `src/noisy_label_gb_grace/training/ball_coupling.py`。

## 做了什么

从 GRACE checkpoint init，**全 encoder 小 lr（1e-4）**微调 150 epoch：
`L = L_sup_noisy + λ_cl·L_GRACE + λ_gb·L_ball`（λ_cl=λ_gb=λ_sup=1 固定，**不调**）。
`L_ball` = 周期（每 10 ep，detach H 构 dcgb emb_only）+ 仅高质量球（label-free gate）+ confidence-weighted
intra（向可微 H 现算的球中心靠拢）+ 高置信不同-noisy-majority 球中心 margin inter。**无 hard same-ball
positive、无 view generation、clean label 仅诊断。** 同 init 产两份 H（唯一差别 λ_gb）：
`coupled`（λ_gb=1）vs `baseline`（λ_gb=0）。

## 中间诊断（早停门，用户定先做）

| 表示 | dcgb clean_purity | noise_auroc | balls |
|---|---|---|---|
| frozen（原始 GRACE） | 0.8893 | 0.8920 | 265 |
| baseline（sup+GRACE，λ_gb=0） | 0.8856 | 0.8738 | 264 |
| coupled（+球损失） | 0.8856 | 0.8885 | 254 |

- **coupled − frozen：purity −0.37pp、AUROC −0.35pp**（球质量略降，非升）。
- **coupled − baseline：purity ±0.00pp**（加球损失与不加，球质量完全相同）。

**→ 触发预注册早停：coupling 未改善球质量。按规则不跑完整 selection gate、不扩 seed、不调 λ。**

## 归因（诚实、清晰）

1. **不是"球质量变好但不转化"，而是"最小耦合根本没让球质量变好"。** 关键证据是 coupled ≈ baseline
   purity 完全相同——球损失对球质量**零作用**。
2. **根因：球损失被对比损失淹没。** finetune.log：`L_GRACE≈6.84` ≫ `intra≈0.02 / inter≈0.2`，球损失在
   总损失里量级太小，150 epoch 未能把表示推向"球更纯"。而护栏禁止调 λ 刷结果，故止于此。
3. 这是"**最小、不破坏 GRACE 的弱耦合**"的诚实负结果，**不**否定"强耦合"：CGBC/GBmooer 的有效性来自
   **球损失主导的训练**（CGBC `L=L_clustering+L_SCE`，球结构是一等损失；GBmooer 超球对比专为球可分），代价
   是放弃 frozen GRACE 的稳定性、接近从头训。

## 结论与方向

- frozen-space 改球（§12.1/12.5/12.7/12.8）+ 最小弱耦合（本节）均无首信号。**在"保持 GRACE 稳定"的前提
  下，粒球结构无法获得相对 kNN 共识的独立增量。**
- 保留粒球主线只剩**强耦合 / 球损失主导训练**（CGBC/GBmooer 完整形态），这是更大工程、且与项目"frozen 更稳、
  联合训练历史不稳"的经验相悖，风险显著。
- 诚实的替代主线（与全部证据一致）：**"局部一致性可靠性信号 + commit-style 选择"为方法主体，粒球为其一种
  实现 + 一条"为何 frozen 表示下球结构不转化、弱耦合不足"的负结果机制分析。**

## 纪律

预注册早停门先于 selection；λ 固定未刷；clean label 仅诊断（球损失/gate 全 label-free，已单测）；
coupled 与 baseline 同 init 同 split（唯一变量 λ_gb）；未从 seed0 宣称成功（seed0 仅决定是否扩 seed）；
ratio-split symmetric，不与 NoisyGL 混报；ball_coupling 损失经单测（可微/无 clean/空 mask 安全/打乱升 intra）。
