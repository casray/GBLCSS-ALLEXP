# S3 Phase 1 — 现象稳健性(commit-style ball-guided selection, Cora 0.4 × 5 seed)

> **多 seed smoke**(Cora / `ratio_train0.2_val0.1` / split0 / symmetric 0.4 / seeds 0–4,每 seed s
> 用 pretrain{s}/noise{s}/clf{s} / frozen GRACE)。**NOT planetoid, NOT NoisyGL。** 只验现象、不解释、
> 不扩表示轴(按用户 2026-06-07 拍板)。证据 `reports/s3_phase1_20260607/s3_phase1_rows.csv`、
> `.../s3_phase1_summary.json`、`scripts/run_s3_phase1.py`。固定 schedule W20/ramp20/ρ=0.6,所有信号同
> schedule,仅 rel 来源不同。

## per-seed test acc

| seed | ce | gb_soft | clean_oracle | **sel_dcgb_emb** | sel_gb | sel_topk | sel_noball | sel_shuffled | sel_random |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7878 | 0.7962 | 0.8126 | 0.7788 | 0.8094 | 0.7930 | 0.6642 | 0.6705 | 0.6774 |
| 1 | 0.7973 | 0.7962 | 0.8105 | 0.8073 | 0.7899 | 0.8010 | 0.6874 | 0.7122 | 0.7165 |
| 2 | 0.7925 | 0.7962 | 0.8157 | 0.8115 | 0.8031 | 0.7988 | 0.7371 | 0.7307 | 0.6990 |
| 3 | 0.8068 | 0.8073 | 0.8099 | 0.7978 | 0.7239 | 0.7941 | 0.7291 | 0.7181 | 0.7614 |
| 4 | 0.7830 | 0.7909 | 0.8226 | 0.8057 | 0.7249 | 0.7978 | 0.7233 | 0.7239 | 0.7254 |
| **mean** | 0.7935 | 0.7974 | 0.8143 | **0.8002** | 0.7702 | 0.7969 | 0.7082 | 0.7111 | 0.7159 |

## 预注册判定(机械)

| 现象 | 标准 | 结果 | 判定 |
|---|---|---|---|
| **A** `sel_gb > sel_dcgb_emb` | ≥4/5 且 mean Δ≥1.0pp | wins **1/5**,mean Δ **−3.00pp** | **FAILS** |
| **B** `sel_gb > gb_soft` | ≥4/5 且 mean Δ≥0.5pp | wins **2/5**,mean Δ **−2.71pp** | **FAILS** |
| **C** `sel_dcgb_emb > shuffled / random / noball` | 各 ≥4/5 | **5/5 / 5/5 / 5/5**(mean +8.9 / +8.4 / +9.2pp) | **HOLDS** |

## 结论(只陈述现象,不解释)

1. **A 不成立 → seed0 的 `sel_gb > sel_dcgb` 是 seed 噪声。** 跨 seed 反转:`sel_dcgb_emb` 0.8002 >
   `sel_topk` 0.7969 > `sel_gb` 0.7702。`sel_gb` 在 seed3/4 崩到 0.72。**不能围绕 `sel_gb > sel_dcgb`
   写论文;`sel_gb` 不重新预注册为方法。**
2. **B 不成立 → `sel_gb` 不稳超 gb_soft**(2/5,mean −2.71pp)。先前"sel_gb 吃到 63% headroom"是单格。
3. **C 成立(5/5 满分)→ commit-style selection + reliable 球本身是稳健正信号。** 破坏球结构
   (shuffled/random)purity 掉到 ~0.51、test 掉到 ~0.71;loss-only(noball)0.708。球结构对 selection
   有真实因果贡献。

## 球质量 vs 转化(跨 seed 均值;先记录、Phase 2 再解释)

| method | test | sel_precision | noise_excl_recall | clean_purity | rel_noise_auroc |
|---|---|---|---|---|---|
| **sel_dcgb_emb** | 0.8002 | 0.9040 | 0.8562 | 0.8919 | 0.8463 |
| sel_gb | 0.7702 | 0.8002 | 0.5889 | 0.8934 | 0.7934 |
| sel_topk_matched | 0.7969 | 0.9225 | 0.8839 | – (无显式球) | 0.9022 |
| sel_shuffled_matched | 0.7111 | 0.6753 | 0.2710 | 0.5107 | 0.5261 |
| sel_random_matched | 0.7159 | 0.7632 | 0.5779 | 0.5066 | 0.5153 |
| sel_noball | 0.7082 | 0.6886 | 0.3207 | – | – |
| sel_clean_oracle (诊断) | 0.8143 | 1.0000 | 1.0000 | – | – |

**关键观察(留给 Phase 2)**:dcgb 与 gb 的 group-purity 几乎相同(0.892 vs 0.893),但 dcgb 的
**node-level selection precision 0.904 ≫ gb 0.800**、noise-excl 0.856 ≫ 0.589。**group-purity 不区分
两者,node-level 选择质量才区分** —— 这正是"球质量指标与转化错位"的具体落点。另:`sel_topk`(kNN 共识,
不显式建球)rel-AUROC 0.902 最高、sel-precision 0.923 最高、test 0.797 接近 dcgb —— Phase 2 应查
"显式建球 vs 纯 kNN 共识"差异来自哪。

## 决策

- 按预注册:A/B 不成立 → **不把 `sel_gb` 当方法,不围绕单格倒挂写结论**。
- C 成立 → commit-style selection + reliable 球稳健,**S2/S3 不收缩**。
- 下一步(待用户定):Phase 2 机制研究锚定**"为什么 group-purity 相同而 node-level selection 质量不同"**
  + "建球 vs kNN 共识"——而不是 seed0 那个已被证伪的 `sel_gb` 倒挂。

## 纪律

预注册多 seed gate 机械判定;固定 schedule 全信号一致(无调参);clean label 仅 eval/诊断;ratio-split
symmetric,未与 NoisyGL 混报;未从单格写结论;`sel_gb` 未被宣称为方法。
