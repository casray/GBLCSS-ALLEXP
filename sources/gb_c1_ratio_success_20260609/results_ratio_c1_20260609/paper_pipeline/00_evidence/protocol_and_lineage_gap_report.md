# Protocol and manuscript-table lineage gap report

## 1. Scope and verdict

本次只读取既有资产，不训练、不联网、不修改任何原始结果、论文正文、`material_manifest.md`、claim ledger 或 issue traceability。输出覆盖 `paper_final_zh_v30_information_sciences.md` **Tables 1–6 的数据单元格**，并把 Table 3 的“训练资源”与 Table 6 的“推理需图”元数据单元格作为显式 GAP 记录。

- **PKG-P1-06：PARTIAL。** 1,200 条 phase4 method/seed 行均已绑定到真实选择文件、真实 source CSV、source SHA-256、精确过滤键和直接取值公式；但完整必需协议字段只闭合 300/1200 行（25.00%）。
- **PKG-P1-07 / EXP-10 的数值 lineage：PARTIAL。** Tables 1–6 的 253/253 个数据单元格（100.00%）与稿件匹配且有 path/hash/filter/formula；数值 mismatch 为 0。
- **完整逐 cell（含资源元数据）lineage：PARTIAL。** 253/270（93.70%）闭合；17 个元数据 cell 明确为 GAP；数值 mismatch 为 0。
- 这不等于 encoder-to-result 端到端复现闭合。MET-04 所列源码入口、环境锁定、全部配置映射和缺失 checkpoint 问题仍然存在。

## 2. Protocol ledger coverage

| Audit item | Closed | Target | Coverage |
|---|---:|---:|---:|
| source path/bytes/SHA-256 + exact source-row test match | 1200 | 1200 | 100.00% |
| split_hash + noise_hash（row） | 975 | 1200 | 81.25% |
| modified train/val/test counts（row） | 525 | 1200 | 43.75% |
| all required protocol fields（row） | 300 | 1200 | 25.00% |
| primary-statistics subset all required fields | 100 | 500 | 20.00% |
| split/noise hashes complete（setting） | 20 | 25 | 80.00% |
| pollution counts complete（setting） | 10 | 25 | 40.00% |
| all required fields complete（setting） | 5 | 25 | 20.00% |
| instance overall realization hash | 180 | 225 | 80.00% |
| explicit instance generator seed/hash | 0 | 225 | 0.00% |

验证结果：

- phase4 unique key `(dataset,noise_type,noise_rate,method,seed)`：1200 个，无重复。
- `primary_statistics_raw.csv` 的 500 行全部精确连接，test 值容差 `1e-12` 内一致。
- 对已有的 split/noise hash，同一 `(dataset,setting,seed)` 内未发现冲突；冲突数 0。
- `source_sha256` 是对当前真实 source CSV 重新计算，不从 manifest 抄写。

## 3. Table lineage coverage

| Manuscript table | Tracked cells | Closed | GAP | Mismatch |
|---|---:|---:|---:|---:|
| Table 1 | 45 | 45 | 0 | 0 |
| Table 2 | 30 | 30 | 0 | 0 |
| Table 3 | 54 | 45 | 9 | 0 |
| Table 4 | 63 | 63 | 0 | 0 |
| Table 5 | 30 | 30 | 0 | 0 |
| Table 6 | 48 | 40 | 8 | 0 |

表号映射已显式固化：

- external `table3_recomputed.csv` → manuscript Table 3；
- external `table4_recomputed.csv` → manuscript Table 5；
- external `table5_recomputed.csv` → manuscript Table 6；
- manuscript Table 4 直接回到 `latency_memory_seed0.csv`，FP32 MiB 与吞吐量公式已逐 cell 物化在 lineage CSV。
- `tables3_5_evidence_manifest.csv` 的 36/36 个 path/bytes/SHA-256 与当前文件重算一致，并且恰好覆盖 audited source-chain 的 36 个不同源文件。

## 4. Unclosed gaps

### GAP-PROTOCOL-01: CoraFull lacks split/noise hashes

`mlp21_raw.csv` 有 `noise_scope` 与修改计数，但没有 `split_hash`、`noise_hash`。因此 CoraFull 的 method/seed 行不能独立证明 shared realization。没有从其他方法或 setting 借值。

### GAP-PROTOCOL-02: Cora/CiteSeer/Coauthor-CS lack pollution counts in selected source

`mlp10_combined_raw.csv` 有 split/noise hash 与 `actual_train_noise_rate`，但没有 `modified_*_count`。因此这些行不因 hash 已知而被伪装成“完整协议闭合”。

### GAP-PROTOCOL-03: instance generator provenance

已有 `noise_hash` 时将其原样复制为 `instance_realization_hash`；没有把 downstream seed 推断成 generator seed。所有 instance 行的显式 `instance_generator_seed/hash` 仍为 GAP；CoraFull 连 overall realization hash 也缺失。

### GAP-FREEZE-01: freeze boundary remains non-temporal

现有 beta 资产真实记录候选值/验证结果与“beta=0.5 作为 conservative fixed default”的决策：

- `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p0a_beta_validation_audit.csv`；
- `../../external/NoisyGL/runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_decision.csv`；
- `../../external/NoisyGL/runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_report.md`。

但仍缺：冻结时间或 commit、Cora/CiteSeer 如何合并为唯一选择规则、`B_max/n_min/delta/s_min` 的完整候选网格与选择记录、哪些 encoder 参数允许 dataset-specific 的正式表。稿件不能据此把“全部构造参数冻结”升级为已审计事实。

### GAP-TABLE-01: nonnumeric resource metadata

Table 3 的 9 个“训练资源”cell 和 Table 6 的 8 个“推理需图”cell 没有逐 cell 配置/资源 manifest。lineage CSV 逐一写为 `GAP`，未用论文文本自证论文文本。

### GAP-TABLE-02: Table 4 environment metadata

Table 4 的数值 cell 均有直接 raw/formula；FP32 MiB 使用 `parameter_count*4/1048576`，吞吐量使用节点数除以 full-graph latency。raw 明确 seed 0、`inference_repeats=100`、一个 ROCm GPU；现有 latency raw/report 未给 exact GPU model、ROCm、PyTorch、PyG 版本或同步/warm-up 细节，因此跨环境复现实验仍未闭合。

### GAP-TABLE-03: breadth-source protocol columns

`mlp23_raw.csv` 的四个扩展数据集数值可按 dataset/method/seed 精确聚合，但文件自身没有 `noise_type/noise_rate` 字段。对应 cell 的数值来源可追踪；“该 raw 行独立声明 uniform 0.4”仍是协议 GAP。

### GAP-REPRO-01: end-to-end reproduction

本次没有创建不存在的源码入口、环境锁文件、全数据集 config-ID 映射或 checkpoint。现有资产支持 seed-row/aggregate replay，不自动支持从原始图重建全部 H 并复跑 Tables 1–6。

## 5. Existing 75-test Holm sensitivity

基于 `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_holm.csv` 中现有 75 个 `paired_t_p`，按单一 75-test Holm 家族机械重算：

- 全局显著：42/75；
- LCSS vs CE：19/25（原分家族为 24/25）；
- LCSS vs JoCoR：6/25（原分家族为 8/25）；
- LCSS vs shuffled：17/25（原分家族为 25/25）。

公式：`sort all 75 paired_t_p ascending; raw Holm=(75-rank+1)*p; adjusted=max(previous,raw Holm), clipped at 1`。这是既有 p 值的敏感性重算，不是新实验，也不把三个原作者定义族伪装成外部预注册。

## 6. Key audited inputs

| Path | Bytes | SHA-256 |
|---|---:|---|
| `paper_final_zh_v30_information_sciences.md` | 74746 | `CA9048651A1E3E400EBFA670F62DAACAAE4EC0861F18609CC26CA2DECBF0E86B` |
| `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_selected_raw.csv` | 141091 | `A644DC63BAAAC26B374A4FDAB9C5040E77E360D74A4B9F455FD053798D9B99F4` |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_raw.csv` | 24051 | `42B7968222CF2340E7E761DE5FF2C9CC77AB2D969D301619BF34F4ECBA045D27` |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_holm.csv` | 12532 | `7AE75564FC1C9FF98EDB3CC0FCC401C8AB711FA401A0C3EAE484F4F9F30C8C9C` |
| `latest_baselines_trainonly_20260622_continuation/formal_main_raw.csv` | 44411 | `1CFF25CD2781BF7B8AC8189D3B66CAB4CAF99F115D080CA2F3D2CDABB7F5D044` |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv` | 7514 | `BAB7E4D724459037F2A827774FFF642D85A4EA210061F7D9A82E665D359B535E` |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/latency_memory_seed0.csv` | 1931 | `82677DBF5B90C319876D63EFAC7C9D166CB57FF5FBE287DE6B96A8B0F19DF018` |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p0a_beta_validation_audit.csv` | 1737 | `6198DC4A53D610FE8AA6385C342D0C779A78B183390BDDAF662F9494C9FB8C2C` |
| `../../external/NoisyGL/runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_decision.csv` | 1841 | `D4240F02DC64386A365832F8CE3B252CC837FA48FCDAD30CC1D206E2C6DD1DFA` |

## 7. Generated assets

| Path | Bytes | SHA-256 |
|---|---:|---|
| `paper_pipeline/00_evidence/protocol_ledger.csv` | 1025817 | `73CEC921C092EC398FC13EC4E1C4A85A43BC3792298F702930222E1524ADCB90` |
| `paper_pipeline/00_evidence/manuscript_table_lineage.csv` | 251937 | `39A809405F6C8386E688A55EB9919246F10E4943BE86D08FB6D834DA510474A7` |

生成/复核命令：

```powershell
py paper_pipeline/scripts/generate_protocol_and_lineage.py --manuscript paper_final_zh_v30_information_sciences.md
py paper_pipeline/scripts/generate_protocol_and_lineage.py --manuscript paper_final_zh_v30_information_sciences.md --verify-only
```
