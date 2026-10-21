# Material Manifest

生成日期：2026-07-27  
工作目录：`results_ratio_c1_20260609`

## 可信级别

- `PRIMARY_RESULT`：原始或汇总实验数据，是性能数字的首要依据。
- `AUDIT`：协议、结果或稿件事实审计，用于解释和交叉核对，不能替代原始结果。
- `MANUSCRIPT`：待审稿文本，只能作为 claim 提取来源，不能自证。
- `LITERATURE_MAP`：项目文献脉络；正式引用仍需逐条核对原论文。
- `STYLE_GUIDANCE`：写作规则，不构成科学证据。

## 冻结输入

| Path | Class | Bytes | SHA-256 | Intended use |
|---|---:|---:|---|---|
| `paper_final_zh_v29_human_style_20260727.md` | MANUSCRIPT | 67700 | `BAB34F9D35A54C2CC7D3CFC58DBF61B72BA8D8C2C400FAA744C01C6C80129665` | Canonical claim source and immutable baseline |
| `experiments_results_conclusion_audit_20260727.md` | AUDIT | 5775 | `166AD1228CBFF42C756DA6036839E69BF155F57CF73ABC75C6DB2D2303141DF2` | Sections 4–7 fact and prose audit |
| `xhs_ai_paper_writing_synthesis_20260727.md` | STYLE_GUIDANCE | 4465 | `7A49B2509BAFF7A5E905CCEC6157031EDD3AD515FBDF658F2BAA0E4950811B7A` | Defensive-writing and de-AI rules |
| `01_literature_and_research_map.md` | LITERATURE_MAP | 8326 | `C3B9A4FA610FC2BAF933B302FD1B2A4FA102876E4CA02EF4F3F2C9384D52417A` | Literature orientation and claim-risk discovery |
| `latest_baselines_trainonly_20260622_continuation/formal_main_raw.csv` | PRIMARY_RESULT | 44411 | `1CFF25CD2781BF7B8AC8189D3B66CAB4CAF99F115D080CA2F3D2CDABB7F5D044` | Formal main comparison rows |
| `latest_baselines_trainonly_20260622_continuation/paired_deltas.csv` | PRIMARY_RESULT | 2119 | `D36346A5806588E8BF26C14594C00BE3CB38DC394A76781F5D1F2A3766AB9C0E` | Paired seed deltas |
| `latest_baselines_trainonly_20260622_continuation/appendix_raw.csv` | PRIMARY_RESULT | 27069 | `13C5FAB3A7AF3EE0B69E2A6C89BEB3DBCA8ABB995F3D57A97CE22417E0E6306A` | Ablation and appendix-level raw results |
| `latest_baselines_trainonly_20260622_continuation/protocol_audit_report.md` | AUDIT | 427 | `D37203B5E8E90EAEC374A230E46AE7EFC52A2A43D45BF3AE0C8A7D1D89EE11AC` | Protocol consistency verdict |
| `latest_baselines_trainonly_20260622_continuation/r2lp_and_legacy_baseline_audit_20260623.md` | AUDIT | 3677 | `0362517B13603295627788C172404D05EB46A906E358224ABBADCC902C5F0768` | R2LP and legacy baseline audit |

## 扩展实验资产

以下路径相对当前工作目录。它们补足 v29 中 11 个数据集、主统计、机制消融、稳定性和效率 claim；上面的 continuation 目录只覆盖三数据集、uniform 0.4 的部分图方法，不能独立支撑整篇稿件。

| Path | Class | Bytes | SHA-256 | Claim group |
|---|---:|---:|---|---|
| `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_selected_raw.csv` | PRIMARY_RESULT | 141091 | `A644DC63BAAAC26B374A4FDAB9C5040E77E360D74A4B9F455FD053798D9B99F4` | 五核心数据集完整噪声谱 |
| `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_strongest_formal_baselines.csv` | PRIMARY_RESULT | 3761 | `982278A8A3CCDC7748147355192D47E887EA72F7FEC3DB3A4250FEDE5EB2073D` | 25 个设置及最强基线 |
| `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_paired_deltas.csv` | PRIMARY_RESULT | 11728 | `003D6AECDDEA5BB6E8AF4CD233E37CF6A6D8DD6195AD640A94B158EE7289CA89` | 完整噪声谱配对差值 |
| `../../external/NoisyGL/runs/phase4_faithful_claims_20260712/phase4_report.md` | AUDIT | 2797 | `E82F832C18ED5088A8A69AD6F13728C6B6EED01FB926BCB6DCCAB6F3BAB5B6B9` | 完整噪声谱审计说明 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/core_main_recomputed.csv` | AUDIT | 25580 | `6D66847788A6DF0283F39ED30C7B64004F7EEE3780195253F0E932426E907D18` | 主结果重算 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/core_evidence_manifest.csv` | AUDIT | 20546 | `111CBE8C4DC376E123C2A4C20C998A31E2EFE7B09B362472B887724E676D717B` | 主结果来源追溯 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table3_recomputed.csv` | AUDIT | 3700 | `1A04E25703954CA2D20A62F34FAADF46460079E165D48535545E91658F44FEC0` | 当前 Table 3 重算 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table4_recomputed.csv` | AUDIT | 2583 | `023C006C6EA6448C200D1462761D2E5119DAE7AE956C01F2E172CDA6AB073D51` | 当前 Table 5 重算 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/table5_recomputed.csv` | AUDIT | 2951 | `CB44069404602BBA3A7FFC8111C75599917BFD3BFEB04A020F0093852F04A83D` | 当前 Table 6 重算 |
| `../../external/NoisyGL/runs/submission_integrity_v12_20260712/tables3_5_evidence_manifest.csv` | AUDIT | 7514 | `BAB7E4D724459037F2A827774FFF642D85A4EA210061F7D9A82E665D359B535E` | Table 3/5/6 原始来源追溯 |
| `../../external/NoisyGL/runs/mlp26_ffg2m_fidelity_20260712/mlp26_combined_raw.csv` | PRIMARY_RESULT | 16201 | `2151C28E95FA6AAF87284A7776879338D8708E0ACB2EA3C44299CAFC6F940168` | FF-G2M fidelity 结果 |
| `../../external/NoisyGL/runs/mlp27_graphmlp_fidelity_20260712/mlp27_combined_raw.csv` | PRIMARY_RESULT | 14250 | `AAF365EB8783C856D5D65D8B405E0318056083F66356645FE5BE17DDD92D60E5` | GraphMLP fidelity 结果 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_raw.csv` | PRIMARY_RESULT | 24051 | `42B7968222CF2340E7E761DE5FF2C9CC77AB2D969D301619BF34F4ECBA045D27` | 主要比较配对数据 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_holm.csv` | PRIMARY_RESULT | 12532 | `7AE75564FC1C9FF98EDB3CC0FCC401C8AB711FA401A0C3EAE484F4F9F30C8C9C` | Holm 校正 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/primary_statistics_report.md` | AUDIT | 463 | `09819E4AA9A2676FE0DF7E9C1541014073115B46E298D720C025FD8EFE40A96F` | 主要统计审计说明 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_raw.csv` | PRIMARY_RESULT | 152709 | `224ACF82712306EE547C78B49F047590A13DAA19E67050C86D50F813DC136AEF` | LOO、Brier、NCR/CDR |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_aggregate.csv` | PRIMARY_RESULT | 13593 | `27ABCF0D7527696F86A81895F7797EF71A5A47D911B34B3966ED8B1F2535C375` | LOO、Brier、NCR/CDR 汇总 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/mlp36_paired_deltas.csv` | PRIMARY_RESULT | 5669 | `53BB5A4037E741F256F5168E3BFDE7BE3994BA69A27A89C6AF847B742D8A6923` | LOO 配对差值 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/latency_memory_seed0.csv` | PRIMARY_RESULT | 1931 | `82677DBF5B90C319876D63EFAC7C9D166CB57FF5FBE287DE6B96A8B0F19DF018` | 下游延迟和显存 |
| `../../external/NoisyGL/runs/mlp36_review_defense_20260715/latency_memory_report.md` | AUDIT | 1106 | `B3BA0DB1C067F90968EA374770AAB74D58CEDBF9A73EB12E850BE47AB8BB38DB` | 延迟和显存审计说明 |
| `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_raw.csv` | PRIMARY_RESULT | 78488 | `654767AD176ABAB04ED01CE922EC4ECB1FA1A9D2E67DFE4CB364D5C513413674` | GRACE seed 稳定性 |
| `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_variance.csv` | PRIMARY_RESULT | 748 | `1D948D57A1C0F949117C3CAD9543E88A600AB09BCA0E820E4BFC2B665643B5D9` | H-seed 方差 |
| `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_assignment_agreement.csv` | PRIMARY_RESULT | 971 | `9B3F435D0F3DD573264485BE6E4C8851AE128292E068615656E06B311E8D1546` | 粒球分配 ARI |
| `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_grace_costs.csv` | PRIMARY_RESULT | 2811 | `91F47CF2D003DAFD81BCCAADC7CD6FE759D2B60E5D9540CDCA98184587247F5A` | GRACE 一次性预训练成本 |
| `../../external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/grace_pretrain_manifest.csv` | AUDIT | 8089 | `53DC7296319068E5211DAAEF8386591A830E0740CF643D50D1CBFF6F05167E16` | 预训练资产追溯 |
| `../../external/NoisyGL/runs/mlp23_dataset_breadth_20260710/mlp23_raw.csv` | PRIMARY_RESULT | 34222 | `62B8B144823700B72A393FACA8D5909EB40F9B19D47E04666C0B442A5DB63456` | 扩展数据集 |
| `../../external/NoisyGL/runs/mlp21_dataset_replacement_20260710/mlp21_raw.csv` | PRIMARY_RESULT | 105559 | `244A298C6CFEACB0294CDECA94B062A26182A34876FB23E2E29244A08ADC120C` | Amazon-Computers 等扩展结果 |
| `../../external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_combined_raw.csv` | PRIMARY_RESULT | 140864 | `68D5571FC44A4BA763EBF58DB468BFFABDCE572A1D0B980A187A48268DF6B823` | PubMed 与主表扩展结果 |
| `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_combined_raw.csv` | PRIMARY_RESULT | 33513 | `2EB3616156C2FA80DAC8E49C3FEE7AD1DD20D1D1B0C41370C8832F89059F1DF2` | kNN 敏感性 |
| `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_knn_curve.csv` | PRIMARY_RESULT | 2524 | `103CF50C8DFD99BE91CE2B6E44036253C4A8F5B813546FD5540705FC3619A796` | kNN 曲线汇总 |
| `../../external/NoisyGL/runs/mlp11_k_sensitivity_20260702/mlp11_report.md` | AUDIT | 4306 | `AAC35758B4225C8C0692833EF89796FB059E72AC404391181EF743EFAF44F63D` | kNN 诊断说明 |
| `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_raw.csv` | PRIMARY_RESULT | 36306 | `A4D7FEF627B10CA7BA46443E252190D4EE95975D9CC161D4E74C8C4B8ABC86AD` | 粒球最小尺寸与票数 |
| `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_summary.csv` | PRIMARY_RESULT | 3163 | `73AEF3FB3D7502702FCF5306D6941C83FF8008CC1FBD84EC0096B71EC28730E6` | 粒球票数、纯度与性能汇总 |
| `../../external/NoisyGL/runs/mlp13_ballparam_sensitivity_20260703/mlp13_stageB_paired_deltas.csv` | PRIMARY_RESULT | 7218 | `5CD16EE6E5A9D5F4AF0FC9023A89BDC5267BA9D55628037D6CA362C3E650880B` | 粒球参数配对差值 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p0c_linear_capacity_raw.csv` | PRIMARY_RESULT | 14206 | `8BC57C4E13F5885FB1D9134C55A1DAAE22E45206C6E6FC25F6EFE0D64109ECB7` | 线性分类器 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_uniform0p0_seeds_0_1_2_3_4.csv` | PRIMARY_RESULT | 23997 | `3DB39C9A963BD1486A7EF2ACA06322C0EA2CCE39E9AFABD91736386B3E8103B2` | 零噪声边界 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_aggregate.csv` | PRIMARY_RESULT | 1975 | `7D2CE758806E874F912AEDE57E9A419C59F485E6E4F8646529C25A46A5722D83` | 零噪声汇总 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p2_clean_paired_deltas.csv` | PRIMARY_RESULT | 2578 | `00A42DF450251C22C35C7B6D6C4D373B288A2030E5B9769A3ECADCA3E0D3887E` | 零噪声配对差值 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_uniform0p4_seeds_0_1_2_3_4.csv` | PRIMARY_RESULT | 5576 | `D97CA500F15DAB75DFEE593DEAFC76F0C7AC906598E151A27FEDC8BF854C990D` | 共识空间目标 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_aggregate.csv` | PRIMARY_RESULT | 1583 | `C4313048A69BD1CDA75875C040A65B0798CE847D9919855039DC7D24EA30B4D6` | 共识空间汇总 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p3_lp_target_paired_deltas.csv` | PRIMARY_RESULT | 2279 | `8DF02BE3F8FBD25C108FBE3CD72B8EF6BF0C8CA0272DB6BBD3F8FEEF4034EF6F` | 共识空间配对差值 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_aggregate.csv` | PRIMARY_RESULT | 4699 | `5661E7118A05352F4AC84A2447F2C54666D88C497E9916F0705407231921058B` | PubMed 线性结果 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_best_robust_summary.csv` | PRIMARY_RESULT | 602 | `6CC1B9B2ED44DC82CF205F7740B5F439AAC776C1988A7FA98C8BBC7BE17DEBE4` | PubMed 最强线性鲁棒基线 |
| `../../external/NoisyGL/runs/mlp15_review_defense_20260704/p4_pubmed_linear_paired_deltas.csv` | PRIMARY_RESULT | 8107 | `C8AA878FB85AAC3E53FA25B0A5BABC03BF287995111EFBC009B3700CBFF0236C` | PubMed 线性配对差值 |
| `../../external/NoisyGL/runs/mlp15_gcn_hidden_balls_20260703/mlp15_raw.csv` | PRIMARY_RESULT | 23085 | `819E3CC427F0A087AEBABF283CCBCCFFD31B8479DBF05BE70BE6B764E7919087` | GCN-hidden 粒球 |
| `../../external/NoisyGL/runs/grace_h_ablation_20260704/grace_h_ablation_uniform0p4_seeds_0_1_2_3_4.csv` | PRIMARY_RESULT | 15544 | `00972037F5D43111CF2B946ABDE97793CFE09AC5C6F70083C6EB9B1EB8FC5895` | H-access 消融 |
| `../../external/NoisyGL/runs/grace_h_ablation_20260704/grace_h_ablation_aggregate.csv` | PRIMARY_RESULT | 1894 | `1C5D4220DB99209D36AD1309B8C4E9632A6BC1456B97F0C3BF41954066DB6AC0` | H-access 汇总 |
| `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_combined_raw.csv` | PRIMARY_RESULT | 86820 | `BB02F72CF79096DF379A59DE9175DDB02ABF98F771C07EEBDBD9932E83C687A0` | 标签率与 H-access 鲁棒基线 |
| `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_aggregate.csv` | PRIMARY_RESULT | 10143 | `D73ACABE91381B446BDC7AA4E8B50B65E8505C8C2755E4D3A34070F69F573A01` | 标签率与 H-access 汇总 |
| `../../external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/mlp6_paired_deltas.csv` | PRIMARY_RESULT | 13974 | `5722CDB1CCE04E7B582580861BFB3583FBA172288EDF1E3C0DE5F523F03D929C` | 标签率配对差值 |
| `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_raw.csv` | PRIMARY_RESULT | 78211 | `DB8F14B11884340A5D27AB2DD71598B0710646BC187D26E4B7C8153A09E875A9` | 特征噪声边界 |
| `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_aggregate.csv` | PRIMARY_RESULT | 13860 | `7BADA2AB9C3CD80EB466BBD4D251C9F50A122D27FF8058E9B968A20DBE3D8B0B` | 特征噪声汇总 |
| `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_deltas.csv` | PRIMARY_RESULT | 9374 | `431D1DFBF41822C3941EB4B027B95EC15DB53A296C41F4043E0FA5ECE91A8C58` | 特征噪声配对差值 |
| `../../external/NoisyGL/runs/mlp16_cutvalue_featnoise_20260706/mlp16_feature_noise_slopes.csv` | PRIMARY_RESULT | 2261 | `DEC34DC9089ADE87591A9BDE3F4269C98D9D2D8F94D609F77BB2C8221C47329C` | 特征噪声退化斜率 |
| `../../external/NoisyGL/runs/featuregb_mlp_diagnostic_20260618/featuregb_offline_raw.csv` | PRIMARY_RESULT | 1986 | `F3B96A2F9520BB8C99F381C1F77EE0A4A988C2209BF942B341F20FFB094EBFFE` | X-space 退化诊断 |
| `../../external/NoisyGL/runs/ssl_swap_seed0_20260612/ssl_ball_diagnostics_seed0.csv` | PRIMARY_RESULT | 1292 | `80A04029874EC533529E076D8DFF5F012391DBFD4CF9684B27F31FB5061924F4` | Cora seed-0 SSL 表征粒球诊断 |
| `../../external/NoisyGL/runs/ssl_swap_seed0_20260612/ssl_swap_seed0_report_20260612.md` | AUDIT | 5488 | `5BFAF86BBFA97B73E2464C3046CC09F86BFD7A4E21CDCDE12030C3D7E363169A` | SSL 表征交换审计说明 |
| `../../external/NoisyGL/runs/ssl_probe_20260612/probe_aggregate.csv` | PRIMARY_RESULT | 1856 | `8BC680DB6C2F6841B833F2CA317D2BD268E87B52A4B6E5AC91190F3673702EAC` | SSL 线性探针汇总 |
| `../../external/NoisyGL/runs/mlp32_robust_student_distill_20260712/resource_table_raw.csv` | PRIMARY_RESULT | 46260 | `BC7C37ABA202DEB2D29B35FA371963437AF23B05EEFB3A66E8628FE27FCDB3C5` | 蒸馏及资源参照 |
| `../../external/NoisyGL/config/mlp_distill_softdist_xh/mlp_distill_softdist_xh_cora.yaml` | AUDIT | 828 | `C02C643A0D27B0D7A787F856AB8845C5BFC420156624A65973771C446CEF51AB` | 蒸馏教师 `target_beta`、输入与目标配置 |

## 使用优先级

1. 性能数值：`PRIMARY_RESULT`
2. 协议和可比性：`AUDIT`，并回查结果列
3. claim 表述：从 `MANUSCRIPT` 提取，由前两类材料验证
4. novelty 与 related work：`LITERATURE_MAP` 仅用于定位，投稿引用必须另行验证
5. 语言和结构：`STYLE_GUIDANCE`

## 已知边界

- 噪声只施加于训练节点。
- 只比较相同数据划分、训练协议、评估流程和可比 seed 的结果。
- 小于标准差尺度的差异不能描述为稳定或显著提升。
- 审计文档中的结论必须能回到原始 CSV/JSON；无法回查时标记为 `UNVERIFIED`。
- 本阶段不运行新实验，也不改变 baseline evaluation、split 或噪声协议。
- Table 0 的数据集统计尚缺单一正式 `dataset_statistics.csv`；正式定稿前应从数据加载代码或权威数据源逐项核验。
- 当前 Table 4 的 FP32 MiB 和吞吐量是由参数量、节点数和延迟派生，尚未物化为单一最终表资产。
- `submission_integrity_v12_20260712/table4_recomputed.csv` 实际对应 v29 的 Table 5，`table5_recomputed.csv` 实际对应 v29 的 Table 6。
- X-space 单球退化只在 Cora/CiteSeer、seed 0 的预注册停止诊断中验证，不得泛化成跨数据集定律。
- GRACE 稳定性实验只有 3 个 H-seed，只支持方向稳定性，不支持高精度不确定性估计。
- v29 §5.3.2 的 kmeans 结果尚未定位到可追溯实验行；找到原始资产前不得保留为已验证比较。
- v29 §5.4 的 CCA-SSG/GraphMAE 单组退化只由 Cora seed 0 支持；CiteSeer 的 GraphMAE 并未单组塌缩。
- v29 §5.4 的“带噪 CE-GRACE 52→1 球、纯度 0.79→0.30、精度下降 12.6pp”和“训练后 MLP hidden 单组”尚缺 raw/aggregate locator。
