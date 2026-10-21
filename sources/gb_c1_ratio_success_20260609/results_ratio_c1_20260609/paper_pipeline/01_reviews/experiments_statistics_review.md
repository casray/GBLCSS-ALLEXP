# Experiments and Statistics Review

## Review scope

本审查按 Elsevier *Information Sciences* 计算机实验论文的实验设计、统计有效性、公平比较与可复现性标准执行。审查严格限定于 `paper_final_zh_v29_human_style_20260727.md`、`paper_pipeline/00_evidence/claim_evidence_ledger.md`、`paper_pipeline/00_evidence/material_manifest.md`，并仅沿 ledger 指定路径抽查 raw / aggregate / audit。未联网、未运行新实验、未读取其他 reviewer 报告、未修改稿件。

## Overall recommendation

**Major Revision（在当前版本不建议接收）**。

核心性能结果不是主要问题：五个主数据集的 25 个设置、七个正式同身份基线、24/25 与 14/15 的均值排名，以及三组预设比较的平均差和 Holm 计数均能从现有聚合资产复算。当前没有会在最小文字或资产修复后仍使核心主结果失效的 P0；主要修订阻断项均为 P1 claim boundary 与证据链问题：稿件一方面声称过程“不依赖干净标签”，另一方面明确使用干净验证标签选模；另一方面，§5.4 仍保留 ledger 已判定为 `PARTIAL` 且缺少 raw locator 的 noisy-CE GRACE、MLP-hidden 与跨表征泛化表述。另有 pair/instance 主结果缺少最终逐行 split/noise hash 审计、统计族定义偏乐观、代理实现与跨部署资源比较容易被误读等问题。

这些问题的最低可接受修复大多不要求重跑实验：删除或收缩不受支持的 claim，补齐现有日志导出的审计表，重新组织表格和统计解释即可。若作者坚持“完全不使用任何干净标签”“GRACE 普遍优于其他表征”或“与图模型等效”等更强主张，则必须另行设计实验；本报告不建议保留这些主张。

## Confidence

**High（0.93）**。三份核心文档已完整阅读；14 个关键冻结资产的 SHA-256 与 manifest 全部一致；主结果、配对统计、LOO、kNN/粒球、H-access、clean/feature-noise、H-seed、效率与蒸馏表均做了定向抽查。置信度未设为满分，是因为部分最终表只提供汇总重算表或资产级 hash 清单，尚缺逐表格单元的来源映射；本审查也未被授权核查代码执行路径或运行新实验。

## Strengths

1. **主结果的数值主干可复核。** `phase4_strongest_formal_baselines.csv` 有 25 个完整设置、每个设置 5 个配对 seed；GB-LCSS 的均值胜出计数为 24/25，三个非开发数据集为 14/15，唯一均值落后是 Coauthor-CS uniform 0.2（−0.25 pp）。这与 v29 的限定性表述一致。
2. **七个正式同身份基线与补充代理基本分开。** 25-setting 正式集合稳定包含 CE、Small-loss、Co-teaching、JoCoR、SCE、ELR、LogitClip；ANL-CE、GMM+CT、FINE 只在 uniform 0.4 补充，不改变最强对照。KRD/VQGraph 被标为 inspired proxy，稿件没有把它们计入 24/25。
3. **主要统计数字复算一致。** `primary_statistics_raw.csv` 的四种方法均有 25 settings × 5 seeds；相对 CE、JoCoR、shuffled 的跨 setting 平均差分别为 +7.16、+2.87、+7.83 pp，按各自 25-test Holm 族得到 24、8、25 个校正后显著设置，且至少 4/5 seed 同向的设置数为 24、23、25。
4. **负结果与资源边界披露优于常见稿件。** Table 2 明确给出 PubMed、Amazon-Computers、Attributed-Wiki 等数据集上的落后；clean-label 下 Coauthor-CS/Amazon-Photo 为负增益；feature-noise 下 Cora 的绝对退化大于 CE；H-access 被明确列为更强推理资源；Table 4 也明确限定为 seed 0、单一 ROCm 硬件。
5. **机制对照具有实际信息量。** shuffled grouping 在 25/25 settings 上低于真实分组；LOO、两种显式平滑、Brier/NCR/CDR、H-seed 与蒸馏教师分解共同说明“其他球内训练节点提供额外监督信息”是有数据支撑的窄结论。C-008 的逐数据集最强图依赖参照也核对正确：Cora/CiteSeer/CoraFull 为 NRGNN，Coauthor-CS 为 NRGNN[X‖H]，Amazon-Photo 为 GCN[X‖H]。

## Top rejection blockers

当前没有 P0。以下是若不修复会阻断接收、但可通过收缩 claim、补齐资产或重算现有统计解决的主要 P1 问题。

1. **监督资源定义自相矛盾。** 方法叙述称“不依赖干净标签”，实验却用干净验证标签选择 checkpoint。这不是措辞小问题，而是任务资源边界发生变化。
2. **C-025 的缺失资产 claim 仍留在正文。** noisy-CE GRACE 的 52→1 球、0.79→0.30 纯度、−12.6 pp，以及 MLP-hidden 单球退化没有 ledger 可定位的 raw/aggregate；CCA-SSG/GraphMAE 也只支持 Cora seed 0 的窄诊断。
3. **C-024 仍混入未追溯 kmeans、测试集挑选的 kNN “最优”与跨数据集票数泛化。** 现有证据只能支持若干数据集特定、探索性的“与票数解释相容”。
4. **完整 25-setting 的配对协议尚缺最终逐行 hash 审计。** phase4 selected raw、mlp21/mlp23 扩展 raw 没有同时携带 split_hash 与 noise_hash，pair/instance 的共同 realization 不能由最终冻结表独立复核。
5. **显著性叙述依赖 n=5 配对 t 检验和三个分别校正的 25-test 族。** 数值计算本身正确，但“预声明”没有可审计的预注册时间戳，且 75 个检验的整体 FWER 未被控制。

## Detailed issues

### EXP-01
- Severity: P1
- Manuscript location: §1 方法概述（v29 L31–35）、§3.4.2（L214–216）、§4.2（L377–381）、§4.3（L383–387）
- Problem: 稿件称“整个过程不依赖干净标签”“不额外使用干净标签”，但实验协议明确保持验证标签干净，并用验证集性能选择 checkpoint。干净验证标签参与模型选择，属于训练流程的监督资源；因此只能声称软目标构造不读取 clean labels，不能声称完整训练/选模过程完全不依赖 clean labels。L216 还将 clean labels 的用途写成仅“实验后的诊断与最终评测”，遗漏验证选模。
- Evidence: `protocol_audit_report.md` 明确写明 “Validation: clean labels, used for model/config selection”；v29 L381、L385 也明确写明验证集用于模型选择。C-002/C-021 的 narrow claim 实际只允许“clean validation/test 不作为训练目标”，并不支持“完全不使用 clean labels”。
- Rejection impact: 该矛盾改变了论文任务定义和可部署性：有干净验证集的 noisy-train setting 与完全无干净标签 setting 不是同一个问题。若不修复，摘要、方法和实验协议无法同时成立。
- Minimal fix: 将所有“不依赖/不额外使用干净标签”改为“软目标构造与损失不使用干净训练标签；实验使用干净验证标签进行 checkpoint 选择”；在问题设定、算法和实验设置中统一写清监督资源。若作者要保留“完全不使用任何干净标签”，则需另设无需 clean validation 的选模协议，但最低可接受修复是收缩 claim。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-02
- Severity: P1
- Manuscript location: §5.4“冻结无监督对比表征的经验选择”（v29 L514–518）
- Problem: 正文仍保留 C-025 已判定为 `PARTIAL` 的扩展叙述：noisy-CE GRACE 的“52 个球塌缩为 1 个、纯度 0.79→0.30、精度下降 12.6 pp”及“训练后 MLP 隐层退化为单一分组”没有可定位 raw/aggregate。CCA-SSG/GraphMAE 的单球诊断也只能限定为 Cora、encoder seed 0；CiteSeer 的 GraphMAE 实际形成 64 个球而非单球。
- Evidence: `ssl_ball_diagnostics_seed0.csv` 只直接支持 Cora seed 0：GRACE 52 球、纯度 0.7873；CCA-SSG/GraphMAE 各 1 球、纯度 0.3021。`probe_aggregate.csv` 支持 Cora 上 CCA-SSG/GraphMAE 的 clean/noisy probe 高于 GRACE。相同诊断中 CiteSeer GraphMAE 为 64 球、纯度 0.5266。ledger C-025 与 manifest L115–116 均明确指出 noisy-CE GRACE 和 MLP-hidden locator 缺失。
- Rejection impact: 这是正文中的可证伪数值和对象级 claim，而不是讨论性猜测。保留无法追溯的实验数字会直接损害结果可信度；将 Cora seed-0 诊断泛化成表征方法的一般排序也不成立。
- Minimal fix: 删除 noisy-CE GRACE 的 52→1、0.79→0.30、−12.6 pp 与 MLP-hidden 单球句；将剩余结论改为“在 Cora encoder seed 0、相同构球参数下，CCA-SSG/GraphMAE 虽有更高线性探针精度，却各形成单球，因此线性可分性不保证当前参数下的局部可聚类性”。不得恢复缺失资产 claim。
- New experiment required: NO
- Claim decision: DELETE

### EXP-03
- Severity: P1
- Manuscript location: §5.3.2“分组形式与有效票数”（v29 L477–481）
- Problem: 该段把四类不同证据拼成近似因果结论。首先，kmeans 没有可追溯 raw，不能保留“kmeans 与粒球相近”。其次，“默认球平均约 10 个训练节点”和“纯度 0.505→0.477、精度提高”是 Cora/CiteSeer 或 Cora 特定数字，不是五数据集通用事实；Coauthor-CS、PubMed 默认平均票数实际约 57.3、61.6。再次，“四个数据集最优 k 均为 20”是按测试均值事后选择；按验证均值，CiteSeer 的最佳 k 为 50，PubMed 为 100。最后，票数、球数、最小球规模与 split-improvement 同时变化，现有消融不能识别“票数比纯度更关键”的唯一因果路径。
- Evidence: `mlp11_knn_curve.csv` 的测试均值峰值均在 k=20，但验证峰值为 Cora 20、Coauthor-CS 20、CiteSeer 50、PubMed 100。`mlp13_stageB_summary.csv` 中 Cora 默认→大球为 52.4→22 球、10.325→24.611 票、0.505→0.477 纯度、55.59→56.56%；CiteSeer 默认约 10.42 票，Coauthor-CS/PubMed 默认则远高于 10。ledger C-024 明确要求保持 `PARTIAL` 并删除 kmeans 实证 claim。
- Rejection impact: 该问题不推翻主性能表，但削弱论文对“为何有效”的核心解释，并引入使用测试集选择超参数的印象。
- Minimal fix: 删除 kmeans 比较；每个数字显式标注数据集；把“最优 k”改成“所测网格中的测试曲线峰值（探索性诊断）”，或用现有 validation 列选择 k 后报告对应 test；把因果句改为“这些受限观察与有效票数不足会限制多数统计的解释相容”，不得写成已证明的主导机制。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-04
- Severity: P1
- Manuscript location: §4.2–4.4（v29 L377–395）、§5.1 pair/instance 解释（L419–423）、附录 A 的协议审计承诺（L726）
- Problem: 直接抽查支持 train-only 和共同 realization 的 `formal_main_raw.csv` 只覆盖 Cora/CiteSeer/Coauthor-CS 的 uniform 0.4 图方法。完整 25-setting 的 `phase4_selected_raw.csv` 只有 `dataset,noise_type,noise_rate,method,seed,test,source`，没有 noise_scope、split_hash、noise_hash 或 modified_val/test；`mlp23_raw.csv` 没有 noise_scope/split_hash/noise_hash，`mlp21_raw.csv` 虽有 noise_scope 和修改计数，也没有 split/noise hash。因此，pair/instance 以及 CoraFull/Amazon-Photo/扩展数据集的“同一 seed 共享同一 split/noise realization”不能从最终冻结主表逐行独立审计。
- Evidence: `formal_main_raw.csv` 的 135 行均为 `noise_scope=train`、modified_val/test=0，且每个 dataset–method 有 seed 0–4、同 setting–seed 的 split_hash/noise_hash 唯一；但其数据集仅为 Cora/CiteSeer/Coauthor-CS、噪声仅 uniform 0.4。phase4、mlp21、mlp23 的冻结 CSV 表头如上，缺少至少一组关键审计列。instance 噪声又由 X、类条件随机映射 \(W_y\) 和截断正态 \(q_v\) 共同生成，尤其需要记录生成 seed/hash。
- Rejection impact: 配对检验、公平排名和 pair/instance 的数据依赖解释都以共同 realization 为前提。最终证据包不能逐行证明这一前提，会成为复现与审稿质疑的集中点。
- Minimal fix: 从现有日志导出一张不可变的 protocol matrix，逐行包含 `dataset, noise_type, noise_rate, method, seed, split_hash, noise_hash, noise_scope, modified_train/val/test_count, actual_train_noise_rate, source_hash`；instance 另记录 \(W_y\)/noise generator seed 或整体 realization hash。该工作是资产补齐，不需要重跑训练。
- New experiment required: NO
- Claim decision: TEST

### EXP-05
- Severity: P1
- Manuscript location: §4.4“统计报告与复现信息”（v29 L389–393）、§5.1（L419–423）、附录 B（L727）
- Problem: 配对 t 检验和 Holm 数值计算正确，但统计解释仍偏强。每个 setting 只有 n=5，t 检验对配对差近似正态和异常值高度敏感；三个 comparator 分别构成 25-test Holm 族，未控制 75 个确认性检验的整体 FWER。“预声明主比较族”也没有在 ledger/manifest 中对应一个带时间戳、先于结果的注册记录。因此，24/8/25 应解释为作者定义的三个族内校正结果，而不是全论文层面的统一显著性保证。
- Evidence: `primary_statistics_raw.csv` 有 500 行且所有 method–setting 单元均为 seed 0–4；`primary_statistics_holm.csv` 复算得到 +7.16/+2.87/+7.83 pp 与 24/8/25。最小 raw p 值约 \(8.35\times10^{-8}\)，Holm p 从不小于 raw p，算法实现无明显错误。问题在检验族定义和 n=5 推断强度，而非算术错误。LOO 结果也显示未校正 p<0.05 的 Cora/CiteSeer 在六 setting Holm 后均不显著，说明小 n 结论会随族定义明显改变。
- Rejection impact: 若把族内 p 值包装成“方法普遍显著优于基线”，会高估统计证据，尤其是 JoCoR 仅 8/25 显著。数值排名仍可保留，但显著性是投稿级主要方法 claim，必须更保守。
- Minimal fix: 明确说明三个族是作者定义、非外部预注册；将“显著提升”限定到具体校正族和 setting；提供 75-test 单一 Holm（或层级/跨 setting 模型）作为敏感性分析，并继续同时报告配对差、95% CI 与方向计数。所有操作均可用现有 seed-level 数据完成。
- New experiment required: NO
- Claim decision: TEST

### EXP-06
- Severity: P1
- Manuscript location: §5.2 Table 3（v29 L438–458）、§6 Table 5–6（L543–580）
- Problem: Table 3 将 protocol port、common-backbone reimplementation、mechanism adaptation、inspired proxy 与不同训练资源的方法放在同一视觉排名中；尽管文字做了说明，统一粗体仍容易把 VQGraph/KRD proxy 当成正式复现结果。Table 6 又把 X-only 学生与推理时依赖图的 GCN/NRGNN/本地适配方法并列，且 TFR/ProCon 在若干数据集出现极端退化。该表可以作为资源差距审计，不能作为同资源公平排名，也不能由均值接近推出统计等效。
- Evidence: Table 3 的 VQGraph-inspired proxy 在 Amazon-Photo 90.54 为全表粗体，而正式 graph-free 组最高是 GB-LCSS 蒸馏 90.49；C-006 明确禁止把 proxy 当原方法官方结果。Table 6 的逐数据集最佳图参照核对为：Cora NRGNN 81.76、CiteSeer NRGNN 71.03、CoraFull NRGNN 62.70、Coauthor-CS NRGNN[X‖H] 93.72、Amazon-Photo GCN[X‖H] 91.87。学生对应差为 −1.16、−1.63、按全精度 −4.47（用显示到两位的小数相减为 −4.48）、+0.03、−1.38 pp。没有 non-inferiority/equivalence 检验。
- Rejection impact: 比较资源和实现身份不清会被视为不公平 baseline presentation。尤其“达到 GCN 水平”“具有竞争力”若不附资源与非等效限定，容易被读成跨资源胜出或统计等效。
- Minimal fix: 将 Table 3 分成 identity-preserving/ported、mechanism proxy 两个视觉块，只在可比的正式块内标粗；Table 6 标题和正文统一使用“跨部署资源差距”，按 C-008 每个数据集的最强图参照报告差值，不使用等效或全面优于语言；最终差值统一由全精度单元计算并在表注说明四舍五入。若不做新的等效检验，只保留描述性“数值接近”。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-07
- Severity: P1
- Manuscript location: §5.3.6“标签率与输入扰动”（v29 L508–512）
- Problem: “标签率从 65% 降至 5% 时，GB-LCSS 相对最强样本选择方法的优势由 0.2–1.4 pp 扩大到 5.6–6.7 pp”中的 6.7 与“最强方法”口径不一致。5% 时 Cora/CiteSeer 的最强 tested selector 均为 Small-loss，对应增量是 5.61/5.80 pp；6.68 pp 是 Cora 相对 JoCoR 的差，而 JoCoR 在该点不是最强均值。
- Evidence: `mlp6_aggregate.csv` 在 5% 标签率下：Cora GB-LCSS 43.70、Small-loss 38.09、JoCoR 37.02；CiteSeer GB-LCSS 43.34、Small-loss 37.54、JoCoR 37.23。65% 下相对最强者的差为 Cora 0.18、CiteSeer 1.35 pp。`mlp6_paired_deltas.csv` 也给出 Cora 5% 相对 Small-loss 5.615 pp、相对 JoCoR 6.684 pp。
- Rejection impact: 这不会改变“低标签率下总体优势扩大”的方向，但属于正文精确数字与比较对象错配，审稿人复算后会质疑结果筛选口径。
- Minimal fix: 将范围改为“约 0.2–1.4 pp 扩大到约 5.6–5.8 pp”，并明确每个标签率均在同一候选集合中按验证/预定规则确定比较对象；不要在不同点切换成较弱基线以扩大差值。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-08
- Severity: P1
- Manuscript location: 摘要（v29 L17–21）、贡献 3（L39–43）、§4.1（L357–360）、§5.1（L401–436）、结论（L584–590）
- Problem: 稿件正文虽然在 §4.1 给出主/扩展范围，但摘要和贡献中的“11 个数据集和多种噪声协议”仍可能被读成 11 个数据集都完成完整噪声谱。实际证据结构是五个主数据集 × 五种噪声设置，另六个扩展数据集只以 uniform 0.4 为统一宽度审计；PubMed虽另有五噪声谱边界分析，但不进入 24/25。另一个表述风险是“参数冻结后的三个数据集”：14/15 的数值可复算，但现有 manifest 没有先于结果的冻结时间戳/commit，不能独立证明开发—确认的时间顺序。
- Evidence: `phase4_selected_raw.csv` 只含 Cora/CiteSeer/CoraFull/Coauthor-CS/Amazon-Photo；mlp23 覆盖 Cora_ML/Attributed-Wiki/CiteSeerFull/Coauthor-Physics，mlp21 覆盖 Amazon-Computers；Table 2 的六数据集均只在 uniform 0.4 汇总。24/25 和 14/15 在正式候选集内成立，但 C-003/C-005 明确禁止外推到 11 数据集普遍 SOTA。
- Rejection impact: 数据集覆盖是论文主要贡献之一。混淆 full-spectrum confirmation 与 breadth audit 会夸大外部有效性；无法审计的“冻结后”措辞还可能被质疑为事后定义 held-out datasets。
- Minimal fix: 在摘要、贡献和结论统一写成“5 个主数据集的完整五噪声谱 + 6 个扩展数据集的 uniform 0.4 宽度审计”；24/25、14/15 后始终附候选集和数据集集合。若无带时间戳的预先冻结记录，将“参数冻结后”改为“使用同一固定配置评测的三个额外主数据集”。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-09
- Severity: P2
- Manuscript location: §5.4 GRACE seed 稳定性（v29 L517–518）、§4.4（L391–395）
- Problem: “预训练随机性没有改变上述结论”“对具体成员划分并不敏感”超过 n=3 H-seed 的证据分辨率。实验只覆盖 Cora/CiteSeer/CoraFull、uniform 0.4、3 个 H-seed × 3 个下游 seed；0.53/0.07/0.44 pp 是三个 H-seed 均值的样本标准差，不能作为高精度不确定性估计。低 ARI 只说明离散分区不同，不证明所有局部统计或所有预训练配置都稳定。
- Evidence: `mlp37_variance.csv` 给出 h-seed std 0.5286/0.0716/0.4367 pp，三数据集的最小 ours−shuffled/CE/JoCoR 均为正；`mlp37_assignment_agreement.csv` 的 ARI 范围约 0.25–0.36。`mlp37_raw.csv` 对每个 ours/shuffled 单元仅有 H-seed 0–2、downstream seed 0–2。manifest L113 已明确“只有 3 个 H-seed，只支持方向稳定性”。
- Rejection impact: 这是机制稳定性表述的精度问题，不推翻主结果；但过度外推会削弱复现可信度。
- Minimal fix: 改为“在三个数据集、三个已测 H-seed 的 uniform 0.4 诊断中，增益方向保持为正”；将 ARI 解释限定为“性能比具体离散分区更稳定的观察”，删除“预训练随机性没有改变结论”和一般性的“不敏感”语言。
- New experiment required: NO
- Claim decision: WEAKEN

### EXP-10
- Severity: P1
- Manuscript location: Table 1–6 的数据来源、§8“数据、代码与结果资产”（v29 L594–596）、附录索引（L722–734）
- Problem: 数值表大多能追到冻结汇总，但最终 table lineage 仍不是逐单元可复现的。`tables3_5_evidence_manifest.csv` 只是 input path/hash 清单，不含 manuscript table–dataset–method–source row 映射；外部 `table4_recomputed.csv` 实际对应稿件 Table 5，`table5_recomputed.csv` 对应稿件 Table 6。Table 4 的 FP32 MiB 与吞吐量还是派生列，manifest 明确说明尚未物化为单一最终表资产，计时环境和派生公式也没有在一张最终表资产中闭合。
- Evidence: 抽查的关键文件 hash 与 material manifest 匹配，说明资产未漂移；问题是 lineage 粒度。当前 C-005 已由 `../../external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_combined_raw.csv` 闭合 PubMed 的 claim-level locator：过滤 `dataset=pubmed, noise_type=uniform, noise_rate=0.4` 并按 `method` 聚合 `test_acc`。仍未闭合的是逐 manuscript cell 来源；manifest L109–111 明确记录 Table 4 派生列及 external/manuscript table 编号错位。
- Rejection impact: 审稿阶段人工核对和 claim-level locator 不等于投稿补充材料可逐单元、一键复现。表号错位、逐单元映射缺失和未物化的派生列会使读者难以确定最终表格每个 cell 的 raw 来源、过滤条件和计算公式。
- Minimal fix: 生成并冻结一张 `manuscript_table_lineage.csv`，字段至少包含 `manuscript_table_id, external_table_id, row_label, column_label, dataset, noise, method, n, mean, std, raw_path, raw_sha256, filter_expression, aggregation_formula`；显式记录 external Table 4/5 到 manuscript Table 5/6 的映射；为 Table 4 物化参数 MiB、吞吐量及公式，并记录 seed 0、100 次 full-batch、硬件、ROCm/PyTorch 版本和计时口径。无需重跑训练。
- New experiment required: NO
- Claim decision: TEST

### EXP-11
- Severity: P2
- Manuscript location: §5.3.1“局部共识与标签平滑”（v29 L462–475）
- Problem: LOO 的数值和 Holm 结论可复核，但 Brier、NCR、CDR 在正文没有正式定义计算对象、分母和 clean-label 使用边界；“其他球内训练节点确实提供了独立监督信息”中的“独立”也不准确，因为同一球内节点及其表征具有统计依赖，LOO 只排除了节点自身投票。
- Evidence: `mlp36_aggregate.csv` 支持 Cora Brier 0.4876→0.4503、NCR 0.6850→0.7107、CDR 0.2151→0.3020；`mlp36_paired_deltas.csv` 支持六数据集的 LOO−current 均未通过六检验 Holm。C-009 允许的窄解释是“球内其他成员含有非随机监督信息”及 NCR/CDR 权衡，不允许把相关变化当成唯一因果机制。
- Rejection impact: 不定义诊断指标会妨碍复现，也可能让读者误以为 clean labels 被用于训练；“独立监督”则把 LOO 控制的作用说得过强。
- Minimal fix: 在方法或附录给出 Brier、NCR、CDR 的公式、分母、取值范围和 clean-label 仅用于 post-hoc diagnostic 的说明；把“独立监督信息”改为“来自其他球内训练节点的额外、非随机监督信息”。
- New experiment required: NO
- Claim decision: WEAKEN

## Consolidated verdict on requested checkpoints

- **11 数据集覆盖：** 五主数据集完整噪声谱成立；六扩展数据集 uniform 0.4 成立；不得把 24/25 外推到 11 数据集。
- **train-only、20/10/70、seed 可比：** 已抽查 formal raw 的范围内成立；完整 25-setting 最终表仍需逐行 hash 审计。
- **七个正式基线：** 正式候选集合与统一 X-only MLP 骨干成立；双网络/known-noise-rate 方法具有不同训练资源，不能称计算预算相同。代理和跨部署表必须与正式排名隔离。
- **统计：** mean±std、配对 seed、三组平均差和各 25-test Holm 计算正确；n=5、族定义与非预注册性质必须限制解释。
- **C-008：** 每数据集最佳图依赖参照选择正确；CoraFull 的 −4.48 是显示值相减，按全精度为约 −4.47，最终表应统一口径。
- **C-024/C-025：** 必须保持 `PARTIAL`；删除 kmeans、noisy-CE GRACE、MLP-hidden 扩展 claim，不得恢复。
- **LOO/Brier/NCR/CDR：** 数值与“纠错—误伤权衡”成立；需补公式并避免“独立”措辞。
- **kNN/票数/纯度：** 只能作为数据集特定、探索性相容证据；测试集峰值不能写成验证选择后的最优超参数。
- **H-access、clean/feature-noise 负结果：** 报告完整且边界正确，是稿件的明显优点。
- **GRACE H-seed：** n=3 只支持已测范围内方向稳定，不支持普遍不敏感。
- **效率：** seed 0、单硬件、GRACE 单独计费的范围披露正确；不得外推跨平台加速，最终需物化派生列与计时环境。
- **蒸馏：** `target_beta=0`、teacher_use_xh、学生 X-only 的资源流与 Table 5/6 汇总一致；H 输入是主要相关增量，共识附加收益具有 setting 依赖，不能写成唯一因果。

## Issue count self-check

- Total issues: 11
- P0: 0
- P1: 9
- P2: 2
- Schema fields per issue: 8/8 complete
- Experiment recommendations: YES 0 / NO 11
- Claim decisions: KEEP 0 / WEAKEN 7 / DELETE 1 / TEST 3
