# 1. Current Overall Judgment

本文件只覆盖当前项目目录 `/Users/qwwr/Downloads/codex-code_2` 中的日志、配置、源码和结果；不代表其他项目目录或全部 GB-GCL 项目的最终总判断。

- Confirmed: 当前项目至少完成 6 个方向级尝试：GB 视图生成/扰动、GB-FN-GRACE 负样本抑制、GB-BGRL/GBT 目标替换、GB edge denoise、x-space percentile 噪声鲁棒性、GB 构球/可靠球诊断。
- Confirmed: 直接让 GB 深度参与 GRACE 视图生成在长训练下不如 GRACE；Cora long comparison 中 `gb_gcl_mainline=0.7656±0.0109` 低于 `grace=0.7746±0.0113`。
- Confirmed: GB-FN-GRACE v2 topk-soft 是当前项目里最可写成主线的 GB 方法，但增益很小；Cora/CiteSeer 小幅高于 GRACE，PubMed 持平，且多数差值小于或接近 std。
- Confirmed: 纯 GB cross-ball edge denoise 命中噪声边太弱，不值得 formal；x-space percentile edge weighting 在噪声图上更强，但它不是纯 GB 主机制。
- Likely: GB 更适合作为温和结构先验或诊断工具，而不是硬 positive label、强视图替换、或独立主导图边重写机制。
- Inconclusive: Amazon/Coauthor 扩展仍不足；Amazon-Photo 单 seed 低于 GRACE，Amazon-Computers 单 seed 略高，Coauthor-CS 只有 sanity。
- Risk: 当前最强论文 claim 如果写成“GB 显著提升候选纯度或显著优于 GRACE”会被结果削弱；no-ball topk 与 ball-topk 非常接近。
- Next priority: 若继续 GB 主线，优先做“GB as conservative objective prior”的机制诊断和更严格多 seed/多数据集；若转噪声鲁棒性，优先围绕 percentile/edge scorer，而不是继续纯 same-ball edge denoise。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation / perturbation | 改视图生成、同球/跨球边扰动、anchor/explore | mainline, decay, freeze, anchor-explore v2, long/stability | Cora | 短训可激活，长训低于 GRACE | GB 视图扰动后期损伤 GRACE objective | abandon | negative result |
| DIR-02 | GB-FN-GRACE false-negative suppression | loss 分母中同球 Top-k soft suppression | hard, v1 full-soft, v2 topk, ratio, alpha schedule, strict quality, v2.1/v2.2/v2.3, no-ball | Cora, CiteSeer, PubMed, Amazon-Photo/Computers, Coauthor-CS sanity | Cora/CiteSeer 小幅正向，PubMed 持平 | 增益小且与 no-ball topk 接近 | modify | main-method candidate |
| DIR-03 | GB-BGRL / GB-GBT target replacement | 用 GB 视图接入 bootstrap 或 Barlow Twins | gb_bgrl, gb_gbt | Cora | 均低于 GRACE 和 GB-GCL stronger | 目标函数与 GB 视图分布错配 | abandon | negative result |
| DIR-04 | GB edge denoise by ball membership | 用同球/跨球关系给边加权或降权 | w025/w050/w075, hybrid ball-percentile | Cora edge_noise=0.4 | pure GB quick val 低于 GRACE；hybrid formal 仍低于更优 percentile | 噪声边命中弱、干净边 collateral damage | abandon | negative result |
| DIR-05 | x-space percentile noise robustness | 非 GB 或弱 GB 的相似度百分位边权 | pct soft, lr/mw ablations, noise-gated, DropEdge comparison | Cora, CiteSeer, PubMed, Photo | 噪声图多数据集稳定正向 | clean Cora 退化，机制不再是 GB 主贡献 | continue | main-method candidate |
| DIR-06 | GB diagnostics and reliable-ball topology | 诊断 GB 构球稳定性、候选命中率、可靠 topology | fn candidate precision, edge hit-rate, reliable-ball pct, ball partition diag | Cora, CiteSeer, PubMed | 提供失败解释和机制边界 | 诊断结果多为“GB 增益温和/构球不敏感” | diagnostic-only | diagnostic |

# 3. Direction-Level Detailed Records

## DIR-01: GB-guided View Generation / Perturbation

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB-guided view generation / perturbation
- Time period: Unknown；当前项目中已存在多轮 Cora 结果。
- Target problem: 让 granular ball 指导 GRACE 双视图增强，保留同球结构、扰动跨球结构，期望更语义一致的正样本视图。
- Core hypothesis: GB hierarchy 能提供比随机 DropEdge/feature mask 更好的结构化增强。
- Baseline: GRACE, 同一项目统一线性评估。
- Main related files: `src/augment/view_generator.py`, `src/augment/granular_ball.py`, `configs/cora_decay_*.yaml`, `configs/cora_anchor_v2_*.yaml`
- Result files: `outputs/cora_stability_comparison/*`, `outputs/cora_long_comparison/*`, `outputs/cora_decay_tuning_round/*`, `outputs/cora_hierarchy_freeze_round/*`, `outputs/cora_anchor_explore_v2_round/*`
- Raw logs: 各 run 下 `seed_*/train.log`，未逐条展开。
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 随机视图增强可能破坏局部语义结构。
- Why GB might help: GB 可提供局部结构/表示一致的节点组，理论上能保护同球边、强化跨球扰动。
- Expected improvement: 更稳定的正样本视图和更强鲁棒性。
- Literature or previous experiment inspiration: README 中项目目标是平滑接入 GB-GCL；具体外部文献锚点本项目未提供。
- Expected paper claim: 粒球更适合指导 graph view augmentation。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| mainline stability | GB 结构化视图增强 | Cora | `gb_gcl=0.7356±0.0112` vs `grace=0.7286±0.0235` | 短程/稳定性 | 可激活多球并短期小幅正向，但不是最终结论 |
| long comparison | 长训练对比 | Cora | `gb_gcl=0.7656±0.0109` vs `grace=0.7746±0.0113` | 长训练是否保持优势 | 长训练不能保持优势 |
| decay / stronger recovery | 后期降低扰动 | Cora | best `0.7482±0.0083`，仍低于 `grace=0.7532` | 减轻后期扰动 | 追回一部分但不足 |
| freeze hierarchy | 后期冻结构球 | Cora | `0.7460±0.0077`，低于 stronger recovery | 降低分区漂移 | 冻结不能解决性能损伤 |
| anchor-explore v2 | anchor 稳定、explore 扰动 | Cora | `0.7354±0.0146`，低于 stronger `0.7430±0.0157` | 弱/强视图分工 | anchor 过稳，正样本信号变弱 |

### 3.4 Role of Granular Ball

- GB role: 直接控制视图生成中的边扰动与 feature mask。
- Where GB is generated: `src/augment/view_generator.py` 调用 `GranularBallBuilder`。
- Representation space: 配置可用 `x` 或 `z`；当前结果多依赖训练期表示，具体每个版本需查对应 YAML。
- Affects loss: 间接影响 InfoNCE 输入视图。
- Affects augmentation: 是。
- Affects graph structure: 是，改变边保留率和同球/跨球边分布。
- Affects sampling: 否。
- Affects evaluation: 否，评估协议保持统一。

### 3.5 Result Summary

- Best result: 短程 stability round 中 GB-GCL `0.7356±0.0112` 高于该轮 GRACE `0.7286±0.0235`。
- Worst result: anchor-explore v2 `0.7354±0.0146`，明显低于 GRACE `0.7586±0.0112`。
- Average trend: 一旦进入更健康/长训练基线，GB 视图增强不能超过 GRACE。
- Mean/std: 见 `outputs/cora_long_comparison/summary.json`, `outputs/cora_decay_tuning_round/summary.csv`。
- Baseline mean/std: GRACE `0.7746±0.0113` in long comparison；GRACE `0.7532±0.0074` in perturbation decay comparison。
- Delta: long comparison 约 `-0.0090`；decay best 约 `-0.0050`。
- Std overlap: 多数重叠，不能说显著失败；但趋势方向反复低于 GRACE。
- Datasets improved: Cora 短程 stability round。
- Datasets worsened: Cora long/decay/freeze/anchor-explore。
- Per-seed stability: 多 seed 有；但不同轮次 baseline 绝对值不同。
- Runtime/memory cost: 比 GRACE 更复杂，需要构球和视图统计；具体 runtime Unknown。
- Raw evidence: `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前项目直接支持：GB 视图生成能激活非退化多球结构，但这不自动转化为更好的线性评估；长训练下 GB-GCL 视图生成低于 GRACE。

### 3.7 Reasonable Interpretation

解释：GB 结构化扰动在训练早期提供局部结构先验，但后期会减少有用增强多样性或改变正样本分布，使 GRACE 的 uniformity/alignment 平衡受损。

### 3.8 What Cannot Be Concluded

不能结论为“所有 GB augmentation 都无效”；当前只证明本项目这些同球/跨球扰动、decay、freeze、anchor-explore 形式不适合作为主线。也不能把短程 Cora 正向推广到其他数据集。

### 3.9 Failure Analysis

- Main failure reason: GB 深度参与视图生成后损伤原始 GRACE objective，长训练无法保持优势。
- Secondary failure reason: freeze/decay/anchor-explore 改动只缓解扰动强度，没有解决视图分布错配。
- Evidence: `outputs/cora_long_comparison/summary.json`, `outputs/cora_anchor_explore_v2_round/summary.csv`
- Alternative explanation: 部分 baseline 轮次绝对值不一致，可能存在训练长度或配置差异；但多轮结论方向一致。
- Minimum check needed: 若要复活，需在 Cora/CiteSeer/PubMed 同协议 5 seeds 下证明超过 GRACE，且保留增强多样性诊断。

### 3.10 Lessons Learned

- Lesson: 多球激活不是成功证据；evidence: long comparison 中 `mean_num_balls=30.944` 仍低于 GRACE；future constraint: 必须同时报告 accuracy 与机制统计。
- Lesson: 后期扰动衰减不能单独修复目标错配；evidence: stronger late recovery `0.7482±0.0083` 仍低于 GRACE；future constraint: 不要继续只调 decay schedule。
- Lesson: anchor 过稳会削弱对齐信号；evidence: anchor-explore v2 低于 stronger；future constraint: 不要把弱视图退化成近乎原图。

### 3.11 What Is Still Useful

可保留 `view_generator.py` 中的 GB 统计、同球/跨球边比例、anchor/explore 诊断字段，用作失败分析和附录 negative result。

### 3.12 Decision

- Decision: abandon
- Reason: 多轮 Cora formal/long 结果无法超过 GRACE，机制复杂度高。
- What should not be tried again: 不要继续只通过调同球/跨球 edge keep prob、freeze、decay 来挽救 GB view generation。
- Variant still worth trying: 只在噪声或强扰动设置中作为 diagnostic/guard，而非 clean graph 主线。
- Evidence required to revive this direction: 多数据集 5 seeds、同协议、无 label leakage、显著超过 GRACE/GCA，并提供增强多样性诊断。

## DIR-02: GB-FN-GRACE False-Negative Suppression

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB-FN-GRACE false-negative suppression
- Time period: 当前项目论文计划主线，`PAPER_PLAN.md` 日期为 2026-04-09。
- Target problem: 标准 InfoNCE 把所有非正样本当负样本，可能误伤语义相近节点。
- Core hypothesis: GB same-ball 提供候选范围；只对高质量同球 Top-k 高相似节点做 soft downweight 比整球 hard/full suppression 更稳。
- Baseline: GRACE；附加对照包括 no-ball Top-k、GCA、DGI。
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `src/augment/granular_ball.py`, `configs/*fn_grace*.yaml`
- Result files: `outputs/cora_fn_grace_*`, `outputs/fn_grace_v2_mainline_round/*`, `outputs/noball_topk_round/*`, `outputs/pubmed_formal_round/*`, `outputs/multi_dataset_round/*`, `outputs/fn_candidate_precision_round/*`
- Raw logs: 各 `outputs/*gb_fn_grace/seed_*/train.log`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: false negative 被过强排斥。
- Why GB might help: 同球节点更可能是局部结构/语义相近候选。
- Expected improvement: 在不改视图生成的前提下，小幅但稳定提升或至少减少 GB-GCL 视图扰动风险。
- Literature or previous experiment inspiration: `PAPER_PLAN.md` 将其定位为 objective correction，不是 view generation。
- Expected paper claim: GB 更适合作为 loss-side false-negative correction 的温和结构先验。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| hard | hard mask same-ball negatives | Cora | `0.7520±0.0217` | 硬删除是否可行 | 不如 soft，方差大 |
| v1 full-soft | 整球 soft downweight | Cora | `0.7580±0.0167` | 同球全量降权 | 接近 GRACE，但过宽 |
| v2 topk fixed-alpha | same-ball Top-k, alpha=0.25 | Cora | `0.7600±0.0118` vs GRACE `0.7586±0.0112` | 精细降权 | 当前最稳 GB-FN 版本 |
| v2 ratio | 按比例扩展 suppression | Cora | `0.7544±0.0118` | 更宽候选 | 过宽变差 |
| strict quality | 提高 quality threshold | Cora | `0.7600±0.0118` | 严格球质量 | 未改变主结果 |
| alpha schedule | 动态 alpha | Cora | `0.7600±0.0113` | 抑制强度 schedule | 与固定 alpha 基本等价 |
| v2.1 similarity-adaptive | 候选内连续权重 | Cora | `0.7580` | 更复杂权重 | 不优于 fixed-alpha |
| v2.2 mass preserve | 负压力守恒 | Cora | `0.7592±0.0112` | 修正 denominator mass | 无明显增益 |
| v2.3 cross-view | cross-view ranking/scope | Cora | best `0.7598±0.0120` | 排序来源与 scope | 不优于 same-view both |
| no-ball topk | 全体负样本相似度 Top-k | Cora/CiteSeer/PubMed | 极接近主线 | GB 候选是否有额外价值 | GB 额外收益很小 |

### 3.4 Role of Granular Ball

- GB role: loss-side candidate prior；不是 view generation。
- Where GB is generated: `build_false_negative_suppression_state` 从 `GranularBallHierarchy` 选择首个非平凡层级。
- Representation space: 默认 `ball_source=z`，训练期 encoder embedding；源码中 builder detach 表示，非梯度预处理。
- Affects loss: 是，InfoNCE denominator 权重。
- Affects augmentation: 否，保留 GRACE view generation。
- Affects graph structure: 否。
- Affects sampling: 只影响 negative candidate selection。
- Affects evaluation: 否。

### 3.5 Result Summary

- Best result: Cora topk=5 alpha=0.35 `0.7602±0.0114`，但只比主线 `+0.0002`；主线保持 `topk=5 alpha=0.25`。
- Worst result: v2 ratio `0.7544±0.0118`，hard `0.7520±0.0217`。
- Average trend: Cora/CiteSeer 小幅高于 GRACE，PubMed 持平，Amazon mixed。
- Mean/std: Cora `0.7600±0.0118`; CiteSeer `0.6766±0.0128`; PubMed `0.7790±0.0173`。
- Baseline mean/std: Cora GRACE `0.7586±0.0112`; CiteSeer GRACE `0.6756±0.0140`; PubMed GRACE `0.7790±0.0173`。
- Delta: Cora `+0.0014`; CiteSeer `+0.0010`; PubMed `0.0000`。
- Std overlap: 是，主线增益统计上很弱。
- Datasets improved: Cora, CiteSeer 微弱；Amazon-Computers 单 seed `+0.0030`。
- Datasets worsened: Amazon-Photo 单 seed `-0.0220`。
- Per-seed stability: Cora/CiteSeer 5 seeds，PubMed 3 seeds，Amazon mostly 1 seed。
- Runtime/memory cost: PubMed CPU 训练成本高；具体 runtime Unknown。
- Raw evidence: `outputs/fn_grace_v2_mainline_round/summary.csv`, `outputs/noball_topk_round/conclusion.txt`, `outputs/pubmed_formal_round/conclusion.txt`, `outputs/paper_ready_note.txt`

### 3.6 What Is Directly Supported

当前项目直接支持：相比 GB view generation，loss-side Top-k soft suppression 更稳；v2 topk 优于 full-soft/ratio/复杂加权变体；在 citation graphs 上不崩，并在 Cora/CiteSeer 小幅高于 GRACE。

### 3.7 Reasonable Interpretation

解释：收益主要来自“少量高相似 Top-k 的保守修正”，而不是来自 GB 大幅提高候选纯度。GB 可能提供轻量结构正则化或候选边界约束，但效果温和。

### 3.8 What Cannot Be Concluded

不能说 GB-FN-GRACE 显著优于 GRACE；不能说 GB 显著提高 false-negative 标签命中率；不能把 Amazon/Coauthor 写成 formal 支持；不能忽略 no-ball topk 近似追平主线。

### 3.9 Failure Analysis

- Main failure reason: GB same-ball 候选相对纯相似度 Top-k 的额外信息很弱。
- Secondary failure reason: 当前构球常只有 1-2 个球，masked_negative_ratio 极低，机制影响很保守。
- Evidence: `outputs/fn_candidate_precision_round/conclusion.txt` 显示 ball-topk Precision@k 相对 no-ball 不一致；`outputs/pubmed_formal_round/summary.json` 显示 PubMed mainline 与 GRACE 持平。
- Alternative explanation: 小幅收益可能受 best-val checkpoint 和 seed 方差影响。
- Minimum check needed: paired seed 显著性、主线重新收集 selected similarity、更多 noisy/heterophily 场景。

### 3.10 Lessons Learned

- Lesson: 整球 full-soft 太粗；evidence: v1 `masked_negative_ratio≈0.13`，v2 `≈0.0012` 且结果更好；future constraint: 不再扩大 suppression 范围作为默认方向。
- Lesson: 固定 alpha 已足够；evidence: similarity-adaptive、mass-preserve、cross-view-aware 均未超过 v2；future constraint: 停止堆复杂公式，除非有新诊断。
- Lesson: no-ball 是必须 baseline；evidence: Cora no-ball `0.7592±0.0114` vs mainline `0.7600±0.0118`; future constraint: 所有 GB 候选论文 claim 必须报告 no-ball topk。

### 3.11 What Is Still Useful

当前主线代码、no-ball 对照、候选 precision 诊断、机制统计都可保留。论文可保守写成“GB as mild structural prior for objective correction”，而不是大幅性能 claim。

### 3.12 Decision

- Decision: modify
- Reason: 是当前最好的 GB 主线，但证据强度不足以夸大；需要更诚实定位和补机制诊断。
- What should not be tried again: 不要继续 v2.1/v2.2/v2.3 式复杂公式微调；不要扩大 ratio/full-ball suppression。
- Variant still worth trying: 在噪声图或强增强设置中测试 GB-FN 是否比 no-ball 更稳；或增加候选稳定性/时机诊断。
- Evidence required to revive this direction: 多数据集 paired 5 seeds、相对 GRACE/no-ball/GCA 的明确非重叠收益、GB ablation、候选质量与最终性能一致机制解释。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer, PubMed；Amazon-Photo/Computers partial；Coauthor-CS sanity。
- Seeds: Cora/CiteSeer 5 seeds；PubMed 3 seeds；Amazon mostly 1 seed。
- Epochs: 多数 100 pretrain / 100 eval；部分 quick 30 epoch。
- Encoder: GCN + projection head。
- Base loss: InfoNCE。
- Added mechanism: same-ball Top-k soft negative downweight。
- GB loss weight: Not applicable；通过 denominator weight `alpha=0.25`。
- Augmentation: GRACE edge dropout + feature mask。
- Ball generation method: representation/topology joint split；`GranularBallBuilder`。
- Evaluation protocol: frozen linear evaluation。
- Train/val/test split: Planetoid/RandomNodeSplit 按当前项目配置；Amazon 与文献协议不同。
- Hardware/runtime: CPU 下 PubMed 成本高；具体 Unknown。

### Baseline Comparability Check

- Same dataset: yes for formal Cora/CiteSeer/PubMed。
- Same split: appears yes within current configs; exact split seeds TODO。
- Same seeds: yes for Cora/CiteSeer；PubMed formal 3 seeds；Amazon only 1 seed。
- Same encoder/projector: yes。
- Same augmentation: yes，GB-FN 不改 view。
- Same evaluation script: yes。
- Same linear evaluation: yes。
- Baseline rerun: yes in current outputs。
- Comparable conclusion: partial；Cora/CiteSeer/PubMed comparable，Amazon only exploratory。

### Label Leakage Check

- Labels used in GB generation: no evidence；GB quality uses representation, structural density, boundary mixing。
- Labels used in purity: no training-time purity in `ball_quality.py`；post-hoc `fn_candidate_precision.py` uses labels only for analysis。
- Labels used in sampling/coarsening: no evidence。
- Validation/test labels used outside evaluation: no evidence in reviewed source。
- Supervised GB mechanism used in pretraining: no evidence。
- Leakage risk: low for training; diagnostic analyses are label-aware and must stay post-hoc only。

### Granular-Ball Diagnostics

- Number of balls: Cora GB-FN mean `1.65`; CiteSeer mean `1.75`; PubMed mean `1.72`, max `2`。
- Average ball size: Cora high-quality size around `880.1` in `fn_grace_v2_mainline_round`；PubMed detailed sizes Unknown。
- Max ball size: Unknown/TODO。
- Singleton ball ratio: Unknown/TODO。
- Radius mean/std: Unknown/TODO。
- Coverage: high-quality coverage often implied by candidate stats; exact coverage TODO。
- Relation density: same_ball_pair_ratio Cora around `0.36`; CiteSeer around `0.384`; PubMed `0.403`。
- Confidence distribution: ball quality threshold tested but detailed distribution Unknown。
- Cross-epoch stability: suppression epochs Cora 500 selection events; PubMed `71.67` epochs average。
- Cross-seed stability: mean/std available; assignment stability TODO。
- GB loss scale vs contrastive loss: Not applicable。
- Too coarse: likely on PubMed/Cora first nontrivial partition often only 2 balls。
- Too fragmented: not for GB-FN; hybrid edge denoise fragmented into thousands of balls, separate direction。
- TODO: collect ball sizes/radii/stability consistently.

### Next Minimal Experiment If Modified

- New hypothesis: GB-FN 的价值在噪声/强增强下可能来自候选时机和结构稳定性，而不是 clean graph candidate purity。
- Change only: run GB-FN v2, no-ball topk, GRACE on one noisy setting with identical seeds and collect candidate overlap/stability.
- Keep fixed: encoder, view augmentation, alpha=0.25, topk=5, eval protocol。
- Minimal datasets: Cora edge_noise=0.4 plus CiteSeer edge_noise=0.4。
- Minimal seeds: 5 paired seeds。
- Success criterion: GB-FN beats no-ball and GRACE by delta exceeding overlapping std or paired p<0.05, with stable candidate diagnostics。
- Failure criterion: GB-FN remains tied with no-ball/GRACE or candidate purity/overlap shows no GB effect。
- Why this is the smallest useful test: It directly tests whether GB adds value beyond similarity Top-k where robustness signal might matter.

## DIR-03: GB-BGRL / GB-GBT Target Replacement

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: GB-BGRL / GB-GBT target replacement
- Time period: Unknown；current outputs include Cora rounds。
- Target problem: 用 non-InfoNCE objective 缓解 GRACE false negative 或视图不稳问题。
- Core hypothesis: BGRL/Barlow Twins 风格目标配合 GB view 更稳定。
- Baseline: GRACE and GB-GCL stronger_late_recovery。
- Main related files: `src/gb_gcl/models/contrastive.py`, `src/gb_gcl/losses/bgrl_loss.py`, `src/gb_gcl/losses/barlow_twins.py`, `src/augment/view_generator.py`
- Result files: `outputs/cora_gb_bgrl_round/*`, `outputs/cora_gbt_round/*`
- Raw logs: `outputs/cora_gb_bgrl_*/*/train.log`, `outputs/cora_gb_gbt_*/*/train.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: InfoNCE false negatives 和视图增强目标错配。
- Why GB might help: GB 可产生 weak/strong 视图或 teacher/student 视图分工。
- Expected improvement: 避免负样本假设，提升稳定性。
- Literature or previous experiment inspiration: BGRL/Barlow Twins family；本项目具体文献锚点 Unknown。
- Expected paper claim: GB 可用于非对比 self-supervised graph objective。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-BGRL | 原图 teacher + 强 GB student | Cora | `0.4164±0.0456` | bootstrap 是否稳定 | 严重失败，loss 快速近零但表示差 |
| GB-GBT | GB/weak view + Barlow Twins | Cora | `0.7304±0.0138` | 去负样本冗余约束 | 比 BGRL 好但仍低于 GRACE/GB-GCL |

### 3.4 Role of Granular Ball

- GB role: 生成 student/weak/GB view。
- Where GB is generated: `src/augment/view_generator.py` 中 `gb_bgrl`, `gb_gbt` 分支。
- Representation space: 当前配置需查 YAML；总体基于 GB hierarchy。
- Affects loss: 目标函数替换为 BGRL or Barlow Twins。
- Affects augmentation: 是。
- Affects graph structure: 是。
- Affects sampling: 否。
- Affects evaluation: 否。

### 3.5 Result Summary

- Best result: GB-GBT `0.7304±0.0138`。
- Worst result: GB-BGRL `0.4164±0.0456`。
- Average trend: 均低于 GRACE `0.7586±0.0112`。
- Mean/std: 见 `outputs/cora_gb_bgrl_round/summary.json`, `outputs/cora_gbt_round/summary.json`。
- Baseline mean/std: GRACE `0.7586±0.0112`; GB-GCL stronger `0.7430±0.0157`。
- Delta: GB-GBT vs GRACE `-0.0282`; GB-BGRL vs GRACE `-0.3422`。
- Std overlap: GB-GBT 与 GRACE 基本不支持正向；GB-BGRL 清楚失败。
- Datasets improved: None。
- Datasets worsened: Cora。
- Per-seed stability: 5 seeds。
- Runtime/memory cost: Unknown。
- Raw evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前最小 GB-BGRL 和 GB-GBT 都不适合作为本项目主线；把目标函数替换掉比保持 GRACE 并做 loss-side 小修更危险。

### 3.7 Reasonable Interpretation

BGRL 失败可能来自 teacher/student gap 过大且 teacher 原图过易，导致 loss 近零但表示无信息；GB-GBT 问题更像目标函数与当前视图分布错配，而非构球完全失效。

### 3.8 What Cannot Be Concluded

不能结论为 BGRL/Barlow Twins 在图上无效；只能说当前 GB 接入方式失败。也不能把 GB-GBT 稳定激活解释为有效表示学习。

### 3.9 Failure Analysis

- Main failure reason: 目标替换引入更大训练动力学错配。
- Secondary failure reason: GB 视图仍然继承 DIR-01 的视图分布风险。
- Evidence: GB-BGRL `0.4164±0.0456`; GB-GBT `0.7304±0.0138`。
- Alternative explanation: 超参没有充分调优；但差距过大，不宜继续投入。
- Minimum check needed: 若重启，需要先在 vanilla BGRL/GBT 健康 baseline 上证明当前实现正常。

### 3.10 Lessons Learned

- Lesson: 避免同时改 view 和 objective；evidence: GB-BGRL 极低；future constraint: 新方向一次只改一个机制。
- Lesson: loss 下降不等于表示好；evidence: GB-BGRL conclusion 中 loss 接近 0 但 test 低；future constraint: 记录 collapse/uniformity diagnostics。
- Lesson: 非对比路线应先有 vanilla strong baseline；evidence: 当前只显示 GB 版本差；future constraint: 没有健康 vanilla BGRL/GBT 不进入主线。

### 3.11 What Is Still Useful

可作为 negative result 支持“保持 GRACE 主体、只做最小 loss correction”这一研究选择。

### 3.12 Decision

- Decision: abandon
- Reason: Cora 5 seeds 远低于 GRACE，且机制比 GB-FN-GRACE 更复杂。
- What should not be tried again: 不要在没有 vanilla BGRL/GBT 健康基线前继续改 GB-BGRL/GB-GBT。
- Variant still worth trying: Diagnostic-only collapse analysis。
- Evidence required to revive this direction: 健康 vanilla bootstrap/BT baseline + GB 接入后多 seed 正向。

## DIR-04: GB Edge Denoise by Ball Membership

### 3.1 Direction Summary

- Direction ID: DIR-04
- Direction name: GB edge denoise by ball membership
- Time period: 当前项目噪声鲁棒性探索阶段。
- Target problem: 输入图有 edge noise 时，利用 GB same/cross-ball 信息降低疑似噪声边权重。
- Core hypothesis: 跨球边更可能是噪声边，同球边更可信。
- Baseline: GRACE quick/formal, DropEdge, percentile edge weighting。
- Main related files: `src/augment/soft_edge_denoise.py`, `configs/cora_gb_edge_denoise_*.yaml`
- Result files: `outputs/soft_edge_denoise_round/*`, `outputs/edge_denoise_precision_round/*`, `outputs/hybrid_ball_quick_summary.json`, `outputs/cora_hybrid_aggressive_formal_gb_edge_denoise/summary.json`
- Raw logs: `outputs/cora_gb_edge_denoise_*/*/train.log`, `outputs/cora_hybrid_aggressive_formal_run.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 噪声边会污染 message passing 和增强。
- Why GB might help: 跨球边可能是局部结构异常，边权下调可去噪。
- Expected improvement: edge_noise=0.4 下超过 GRACE。
- Literature or previous experiment inspiration: 项目内部从 GB-FN 失败边界转向噪声鲁棒性。
- Expected paper claim: GB denoising improves robustness under graph noise。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| pure GB w025/w050/w075 | 跨球边固定降权 | Cora edge_noise=0.4 quick | best val `0.5880` vs GRACE quick `0.5980` | same/cross-ball 是否能去噪 | val 不过 GRACE |
| precision diagnosis | 噪声边命中率 | Cora edge_noise=0.4 | precision `0.5596`, recall `0.2272` | 是否真命中噪声边 | 只是弱统计偏差 |
| hybrid ball-percentile quick | pct + GB cross-ball penalty | Cora edge_noise=0.4 quick | aggressive test `0.5940`, val `0.5680` | GB 是否增强 percentile | quick 正向但机制极端 |
| hybrid aggressive formal | 5 seed formal | Cora edge_noise=0.4 | `0.6344±0.0182` | quick 能否 formal | 低于更优 percentile `0.6518±0.0178` |

### 3.4 Role of Granular Ball

- GB role: 边权计算，same-ball 保留/boost，cross-ball penalty/downweight。
- Where GB is generated: edge denoise path 中 refresh hierarchy。
- Representation space: x or z depending config；hybrid formal 显示 `mean_num_balls≈2622`，近似极碎。
- Affects loss: 间接通过 encoder edge_weight。
- Affects augmentation: 可改变输入边权。
- Affects graph structure: 是。
- Affects sampling: 否。
- Affects evaluation: 否。

### 3.5 Result Summary

- Best result: hybrid aggressive formal `0.6344±0.0182` on Cora edge_noise=0.4。
- Worst result: pure GB w025/w050/w075 quick val 均低于 GRACE quick。
- Average trend: 纯 GB denoise 不足；hybrid 不如纯 percentile 最优。
- Mean/std: hybrid formal `0.6344±0.0182`; best percentile same setting `0.6518±0.0178`。
- Baseline mean/std: Cora edge_noise=0.4 GRACE formal `0.6126±0.0286`；pct lr055/mw00 `0.6518±0.0178`。
- Delta: hybrid vs GRACE `+0.0218`; hybrid vs pct `-0.0174`。
- Std overlap: hybrid vs pct 可能重叠，但 best-val 不支持替代。
- Datasets improved: Cora noisy vs GRACE。
- Datasets worsened: vs percentile。
- Per-seed stability: hybrid formal 5 seeds；pure GB quick 1 seed。
- Runtime/memory cost: 构球碎片化，复杂度高；具体 Unknown。
- Raw evidence: `outputs/soft_edge_denoise_round/conclusion.txt`, `outputs/edge_denoise_precision_round/conclusion.txt`, `outputs/cora_hybrid_aggressive_formal_gb_edge_denoise/summary.json`

### 3.6 What Is Directly Supported

当前项目直接支持：跨球边确实稍微富集噪声边，但 precision/recall 不够，不适合作为主机制；hybrid 的收益主要可能来自 percentile similarity，而不是 GB。

### 3.7 Reasonable Interpretation

GB cross-ball 信号过粗或过碎。hybrid 中 `same_ball_edge_ratio≈0.001`，几乎所有边都是 cross-ball，GB penalty 近似全局额外下压，不是有解释力的结构去噪。

### 3.8 What Cannot Be Concluded

不能说“GB denoising 已成功”；也不能把 hybrid quick 正向当作正式 GB 贡献。不能把 label/noise-origin 诊断用于 unsupervised training claim。

### 3.9 Failure Analysis

- Main failure reason: same/cross-ball 对噪声边只是弱分类器，clean collateral damage 高。
- Secondary failure reason: hybrid 构球极碎，GB 信号退化为近乎全边 penalty。
- Evidence: downweight precision `0.5596`, recall `0.2272`; hybrid formal low vs percentile。
- Alternative explanation: 更强 scorer 可能有效，但那会变成 edge scorer 方向。
- Minimum check needed: 若重启，先证明 GB edge score 的 AUC/precision 显著高于 x-space percentile。

### 3.10 Lessons Learned

- Lesson: 跨球不等于噪声；evidence: precision 仅 modest above base rate 0.4；future constraint: 不要再用 hard cross-ball rule。
- Lesson: 高碎片化球不可解释；evidence: hybrid `mean_num_balls≈2622`, `same_ball_edge_ratio≈0.001`；future constraint: 报告 singleton/fragmentation。
- Lesson: 与 percentile 比较是必要的；evidence: percentile formal 更强；future constraint: GB denoise 不能只比 GRACE。

### 3.11 What Is Still Useful

edge precision 诊断代码、noise-origin hit-rate 分析、hybrid negative result 可用于说明为什么转向 percentile/edge scorer。

### 3.12 Decision

- Decision: abandon
- Reason: 纯 GB 机制弱，hybrid 不如更简单 percentile。
- What should not be tried again: 不要用 same-ball/cross-ball 固定边权作为主线。
- Variant still worth trying: 只作为 edge scorer 的一个 feature，而非单独规则。
- Evidence required to revive this direction: GB feature 在 edge-noise AUC/AP 和 formal accuracy 上都超过 x-space percentile。

## DIR-05: X-space Percentile Noise Robustness

### 3.1 Direction Summary

- Direction ID: DIR-05
- Direction name: x-space percentile noise robustness
- Time period: 当前项目后期噪声鲁棒性方向。
- Target problem: edge noise 下 GRACE 性能下降。
- Core hypothesis: 端点特征/表示相似度低的边更可能是噪声，按 percentile 进行 soft edge weighting 可提升鲁棒性。
- Baseline: GRACE, DropEdge, always-on vs gated。
- Main related files: `src/augment/soft_edge_denoise.py`, `scripts/summarize_pct_multi_noise_formal.py`, `scripts/analyze_pct_edge_hit_rate.py`
- Result files: `outputs/pct_formal_round/*`, `outputs/pct_multi_noise_formal/*`, `outputs/statistical_boundary_round/*`, `outputs/noise_gate_formal_round/*`, `outputs/new_params_formal_summary.json`, `outputs/supplement_summary.json`
- Raw logs: `outputs/*pct*_gb_edge_denoise/seed_*/train.log`
- Current decision: continue

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: noisy graph edges inject wrong neighborhoods。
- Why GB might help: 原始起点来自 GB edge denoise，但最终有效机制主要是 x-space percentile。
- Expected improvement: 噪声图上稳定超过 GRACE 和 DropEdge，clean graph 上通过 gate 避免伤害。
- Literature or previous experiment inspiration: 当前项目内部诊断显示 pure GB denoise 覆盖不足，percentile 改善 coverage。
- Expected paper claim: 简单相似度边权能提供噪声鲁棒性；若写 GB 论文需谨慎，因为 GB 角色较弱。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| pct soft initial | low_ratio=0.35/min_weight=0.2 | Cora/CiteSeer noisy | 正向但非最优 | percentile 是否有效 | 噪声图有信号 |
| new params | low_ratio=0.55/min_weight=0.0 | Cora/CiteSeer noisy | 4/4 settings 胜旧参数 | 参数是否收束 | 更强默认 |
| statistical boundary | 多噪声/多数据集 formal | Cora/CiteSeer/PubMed/Photo | noisy 8/8 positive | 统计边界 | noisy robustness 最强证据 |
| noise gate | clean off/noisy on | Cora clean/noise=0.4 | clean 等于 GRACE，noisy 保留收益 | 避免 clean Cora 伤害 | gate 可采用 |
| clean graph risk | clean graph 标签诊断 | Cora/CiteSeer/PubMed | Cora clean 退化 | 为什么 clean 风险 | same-label collateral damage |

### 3.4 Role of Granular Ball

- GB role: 当前主机制基本不是 GB；在代码路径中 method 名仍为 `gb_edge_denoise`，但 `use_percentile_scoring=true` 且 ball effect 可为 none。
- Where GB is generated: 部分 configs 仍会构球，但 percentile weight 不依赖 same-ball。
- Representation space: x-space cosine percentile in diagnostics。
- Affects loss: 间接通过 edge weights。
- Affects augmentation: 是，改变 edge weights。
- Affects graph structure: 是。
- Affects sampling: 否。
- Affects evaluation: 否。

### 3.5 Result Summary

- Best result: statistical boundary Cora noise=0.5 delta `+0.0478`; CiteSeer noise=0.4 delta `+0.0532`; PubMed noise=0.4 delta `+0.0438`。
- Worst result: clean Cora `-0.0156`。
- Average trend: noisy settings consistently positive；clean mixed。
- Mean/std: see `outputs/statistical_boundary_round/summary.json`。
- Baseline mean/std: GRACE same settings。
- Delta: noisy 8/8 positive；Cora noisy trend monotonic 0.1-0.5。
- Std overlap: some small deltas overlap；Cora noise=0.3/0.4/0.5 p-values `0.0163/0.0110/0.0018`。
- Datasets improved: noisy Cora, CiteSeer, PubMed, Photo；clean CiteSeer/PubMed。
- Datasets worsened: clean Cora。
- Per-seed stability: formal 5 seeds for statistical boundary。
- Runtime/memory cost: edge weight computation overhead；specific Unknown。
- Raw evidence: `outputs/statistical_boundary_round/conclusion.txt`, `outputs/noise_gate_formal_round/conclusion.txt`, `outputs/pct_edge_hit_rate_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前项目最强直接支持是：x-space percentile edge weighting 是有效噪声鲁棒性机制，在 noisy settings 明显强于 GRACE/DropEdge；noise-gated 版本可避免 clean Cora always-on 退化。

### 3.7 Reasonable Interpretation

低相似边在注入 edge noise 后富集噪声边，因此 percentile weighting 是一个有用统计偏置；但它也会压低部分干净同类边，所以 clean graph 尤其 Cora 会受损。

### 3.8 What Cannot Be Concluded

不能把这个方向写成“GB 提供 robust local structure prior”的直接证据；主机制不是 same-ball GB。也不能说 clean graph 一定提升，Cora clean 明确退化。

### 3.9 Failure Analysis

- Main failure reason: clean graph 下 downweight 覆盖过大，误伤 same-label edges。
- Secondary failure reason: 当前规则是统计 bias，不是 clean separator。
- Evidence: `outputs/pct_edge_hit_rate_round/conclusion.txt` 中 clean downweight rate `0.3888` across noisy settings；clean risk round 中 Cora same-label downweight rate `0.5226`。
- Alternative explanation: clean Cora 退化可能与同质性/特征分布有关。
- Minimum check needed: clean/noisy gate 的自动判别方式不能依赖标签或已知噪声。

### 3.10 Lessons Learned

- Lesson: noisy graph 上简单统计规则比纯 GB 边规则更强；evidence: pct formal 多设置正向；future constraint: 噪声鲁棒性 baseline 必须包括 percentile。
- Lesson: always-on 会伤 clean Cora；evidence: clean Cora `-0.0156`; future constraint: 需要 gate 或只声称 noisy robustness。
- Lesson: 需要同时报告 precision/recall/collateral；evidence: pct edge hit-rate 仍有 clean collateral；future constraint: 论文不能只报 accuracy。

### 3.11 What Is Still Useful

这是当前项目最强的鲁棒性信号；若论文方向允许从 GB 转为 robust edge weighting，可作为主线候选。若论文必须是 GB-GCL，则它更适合作为对照/启发或未来方向。

### 3.12 Decision

- Decision: continue
- Reason: 多数据集 noisy formal 证据比其他方向更强。
- What should not be tried again: 不要在 clean graph always-on；不要称其为纯 GB 方法。
- Variant still worth trying: unsupervised noise gate、attention/MLP edge scorer，但标签/噪声标记只能 diagnostic-only。
- Evidence required to revive this direction: 自动 gate、clean 不退化、noisy 多数据集显著正向。

## DIR-06: GB Diagnostics and Reliable-Ball Topology

### 3.1 Direction Summary

- Direction ID: DIR-06
- Direction name: GB diagnostics and reliable-ball topology
- Time period: 当前项目中后期诊断。
- Target problem: 解释 GB-FN/edge-denoise 为什么有效或失败。
- Core hypothesis: 分析候选命中率、构球稳定性、可靠 topology 是否改变球划分，可定位瓶颈。
- Baseline: Diagnostic-only。
- Main related files: `src/gb_gcl/analysis/fn_candidate_precision.py`, `src/gb_gcl/analysis/edge_denoise_precision.py`, `src/gb_gcl/analysis/ball_partition_diagnosis.py`, `src/augment/granular_ball.py`
- Result files: `outputs/fn_candidate_precision_round/*`, `outputs/edge_denoise_precision_round/*`, `outputs/reliable_ball_percentile_round/*`, `outputs/ball_partition_diag_round/*`, `outputs/pct_edge_hit_rate_round/*`
- Raw logs: Diagnostic outputs and CSVs。
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 结果差异太小，必须解释机制而非只调参。
- Why GB might help: GB 统计能揭示 same-ball 候选、构球稳定性、拓扑敏感性。
- Expected improvement: 不直接改善模型，而是约束下一步实验。
- Literature or previous experiment inspiration: project-internal failed variants。
- Expected paper claim: GB diagnostics explain why objective correction is safer than view generation。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| fn candidate precision | label-only post-hoc Precision@k | Cora/CiteSeer/PubMed | ball-topk not consistently above no-ball | GB 候选纯度 | GB 没有显著纯度跃迁 |
| edge denoise precision | noisy-origin hit-rate | Cora noise=0.4 | precision modest, recall low | 跨球是否命中噪声 | pure GB denoise 弱 |
| reliable-ball pct | 构球前删 topology edges | Cora noise=0.4 quick | pct70/80/90 完全同结果 | topology filter 是否影响构球 | 不改变球/候选统计 |
| ball partition diagnosis | raw vs reliable assignment | Cora noise=0.4 | assignment change max `0.001108` | 为什么 topology 不敏感 | feature term 主导 |
| pct edge hit-rate | percentile noisy-edge hit | multi noisy | AUC/AP 有信号但 collateral | percentile 机制 | 支持 edge scorer/gate |

### 3.4 Role of Granular Ball

- GB role: diagnostic-only。
- Where GB is generated: analysis scripts replay checkpoints/configs。
- Representation space: mostly encoder embedding or x-space depending diagnostic。
- Affects loss: no。
- Affects augmentation: no。
- Affects graph structure: no training changes in diagnostics。
- Affects sampling: no。
- Affects evaluation: post-hoc only。

### 3.5 Result Summary

- Best result: 诊断解释清晰；例如 ball partition diag 表明 feature margin `0.119535` 大于 topology delta `0.004603`。
- Worst result: reliable-ball topology route无早期正向。
- Average trend: diagnostics mostly weaken strong GB claims, but provide useful anti-repetition evidence。
- Mean/std: Not generally applicable。
- Baseline mean/std: Not applicable。
- Delta: Not applicable。
- Std overlap: Not applicable。
- Datasets improved: Not applicable。
- Datasets worsened: Not applicable。
- Per-seed stability: some diagnostics single seed；fn precision multi dataset。
- Runtime/memory cost: modest, but some replay diagnostics。
- Raw evidence: Listed above。

### 3.6 What Is Directly Supported

标签只用于 post-hoc 的 candidate precision；它直接支持“Top-k 相似度远好于 random，但 GB ball-topk 相对 no-ball topk 纯度优势不一致”。可靠 topology 删除真实发生，但对构球结果几乎没有影响。

### 3.7 Reasonable Interpretation

当前 GB 构球更受 feature/representation margin 主导，局部 topology 的小扰动不足以改变首个非平凡分区；因此“只改构球 topology”不是有效路径。

### 3.8 What Cannot Be Concluded

不能把 post-hoc label precision 当训练证据；不能使用 label/noise-origin 指标设计 unsupervised training 规则，除非明确为 oracle/diagnostic。

### 3.9 Failure Analysis

- Main failure reason: 诊断显示部分 GB claim 不成立，例如候选纯度没有明显强于 no-ball。
- Secondary failure reason: 一些诊断是 single seed/quick，不能过度统计化。
- Evidence: `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/ball_partition_diag_round/conclusion.txt`
- Alternative explanation: 不同 checkpoint/epoch 的候选可能不同；需更完整轨迹。
- Minimum check needed: 若用于论文，补齐主线 selected similarity 与 candidate overlap 诊断。

### 3.10 Lessons Learned

- Lesson: 诊断必须在论文 claim 前运行；evidence: fn precision 改写了“GB 提高候选纯度”的叙事；future constraint: 每个新方向必须有机制诊断。
- Lesson: 改 topology 不等于改构球；evidence: pct70/80/90 assignment 几乎不变；future constraint: 构球相关方向必须报告 assignment change。
- Lesson: 标签/噪声标记只能用于 post-hoc；evidence: analysis scripts 明确 post-hoc；future constraint: 训练代码不得引入 label purity。

### 3.11 What Is Still Useful

所有诊断脚本和 CSV 是未来 AI run 的关键记忆，尤其是防止重复尝试“可靠 topology filter”和“GB 显著提升候选纯度”的强 claim。

### 3.12 Decision

- Decision: diagnostic-only
- Reason: 不作为方法主线，但解释力强。
- What should not be tried again: 不要把可靠 topology pct70/80/90 直接扩 formal。
- Variant still worth trying: candidate overlap/stability over epochs；GB partition entropy/singleton diagnostics。
- Evidence required to revive this direction: 诊断发现可操作且无泄漏的新信号。

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: GB as Direct View Generator Is Unstable

### Status
confirmed

### Related Directions
DIR-01, DIR-03

### Observation
GB-GCL 视图生成短期可激活多球，但 long/decay/freeze/anchor-explore 多轮不能超过 GRACE。

### Possible Reason
结构化视图减少增强多样性或改变 alignment/uniformity 平衡。

### Evidence
`outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`

### Future Constraint
不要把 GB 深度参与 view generation 作为默认主线。

### When to Revisit
仅在 noisy graph/strong augmentation 下，并且包含增强多样性诊断。

## FP-2: Hard or Broad Same-Ball Suppression Is Too Coarse

### Status
confirmed

### Related Directions
DIR-02

### Observation
hard 和 full-soft 不如 Top-k；ratio 扩宽后退化。

### Possible Reason
same-ball 不等于同语义，整球降权误伤有效负样本。

### Evidence
`outputs/fn_grace_v2_mainline_round/summary.csv`

### Future Constraint
保持 Top-k 小候选，不要扩大 suppression 范围。

### When to Revisit
只有当候选置信度诊断显著提高且有 no-ball 对照时。

## FP-3: More Complex FN Formula Does Not Add Value

### Status
confirmed

### Related Directions
DIR-02

### Observation
similarity-adaptive、mass-preserve、cross-view-aware 都未超过 v2 fixed-alpha。

### Possible Reason
候选集已极小，额外权重/归一化对训练动力学影响弱或削弱负压力。

### Evidence
`outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt`

### Future Constraint
停止公式微调，除非新诊断指出明确瓶颈。

### When to Revisit
在不同数据集/噪声条件下 fixed-alpha 明确失败但候选质量仍强时。

## FP-4: GB Candidate Purity Claim Is Weak

### Status
confirmed

### Related Directions
DIR-02, DIR-06

### Observation
ball-topk Precision@k 相对 no-ball topk 不一致；Cora/PubMed 略低，CiteSeer 略高。

### Possible Reason
纯相似度 Top-k 已经很强，GB 同球范围没有增加明显标签纯度。

### Evidence
`outputs/fn_candidate_precision_round/conclusion.txt`

### Future Constraint
论文不要写“GB 显著提高候选纯度”。

### When to Revisit
如果新的 GB confidence 或 boundary-distance 让 Precision@k 在多数据集稳定超过 no-ball。

## FP-5: Pure Same/Cross-Ball Edge Denoise Is Too Weak

### Status
confirmed

### Related Directions
DIR-04

### Observation
downweight precision 仅 `0.5596`、recall `0.2272`，quick val 低于 GRACE。

### Possible Reason
跨球边只是噪声边弱统计偏置，误伤干净边。

### Evidence
`outputs/edge_denoise_precision_round/conclusion.txt`, `outputs/soft_edge_denoise_round/conclusion.txt`

### Future Constraint
不要用 same/cross-ball fixed edge weights 作为主线。

### When to Revisit
作为 edge scorer 的一个 feature，而非规则本身。

## FP-6: Reliable Topology Filtering Does Not Move Ball Assignments

### Status
confirmed

### Related Directions
DIR-06

### Observation
pct90/80/70 实际删边，但 changed_assignment_ratio 最大只有 `0.001108`。

### Possible Reason
feature margin 主导 assignment，topology margin delta 太小。

### Evidence
`outputs/ball_partition_diag_round/conclusion.txt`

### Future Constraint
不要直接扩大 reliable-ball topology formal。

### When to Revisit
只有在构球权重或表示空间改变后 assignment change 明显增加。

## FP-7: Clean Graph Always-On Edge Weighting Can Hurt

### Status
confirmed

### Related Directions
DIR-05

### Observation
Clean Cora pct test `0.7432` 低于 GRACE `0.7588`。

### Possible Reason
downweight 了大量 same-label clean edges。

### Evidence
`outputs/clean_graph_risk_round/conclusion.txt`

### Future Constraint
噪声鲁棒性方法必须 gate 或限定 noisy setting。

### When to Revisit
有无监督 clean/noisy 判别或 edge scorer 能降低 clean collateral。

# 5. Cross-Direction Positive Signal Ledger

## PS-1: Loss-Side GB Prior Is Safer Than View-Side GB

### Status
confirmed

### Related Directions
DIR-01, DIR-02

### Observation
GB-FN-GRACE v2 在 Cora/CiteSeer 小幅高于 GRACE，且不改 augmentation；GB-GCL view generation 长训低于 GRACE。

### Why It May Matter
这支持“minimal objective correction”作为 GB-GCL 的更安全使用位置。

### Required Verification
paired significance、no-ball 对照、更多数据集。

### Possible Paper Use
main method

## PS-2: Top-k Similarity Selection Is Strong

### Status
confirmed

### Related Directions
DIR-02, DIR-06

### Observation
no-ball topk 和 ball-topk 都远好于 random topk 的 Precision@k。

### Why It May Matter
false negative suppression 的核心可能是少量高相似候选，而非整球关系。

### Required Verification
训练期候选 overlap 与最终性能的关系。

### Possible Paper Use
motivation

## PS-3: X-space Percentile Edge Weighting Has Robustness Value

### Status
confirmed

### Related Directions
DIR-05

### Observation
noisy settings 8/8 positive，Cora noisy delta 随噪声增强。

### Why It May Matter
可形成更强鲁棒性论文或作为 GB 失败后的替代方向。

### Required Verification
自动 gate、clean 不退化。

### Possible Paper Use
main method

## PS-4: Diagnostics Explain Failure Before More Runs

### Status
confirmed

### Related Directions
DIR-04, DIR-06

### Observation
edge denoise precision、fn candidate precision、ball partition diagnosis 都直接阻止了低价值 formal 扩展。

### Why It May Matter
未来实验应先做机制诊断，减少盲目调参。

### Required Verification
每个新方向必须预定义机制指标。

### Possible Paper Use
diagnostic

## PS-5: Noise Gate Preserves Noisy Gain and Avoids Clean Cora Regression

### Status
likely

### Related Directions
DIR-05

### Observation
formal Cora clean gated 等于 GRACE，noise=0.4 gated 等于 always-on pct。

### Why It May Matter
解决 always-on clean graph risk。

### Required Verification
gate 触发不能依赖已知噪声标签；多数据集验证。

### Possible Paper Use
ablation

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GB view generation schedule/freeze 微调 | 长训练仍低于 GRACE | `outputs/cora_long_comparison`, `outputs/cora_hierarchy_freeze_round` | 多数据集 5 seeds 超过 GRACE，且增强多样性诊断成立 |
| full-ball / ratio negative suppression | 候选过宽，误伤有效负样本 | `outputs/fn_grace_v2_mainline_round/summary.csv` | 候选置信度显著提升，并超过 topk/no-ball |
| v2.1/v2.2/v2.3 公式微调 | 复杂度增加但无增益 | `outputs/cora_fn_grace_v21_round`, `v22`, `v23` | 明确机制瓶颈和跨数据集收益 |
| pure same/cross-ball edge denoise | 噪声命中弱、recall 低、clean collateral | `outputs/edge_denoise_precision_round` | GB edge score AUC/AP 超过 percentile |
| reliable-ball topology pct70/80/90 | 删边不改球划分 | `outputs/ball_partition_diag_round` | assignment change 明显且 accuracy 正向 |
| clean graph always-on percentile | clean Cora 退化 | `outputs/clean_graph_risk_round` | 无监督 gate 或 clean collateral 明显降低 |
| 把 Amazon partial 当 formal | 单 seed、协议与文献不同 | `outputs/amazon_protocol_check_round` | 当前协议下多 seed formal 或单独文献协议复现 |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB view generation | GB 可描述局部结构和扰动统计 | 作为 diagnostic 或 augmentation guard |
| Hard/full same-ball suppression | 同球可能包含 false negatives | Top-k soft + confidence + no-ball control |
| Pure cross-ball denoise | 跨球边略富集噪声 | 作为 edge scorer feature |
| Reliable topology filtering | 构球 topology 可能影响边界节点 | 先提高 topology term 或用 boundary-distance 诊断 |
| Always-on percentile | 低相似边富集噪声 | noise-gated 或 robust-only claim |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same config, same split, same seeds, same eval。
- Multi-seed evidence: at least 5 seeds for small/medium graphs；PubMed minimum 3 seeds only if cost documented。
- Multi-dataset evidence: not Cora-only。
- Ablation removing GB signal: include no-ball or no-GB version。
- Mechanism diagnostic: candidate precision/overlap, ball stats, or edge hit-rate。
- No label leakage: labels/noise-origin only post-hoc unless explicitly oracle。

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation when comparing a new GB mechanism.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as reliable semantic positive without confidence or soft weighting.
- Every new direction must define the exact GB role before coding.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing the GB signal.
- If improvement is smaller than overlapping std, mark it `inconclusive` or `weak`.
- If a direction only works for one seed, do not move it into the paper main line.
- If clean graph gain is tiny, test noisy graph or strong augmentation before abandoning robustness claims.
- If complexity increases, require stronger evidence than a simple baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.
- Always include no-ball Top-k when claiming GB helps false-negative suppression.
- Always report whether ball partitions are too coarse or too fragmented.
- Never promote post-hoc label/noise-origin diagnostics into training rules without marking oracle/leakage risk.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-01 | GB controls views | Active but long-train weaker than GRACE | View distribution/objective mismatch | Diagnostics and negative result | abandon |
| DIR-02 | GB scopes Top-k false negatives | Best GB mainline but small gains | GB adds little beyond similarity Top-k | Main-method candidate with conservative claims | modify |
| DIR-03 | GB with BGRL/GBT | Fails on Cora | Objective/view mismatch | Negative support for minimal correction | abandon |
| DIR-04 | GB same/cross-ball edge denoise | Pure GB weak; hybrid below percentile | Weak noisy-edge separator | Edge diagnostics | abandon |
| DIR-05 | Percentile edge robustness | Strong noisy-graph signal | Clean Cora collateral | Main robustness candidate | continue |
| DIR-06 | Diagnostics/reliable topology | Explains failures | Not a method | Essential experiment memory | diagnostic-only |

# 9. Recommended Next Directions

## NEXT-DIR-1: Conservative GB-FN Robustness Recheck

### Why This Direction Is Recommended
It is the smallest way to determine whether GB-FN adds value beyond no-ball Top-k under conditions where structure may matter more.

### Previous Failure It Addresses
Clean graph GB-FN gains are too small and no-ball Top-k nearly matches it.

### Core Hypothesis
GB candidate scope may be more useful under edge noise or strong augmentation, where pure similarity Top-k becomes greedier or less stable.

### GB Role
Loss-side candidate prior only。

### What Must Be Different From Failed Directions
No view-generation changes, no formula micro-tuning, mandatory no-ball baseline。

### Minimal Implementation
Reuse GB-FN v2 topk alpha=0.25; run GRACE, no-ball topk, GB-FN on Cora/CiteSeer edge_noise=0.4。

### Minimal Datasets and Seeds
Cora and CiteSeer, 5 paired seeds。

### Success Criteria
GB-FN beats both GRACE and no-ball with meaningful paired delta and stable candidate diagnostics。

### Failure Criteria
Tie with no-ball/GRACE or improvement within overlapping std。

### Paper Claim If Successful
GB provides a conservative structural prior for false-negative suppression under noisy graph conditions。

### Reason to Stop Early
If first dataset shows no difference from no-ball and candidate overlap >95%。

## NEXT-DIR-2: Unsupervised Noise Gate for Percentile Edge Weighting

### Why This Direction Is Recommended
Percentile has the strongest noisy signal but clean Cora regression blocks broad claims.

### Previous Failure It Addresses
Always-on percentile hurts clean Cora。

### Core Hypothesis
A label-free gate based on graph/edge similarity distribution can disable percentile weighting on clean graphs.

### GB Role
None or diagnostic-only；do not sell as GB unless GB feature is explicitly used and ablated。

### What Must Be Different From Failed Directions
No known noise labels at train time。

### Minimal Implementation
Compute unsupervised distribution statistic; gate pct on/off; compare to oracle gate and always-on。

### Minimal Datasets and Seeds
Cora clean/noise=0.4, CiteSeer clean/noise=0.4, 5 seeds。

### Success Criteria
Matches GRACE on clean and retains pct gain on noisy。

### Failure Criteria
False positive gate on clean or false negative gate on noisy。

### Paper Claim If Successful
Simple unsupervised edge-reliability gating improves robust GCL under edge noise。

### Reason to Stop Early
If gate statistic cannot separate clean/noisy on Cora without labels。

## NEXT-DIR-3: GB as Edge-Scorer Feature, Not Rule

### Why This Direction Is Recommended
Pure GB edge denoise is weak, but GB may still help as a feature combined with x-space statistics.

### Previous Failure It Addresses
same/cross-ball hard rule has low recall and high collateral damage。

### Core Hypothesis
GB features such as boundary distance, ball size, or confidence may improve a non-oracle edge reliability score.

### GB Role
Feature in edge scorer, fully ablated。

### What Must Be Different From Failed Directions
Do not use fixed same/cross-ball weights; compare against percentile-only。

### Minimal Implementation
Compute edge score from cosine percentile plus GB boundary/confidence features; no labels in training if unsupervised。

### Minimal Datasets and Seeds
Cora edge_noise=0.4 quick first, then 5 seeds only if edge AUC/proxy improves。

### Success Criteria
Better noisy-edge diagnostic than percentile-only and better formal accuracy。

### Failure Criteria
GB feature does not improve edge diagnostic。

### Paper Claim If Successful
GB diagnostics can enrich edge reliability estimation。

### Reason to Stop Early
If GB feature importance/ablation is zero or harmful。

## NEXT-DIR-4: Ball Partition Stability Diagnostic

### Why This Direction Is Recommended
Many failures depend on too coarse or too fragmented balls, but diagnostics are incomplete.

### Previous Failure It Addresses
GB-FN uses 1-2 balls while hybrid denoise fragments into thousands。

### Core Hypothesis
There is a usable middle regime of ball granularity; outside it, GB becomes no-op or noise.

### GB Role
Diagnostic-only。

### What Must Be Different From Failed Directions
No new model until diagnostic thresholds are known。

### Minimal Implementation
Log ball count, singleton ratio, size distribution, assignment stability, candidate overlap for existing checkpoints。

### Minimal Datasets and Seeds
Cora/CiteSeer/PubMed existing runs。

### Success Criteria
Clear relationship between ball granularity and method effect。

### Failure Criteria
No consistent relationship。

### Paper Claim If Successful
GB usefulness depends on a measurable granularity regime。

### Reason to Stop Early
If existing logs lack enough assignment snapshots and reruns are too costly。

# 10. Paper Claim Implications

## Claim-1

### Claim
GB provides robust local structure prior for node-level GCL.

### Supported By Which Directions
DIR-02 weakly; DIR-06 diagnostics partially。

### Weakened By Which Directions
DIR-01, DIR-04, no-ball results in DIR-02。

### Current Evidence
Cora/CiteSeer GB-FN small positive; PubMed tie；no-ball nearly same。

### Missing Evidence
Significant gains, stronger candidate-quality proof, larger graph evidence。

### Required Experiments
Paired multi-seed GB-FN vs no-ball under clean/noisy settings。

### Risk
High if claim is written strongly。

### Current Readiness
weak

## Claim-2

### Claim
GB-guided augmentation preserves local semantics.

### Supported By Which Directions
DIR-01 short stability only。

### Weakened By Which Directions
DIR-01 long/decay/freeze/anchor-explore。

### Current Evidence
Long training lower than GRACE。

### Missing Evidence
Multi-dataset positive view-generation results。

### Required Experiments
Only if direction revived；currently not recommended。

### Risk
Very high。

### Current Readiness
not ready

## Claim-3

### Claim
Weak-view GB relation stabilizes strong-view contrast.

### Supported By Which Directions
GB-GBT weak view was more stable structurally。

### Weakened By Which Directions
GB-GBT accuracy low; anchor-explore v2 low。

### Current Evidence
Structural stability did not improve downstream performance。

### Missing Evidence
Accuracy gain and objective diagnostics。

### Required Experiments
Healthy vanilla BGRL/GBT baseline first。

### Risk
High。

### Current Readiness
not ready

## Claim-4

### Claim
GB denoising improves robustness under graph noise.

### Supported By Which Directions
DIR-04 weakly only as hybrid vs GRACE。

### Weakened By Which Directions
pure GB edge denoise weak; percentile stronger。

### Current Evidence
Same/cross-ball denoise precision/recall insufficient。

### Missing Evidence
GB-specific edge scorer outperforming percentile。

### Required Experiments
GB-feature edge scorer with percentile-only ablation。

### Risk
High。

### Current Readiness
not ready

## Claim-5

### Claim
Hard same-ball positive or broad same-ball suppression is not reliable for node-level GCL.

### Supported By Which Directions
DIR-02, DIR-01。

### Weakened By Which Directions
None in current project。

### Current Evidence
hard/full/ratio weaker than topk; view generation weaker long-term。

### Missing Evidence
More datasets for hard variant。

### Required Experiments
Optional; current negative evidence enough for discussion。

### Risk
Low if scoped to current project。

### Current Readiness
moderate

## Claim-6

### Claim
GB is more useful as an auxiliary prior than as a direct replacement for contrastive positives or views.

### Supported By Which Directions
DIR-01 negative, DIR-02 weak positive, DIR-03 negative。

### Weakened By Which Directions
No-ball Top-k near tie weakens GB-specific strength but not auxiliary-prior framing。

### Current Evidence
Loss-side minimal correction is safest GB method in current project。

### Missing Evidence
Stronger statistical separation。

### Required Experiments
GB-FN vs no-ball paired tests and diagnostics。

### Risk
Medium。

### Current Readiness
moderate

## Claim-7

### Claim
X-space percentile edge weighting improves robustness under edge noise.

### Supported By Which Directions
DIR-05。

### Weakened By Which Directions
clean Cora regression。

### Current Evidence
Noisy settings 8/8 positive; Cora noisy delta monotonic；noise-gate formal Cora clean/noisy works。

### Missing Evidence
Unsupervised gate and broader clean datasets。

### Required Experiments
Gate without known noise labels。

### Risk
Medium。

### Current Readiness
moderate

# 11. TODO List

## Missing raw logs

- Summarize selected raw `train.log` anchors for DIR-01 and DIR-02 if paper needs exact epoch behavior.
- Link final paper tables to exact run directories; `outputs/final_main_results.json` mentioned in `PAPER_PLAN.md` is not present in current file list.

## Missing baseline checks

- Verify exact train/val/test split seeds for all formal Cora/CiteSeer/PubMed comparisons.
- Separate current unified Amazon protocol from literature protocol in any paper table.

## Missing seeds/std

- Amazon-Photo and Amazon-Computers GB-FN evidence is 1 seed only.
- Coauthor-CS is only 1 epoch sanity.
- PubMed formal is 3 seeds, not 5 seeds.

## Missing datasets

- No formal Coauthor-CS/Physics result for GB-FN.
- No formal Amazon multi-seed result under current protocol.

## Missing ablations

- GB-FN mainline lacks complete re-run with selected similarity metric comparable to no-ball.
- Percentile noise gate lacks unsupervised gate ablation.
- GB edge scorer as feature has not been tested.

## Missing GB diagnostics

- Ball radius mean/std unavailable.
- Singleton ball ratio unavailable for key GB-FN runs.
- Cross-epoch assignment stability unavailable except ball partition diagnostic subset.
- Cross-seed partition stability unavailable.

## Missing visualizations

- Candidate overlap/stability plots for GB-FN vs no-ball.
- Ball granularity distribution across datasets.
- Clean collateral vs noisy recall plots for percentile/default/gated.

## Missing explanation for failed directions

- GB-BGRL collapse needs representation variance/collapse metrics if discussed in paper.
- GB-GBT needs vanilla GBT baseline before attributing failure to GB.

## Missing paper-claim evidence

- Strong GB-specific gain over no-ball Top-k is missing.
- Strong GB candidate-purity gain is missing.
- Pure GB edge-denoise robustness evidence is missing.
- Clean/noisy automatic gate evidence is missing.

# 12. Experiment 7 Linkage Note

- Added on 2026-05-19: `02_experiment_history_and_lessons 7 .md` now records `实验7noisy_label_gb_grace` as a separate direction-level history.
- Do not merge that file into this document's direction count or DIR numbering; this document remains a separate historical review.
- Relevant new conclusion: frozen GRACE + strict GB local label smoothing under noisy train labels is Cora-positive only in selected medium-label high-noise settings and is cross-dataset unstable on CiteSeer.
- Future constraint imported from Experiment 7: any noisy-label GB direction must report effective smoothing coverage and must beat CE/global/kmeans/topk_matched, not only global smoothing.
