# Baseline code and reproduction provenance audit

Audit date: 2026-08-28

Scope: every method that appears as a comparison method or an algorithmic control in Tables 2--4 and the supplementary grouping/representation comparisons of `paper_final_en_v26_advisor_compressed_revised_20260828.md`.

## Status definitions

- **Official repository**: a repository maintained by the paper authors or their institution and explicitly associated with the paper.
- **Official-code integration**: the submitted experiment imports or mirrors a defined path from the official code, while adapting the surrounding data and model pipeline.
- **Common-backbone reimplementation**: the published algorithm is reimplemented under the paper's shared MLP/teacher protocol.
- **Protocol port**: the core objective is retained, but the original architecture, data split, noise protocol, or optimization setup is changed.
- **Mechanism adaptation**: only the central mechanism is retained. This is not an official or full paper reproduction.

## Table 2: noisy-label methods with a matched deployed MLP

| Method | Author-maintained code | Code source | Identity of the results in this paper | Audit conclusion |
|---|---|---|---|---|
| MLP+CE | Not applicable | Standard project baseline | Unified project implementation | No external reproduction claim is needed. |
| Small-loss | No canonical standalone paper/repository | Project implementation | Generic per-sample CE ranking with a forget-rate schedule | Treat as a defined strategy, not as an officially reproduced named paper. |
| Co-teaching | Yes | https://github.com/bhanML/Co-teaching | Common-MLP reimplementation using a NoisyGL-derived schedule/configuration | The original paper has official code, but the table does not run that repository directly. |
| JoCoR | Yes | https://github.com/hongxin001/JoCoR | Common-MLP reimplementation using a NoisyGL-derived schedule/configuration | Official code exists; current results are a matched-backbone reimplementation. |
| SCE | Yes | https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels | Local implementation of CE + reverse CE | Official code exists; current results are an objective-level reimplementation. |
| ELR | Yes | https://github.com/shengliu66/ELR | Local implementation of temporal target history and the ELR penalty | Official code exists; current results are an objective-level reimplementation. |
| LogitClip | Yes | https://github.com/hongxin001/LogitClip | Local implementation of L2 logit clipping followed by CE | Official code exists; current results are an objective-level reimplementation. |
| GB-LCSS | This paper | Submission repository | Current formal implementation | Primary method. |

NoisyGL is itself an official benchmark repository (https://github.com/eaglelab-zju/NoisyGL) and contains benchmark reimplementations of Co-teaching, SCE, and JoCoR. It is not the original authors' repository for those methods.

## Table 3: structure-assisted X-only methods

| Method | Author-maintained code | Code source | Identity of the results in this paper | Audit conclusion |
|---|---|---|---|---|
| PosteL | Yes | https://github.com/ml-postech/PosteL | Official target-construction path integrated into the project; iterative pseudo-labeling disabled | This is the closest row to an official-code integration, but it is still not the full official training pipeline. |
| ALS | No author repository located | Paper: https://arxiv.org/abs/2108.13555 | Mechanism adaptation of label propagation, global class relevance, KL regularization, and pacing | Paper-derived implementation; not official reproduction. |
| SALS | No author repository located | Paper: https://arxiv.org/abs/2112.00499 | Published target equation adapted to the common X-only MLP | Paper-derived mechanism adaptation; not official reproduction. |
| Graph-MLP | Yes | https://github.com/yanghu819/Graph-MLP | NContrast protocol port with a matched MLP and the paper's noise protocol | Official code exists; table values are not official-repository results. |
| GSDN | No author repository located | Paper: https://arxiv.org/abs/2210.02097 | Project mechanism adaptation of feature- and label-level neighborhood self-distillation | Highest provenance risk in Table 3; do not call it an official or faithful reproduction. |
| GB-LCSS | This paper | Submission repository | Current formal implementation | Primary method. |

The searches for ALS, SALS, and GSDN covered the paper/arXiv pages, author names, exact titles, GitHub repository search, and public code indexes. No author-maintained repository was found as of the audit date. This is an evidence-of-search statement, not proof that private or unindexed code does not exist.

## Table 4: graph-to-MLP knowledge-transfer routes

| Method | Author-maintained code | Code source | Identity of the results in this paper | Audit conclusion |
|---|---|---|---|---|
| GCN(X)+CE teacher (GLNN-style) | Yes, for GLNN | https://github.com/snap-research/graphless-neural-networks | Project GCN/MLP response-distillation implementation in the style of GLNN | Keep the suffix `GLNN-style`; this is not an official GLNN reproduction. |
| FF-G2M | Yes | https://github.com/LirongWu/FF-G2M | Common-teacher/student reimplementation retaining low- and high-frequency objectives | Stronger than a proxy, but not a direct run of the official repository. |
| KRD | Yes | https://github.com/LirongWu/KRD | Simplified/common-protocol reliability mechanism adaptation | Official code exists; current row is not a full KRD reproduction. |
| VQGraph | Yes | https://github.com/YangLing0818/VQGraph | Soft-codebook mechanism adaptation under the common protocol | Official code exists; current row is not a full VQGraph reproduction. |
| Granular-ball consensus teacher | This paper | Submission repository | Current extension | Proposed extension. |

## Supplementary algorithms and controls

| Component/control | Public source status | Current identity |
|---|---|---|
| GRACE | Author repository: https://github.com/CRIPAC-DIG/GRACE | Frozen-representation component, not a baseline ranking row. |
| GCN | Canonical implementations are public; e.g. https://github.com/tkipf/gcn and the PyTorch port https://github.com/tkipf/pygcn | Local project backbone. |
| kNN | Standard algorithm; scikit-learn implementation/documentation is public | Alternative grouping interface, not a reproduction of a paper-specific method. |
| k-means / 2-means | Standard algorithm; scikit-learn implementation/documentation is public | Granular-ball split primitive and alternative grouping control. |
| Random size-matched grouping | Internal control | No external repository is expected. |
| Uniform/prior smoothing and self-vote ablation | Internal controls | No external repository is expected. |

## Bottom line

1. It is false that PosteL is the only compared method whose original paper has source code. Eleven of the fourteen externally named methods in the main comparison tables have an author-maintained repository: Co-teaching, JoCoR, SCE, ELR, LogitClip, PosteL, Graph-MLP, GLNN, FF-G2M, KRD, and VQGraph.
2. It is approximately true that PosteL is the only row in the current experiments that integrates a specific official-code path. Even PosteL is a partial integration because iterative pseudo-labeling is disabled and the common project MLP/protocol is used.
3. ALS, SALS, and GSDN have no verified author repository in this audit. Their rows are paper-derived adaptations.
4. The current FF-G2M row is a common-backbone reimplementation. Co-teaching, JoCoR, SCE, ELR, and LogitClip are objective/training-rule reimplementations. Graph-MLP is a protocol port. GLNN, KRD, VQGraph, ALS, SALS, and GSDN must not be described as official reproductions.
5. The manuscript's Appendix H is directionally correct, but the main table labels should retain explicit qualifiers wherever a bare method name could be mistaken for official results.
