# Submission Integrity Audit - Phase 3: Method Fidelity

Date: 2026-07-11

## 1. Scope and verdict

This phase asks a different question from the numerical audit: do the method names in Tables 1, 3, and 5 denote implementations that are sufficiently close to the cited algorithms?

The stored numbers remain real outputs of the local code under the audited train-only protocol. However, several rows are not faithful implementations of the papers whose names they use. This is a submission-level problem because a reviewer will interpret a row named `FF-G2M`, `KRD`, `VQGraph`, `CT`, or `ANL` as evidence about that published method, not merely about a locally invented loss inspired by it.

**Overall verdict: FAIL for method-name fidelity in the current manuscript.** The paper must either reimplement/port the defining algorithm or rename/remove the affected rows before submission.

## 2. Fidelity scale

| Grade | Meaning | Permitted manuscript use |
|---|---|---|
| A | Official code or a semantics-preserving port | May use the method name, with protocol/config changes disclosed |
| B | Paper-faithful reimplementation of the defining objectives and training loop | May use the method name with `reimplemented` |
| C | Core-mechanism adaptation that keeps the central idea but changes material details | Must use `adapted core`; not suitable for official-ranking or SOTA claims |
| D | Inspired proxy that omits or replaces a defining component | Must not be presented as the cited method's performance |

## 3. Method-by-method matrix

### 3.1 Main X-only noisy-label table

| Local row | Grade | Finding | Required action |
|---|---:|---|---|
| MLP+CE | A | Standard CE baseline | Keep |
| Small-loss | B | Implements scheduled small-loss selection; it is presented as a criterion rather than a specific paper reproduction | Keep as `Small-loss criterion` |
| Co-teaching | B/C | Two networks exchange small-loss samples; full-batch graph-node adaptation and summed two-network loss change optimization scale | Keep only as `Co-teaching (MLP-adapted)` |
| JoCoR | B/C | Joint CE plus symmetric KL and small-loss selection are present; full-batch adaptation remains | Keep as `JoCoR (MLP-adapted)` |
| ELR | B | EMA prediction target and `CE + lambda log(1-p dot t)` are implemented | Keep with adapted-backbone disclosure |
| SCE | B | CE plus reverse CE is implemented | Keep with adapted-backbone disclosure |
| LogitClip | B | Per-sample L2 logit clipping followed by CE is implemented | Keep with adapted-backbone disclosure |
| CT | D | Local code tracks only an EMA of annotated-class confidence and selects top-k. Official CT tracks every annotated-vs-other confidence-gap time series, applies a Mann-Kendall trend test, and unions CT-selected hard samples with another selector | Remove/rename or faithfully implement |
| ANL | D | Local passive loss is a normalized sum of `-log(1-p_k)` over non-label classes. Official ANL-CE uses `NCE + NNCE`, where NNCE is defined by a vertically flipped and normalized CE with constant `A=-log(p_min)` | Remove/rename or implement official NNCE |
| FINE-adapted | D | Local code centers class features, takes a covariance SVD, and retains a fixed top fraction. Official FINE uses the uncentered class Gram matrix, squared projection on its leading eigenvector, and a per-class two-component GMM/threshold | Rename to an eigenvector-alignment proxy or implement FINE |
| DivideMix-lite | D | Only the two-component loss GMM remains. There is one network, no cross-network division, no MixMatch, no co-refinement, and no co-guessing. The self-pseudo-label term uses the same detached prediction and has essentially zero gradient (`max |grad| = 1.12e-8` in an audit probe) | Do not interpret as DivideMix; rename to `GMM loss reweighting` or implement DivideMix |

Local evidence: `predictor/MLPNoiseBaselines_Predictor.py`, especially lines 258-277 (CT), 310-316 and 45-64 (ANL), 92-138 and 317-334 (FINE/DivideMix-lite).

Primary sources:

- CT: [AAAI 2025 official paper](https://ojs.aaai.org/index.php/AAAI/article/view/34180), which defines confidence-gap trajectories and Mann-Kendall trend testing.
- ANL: [NeurIPS 2023 official paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html), Eqs. 8-12.
- FINE: [NeurIPS 2021 official paper](https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html), Algorithm 1.
- DivideMix: [ICLR 2020 official OpenReview entry](https://openreview.net/forum?id=qBbAV8Wg90K).

### 3.2 Graph-free family table

| Local row | Grade | Finding | Required action |
|---|---:|---|---|
| Graph-MLP (adapted) | D | Local contrastive loss samples one edge positive and one random negative. Official NContrast uses each node's r-hop neighbors as positives and the remaining batch nodes as negatives, with the paper's dedicated LayerNorm/GELU/prediction-head architecture | Rename to `edge-contrastive MLP proxy` or port official Graph-MLP |
| GSDN (adapted) | C | Feature- and label-level neighborhood self-distillation are represented, but the local implementation uses two complete MLPs and locally designed KL/negative terms instead of the paper's shared backbone, two inference heads, and exact batch objectives | Keep only as `GSDN-inspired/adapted core`; do not use as exact method ranking |
| GLNN-style | C/B | All-node GNN-to-MLP soft-target distillation is present. The noisy-label adaptation intentionally removes the original hard-label CE term | The current `GLNN-style` wording is acceptable if this deviation remains explicit |
| FF-G2M (adapted) | D | Local LFD is MSE between edge-wise prediction dot products; local HFD is MSE between scalar L2 prediction differences. Official LFD is neighborhood KL from teacher target-node prediction to student neighbor prediction; official HFD is KL between distributions of element-wise absolute hidden-representation differences | Remove from named comparison or implement Eqs. 6-8 of the paper |
| KRD (adapted) | D | Entropy perturbation reliability is present, but local code globally keeps a fixed top fraction. Official KRD learns a power-law sampling probability, updates its power dynamically with teacher-student agreement and momentum, samples reliable knowledge from each target node's neighborhood, and applies multi-teacher distillation | Remove from named comparison or implement the dynamic neighborhood sampler and KRD loss |
| VQGraph (adapted) | D | Local code jointly learns a soft codebook with a simple hidden-to-quantized MSE and sampled edge BCE. Official VQGraph separately pretrains a VQ-VAE graph tokenizer with nearest-code assignment, node and adjacency decoders, classification, codebook and commitment losses; it then freezes the tokenizer and distills soft code assignments plus class losses | Remove from named comparison or port the official tokenizer/distillation pipeline |
| LCSS distillation | N/A | This is the paper's own adapted GLNN-style extension | Keep, but compare only against baselines whose resource/fidelity levels are explicit |

Primary sources:

- [Graph-MLP paper](https://arxiv.org/abs/2106.04051)
- [GSDN paper](https://arxiv.org/abs/2210.02097)
- [FF-G2M, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26232)
- [KRD, ICML 2023](https://proceedings.mlr.press/v202/wu23m.html)
- [VQGraph, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html)

### 3.3 Graph-dependent reference table

| Local row | Grade | Finding | Required action |
|---|---:|---|---|
| TFR-GCN | A/B | Ported from the official TFR repository at commit `d7e9ff8a58d457b519f068c3d170cd37579c8bd8`; `torch_scatter` mean was replaced with an equivalent `index_add_` mean. The official repository has no Cora/CiteSeer configs, so the DBLP GCN config was transferred | Keep as `official-code port; DBLP config transferred`, not as official tuned performance |
| ProCon | B/C | The paper's peer similarity, two-hop topology factor, prototype pseudo-labels, GMM selection, EMA prototypes, and BGRL-style SSL are present. Graphs above a threshold use a local LSH-bounded approximate peer search rather than the paper's full similarity matrix | Keep as `paper reimplementation`; disclose approximate peer search and protocol adaptation |
| GD2 | B | Ego/structure/mixed views, prediction discrepancy, two-GMM clean selection, pseudo-label purification, and view-specific losses match the paper's central pipeline | Keep as `paper reimplementation`, not official code |
| NRGNN | Not yet graded here | Existing NoisyGL method; separate provenance audit still required | Do not infer official fidelity from this phase |
| NRGNN[X||H] | N/A | Local feature-input modification, not an official NRGNN variant | Always label as the paper's controlled adaptation |

Primary sources:

- TFR official repository: [eaglelab-zju/TFR](https://github.com/eaglelab-zju/TFR)
- ProCon: [IJCAI 2025 official paper](https://www.ijcai.org/proceedings/2025/623)
- GD2: [NeurIPS 2025 official paper](https://papers.nips.cc/paper_files/paper/2025/hash/c132c02176577c4319a878f6417a331a-Abstract-Conference.html)

## 4. Submission-critical consequences

### P0-1: Table 3 currently attributes proxy results to published methods

The manuscript sentence that FF-G2M, KRD, and VQGraph "significantly raise the strength of the graph-free family" is not supported by faithful implementations. The rows demonstrate that three local proxies can run under the protocol; they do not establish the performance of FF-G2M, KRD, or VQGraph.

Safe choices:

1. Preferred: port the official implementations and rerun the same five-seed protocol.
2. If time is insufficient: remove the three quantitative rows and retain the papers in Related Work.
3. Least desirable: rename them as `FF-G2M-inspired frequency proxy`, `KRD-inspired reliability proxy`, and `VQGraph-inspired token proxy`, state that they are not reproductions, and exclude them from ranking language.

### P0-2: Four named main-table baselines are not the cited algorithms

`CT`, `ANL`, `FINE-adapted`, and `DivideMix-lite` cannot collectively support the claim of comparison against those four published methods. This does not invalidate the LCSS raw numbers, but it weakens the central baseline claim and creates an avoidable reviewer-reproducibility failure.

At minimum, rename the rows to their actual mechanisms:

- `EMA-confidence top-k`
- `NCE + normalized complementary loss proxy`
- `classwise covariance-eigenvector top-k`
- `GMM loss reweighting`

These renamed rows may remain as ablations/proxies, but they should not be counted as published-method baselines.

### P0-3: The bibliography itself discloses an unfinished audit

Reference [25] is still a placeholder even though the exact AAAI 2025 paper is identifiable. References [28], [30]-[33], [35], and [36] also contain placeholders or unverified status. A submission with these entries is not ready.

The complete CT citation is:

> W. Pan, W. Wei, F. Zhu, and Y. Deng, "Enhanced Sample Selection with Confidence Tracking: Identifying Correctly Labeled Yet Hard-to-Learn Samples in Noisy Data," Proceedings of the AAAI Conference on Artificial Intelligence, 39(19), 19795-19803, 2025. DOI: 10.1609/aaai.v39i19.34180.

## 5. What remains valid

- The numerical audit still certifies that Tables 1 and 3-5 are recomputable from stored five-seed rows under the stated train-only label-noise protocol.
- LCSS versus MLP+CE and versus shuffled grouping remains a valid controlled comparison.
- Comparisons against formula-faithful SCE, ELR, LogitClip, and the core Co-teaching/JoCoR adaptations remain informative when the adapted MLP backbone is disclosed.
- TFR, ProCon, and GD2 can remain as explicitly qualified reference rows at their audited fidelity levels.
- No evidence in this phase suggests that the LCSS implementation itself fabricated results. The failure is primarily in baseline naming and interpretation.

## 6. Next audit step

Audit the provenance of every frozen GRACE embedding: confirm that pretraining never reads labels, that dataset node ordering matches NoisyGL exactly, that each `H.pt` has the expected row count and finite normalized values, and that no embedding from a different split/dataset is silently reused.
