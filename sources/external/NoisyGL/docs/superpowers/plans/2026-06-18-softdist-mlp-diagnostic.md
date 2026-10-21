# SoftDist MLP Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 验证 SoftDist 的粒球软目标在分类器不接收冻结表示 H、且不使用图卷积时，是否仍带来独立于 CE 和 shuffled control 的增量。

**Architecture:** 冻结 GRACE 表示 H 只用于构建粒球和软目标，分类器输入严格为原始特征 X。沿用现有 `_FrozenGraceGBGCNBase` 训练、评估和 SoftDist 目标代码，仅新增 MLP backbone 与三种可归因方法开关。

**Tech Stack:** Python, PyTorch, PyTorch Geometric, unittest/pytest, YAML

---

## Experiment Preregistration

- **研究假设：** 若 SoftDist 的收益来自粒球共识而非 `[X||H]` 表示增强，则 `SoftDist-MLP(X)` 应优于匹配的 `MLP(X)+CE`，并优于保持相同球统计但打乱球成员关系的 shuffled control。
- **粒球角色：** 基于冻结 H 构建的局部语义关系单元和置信度加权软标签教师，不作为硬标签。
- **方法动机：** 去除分类器侧 H 和图消息传递，隔离粒球软目标本身的贡献。
- **固定协议：** Cora/CiteSeer，percent split 0.2/0.1/0.7，uniform noise 0.4，首轮 seed 0，MLP 两层、hidden 64、dropout 0.5、训练 200 epochs。
- **三组方法：** `mlp_ce`、`frozengrace_gb_mlp_softdist`、`frozengrace_gb_mlp_softdist_shuffled`。
- **预期机制：** 纯球中的多数标签通过 `alpha=(margin-1/C)/(1-1/C)` 控制软目标强度，在不向分类器暴露 H 的条件下抑制局部噪声标签。
- **成功标准：** 两个数据集上 seed 0 的真实 SoftDist 均同时高于 CE 与 shuffled，才扩展 seeds `0,1,2,3,4`；5 seeds 后真实方法对两个控制的均值优势应不小于对应波动量级。
- **失败标准：** 任一数据集上真实 SoftDist 不优于 CE，或不优于 shuffled，则不扩 seed，并判定当前粒球信号不能脱离强下游稳定成立。
- **实验结果：** seed 0 在 Cora/CiteSeer 均同时优于 CE 和 shuffled，按规则扩展到 5 seeds；5-seed 结果见下方 Observed Results。
- **最终决策：** 继续。当前证据支持“粒球软目标的收益不依赖把 H 输入分类器，也不依赖图卷积”，下一阶段应验证噪声类型/噪声率与复杂度，而不是继续调当前两数据集。

### Task 1: MLP Interface Compatibility

**Files:**
- Modify: `predictor/module/GNNs.py`
- Test: `tests/test_mlp_softdist.py`

- [x] **Step 1: Write the failing test**

```python
def test_mlp_accepts_and_ignores_adjacency():
    model = MLP(3, 4, 2, 2, dropout=0.0, use_bn=False).eval()
    features = torch.randn(5, 3)
    adjacency = torch.tensor([[0, 1], [1, 0]])
    assert torch.allclose(model(features), model(features, adjacency))
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mlp_softdist.py::TestMLPSoftDist::test_mlp_accepts_and_ignores_adjacency -q`

Expected: FAIL because `MLP.forward` accepts only `x`.

- [x] **Step 3: Write minimal implementation**

Change the signature to `def forward(self, x, adj=None):`; no computation may use `adj`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mlp_softdist.py::TestMLPSoftDist::test_mlp_accepts_and_ignores_adjacency -q`

Expected: PASS.

### Task 2: SoftDist MLP Predictor Variants

**Files:**
- Modify: `predictor/FrozenGraceGB_GCN_Predictor.py`
- Modify: `single_exp.py`
- Test: `tests/test_mlp_softdist.py`

- [x] **Step 1: Write the failing switch test**

Assert that all three predictor classes exist and that they use `backbone="mlp"`, `use_xh_features=False`; CE skips ball targets, real SoftDist uses non-random soft targets, and shuffled SoftDist uses random-control soft targets.

- [x] **Step 2: Run the switch test to verify it fails**

Run: `python -m pytest tests/test_mlp_softdist.py::TestMLPSoftDist::test_predictor_variants_isolate_soft_targets -q`

Expected: FAIL because the predictor classes do not exist.

- [x] **Step 3: Add the MLP backbone and variants**

Import `MLP`, instantiate it in the `backbone == "mlp"` branch, and add `mlp_ce_Predictor`, `frozengrace_gb_mlp_softdist_Predictor`, and `frozengrace_gb_mlp_softdist_shuffled_Predictor`. Register all three in `single_exp.py`.

- [x] **Step 4: Run the switch test**

Run: `python -m pytest tests/test_mlp_softdist.py::TestMLPSoftDist::test_predictor_variants_isolate_soft_targets -q`

Expected: PASS.

### Task 3: Matched Configurations

**Files:**
- Create: `config/mlp_ce/mlp_ce_{cora,citeseer}.yaml`
- Create: `config/frozengrace_gb_mlp_softdist/frozengrace_gb_mlp_softdist_{cora,citeseer}.yaml`
- Create: `config/frozengrace_gb_mlp_softdist_shuffled/frozengrace_gb_mlp_softdist_shuffled_{cora,citeseer}.yaml`
- Test: `tests/test_mlp_softdist.py`

- [x] **Step 1: Write the failing config test**

For each method/dataset pair, load with `load_conf(None, method, dataset)` and assert the configured method and shared MLP hyperparameters. SoftDist configurations must also contain `target_mode: distribution` and the same ball parameters.

- [x] **Step 2: Run the config test to verify it fails**

Run: `python -m pytest tests/test_mlp_softdist.py::TestMLPSoftDist::test_all_smoke_configs_load_with_matched_mlp_hyperparameters -q`

Expected: FAIL because the configuration files do not exist.

- [x] **Step 3: Add six configurations**

Use `n_hidden=64`, `n_layer=2`, `dropout=0.5`, `lr=1e-2`, `weight_decay=5e-4`, and `n_epochs=200`. Reuse each dataset's existing SoftDist embedding path and ball parameters for both real and shuffled methods.

- [x] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_mlp_softdist.py -q`, then `python -m pytest tests -q`.

Expected: all tests PASS, apart from the existing PyG deprecation warning.

### Task 4: Seed-0 Smoke and Decision

**Files:**
- Create: `runs/softdist_mlp_seed0_20260618/softdist_mlp_seed0_summary.csv`
- Create: `runs/softdist_mlp_5seed_20260618/softdist_mlp_5seed_summary.csv`
- Create: `runs/softdist_mlp_5seed_20260618/softdist_mlp_5seed_raw.log`
- Create: `runs/softdist_mlp_5seed_20260618/softdist_mlp_report.md`
- Modify: `docs/superpowers/plans/2026-06-18-softdist-mlp-diagnostic.md`

- [x] **Step 1: Run the preregistered smoke**

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' scripts/run_ssl_swap_downstream.py `
  --embedding_root 'C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2' `
  --datasets cora,citeseer `
  --ssl_methods grace `
  --base_methods mlp_ce,frozengrace_gb_mlp_softdist,frozengrace_gb_mlp_softdist_shuffled `
  --seeds 0 --noise_type uniform --noise_rate 0.4 --device cuda:0 `
  --output runs/softdist_mlp_seed0_20260618/softdist_mlp_seed0_summary.csv
```

- [x] **Step 2: Apply the decision rule**

Read the six rows and compare within each dataset. Expand to seeds `0,1,2,3,4` only if real SoftDist beats both controls on both datasets; otherwise record the failed condition and stop.

- [x] **Step 3: Record results and decision**

Replace the preregistration result and decision lines with the observed values, preserving raw CSV and the exact command.

## Observed Results

Test accuracy, mean +/- sample std over seeds `0,1,2,3,4`:

| Dataset | MLP(X)+CE | SoftDist-MLP(X) | Shuffled-SoftDist-MLP(X) | SoftDist - CE | SoftDist - shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 45.88 +/- 3.44 | **54.63 +/- 1.46** | 46.51 +/- 1.92 | +8.75 (5/5 wins) | +8.12 (5/5 wins) |
| CiteSeer | 44.53 +/- 4.11 | **52.30 +/- 2.84** | 44.93 +/- 3.46 | +7.77 (5/5 wins) | +7.37 (5/5 wins) |

The 5-seed expansion command was identical to the smoke command except for `--seeds 0,1,2,3,4` and output path `runs/softdist_mlp_5seed_20260618/softdist_mlp_5seed_summary.csv`. A deterministic rerun produced the same rows and was captured in `softdist_mlp_5seed_raw.log`.

**Supported conclusion:** under the fixed Cora/CiteSeer, uniform-0.4, 20/10/70 split protocol, semantic ball soft targets improve an X-only MLP over both CE and shuffled membership controls.

**Reasonable interpretation:** the gain cannot be attributed to exposing frozen H to the downstream classifier or to graph message passing. The near-CE shuffled results indicate that the semantic ball assignment, rather than generic soft-label smoothing, carries the useful signal.

**Not yet supported:** the pipeline is not H-free end to end because frozen H still constructs balls; no claim is made yet for other noise types/rates, larger datasets, runtime advantage, or SOTA accuracy.

---

## 2026-06-18 Continuation: Noise Spectrum, Efficiency, and H-Free Diagnostic

### P0: MLP Noise Spectrum — Completed

Fixed protocol: Cora/CiteSeer, percent split `0.2/0.1/0.7`, seeds `0-4`;
methods `mlp_ce`, `frozengrace_gb_mlp_softdist`, and size-matched shuffled;
new regimes clean, uniform `0.2/0.5/0.6`, pair `0.4`, and instance `0.4`.
Uniform `0.4` reused the existing verified rows.

- Cora: stable positive increment in all `6/6` noisy regimes.
- CiteSeer: stable positive increment in all `6/6` noisy regimes.
- Clean safety passed on both datasets.
- Full evidence: `runs/softdist_mlp_noise_spectrum_20260618/`.

Decision: the uniform-0.4 result is not an isolated sweet spot. The supported wording is
that the H-ball SoftDist target transfers robustly to an X-only MLP across the tested
noise conditions. This still does not support a classifier-independent or
model-independent claim.

### P1: MLP-Specific Efficiency — Completed

Full evidence: `runs/softdist_mlp_efficiency_20260618/`.

- Cora SoftDist-MLP classifier training: `1.151s`, peak allocated memory `140.3 MiB`;
  matched GCN: `2.854s`, `177.4 MiB`; NRGNN: `25.041s`, `406.4 MiB`.
- CiteSeer SoftDist-MLP classifier training: `1.521s`, `209.0 MiB`;
  matched GCN: `3.031s`, `258.4 MiB`; NRGNN: `31.660s`, `505.5 MiB`.
- SoftDist target construction adds `4.507s` on Cora and `5.739s` on CiteSeer, so
  downstream total time is not lower than plain GCN.
- GRACE pretraining is excluded and remains a reusable one-time cost.

Decision: “lightweight classifier stage without inference-time graph propagation” is
supported relative to GCN/NRGNN training and memory. “End-to-end complexity win,”
“fewer parameters than every GCN,” and “no graph computation” are not supported.

### P2: X-Space H-Free Diagnostic — Stopped at Offline Gate

Implementation:

- Added an in-memory ball-feature interface while preserving the frozen-H wrapper.
- Added `ball_feature_source: grace|x`.
- Added `featuregb_mlp_softdist` and its size-matched shuffled control.
- X is converted to dense float and row-L2 normalized before fixed geometric ball
  construction.
- Parameters remained fixed: `max_balls=64`, `min_ball_size=32`,
  `split_improvement=0.05`; no X-specific tuning was performed.

Observed seed-0 offline diagnostics under uniform `0.4`:

| Dataset | Balls | True-all purity real/shuffled | Noisy-train purity real/shuffled | Eligible coverage | Gate |
|---|---:|---:|---:|---:|---|
| Cora | 1 | 0.3021 / 0.3021 | 0.2403 / 0.2403 | 1.0000 | fail |
| CiteSeer | 1 | 0.2107 / 0.2107 | 0.1955 / 0.1955 | 1.0000 | fail |

Both datasets collapsed to one ball. Therefore real and shuffled assignments are
identical, both purity gaps are `0.0`, and the preregistered `+0.10/+0.05` gates fail.
Seed-0 downstream and 5-seed expansion were not run.

Full evidence: `runs/featuregb_mlp_diagnostic_20260618/`.

Final decision: stop the fixed X-space route and record it as a negative result. Do not
tune ball count, split threshold, alpha, or MLP hyperparameters after seeing this
diagnostic. Topology-ball remains a separate future question and requires a new
preregistration.

## Final Claim Boundary

- Supported: SoftDist does not require feeding H to the classifier and does not require
  GCN message passing; its H-ball soft targets transfer to GCN and MLP.
- Supported: the X-only MLP increment is stable across the tested clean/uniform/pair/
  instance conditions on Cora and CiteSeer.
- Not supported: classifier-independent/model-independent stability, because GAT and
  APPNP include negative CiteSeer evidence.
- Not supported: a completely H-free SoftDist pipeline; fixed X-space construction
  failed before downstream evaluation.
- Not supported: an end-to-end efficiency win; GRACE and target-construction costs must
  remain separate.
- Main performance method remains SoftDist-GCN `[X||H]`; SoftDist-MLP remains the
  mechanism and lightweight-classifier variant.
