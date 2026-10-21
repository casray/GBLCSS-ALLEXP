# Phase 20 Custom SSL Report: GRACE + Label-Free Uniformity

Date: 2026-06-15  
Executor: Codex  
Plan: `external/NoisyGL/docs/execution_plan_custom_ssl_20260612.md`

## 1. Research Record

- Research hypothesis: GRACE is suitable for SoftDist [X||H] partly because InfoNCE encourages uniformity. Adding an explicit, label-free uniformity regularizer might produce embeddings that are more separable for granular-ball construction.
- Granular-ball role: downstream geometric diagnostic target. Balls are not used during pretraining; they only evaluate whether the new representation remains useful for SoftDist.
- Method motivation: strengthen the geometry property already suspected to make GRACE useful, without introducing labels, train masks, noisy labels, or ball-vote labels into pretraining.
- Code changes:
  - `gb_graph_aware_minival_6_1/scripts/pretrain_ssl_candidates.py`: added `grace_uniformity`, `uniformity_loss`, and output names `grace_unif0p1`, `grace_unif0p5`, `grace_unif1`.
  - `gb_graph_aware_minival_6_1/tests/test_ssl_candidates.py`: added finite/backprop, deterministic subsampling, lambda-zero regression, and output-name tests.
  - `external/NoisyGL/scripts/diagnose_embedding_geometry.py`: added Phase 20a geometry/probe diagnostic.
  - `external/NoisyGL/tests/test_embedding_geometry_diagnostic.py`: added summary and gate-decision tests.
- Config changes:
  - lambda grid: `{0.1, 0.5, 1.0}`
  - `uniformity_t = 2.0`
  - `uniformity_max_nodes = 2048` for memory-bounded deterministic node sampling.
  - Downstream split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`.
- Expected mechanism: explicit uniformity should reduce embedding crowding and improve ball purity or at least preserve GRACE-level ball count.
- Success standard for Phase 20a:
  - candidate `num_balls_mean >= 0.5 * GRACE num_balls_mean`;
  - candidate `true_all_purity_weighted_mean >= GRACE`;
  - probe accuracy is reference only and is not used for pass/fail.
- Failure standard:
  - all candidates fail either non-collapse or purity gate.
  - If Phase 20a fails, Phase 20b downstream training is not run.
- Final decision: abandon this custom SSL line; do not run Phase 20b.

## 2. Implementation Note

The first full run hit a ROCm OOM on CiteSeer because full pairwise `torch.pdist` creates a large autograd graph. The fix was a deterministic node-sampled estimate with `uniformity_max_nodes=2048`. This keeps the regularizer label-free and makes the run fit on the available GPU.

## 3. Pretraining Runs

Command:

```powershell
$py='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $py scripts\pretrain_ssl_candidates.py `
  --datasets Cora,CiteSeer `
  --methods grace_uniformity `
  --uniformity_weights 0.1,0.5,1.0 `
  --seed 0 `
  --device cuda:0 `
  --log_every 50 `
  --summary checkpoints\embeddings_v2\phase20_grace_uniformity_summary.csv
```

Raw log: `gb_graph_aware_minival_6_1/logs/phase20_grace_uniformity_pretrain_20260615_rerun.log`  
Summary: `gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/phase20_grace_uniformity_summary.csv`

| Dataset | Method | lambda_u | Embedding dim | Epochs | Final loss |
|---|---|---:|---:|---:|---:|
| Cora | grace_unif0p1 | 0.1 | 128 | 200 | 6.4206 |
| Cora | grace_unif0p5 | 0.5 | 128 | 200 | 5.0047 |
| Cora | grace_unif1 | 1.0 | 128 | 200 | 3.2170 |
| CiteSeer | grace_unif0p1 | 0.1 | 256 | 200 | 7.4104 |
| CiteSeer | grace_unif0p5 | 0.5 | 256 | 200 | 5.8860 |
| CiteSeer | grace_unif1 | 1.0 | 256 | 200 | 3.9777 |

## 4. Phase 20a Geometry Diagnostic

Command:

```powershell
$py='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $py scripts\diagnose_embedding_geometry.py `
  --embedding_root C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2 `
  --datasets cora,citeseer `
  --ssl_methods grace,grace_unif0p1,grace_unif0p5,grace_unif1 `
  --seeds 0,1,2 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --probe `
  --probe_epochs 500 `
  --output_dir runs\phase20_custom_ssl_20260615
```

Raw outputs:

- `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_raw.csv`
- `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_summary.csv`
- `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_gate_decisions.csv`
- `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_report.md`
- `external/NoisyGL/runs/phase20_custom_ssl_20260615_geometry.log`

### 4.1 Summary

| Dataset | Method | Num balls | Uniformity | True all purity | Clean probe | Noisy probe |
|---|---|---:|---:|---:|---:|---:|
| CiteSeer | GRACE | 64.0000 | -1.5480 | 0.6991 | 0.7109 | 0.7015 |
| CiteSeer | grace_unif0p1 | 1.0000 | -3.6773 | 0.2107 | 0.6957 | 0.6534 |
| CiteSeer | grace_unif0p5 | 1.0000 | -3.7903 | 0.2107 | 0.7006 | 0.6143 |
| CiteSeer | grace_unif1 | 1.0000 | -3.8045 | 0.2107 | 0.6927 | 0.5968 |
| Cora | GRACE | 52.3333 | -1.0107 | 0.7871 | 0.8294 | 0.7949 |
| Cora | grace_unif0p1 | 1.0000 | -3.3627 | 0.3021 | 0.8303 | 0.7840 |
| Cora | grace_unif0p5 | 1.0000 | -3.5755 | 0.3021 | 0.8218 | 0.7692 |
| Cora | grace_unif1 | 1.0000 | -3.6397 | 0.3021 | 0.8218 | 0.7602 |

### 4.2 Gate Decisions

| Dataset | Candidate | Num balls OK | Purity OK | Pass |
|---|---|---:|---:|---:|
| CiteSeer | grace_unif0p1 | 0 | 0 | 0 |
| CiteSeer | grace_unif0p5 | 0 | 0 | 0 |
| CiteSeer | grace_unif1 | 0 | 0 | 0 |
| Cora | grace_unif0p1 | 0 | 0 | 0 |
| Cora | grace_unif0p5 | 0 | 0 | 0 |
| Cora | grace_unif1 | 0 | 0 | 0 |

## 5. Interpretation

Directly supported:

- The explicit uniformity regularizer does change the geometry metric: uniformity becomes much more negative than GRACE on both datasets.
- That stronger uniformity does not improve granular-ball geometry. All three weights collapse the downstream 2-means ball construction to one ball on both Cora and CiteSeer.
- True weighted all-node ball purity collapses from GRACE-level `0.7871 / 0.6991` to `0.3021 / 0.2107`.
- Cora `grace_unif0p1` clean probe is slightly higher than GRACE (`0.8303` vs `0.8294`), but this candidate still fails the geometry gate. This reinforces the Phase 9 lesson: linear separability is not the same as granular-ball clusterability.

Reasonable interpretation:

- GRACE already supplies a useful balance between alignment and uniformity for this downstream ball-construction procedure. Adding a direct uniformity term over-regularizes the encoder space for the particular recursive 2-means ball splitter.
- The useful property is not "more uniformity is always better"; it is a balanced geometry that remains separable into meaningful local semantic balls.

What cannot be concluded:

- This does not prove every possible custom SSL method is impossible.
- This does not prove uniformity is irrelevant. It shows that the tested explicit uniformity regularizer, with the pre-registered lambda grid and no downstream tuning, destroys ball clusterability.
- This does not provide a downstream accuracy result because Phase 20b was deliberately skipped after Phase 20a failed.

## 6. Decision

Phase 20b is not run. The custom SSL upgrade line is closed under the pre-registered rule.

Paper implication:

- Keep GRACE as the SoftDist [X||H] anchor.
- Use this result in Section 6 to strengthen the "why GRACE" argument: GRACE's implicit geometry is locally suitable, while both SSL replacement and explicit uniformity enhancement can produce embeddings that look acceptable by probe but fail the ball-geometry requirement.
- Do not claim that stronger SSL representations improve SoftDist unless a future method passes the geometry gate first.

