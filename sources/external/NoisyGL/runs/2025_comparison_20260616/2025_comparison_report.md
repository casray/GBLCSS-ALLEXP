# 2025 Comparison Execution Report

Date: 2026-06-16

## Shared Protocol

- Datasets: Cora, CiteSeer
- Split: `percent`, train/val/test = `0.2/0.1/0.7`
- Noise: `uniform`, noise rate = `0.4`
- Seeds: `0,1,2,3,4`
- GPU Python: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`

## Phase 24: NRGNN + [X||H]

Implementation: added `nrgnn_xh`, which feeds NRGNN the same `[X || L2norm(H)]` feature input used by SoftDist [X||H]. NRGNN's training logic, loss, pseudo-label mechanism, and edge mechanism were not changed.

| Dataset | Method | Test acc |
|---|---|---:|
| Cora | NRGNN | 0.8107 ± 0.0230 |
| Cora | NRGNN [X||H] | 0.8139 ± 0.0170 |
| Cora | SoftDist [X||H] | 0.8137 ± 0.0067 |
| CiteSeer | NRGNN | 0.7027 ± 0.0108 |
| CiteSeer | NRGNN [X||H] | 0.7090 ± 0.0187 |
| CiteSeer | SoftDist [X||H] | 0.6891 ± 0.0159 |

Paired deltas:

| Dataset | Comparison | Mean delta | Positive seeds |
|---|---|---:|---:|
| Cora | NRGNN [X||H] - NRGNN | +0.33pp | 3/5 |
| Cora | SoftDist [X||H] - NRGNN [X||H] | -0.02pp | 3/5 |
| CiteSeer | NRGNN [X||H] - NRGNN | +0.64pp | 4/5 |
| CiteSeer | SoftDist [X||H] - NRGNN [X||H] | -1.99pp | 1/5 |

Decision: Phase 24 does not support a strong claim that SoftDist beats a fully H-augmented NRGNN. It supports the narrower claim that SoftDist is essentially tied on Cora with lower variance and a much simpler non-iterative pipeline, while CiteSeer remains stronger for NRGNN [X||H].

Detailed artifacts:

- `runs/phase24_nrgnn_xh_20260616/phase24_fairness_report.md`
- `runs/phase24_nrgnn_xh_20260616/phase24_fairness_aggregate.csv`
- `runs/phase24_nrgnn_xh_20260616/phase24_fairness_paired_deltas.csv`

## Phase 25: TFR (CIKM 2025)

External repository: <https://github.com/eaglelab-zju/TFR.git>

Cloned commit: `d7e9ff8a58d457b519f068c3d170cd37579c8bd8`

Implementation: integrated a GCN-version TFR predictor into local NoisyGL, with a native `index_add_`-based scatter-mean replacement to avoid adding `torch_scatter` as a new runtime dependency. Official TFR does not provide Cora/CiteSeer configs, so the run transfers the official `tfr_gcn_dblp.yaml` template to Cora/CiteSeer without tuning.

| Dataset | Method | Test acc |
|---|---|---:|
| Cora | TFR (CIKM 2025, DBLP-template transfer) | 0.7758 ± 0.0185 |
| CiteSeer | TFR (CIKM 2025, DBLP-template transfer) | 0.6979 ± 0.0129 |

Paired deltas:

| Dataset | Comparison | Mean delta | Positive seeds |
|---|---|---:|---:|
| Cora | TFR - SoftDist [X||H] | -3.79pp | 0/5 |
| Cora | TFR - NRGNN [X||H] | -3.81pp | 0/5 |
| CiteSeer | TFR - SoftDist [X||H] | +0.88pp | 3/5 |
| CiteSeer | TFR - NRGNN [X||H] | -1.12pp | 1/5 |

Decision: TFR is runnable under our protocol after minimal integration. Under a transferred official DBLP template, it does not beat SoftDist on Cora, is slightly above SoftDist on CiteSeer, and remains below NRGNN [X||H]. These numbers should be reported as same-protocol transferred-config results, not definitive tuned TFR performance.

Detailed artifacts:

- `runs/tfr_comparison_20260616/tfr_comparison_report.md`
- `runs/tfr_comparison_20260616/tfr_comparison_aggregate.csv`
- `runs/tfr_comparison_20260616/tfr_comparison_paired_deltas.csv`

## Phase 26: ProCon (IJCAI 2025)

The IJCAI proceedings page and public searches did not expose official code during this execution. No same-protocol reproduction was attempted.

Decision: do not place ProCon in the main result table. Discuss it in related work as a peer-informed label-consistency method; any original-paper number must be labeled as a reference result, not a same-protocol result.

Detailed artifact:

- `runs/procon_comparison_20260616/procon_code_availability_report.md`

## Phase 27: GD²

Not run in this execution. Phase 27 remains optional because the key fairness control and one 2025 same-protocol external method have already been handled, and ProCon lacks an available official implementation source.

## Paper-Claim Update

The paper should not claim broad SOTA over strong H-augmented iterative baselines. A defensible claim is:

SoftDist [X||H] is a simple, non-iterative frozen-representation plus granular-ball soft-correction pipeline. It matches H-augmented NRGNN on Cora with lower variance, remains competitive against transferred-config TFR, and provides a clean attribution chain showing stable ball-specific gains over GCN [X||H] and shuffled balls. CiteSeer remains a limitation where NRGNN [X||H] is stronger.

