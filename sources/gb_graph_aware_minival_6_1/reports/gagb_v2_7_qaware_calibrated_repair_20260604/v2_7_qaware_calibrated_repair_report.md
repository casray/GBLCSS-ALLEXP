# GAGB v2.7 Q-Aware Calibrated Repair Report

## 1. Research Hypothesis

v2.6 showed that `Q(B)` can rank cleaner balls, but quality filtering alone mostly behaves as a safety valve. v2.7 tests whether `Q(B)` becomes more useful when combined with calibrated repair entry conditions:

- high-quality balls can enter boundary-safe / class-safe target repair;
- low-quality balls are blocked before fallback repair;
- clean labels are used only for diagnostics and evaluation.

## 2. Method Role

Granular balls remain weak local repair units, not hard labels or view generators.

New target methods:

- `grace_gagb_qsafe_boundary_alpha09_q50`
- `grace_gagb_qsafe_boundary_alpha09_q70`
- `grace_gagb_qsafe_boundary_alpha07_q50`
- `grace_gagb_qsafe_class_capped_alpha09_q50`
- `grace_gagb_qsafe_class_balanced_alpha09_q50`

Controls:

- `gagb_qsafe_boundary_alpha09_q50`
- `khop_only_qsafe_boundary_alpha09_q50`
- `grace_gagb_shuffled_qsafe_boundary_alpha09_q50`

## 3. Code Changes

- `smoothing/targets.py`
  - Added Q-aware calibrated repair configs.
  - Added `quality_gate_mode="candidate"` to calibrated repair.
  - Low-quality gated candidates are excluded before fallback alpha is applied.
- `run_noisy_classifier.py` / `run_gate_experiment.py`
  - Added new method choices and grouping routes.
- `evaluation/summary.py`
  - Added new formal methods, alpha fallback pairing, and calibrated diagnostics.
- Tests
  - Added tests for low-quality candidate blocking, class-cap budgeting on high-quality candidates, routing, CLI exposure, and summary pairings.

## 4. Experiment Config

- Dataset: `Cora`, `CiteSeer`
- Noise: symmetric `0.2`, `0.4`
- Split: `ratio_train0p2_val0p1_split0`
- Seeds: `0 1 2 3 4`
- Encoder: frozen GRACE embeddings
- Teacher: noisy-train-only linear teacher
- Output:
  - `reports/gagb_v2_7_qaware_calibrated_repair_20260604/gate_summary.csv`
  - `reports/gagb_v2_7_qaware_calibrated_repair_20260604/gate_paired_deltas.csv`

## 5. Main Results

| dataset | noise | method | test mean +- std | NCR | CDR |
|---|---:|---|---:|---:|---:|
| CiteSeer | 0.2 | `gb` | 0.7009 +- 0.0127 | 0.0000 | 0.0000 |
| CiteSeer | 0.2 | `grace_gagb_qfilter_alpha07_q70` | 0.7094 +- 0.0040 | 0.1579 | 0.0285 |
| CiteSeer | 0.2 | `grace_gagb_qsafe_boundary_alpha09_q50` | 0.7069 +- 0.0074 | 0.1564 | 0.0158 |
| CiteSeer | 0.2 | `grace_gagb_qsafe_boundary_alpha09_q70` | 0.7072 +- 0.0053 | 0.1338 | 0.0143 |
| CiteSeer | 0.4 | `gb` | 0.6976 +- 0.0097 | 0.0000 | 0.0000 |
| CiteSeer | 0.4 | `grace_gagb_qfilter_alpha07_q70` | 0.6988 +- 0.0062 | 0.0526 | 0.0040 |
| CiteSeer | 0.4 | `grace_gagb_qsafe_boundary_alpha09_q50` | 0.6979 +- 0.0074 | 0.0173 | 0.0020 |
| CiteSeer | 0.4 | `gagb_qsafe_boundary_alpha09_q50` | 0.6995 +- 0.0084 | 0.0008 | 0.0000 |
| Cora | 0.2 | `gb` | 0.8118 +- 0.0051 | 0.0000 | 0.0000 |
| Cora | 0.2 | `grace_gagb_qfilter_alpha07_q70` | 0.8122 +- 0.0038 | 0.2296 | 0.0078 |
| Cora | 0.2 | `grace_gagb_qsafe_boundary_alpha09_q50` | 0.8120 +- 0.0054 | 0.1833 | 0.0005 |
| Cora | 0.2 | `grace_gagb_qsafe_boundary_alpha09_q70` | 0.8122 +- 0.0058 | 0.1500 | 0.0005 |
| Cora | 0.4 | `gb` | 0.7974 +- 0.0054 | 0.0000 | 0.0000 |
| Cora | 0.4 | `grace_gagb_qfilter_alpha07_q70` | 0.7940 +- 0.0059 | 0.1032 | 0.0031 |
| Cora | 0.4 | `grace_gagb_qsafe_boundary_alpha07_q50` | 0.7968 +- 0.0042 | 0.0387 | 0.0000 |
| Cora | 0.4 | `grace_gagb_qsafe_boundary_alpha09_q50` | 0.7955 +- 0.0052 | 0.0387 | 0.0000 |

## 6. Mechanism Findings

`Q(B)` did change the effective intervention set:

- `CiteSeer 0.2 / q50`: selected gate `57.4`, rejected gate `13.6`.
- `Cora 0.2 / q50`: selected gate `39.4`, rejected gate `10.2`.
- `Cora 0.4 / q50`: selected gate `23.6`, rejected gate `4.6`.
- `CiteSeer 0.4 / q50`: selected gate `23.4`, rejected gate `0`, so Q filtering did not materially change this setting.

Boundary-safe variants reduced `CDR` strongly:

- `Cora 0.4`: qfilter q70 `CDR=0.0031`, qsafe boundary `CDR=0.0000`.
- `Cora 0.2`: qfilter q70 `CDR=0.0078`, qsafe boundary `CDR=0.0005`.
- `CiteSeer 0.2`: qfilter q70 `CDR=0.0285`, qsafe boundary q70 `CDR=0.0143`.

But the reduced damage came with lower `NCR`, so accuracy did not become a stable win.

## 7. Decision

Supported:

- `Q(B) + boundary-safe` is a better safety mechanism than plain qfilter.
- It reduces clean damage while preserving nonzero NCR in most settings.
- It partially repairs the Cora 0.4 failure relative to alpha07/qfilter, but not enough to beat `gb`.

Not supported:

- `grace_gagb_qsafe_*` is not ready as a main method.
- Class-capped and class-balanced qsafe variants are weaker than boundary-safe variants.
- Current GRACE-first GAGB-specific value is still not proven, because `gagb_qsafe` and `khop_only_qsafe` remain competitive.

Final decision:

- Keep `Q(B)` as a diagnostic / guard.
- Keep `boundary-safe qsafe` as the strongest safe repair variant.
- Stop target-repair formula tuning unless the next step changes the ball reliability model or moves to a different host baseline.
