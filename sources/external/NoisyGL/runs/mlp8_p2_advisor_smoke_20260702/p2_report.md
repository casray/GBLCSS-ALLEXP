# MLP-8 P2 Advisor Smoke Report

## Material Passport
- Artifact type: smoke experiment result
- Scope: P2a supervised-CE-assisted GRACE stage-1 and P2b T=2 loop
- Protocol: seed0, train-only uniform label noise, percent split 0.2/0.1/0.7
- Claim limit: smoke only; no 5-seed significance claim

## P2a: GRACE + 0.1 noisy CE

| Variant | Num balls | True all purity | Downstream test |
|---|---:|---:|---:|
| anchor_grace | 52 | 0.7873 | 55.46 |
| grace_ce_l0p1 | 1 | 0.3021 | 42.85 |

P2a gate: **fail**
- candidate - anchor downstream: -12.61pp
- candidate - anchor true_all_purity: -0.4852

## P2b: F0 -> Z1 -> balls -> F1 loop

| Dataset | Variant | Num balls | True all purity | Test | Loop delta |
|---|---|---:|---:|---:|---:|
| cora | F0_graceH | 52 | 0.7873 | 55.46 |  |
| cora | F1_loop_Z1 | 1 | 0.3021 | 42.85 | -12.61pp |
| citeseer | F0_graceH | 64 | 0.6991 | 55.33 |  |
| citeseer | F1_loop_Z1 | 1 | 0.2107 | 43.43 | -11.90pp |

P2b gate: **fail**
- positive datasets: 0/2
- mean loop delta: -12.26pp

## Decision
- P2 is a direct advisor-suggestion smoke, not a new mainline unless the gate passes.
- If either gate fails, do not expand the corresponding direction without a new preregistered mechanism and control.
