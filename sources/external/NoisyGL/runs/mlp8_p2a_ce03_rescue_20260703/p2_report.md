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
| grace_ce_l0p1 | 50 | 0.8202 | 55.09 |

P2a gate: **fail**
- candidate - anchor downstream: -0.37pp
- candidate - anchor true_all_purity: +0.0329

## P2b: F0 -> Z1 -> balls -> F1 loop

| Dataset | Variant | Num balls | True all purity | Test | Loop delta |
|---|---|---:|---:|---:|---:|

P2b gate: **skipped**
- positive datasets: None/None
- mean loop delta: +0.00pp

## Decision
- P2 is a direct advisor-suggestion smoke, not a new mainline unless the gate passes.
- If either gate fails, do not expand the corresponding direction without a new preregistered mechanism and control.
