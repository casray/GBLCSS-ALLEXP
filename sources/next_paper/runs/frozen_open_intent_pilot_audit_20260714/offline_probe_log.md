# Frozen Open-Intent Pilot Offline Audit Log

Date: 2026-07-14

## Commands

```powershell
py -3.12 "开放意图粒球新方向0630\code\offline_probe_B_hparam.py"
py -3.12 "开放意图粒球新方向0630\code\offline_probe_generic.py" "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\banking77_r25_seed0_20260630_172920\features_logits_seed0.npz" 0
py -3.12 "开放意图粒球新方向0630\code\offline_probe_generic.py" "开放意图粒球新方向0630\runs\gb_score_conditional_gain_20260630\stackoverflow_r25_seed0_20260630_201825\features_logits_seed0.npz" 0
```

## BANKING77 hyperparameter robustness

| hyperparameters | balls | real DM | KMeans DM | real distance | KMeans distance | real ALL-safe | KMeans ALL-safe | winner (DM / distance / ALL-safe) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mb16_ms8_si0.05 | 150 | 0.7741 | 0.7867 | 0.7999 | 0.7980 | 0.7525 | 0.7882 | K / R / K |
| mb8_ms8_si0.05 | 127 | 0.7734 | 0.7793 | 0.7949 | 0.7964 | 0.7552 | 0.7612 | K / K / K |
| mb32_ms4_si0.05 | 262 | 0.7713 | 0.7875 | 0.8083 | 0.8115 | 0.7497 | 0.7837 | K / K / K |
| mb16_ms16_si0.05 | 66 | 0.6997 | 0.7390 | 0.7720 | 0.7709 | 0.5932 | 0.6902 | K / R / K |
| mb16_ms8_si0.01 | 152 | 0.7742 | 0.7859 | 0.7997 | 0.7981 | 0.7530 | 0.7873 | K / R / K |
| mb16_ms8_si0.2 | 19 | 0.6610 | 0.6610 | 0.7116 | 0.7116 | 0.5571 | 0.5571 | K / K / K |
| mb64_ms4_si0.01 | 271 | 0.7708 | 0.8009 | 0.8108 | 0.8142 | 0.7530 | 0.7973 | K / K / K |

Summary: real wins `0/7` for dist+margin, `3/7` for distance only, and `0/7` for ALL-safe. The three distance-only wins are at most about 0.002 AUROC and do not satisfy the preregistered joint gate.

## BANKING77 generic probe

- Test: 3,080 samples; unknown 2,320; known 760; known classes 19.
- Real granular balls: 150.
- Components, `real / mean(KMeans seeds 13–15) / shuffled`:
  - margin: `0.6986 / 0.7385 / 0.6161`
  - distance: `0.7999 / 0.7980 / 0.7349`
  - cover: `0.5203 / 0.5215 / 0.5208`
  - density: `0.5927 / 0.6039 / 0.5209`
  - purity: `0.5103 / 0.5013 / 0.4654`
  - overlap: `0.5201 / 0.4607 / 0.4718`
- Dist+margin: `real=0.7741`, `KMeans=0.7867`.
- ALL-safe: `real=0.7525`, `KMeans=0.7882`.
- Preregistered G2: fail.

## StackOverflow generic probe

- Test: 6,000 samples; unknown 4,500; known 1,500; known classes 5.
- Real granular balls: 6.
- Components, `real / mean(KMeans seeds 13–15) / shuffled`:
  - margin: `0.8271 / 0.8271 / 0.8522`
  - distance: `0.8565 / 0.8565 / 0.8570`
  - cover: `0.5000 / 0.5000 / 0.5000`
  - density: `0.6130 / 0.6130 / 0.4970`
  - purity: `0.7211 / 0.7211 / 0.6124`
  - overlap: `0.4716 / 0.4716 / 0.4673`
- Dist+margin: `real=0.8570`, `KMeans=0.8570`.
- ALL-safe: `real=0.8593`, `KMeans=0.8593`.
- Preregistered G2: fail.

## Statistical audit

See `statistics.json`. Paired bootstrap used 3,000 stratified known/unknown resamples and compares real GB against the mean of three size-matched KMeans controls. No rounded p-value is reported as zero; zero exceedances are written as `p<0.001`.

## Test audit

- Core tests not importing BERT: `13 passed, 3 warnings`.
- Full collection is blocked by missing optional `transformers` in the current Python 3.12 environment, before `test_stackoverflow_runner.py` assertions execute.
