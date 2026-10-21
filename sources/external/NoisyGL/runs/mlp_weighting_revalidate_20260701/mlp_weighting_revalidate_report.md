# MLP weighting revalidation report

Scope: Cora/CiteSeer, uniform noise 0.4, percent split 0.2/0.1/0.7, seed0 smoke unless n>1.

## Accuracy

| dataset | method | n | mean | std |
|---|---|---:|---:|---:|
| citeseer | beta=0.5 baseline | 1 | 0.5533 | 0.0000 |
| citeseer | ellipsoid + beta=0.5 | 1 | 0.5485 | 0.0000 |
| citeseer | ellipsoid shuffled + beta=0.5 | 1 | 0.4257 | 0.0000 |
| citeseer | gmm_resp + beta=0.5 | 1 | 0.5451 | 0.0000 |
| cora | beta=0.5 baseline | 1 | 0.5546 | 0.0000 |
| cora | ellipsoid + beta=0.5 | 1 | 0.5662 | 0.0000 |
| cora | ellipsoid shuffled + beta=0.5 | 1 | 0.4459 | 0.0000 |
| cora | gmm_resp + beta=0.5 | 1 | 0.5230 | 0.0000 |

## Paired Deltas

| dataset | comparison | n | mean_delta | positive | seed_deltas |
|---|---|---:|---:|---:|---|
| citeseer | ellipsoid_minus_beta05 | 1 | -0.0047 | 0/1 | -0.004725 |
| citeseer | gmmresp_minus_beta05 | 1 | -0.0082 | 0/1 | -0.008162 |
| citeseer | ellipsoid_minus_shuffled | 1 | +0.1229 | 1/1 | 0.122852 |
| cora | ellipsoid_minus_beta05 | 1 | +0.0116 | 1/1 | 0.011609 |
| cora | gmmresp_minus_beta05 | 1 | -0.0317 | 0/1 | -0.031662 |
| cora | ellipsoid_minus_shuffled | 1 | +0.1203 | 1/1 | 0.120317 |

## Seed0 Decision

| family | complete datasets | positive datasets | seed0 signal |
|---|---|---|---:|
| ellipsoid+beta05 | cora citeseer | cora | 0 |
| gmm_resp+beta05 | cora citeseer |  | 0 |

## Conclusion

No weighting family beats beta=0.5 on all completed seed0 datasets; keep beta=0.5 as main unless a 5-seed rerun is explicitly requested.

Source directory: `runs\mlp_weighting_revalidate_20260701`
