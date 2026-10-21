# MLP Beta Hardness Seed0 Report

Protocol: SoftDist-MLP beta spectrum, uniform noise rates 0.2 and 0.4, percent split 0.2/0.1/0.7, seeds=0.

## Accuracy Matrix

| dataset | noise | beta=0 | beta=0.5 | beta=1 | beta=1 shuffled | beta1-soft | beta1-soft +/n | beta1-shuf | beta1-shuf +/n | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | 0.2 | 0.6153 +/- 0.0000 | 0.6306 +/- 0.0000 | 0.6385 +/- 0.0000 | 0.4976 +/- 0.0000 | 0.0232 | 1/1 | 0.1409 | 1/1 | hard_preferred_5seed |
| cora | 0.4 | 0.5314 +/- 0.0000 | 0.5546 +/- 0.0000 | 0.5620 +/- 0.0000 | 0.4095 +/- 0.0000 | 0.0306 | 1/1 | 0.1525 | 1/1 | hard_preferred_5seed |
| citeseer | 0.2 | 0.6121 +/- 0.0000 | 0.6151 +/- 0.0000 | 0.6022 +/- 0.0000 | 0.5159 +/- 0.0000 | -0.0099 | 0/1 | 0.0863 | 1/1 | mid_preferred_5seed |
| citeseer | 0.4 | 0.5344 +/- 0.0000 | 0.5533 +/- 0.0000 | 0.5528 +/- 0.0000 | 0.4076 +/- 0.0000 | 0.0185 | 1/1 | 0.1452 | 1/1 | hardening_positive_seed0 |
| pubmed | 0.2 | 0.8187 +/- 0.0000 | 0.8244 +/- 0.0000 | 0.8121 +/- 0.0000 | 0.7310 +/- 0.0000 | -0.0066 | 0/1 | 0.0812 | 1/1 | mid_preferred_5seed |
| pubmed | 0.4 | 0.6868 +/- 0.0000 | 0.7224 +/- 0.0000 | 0.7443 +/- 0.0000 | 0.6095 +/- 0.0000 | 0.0575 | 1/1 | 0.1348 | 1/1 | hard_preferred_5seed |

## Directional Counts

- Complete settings: 6/6.
- Best beta counts among real balls: beta=0: 0, beta=0.5: 3, beta=1: 3.
- Hardening-positive settings with beta=1 > beta=0 and beta=1 > shuffled: 1.
- Real beta=1 above shuffled settings: 6/6.

## Conclusion

Seed0 is heterogeneous across datasets or noise rates; do not choose a fixed beta from this smoke alone.

Caution: this is a seed0 smoke test. Any main-method change needs 5-seed confirmation before it can support a paper claim.

## Outputs

- `mlp_beta_hardness_aggregate.csv`
- `mlp_beta_hardness_paired_deltas.csv`
- `mlp_beta_hardness_decision.csv`
