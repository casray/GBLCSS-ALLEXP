# MLP Beta Hardness 5-Seed Report

Protocol: SoftDist-MLP beta spectrum, uniform noise rates 0.2 and 0.4, percent split 0.2/0.1/0.7, seeds=0,1,2,3,4.

## Accuracy Matrix

| dataset | noise | beta=0 | beta=0.5 | beta=1 | beta=1 shuffled | beta1-soft | beta1-soft +/n | beta1-shuf | beta1-shuf +/n | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | 0.2 | 0.6141 +/- 0.0124 | 0.6294 +/- 0.0160 | 0.6368 +/- 0.0136 | 0.5121 +/- 0.0191 | 0.0227 | 5/5 | 0.1247 | 5/5 | hard_preferred_5seed |
| cora | 0.4 | 0.5184 +/- 0.0137 | 0.5559 +/- 0.0150 | 0.5679 +/- 0.0196 | 0.4178 +/- 0.0133 | 0.0495 | 5/5 | 0.1501 | 5/5 | hard_preferred_5seed |
| citeseer | 0.2 | 0.6229 +/- 0.0083 | 0.6338 +/- 0.0120 | 0.6289 +/- 0.0193 | 0.5338 +/- 0.0159 | 0.0060 | 2/5 | 0.0951 | 5/5 | hardening_positive_seed0 |
| citeseer | 0.4 | 0.5162 +/- 0.0270 | 0.5383 +/- 0.0275 | 0.5456 +/- 0.0154 | 0.4105 +/- 0.0231 | 0.0295 | 5/5 | 0.1351 | 5/5 | hardening_positive_seed0 |
| pubmed | 0.2 | 0.8223 +/- 0.0052 | 0.8251 +/- 0.0033 | 0.8125 +/- 0.0052 | 0.7416 +/- 0.0123 | -0.0098 | 0/5 | 0.0710 | 5/5 | mid_preferred_5seed |
| pubmed | 0.4 | 0.7011 +/- 0.0357 | 0.7187 +/- 0.0096 | 0.7424 +/- 0.0064 | 0.6192 +/- 0.0262 | 0.0413 | 4/5 | 0.1232 | 5/5 | hard_preferred_5seed |

## Directional Counts

- Complete settings: 6/6.
- Stable hard datasets (beta=1 >= beta=0 for both noise rates): 1/3.
- Stable half-hard datasets (beta=0.5 >= beta=0 and beta=0.5 >= beta=1 for both noise rates): 0/3.
- Stable hard preference at uniform0.4: 2/3 datasets.
- Stable half-hard preference at uniform0.2: 1/3 datasets.
- Stable beta=0.5 > beta=0 settings: 6/6.
- Stable beta=1 real > shuffled settings: 6/6.
- Mean-best beta counts: beta=0: 0, beta=0.5: 2, beta=1: 4.

## Conclusion

5-seed does not give a stable fixed-hard or adaptive-hardness law. Use beta=0.5 as the conservative fixed default because it is the only hardness that stably improves over beta=0 in every setting; keep beta=1 as an aggressive/mid-noise variant rather than the default.

Caution: these are 5-seed paired results, but any final paper claim still needs alignment with the full MLP robustness table.

## Outputs

- `mlp_beta_hardness_aggregate.csv`
- `mlp_beta_hardness_paired_deltas.csv`
- `mlp_beta_hardness_decision.csv`
