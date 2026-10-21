# MLP label-rate sensitivity report

Scope: Cora/CiteSeer/PubMed, uniform noise 0.4, train-only noise, weak X-only MLP, seed0 smoke unless n>1.

## Accuracy

| dataset | train | method | n | mean | std |
|---|---:|---|---:|---:|---:|
| citeseer | 0.05 | MLP+CE | 1 | 0.3502 | 0.0000 |
| citeseer | 0.05 | MLP+small-loss | 1 | 0.3771 | 0.0000 |
| citeseer | 0.05 | MLP+JoCoR | 1 | 0.3746 | 0.0000 |
| citeseer | 0.05 | SoftDist-MLP beta=0.5 | 1 | 0.4443 | 0.0000 |
| citeseer | 0.05 | SoftDist beta=0.5 shuffled | 1 | 0.3336 | 0.0000 |
| citeseer | 0.10 | MLP+CE | 1 | 0.4040 | 0.0000 |
| citeseer | 0.10 | MLP+small-loss | 1 | 0.4126 | 0.0000 |
| citeseer | 0.10 | MLP+JoCoR | 1 | 0.4081 | 0.0000 |
| citeseer | 0.10 | SoftDist-MLP beta=0.5 | 1 | 0.5036 | 0.0000 |
| citeseer | 0.10 | SoftDist beta=0.5 shuffled | 1 | 0.4006 | 0.0000 |
| citeseer | 0.20 | MLP+CE | 1 | 0.4609 | 0.0000 |
| citeseer | 0.20 | MLP+small-loss | 1 | 0.4905 | 0.0000 |
| citeseer | 0.20 | MLP+JoCoR | 1 | 0.4759 | 0.0000 |
| citeseer | 0.20 | SoftDist-MLP beta=0.5 | 1 | 0.5533 | 0.0000 |
| citeseer | 0.20 | SoftDist beta=0.5 shuffled | 1 | 0.4223 | 0.0000 |
| citeseer | 0.35 | MLP+CE | 1 | 0.5046 | 0.0000 |
| citeseer | 0.35 | MLP+small-loss | 1 | 0.5227 | 0.0000 |
| citeseer | 0.35 | MLP+JoCoR | 1 | 0.5375 | 0.0000 |
| citeseer | 0.35 | SoftDist-MLP beta=0.5 | 1 | 0.5888 | 0.0000 |
| citeseer | 0.35 | SoftDist beta=0.5 shuffled | 1 | 0.4921 | 0.0000 |
| citeseer | 0.50 | MLP+CE | 1 | 0.5368 | 0.0000 |
| citeseer | 0.50 | MLP+small-loss | 1 | 0.5774 | 0.0000 |
| citeseer | 0.50 | MLP+JoCoR | 1 | 0.5895 | 0.0000 |
| citeseer | 0.50 | SoftDist-MLP beta=0.5 | 1 | 0.6188 | 0.0000 |
| citeseer | 0.50 | SoftDist beta=0.5 shuffled | 1 | 0.5308 | 0.0000 |
| citeseer | 0.65 | MLP+CE | 1 | 0.5957 | 0.0000 |
| citeseer | 0.65 | MLP+small-loss | 1 | 0.6330 | 0.0000 |
| citeseer | 0.65 | MLP+JoCoR | 1 | 0.6173 | 0.0000 |
| citeseer | 0.65 | SoftDist-MLP beta=0.5 | 1 | 0.6318 | 0.0000 |
| citeseer | 0.65 | SoftDist beta=0.5 shuffled | 1 | 0.5848 | 0.0000 |
| cora | 0.05 | MLP+CE | 1 | 0.3312 | 0.0000 |
| cora | 0.05 | MLP+small-loss | 1 | 0.3481 | 0.0000 |
| cora | 0.05 | MLP+JoCoR | 1 | 0.3412 | 0.0000 |
| cora | 0.05 | SoftDist-MLP beta=0.5 | 1 | 0.4098 | 0.0000 |
| cora | 0.05 | SoftDist beta=0.5 shuffled | 1 | 0.3286 | 0.0000 |
| cora | 0.10 | MLP+CE | 1 | 0.3795 | 0.0000 |
| cora | 0.10 | MLP+small-loss | 1 | 0.4298 | 0.0000 |
| cora | 0.10 | MLP+JoCoR | 1 | 0.4331 | 0.0000 |
| cora | 0.10 | SoftDist-MLP beta=0.5 | 1 | 0.4751 | 0.0000 |
| cora | 0.10 | SoftDist beta=0.5 shuffled | 1 | 0.4086 | 0.0000 |
| cora | 0.20 | MLP+CE | 1 | 0.4174 | 0.0000 |
| cora | 0.20 | MLP+small-loss | 1 | 0.4950 | 0.0000 |
| cora | 0.20 | MLP+JoCoR | 1 | 0.5082 | 0.0000 |
| cora | 0.20 | SoftDist-MLP beta=0.5 | 1 | 0.5546 | 0.0000 |
| cora | 0.20 | SoftDist beta=0.5 shuffled | 1 | 0.4385 | 0.0000 |
| cora | 0.35 | MLP+CE | 1 | 0.4661 | 0.0000 |
| cora | 0.35 | MLP+small-loss | 1 | 0.5440 | 0.0000 |
| cora | 0.35 | MLP+JoCoR | 1 | 0.5520 | 0.0000 |
| cora | 0.35 | SoftDist-MLP beta=0.5 | 1 | 0.5447 | 0.0000 |
| cora | 0.35 | SoftDist beta=0.5 shuffled | 1 | 0.4385 | 0.0000 |
| cora | 0.50 | MLP+CE | 1 | 0.5254 | 0.0000 |
| cora | 0.50 | MLP+small-loss | 1 | 0.6020 | 0.0000 |
| cora | 0.50 | MLP+JoCoR | 1 | 0.6150 | 0.0000 |
| cora | 0.50 | SoftDist-MLP beta=0.5 | 1 | 0.6270 | 0.0000 |
| cora | 0.50 | SoftDist beta=0.5 shuffled | 1 | 0.4801 | 0.0000 |
| cora | 0.65 | MLP+CE | 1 | 0.5126 | 0.0000 |
| cora | 0.65 | MLP+small-loss | 1 | 0.6411 | 0.0000 |
| cora | 0.65 | MLP+JoCoR | 1 | 0.6529 | 0.0000 |
| cora | 0.65 | SoftDist-MLP beta=0.5 | 1 | 0.6514 | 0.0000 |
| cora | 0.65 | SoftDist beta=0.5 shuffled | 1 | 0.5126 | 0.0000 |
| pubmed | 0.05 | MLP+CE | 1 | 0.5479 | 0.0000 |
| pubmed | 0.05 | MLP+small-loss | 1 | 0.6712 | 0.0000 |
| pubmed | 0.05 | MLP+JoCoR | 1 | 0.6908 | 0.0000 |
| pubmed | 0.05 | SoftDist-MLP beta=0.5 | 1 | 0.6132 | 0.0000 |
| pubmed | 0.05 | SoftDist beta=0.5 shuffled | 1 | 0.5314 | 0.0000 |
| pubmed | 0.10 | MLP+CE | 1 | 0.5932 | 0.0000 |
| pubmed | 0.10 | MLP+small-loss | 1 | 0.7054 | 0.0000 |
| pubmed | 0.10 | MLP+JoCoR | 1 | 0.7170 | 0.0000 |
| pubmed | 0.10 | SoftDist-MLP beta=0.5 | 1 | 0.6387 | 0.0000 |
| pubmed | 0.10 | SoftDist beta=0.5 shuffled | 1 | 0.5666 | 0.0000 |
| pubmed | 0.20 | MLP+CE | 1 | 0.6262 | 0.0000 |
| pubmed | 0.20 | MLP+small-loss | 1 | 0.7746 | 0.0000 |
| pubmed | 0.20 | MLP+JoCoR | 1 | 0.7778 | 0.0000 |
| pubmed | 0.20 | SoftDist-MLP beta=0.5 | 1 | 0.7160 | 0.0000 |
| pubmed | 0.20 | SoftDist beta=0.5 shuffled | 1 | 0.6187 | 0.0000 |
| pubmed | 0.35 | MLP+CE | 1 | 0.7044 | 0.0000 |
| pubmed | 0.35 | MLP+small-loss | 1 | 0.7980 | 0.0000 |
| pubmed | 0.35 | MLP+JoCoR | 1 | 0.8090 | 0.0000 |
| pubmed | 0.35 | SoftDist-MLP beta=0.5 | 1 | 0.7818 | 0.0000 |
| pubmed | 0.35 | SoftDist beta=0.5 shuffled | 1 | 0.6835 | 0.0000 |
| pubmed | 0.50 | MLP+CE | 1 | 0.7461 | 0.0000 |
| pubmed | 0.50 | MLP+small-loss | 1 | 0.8222 | 0.0000 |
| pubmed | 0.50 | MLP+JoCoR | 1 | 0.8142 | 0.0000 |
| pubmed | 0.50 | SoftDist-MLP beta=0.5 | 1 | 0.7989 | 0.0000 |
| pubmed | 0.50 | SoftDist beta=0.5 shuffled | 1 | 0.7028 | 0.0000 |
| pubmed | 0.65 | MLP+CE | 1 | 0.7539 | 0.0000 |
| pubmed | 0.65 | MLP+small-loss | 1 | 0.8273 | 0.0000 |
| pubmed | 0.65 | MLP+JoCoR | 1 | 0.8217 | 0.0000 |
| pubmed | 0.65 | SoftDist-MLP beta=0.5 | 1 | 0.8174 | 0.0000 |
| pubmed | 0.65 | SoftDist beta=0.5 shuffled | 1 | 0.7166 | 0.0000 |

## Paired Deltas

| dataset | train | comparison | n | mean_delta | positive | seed_deltas |
|---|---:|---|---:|---:|---:|---|
| citeseer | 0.05 | beta05_vs_ce | 1 | 0.0941 | 1/1 | 0.094093 |
| citeseer | 0.05 | beta05_vs_small_loss | 1 | 0.0672 | 1/1 | 0.067209 |
| citeseer | 0.05 | beta05_vs_jocor | 1 | 0.0697 | 1/1 | 0.069685 |
| citeseer | 0.05 | beta05_vs_shuffled | 1 | 0.1107 | 1/1 | 0.110718 |
| citeseer | 0.10 | beta05_vs_ce | 1 | 0.0996 | 1/1 | 0.099587 |
| citeseer | 0.10 | beta05_vs_small_loss | 1 | 0.0909 | 1/1 | 0.090943 |
| citeseer | 0.10 | beta05_vs_jocor | 1 | 0.0955 | 1/1 | 0.095453 |
| citeseer | 0.10 | beta05_vs_shuffled | 1 | 0.1030 | 1/1 | 0.102969 |
| citeseer | 0.20 | beta05_vs_ce | 1 | 0.0924 | 1/1 | 0.092354 |
| citeseer | 0.20 | beta05_vs_small_loss | 1 | 0.0627 | 1/1 | 0.062715 |
| citeseer | 0.20 | beta05_vs_jocor | 1 | 0.0773 | 1/1 | 0.077320 |
| citeseer | 0.20 | beta05_vs_shuffled | 1 | 0.1310 | 1/1 | 0.131014 |
| citeseer | 0.35 | beta05_vs_ce | 1 | 0.0842 | 1/1 | 0.084199 |
| citeseer | 0.35 | beta05_vs_small_loss | 1 | 0.0662 | 1/1 | 0.066156 |
| citeseer | 0.35 | beta05_vs_jocor | 1 | 0.0514 | 1/1 | 0.051394 |
| citeseer | 0.35 | beta05_vs_shuffled | 1 | 0.0968 | 1/1 | 0.096774 |
| citeseer | 0.50 | beta05_vs_ce | 1 | 0.0820 | 1/1 | 0.081955 |
| citeseer | 0.50 | beta05_vs_small_loss | 1 | 0.0414 | 1/1 | 0.041353 |
| citeseer | 0.50 | beta05_vs_jocor | 1 | 0.0293 | 1/1 | 0.029323 |
| citeseer | 0.50 | beta05_vs_shuffled | 1 | 0.0880 | 1/1 | 0.087970 |
| citeseer | 0.65 | beta05_vs_ce | 1 | 0.0361 | 1/1 | 0.036101 |
| citeseer | 0.65 | beta05_vs_small_loss | 1 | -0.0012 | 0/1 | -0.001203 |
| citeseer | 0.65 | beta05_vs_jocor | 1 | 0.0144 | 1/1 | 0.014440 |
| citeseer | 0.65 | beta05_vs_shuffled | 1 | 0.0469 | 1/1 | 0.046931 |
| cora | 0.05 | beta05_vs_ce | 1 | 0.0787 | 1/1 | 0.078661 |
| cora | 0.05 | beta05_vs_small_loss | 1 | 0.0617 | 1/1 | 0.061712 |
| cora | 0.05 | beta05_vs_jocor | 1 | 0.0687 | 1/1 | 0.068666 |
| cora | 0.05 | beta05_vs_shuffled | 1 | 0.0813 | 1/1 | 0.081269 |
| cora | 0.10 | beta05_vs_ce | 1 | 0.0956 | 1/1 | 0.095568 |
| cora | 0.10 | beta05_vs_small_loss | 1 | 0.0452 | 1/1 | 0.045245 |
| cora | 0.10 | beta05_vs_jocor | 1 | 0.0420 | 1/1 | 0.042013 |
| cora | 0.10 | beta05_vs_shuffled | 1 | 0.0665 | 1/1 | 0.066482 |
| cora | 0.20 | beta05_vs_ce | 1 | 0.1372 | 1/1 | 0.137203 |
| cora | 0.20 | beta05_vs_small_loss | 1 | 0.0596 | 1/1 | 0.059631 |
| cora | 0.20 | beta05_vs_jocor | 1 | 0.0464 | 1/1 | 0.046438 |
| cora | 0.20 | beta05_vs_shuffled | 1 | 0.1161 | 1/1 | 0.116095 |
| cora | 0.35 | beta05_vs_ce | 1 | 0.0786 | 1/1 | 0.078576 |
| cora | 0.35 | beta05_vs_small_loss | 1 | 0.0007 | 1/1 | 0.000672 |
| cora | 0.35 | beta05_vs_jocor | 1 | -0.0074 | 0/1 | -0.007388 |
| cora | 0.35 | beta05_vs_shuffled | 1 | 0.1061 | 1/1 | 0.106111 |
| cora | 0.50 | beta05_vs_ce | 1 | 0.1016 | 1/1 | 0.101570 |
| cora | 0.50 | beta05_vs_small_loss | 1 | 0.0249 | 1/1 | 0.024931 |
| cora | 0.50 | beta05_vs_jocor | 1 | 0.0120 | 1/1 | 0.012004 |
| cora | 0.50 | beta05_vs_shuffled | 1 | 0.1468 | 1/1 | 0.146814 |
| cora | 0.65 | beta05_vs_ce | 1 | 0.1388 | 1/1 | 0.138848 |
| cora | 0.65 | beta05_vs_small_loss | 1 | 0.0103 | 1/1 | 0.010340 |
| cora | 0.65 | beta05_vs_jocor | 1 | -0.0015 | 0/1 | -0.001477 |
| cora | 0.65 | beta05_vs_shuffled | 1 | 0.1388 | 1/1 | 0.138848 |
| pubmed | 0.05 | beta05_vs_ce | 1 | 0.0653 | 1/1 | 0.065278 |
| pubmed | 0.05 | beta05_vs_small_loss | 1 | -0.0580 | 0/1 | -0.057999 |
| pubmed | 0.05 | beta05_vs_jocor | 1 | -0.0776 | 0/1 | -0.077570 |
| pubmed | 0.05 | beta05_vs_shuffled | 1 | 0.0818 | 1/1 | 0.081807 |
| pubmed | 0.10 | beta05_vs_ce | 1 | 0.0455 | 1/1 | 0.045521 |
| pubmed | 0.10 | beta05_vs_small_loss | 1 | -0.0666 | 0/1 | -0.066633 |
| pubmed | 0.10 | beta05_vs_jocor | 1 | -0.0782 | 0/1 | -0.078235 |
| pubmed | 0.10 | beta05_vs_shuffled | 1 | 0.0721 | 1/1 | 0.072149 |
| pubmed | 0.20 | beta05_vs_ce | 1 | 0.0898 | 1/1 | 0.089776 |
| pubmed | 0.20 | beta05_vs_small_loss | 1 | -0.0586 | 0/1 | -0.058619 |
| pubmed | 0.20 | beta05_vs_jocor | 1 | -0.0619 | 0/1 | -0.061880 |
| pubmed | 0.20 | beta05_vs_shuffled | 1 | 0.0972 | 1/1 | 0.097239 |
| pubmed | 0.35 | beta05_vs_ce | 1 | 0.0775 | 1/1 | 0.077462 |
| pubmed | 0.35 | beta05_vs_small_loss | 1 | -0.0162 | 0/1 | -0.016230 |
| pubmed | 0.35 | beta05_vs_jocor | 1 | -0.0272 | 0/1 | -0.027204 |
| pubmed | 0.35 | beta05_vs_shuffled | 1 | 0.0983 | 1/1 | 0.098303 |
| pubmed | 0.50 | beta05_vs_ce | 1 | 0.0528 | 1/1 | 0.052752 |
| pubmed | 0.50 | beta05_vs_small_loss | 1 | -0.0233 | 0/1 | -0.023332 |
| pubmed | 0.50 | beta05_vs_jocor | 1 | -0.0153 | 0/1 | -0.015344 |
| pubmed | 0.50 | beta05_vs_shuffled | 1 | 0.0961 | 1/1 | 0.096120 |
| pubmed | 0.65 | beta05_vs_ce | 1 | 0.0635 | 1/1 | 0.063502 |
| pubmed | 0.65 | beta05_vs_small_loss | 1 | -0.0099 | 0/1 | -0.009941 |
| pubmed | 0.65 | beta05_vs_jocor | 1 | -0.0043 | 0/1 | -0.004260 |
| pubmed | 0.65 | beta05_vs_shuffled | 1 | 0.1008 | 1/1 | 0.100832 |

## Trend Summary

| dataset | comparison | low mean | mid mean | high mean | low-high | direction |
|---|---|---:|---:|---:|---:|---|
| citeseer | beta05_vs_ce | 0.0968 | 0.0883 | 0.0590 | 0.0378 | low_label_advantage |
| citeseer | beta05_vs_jocor | 0.0826 | 0.0644 | 0.0219 | 0.0607 | low_label_advantage |
| citeseer | beta05_vs_shuffled | 0.1068 | 0.1139 | 0.0675 | 0.0394 | low_label_advantage |
| citeseer | beta05_vs_small_loss | 0.0791 | 0.0644 | 0.0201 | 0.0590 | low_label_advantage |
| cora | beta05_vs_ce | 0.0871 | 0.1079 | 0.1202 | -0.0331 | high_label_advantage |
| cora | beta05_vs_jocor | 0.0553 | 0.0195 | 0.0053 | 0.0501 | low_label_advantage |
| cora | beta05_vs_shuffled | 0.0739 | 0.1111 | 0.1428 | -0.0690 | high_label_advantage |
| cora | beta05_vs_small_loss | 0.0535 | 0.0302 | 0.0176 | 0.0358 | low_label_advantage |
| pubmed | beta05_vs_ce | 0.0554 | 0.0836 | 0.0581 | -0.0027 | flat |
| pubmed | beta05_vs_jocor | -0.0779 | -0.0445 | -0.0098 | -0.0681 | high_label_advantage |
| pubmed | beta05_vs_shuffled | 0.0770 | 0.0978 | 0.0985 | -0.0215 | high_label_advantage |
| pubmed | beta05_vs_small_loss | -0.0623 | -0.0374 | -0.0166 | -0.0457 | high_label_advantage |

## Split Audit

- Missing runs: 0
- Low class-count warnings (min class train count < 3): 0

## Plots

- `runs\mlp_labelrate_20260701\mlp_labelrate_delta_curves_citeseer.png`
- `runs\mlp_labelrate_20260701\mlp_labelrate_delta_curves_cora.png`
- `runs\mlp_labelrate_20260701\mlp_labelrate_delta_curves_pubmed.png`

## Conservative Seed0 Interpretation

- Anti-noise trend counts across dataset-comparison pairs: low=4, high=2, flat=0.
- Because this is seed0 only, any clear pattern is a 5-seed trigger, not a final claim.
- Output directory: `runs\mlp_labelrate_20260701`
