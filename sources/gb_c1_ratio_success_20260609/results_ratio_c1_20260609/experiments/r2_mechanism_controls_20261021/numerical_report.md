# R2 controls: numerical summary

Single protocol/student seed0; no seed standard deviation or significance claims.

| Route | Rule | Noisy accuracy % | Clean accuracy % |
|---|---|---:|---:|
| REAL | CE | 56.6018 | 67.9795 |
| REAL | GB | 57.9214 | 66.2850 |
| REAL | LS | 56.4438 | 67.4236 |
| REAL | GCE | 59.7102 | 66.5648 |
| REAL | BOOT | 56.4658 | 67.9506 |
| REAL | MASS | 58.0492 | 66.7605 |
| REAL | KNN | 59.1339 | 68.1489 |
| REAL | KM | 59.2768 | 67.5692 |
| REAL | RANDOM | 55.7196 | 64.3744 |
| RANDOM_ENCODER | CE | 21.6602 | 23.1908 |
| RANDOM_ENCODER | GB | 21.9585 | 23.1577 |
| SHUFFLED_H | CE | 17.3037 | 19.5380 |
| SHUFFLED_H | GB | 18.0875 | 19.2708 |

## Paired comparisons

- GB_minus_CE: +1.3196pp; positive 8/9; dataset means {'cora': 1.9700967458223413, 'citeseer': 1.002290950744559, 'corafull': 0.9864068326717211}; clean {'cora': -1.477572559366763, 'citeseer': -1.2457044673539457, 'corafull': -2.360158787441364}
- GB_minus_LS: +1.4776pp; positive 8/9; dataset means {'cora': 2.74406332453826, 'citeseer': 0.9020618556701004, 'corafull': 0.7867195958137856}; clean {'cora': -1.2664907651715112, 'citeseer': -0.1718213058419309, 'corafull': -1.9776254059906293}
- GB_minus_GCE: -1.7889pp; positive 1/9; dataset means {'cora': -2.3746701846965714, 'citeseer': -2.7777777777777763, 'corafull': -0.21412245879946804}; clean {'cora': -0.2638522427440648, 'citeseer': -1.9759450171821271, 'corafull': 1.4002165283291177}
- GB_minus_BOOT: +1.4556pp; positive 8/9; dataset means {'cora': 2.1635883905013196, 'citeseer': 1.2457044673539504, 'corafull': 0.9575363887886468}; clean {'cora': -1.477572559366763, 'citeseer': -1.2457044673539457, 'corafull': -2.2735474557921407}
- GB_minus_MASS: -0.1278pp; positive 4/9; dataset means {'cora': -0.5628847845206693, 'citeseer': 0.4152348224513152, 'corafull': -0.23577529171177503}; clean {'cora': -0.1583113456464389, 'citeseer': -0.25773195876288924, 'corafull': -1.0104655359076204}
- GB_minus_KNN: -1.2125pp; positive 0/9; dataset means {'cora': -1.213720316622691, 'citeseer': -1.3316151202749182, 'corafull': -1.0922651269096615}; clean {'cora': -2.638522427440634, 'citeseer': -1.6323024054982795, 'corafull': -1.3208228076506714}
- GB_minus_KM: -1.3554pp; positive 1/9; dataset means {'cora': -1.9525065963060726, 'citeseer': -0.7159221076746837, 'corafull': -1.3978106580055325}; clean {'cora': -1.0554089709762593, 'citeseer': -2.147766323024044, 'corafull': -0.6495849873691881}
- GB_minus_RANDOM: +2.2018pp; positive 8/9; dataset means {'cora': 4.907651715039575, 'citeseer': 0.6872852233676928, 'corafull': 1.0104655359076133}; clean {'cora': 7.651715039577837, 'citeseer': 0.0, 'corafull': -1.919884518224471}
- REAL_CE_minus_RANDOM_ENCODER_CE: +34.9416pp; positive 9/9; dataset means {'cora': 35.00439753737906, 'citeseer': 32.80355097365406, 'corafull': 37.01672079874894}; clean {'cora': 50.712401055408975, 'citeseer': 38.74570446735395, 'corafull': 44.9079754601227}
- REAL_CE_minus_SHUFFLED_H_CE: +39.2981pp; positive 9/9; dataset means {'cora': 37.08003518029903, 'citeseer': 42.0389461626575, 'corafull': 38.775412005292914}; clean {'cora': 49.60422163588392, 'citeseer': 49.52749140893471, 'corafull': 46.19271021291952}
- RANDOM_ENCODER_GB_minus_CE: +0.2982pp; positive 4/9; dataset means {'cora': 0.47493403693931197, 'citeseer': 0.744558991981671, 'corafull': -0.32479249368459023}; clean {'cora': -0.42216358839050017, 'citeseer': 1.1168384879725153, 'corafull': -0.7939372067845536}
- SHUFFLED_H_GB_minus_CE: +0.7838pp; positive 7/9; dataset means {'cora': 0.9498680738786263, 'citeseer': 1.302978235967925, 'corafull': 0.09864068326717194}; clean {'cora': -0.10554089709762238, 'citeseer': -0.5154639175257749, 'corafull': -0.18044027426921616}

## Frozen gates

```json
{
  "selected_generic_regularizer": "GCE",
  "selection_uses": "noisy validation macro only",
  "generic_validation_means": {
    "LS": 58.948453656546604,
    "GCE": 61.644506898015464,
    "BOOT": 58.93346084103726
  },
  "transfer_vs_random": true,
  "transfer_vs_shuffle": true,
  "local_direction_gate": false,
  "generic_regularizer_gate": false,
  "km_gate": false,
  "random_group_gate": true,
  "knn_gate": false,
  "clean_safety_gate": false,
  "per_cell_test_regularizer_envelope_mean_pp": -1.8345683432394742,
  "envelope_is_descriptive_test_oracle_not_selector": true,
  "no_automatic_expansion": true,
  "seed_count": 1,
  "seed_std": null,
  "full_gb_continuation_gate": false
}
```

Interpretation and final research decision are in 综合结论与接续.md.
