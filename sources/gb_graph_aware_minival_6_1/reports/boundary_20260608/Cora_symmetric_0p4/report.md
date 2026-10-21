# Boundary setting: Cora symmetric 0.4

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.7935 | 0.0091 |
| coteaching | 0.7064 | 0.0250 |
| sel_topk | 0.7969 | 0.0033 |
| sel_dcgb | 0.8002 | 0.0130 |
| sel_noball | 0.7082 | 0.0311 |
| sel_shuffled | 0.7111 | 0.0237 |
| sel_random | 0.7159 | 0.0313 |
| sel_clean_oracle | 0.8143 | 0.0052 |

## Boundary Diagnostics

- random clean rate: `0.6000`
- selection precision - random clean rate: `0.3225`
- noise excluded recall: `0.8839`
- topk neighborhood majority disagreement: `0.5572`
- topk neighborhood majority confidence: `0.5636`
- topk pairflip neighborhood consistency: `0.3306`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 9.05 | 8.13 | SIG | 5/5 |
| sel_topk - ce | 0.35 | 0.78 | ns | 4/5 |
| sel_dcgb - sel_topk | 0.33 | 0.71 | ns | 4/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.09049631357192993,
      "mean_delta_pp": 9.049631357192993,
      "std_delta": 0.02487860063105813,
      "t_value": 8.133733559248247,
      "sig_label": "SIG",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.08870118856430054,
        "mean_delta_pp": 8.870118856430054,
        "std_delta": 0.030490746295802448,
        "t_value": 6.5049863125884855,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.0858500838279724,
        "mean_delta_pp": 8.585008382797241,
        "std_delta": 0.021819196335611468,
        "t_value": 8.798061136655605,
        "sig_label": "SIG"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.0034847021102905273,
      "mean_delta_pp": 0.34847021102905273,
      "std_delta": 0.010001060367393861,
      "t_value": 0.7791204646010035,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.003273487091064453,
      "mean_delta_pp": 0.3273487091064453,
      "std_delta": 0.010328805294686544,
      "t_value": 0.7086724408343394,
      "sig_label": "ns"
    }
  },
  "deathmatch_gate": {
    "gate_A_beats_coteaching": {
      "passes": true,
      "candidates": [
        {
          "method": "sel_topk",
          "baseline": "coteaching",
          "wins": 5,
          "mean_delta": 0.09049631357192993,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 5,
          "mean_delta": 0.09376980066299438,
          "passes": true
        }
      ]
    },
    "gate_B_beats_structure_controls": {
      "passes": true,
      "candidates": [
        {
          "method": "sel_topk",
          "parts": [
            {
              "method": "sel_topk",
              "baseline": "sel_noball",
              "wins": 5,
              "mean_delta": 0.08870118856430054,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.0858500838279724,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.08099262714385987,
              "passes": true
            }
          ],
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "parts": [
            {
              "method": "sel_dcgb",
              "baseline": "sel_noball",
              "wins": 5,
              "mean_delta": 0.09197467565536499,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.08912357091903686,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.08426611423492432,
              "passes": true
            }
          ],
          "passes": true
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": 0.00960930585861206,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.7930306792259216,
        "1": 0.8072862029075623,
        "2": 0.8115100860595703,
        "3": 0.7977824807167053,
        "4": 0.805702269077301
      }
    }
  }
}
```
