# Boundary setting: CiteSeer symmetric 0.2

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.7064 | 0.0097 |
| coteaching | 0.7086 | 0.0042 |
| sel_topk | 0.7106 | 0.0046 |
| sel_dcgb | 0.7066 | 0.0095 |
| sel_noball | 0.7086 | 0.0045 |
| sel_shuffled | 0.7052 | 0.0087 |
| sel_random | 0.7079 | 0.0048 |
| sel_clean_oracle | 0.7118 | 0.0124 |

## Boundary Diagnostics

- random clean rate: `0.8000`
- selection precision - random clean rate: `0.1085`
- noise excluded recall: `0.6241`
- topk neighborhood majority disagreement: `0.4865`
- topk neighborhood majority confidence: `0.6356`
- topk pairflip neighborhood consistency: `0.4171`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 0.20 | 4.96 | SIG | 5/5 |
| sel_topk - ce | 0.42 | 1.56 | ns | 4/5 |
| sel_dcgb - sel_topk | -0.40 | -1.77 | ns | 0/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.0019759416580200197,
      "mean_delta_pp": 0.19759416580200198,
      "std_delta": 0.0008907565903420678,
      "t_value": 4.960210134633614,
      "sig_label": "SIG",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.0019759535789489746,
        "mean_delta_pp": 0.19759535789489746,
        "std_delta": 0.0015368021323661526,
        "t_value": 2.875039297421538,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.005412375926971436,
        "mean_delta_pp": 0.5412375926971436,
        "std_delta": 0.00438690241640835,
        "t_value": 2.7587667433004115,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.004209625720977783,
      "mean_delta_pp": 0.4209625720977783,
      "std_delta": 0.00603975478111633,
      "t_value": 1.5585085178239764,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.003951883316040039,
      "mean_delta_pp": -0.39518833160400396,
      "std_delta": 0.0050038760416168705,
      "t_value": -1.7659669544806458,
      "sig_label": "ns"
    }
  },
  "deathmatch_gate": {
    "gate_A_beats_coteaching": {
      "passes": false,
      "candidates": [
        {
          "method": "sel_topk",
          "baseline": "coteaching",
          "wins": 5,
          "mean_delta": 0.0019759416580200197,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 2,
          "mean_delta": -0.0019759416580200197,
          "passes": false
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
              "mean_delta": 0.0019759535789489746,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.005412375926971436,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 4,
              "mean_delta": 0.002663242816925049,
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
              "wins": 2,
              "mean_delta": -0.0019759297370910644,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.0014604926109313965,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": -0.0012886404991149902,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": 0.004209625720977783,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.7109106779098511,
        "1": 0.7139175534248352,
        "2": 0.7027491331100464,
        "3": 0.7139175534248352,
        "4": 0.7113401889801025
      }
    }
  }
}
```
