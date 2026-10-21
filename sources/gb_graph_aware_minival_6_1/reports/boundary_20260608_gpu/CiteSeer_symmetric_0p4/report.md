# Boundary setting: CiteSeer symmetric 0.4

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.6979 | 0.0100 |
| coteaching | 0.6958 | 0.0099 |
| sel_topk | 0.7082 | 0.0026 |
| sel_dcgb | 0.7000 | 0.0046 |
| sel_noball | 0.6977 | 0.0073 |
| sel_shuffled | 0.7040 | 0.0081 |
| sel_random | 0.6944 | 0.0070 |
| sel_clean_oracle | 0.7104 | 0.0038 |

## Boundary Diagnostics

- random clean rate: `0.6000`
- selection precision - random clean rate: `0.2301`
- noise excluded recall: `0.7233`
- topk neighborhood majority disagreement: `0.6571`
- topk neighborhood majority confidence: `0.5359`
- topk pairflip neighborhood consistency: `0.2845`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 1.24 | 2.96 | SIG | 5/5 |
| sel_topk - ce | 1.03 | 2.00 | ns | 4/5 |
| sel_dcgb - sel_topk | -0.82 | -5.31 | SIG | 0/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.01237114667892456,
      "mean_delta_pp": 1.237114667892456,
      "std_delta": 0.009349076058962576,
      "t_value": 2.9588725943861545,
      "sig_label": "SIG",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.010481107234954833,
        "mean_delta_pp": 1.0481107234954834,
        "std_delta": 0.006896996604849058,
        "t_value": 3.398068695632884,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.004123723506927491,
        "mean_delta_pp": 0.4123723506927491,
        "std_delta": 0.007591048160349944,
        "t_value": 1.2147105231220947,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.010309290885925294,
      "mean_delta_pp": 1.0309290885925293,
      "std_delta": 0.011554103588645536,
      "t_value": 1.9951591262693844,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.008161532878875732,
      "mean_delta_pp": -0.8161532878875731,
      "std_delta": 0.003436438739395081,
      "t_value": -5.310655507561357,
      "sig_label": "SIG"
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
          "mean_delta": 0.01237114667892456,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 4,
          "mean_delta": 0.0042096138000488285,
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
              "mean_delta": 0.010481107234954833,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.004123723506927491,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.013745725154876709,
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
              "wins": 3,
              "mean_delta": 0.0023195743560791016,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 1,
              "mean_delta": -0.004037809371948242,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": 0.005584192276000976,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": true,
      "mean_delta": 0.010309290885925294,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.7100515365600586,
        "1": 0.7091924548149109,
        "2": 0.7083333730697632,
        "3": 0.7096220254898071,
        "4": 0.7036082744598389
      }
    }
  }
}
```
