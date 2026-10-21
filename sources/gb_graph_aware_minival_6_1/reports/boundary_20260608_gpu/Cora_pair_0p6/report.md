# Boundary setting: Cora pair 0.6

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.2506 | 0.0086 |
| coteaching | 0.2448 | 0.0448 |
| sel_topk | 0.2721 | 0.0426 |
| sel_dcgb | 0.2592 | 0.0511 |
| sel_noball | 0.2374 | 0.0444 |
| sel_shuffled | 0.2380 | 0.0236 |
| sel_random | 0.2643 | 0.0240 |
| sel_clean_oracle | 0.8048 | 0.0057 |

## Boundary Diagnostics

- random clean rate: `0.4000`
- selection precision - random clean rate: `-0.1172`
- noise excluded recall: `0.4012`
- topk neighborhood majority disagreement: `0.5413`
- topk neighborhood majority confidence: `0.6393`
- topk pairflip neighborhood consistency: `0.4378`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 2.73 | 1.42 | ns | 4/5 |
| sel_topk - ce | 2.15 | 1.30 | ns | 3/5 |
| sel_dcgb - sel_topk | -1.29 | -1.77 | ns | 1/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.027349528670310975,
      "mean_delta_pp": 2.7349528670310974,
      "std_delta": 0.042925055614297505,
      "t_value": 1.4247018293675802,
      "sig_label": "ns",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.03474128842353821,
        "mean_delta_pp": 3.474128842353821,
        "std_delta": 0.036322401069109896,
        "t_value": 2.138732029117524,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.03410770893096924,
        "mean_delta_pp": 3.410770893096924,
        "std_delta": 0.04147710021854701,
        "t_value": 1.8387774295831818,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.021541711688041688,
      "mean_delta_pp": 2.1541711688041687,
      "std_delta": 0.03712701660323427,
      "t_value": 1.2974037801347822,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 1,
      "mean_delta": -0.012882784008979797,
      "mean_delta_pp": -1.2882784008979797,
      "std_delta": 0.016263253015562487,
      "t_value": -1.771280367830505,
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
          "wins": 4,
          "mean_delta": 0.027349528670310975,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 3,
          "mean_delta": 0.014466744661331177,
          "passes": false
        }
      ]
    },
    "gate_B_beats_structure_controls": {
      "passes": false,
      "candidates": [
        {
          "method": "sel_topk",
          "parts": [
            {
              "method": "sel_topk",
              "baseline": "sel_noball",
              "wins": 4,
              "mean_delta": 0.03474128842353821,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.03410770893096924,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": 0.007814154028892517,
              "passes": false
            }
          ],
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "parts": [
            {
              "method": "sel_dcgb",
              "baseline": "sel_noball",
              "wins": 3,
              "mean_delta": 0.02185850441455841,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 2,
              "mean_delta": 0.021224924921989442,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 1,
              "mean_delta": -0.00506862998008728,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": true,
      "mean_delta": 0.02323125898838043,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.25501585006713867,
        "1": 0.3225976824760437,
        "2": 0.21119324862957,
        "3": 0.2697993814945221,
        "4": 0.3104540705680847
      }
    }
  }
}
```
