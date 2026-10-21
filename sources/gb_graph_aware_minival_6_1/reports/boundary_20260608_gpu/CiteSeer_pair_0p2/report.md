# Boundary setting: CiteSeer pair 0.2

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.6906 | 0.0145 |
| coteaching | 0.6970 | 0.0161 |
| sel_topk | 0.7009 | 0.0087 |
| sel_dcgb | 0.7001 | 0.0106 |
| sel_noball | 0.6997 | 0.0106 |
| sel_shuffled | 0.6985 | 0.0109 |
| sel_random | 0.6947 | 0.0169 |
| sel_clean_oracle | 0.7118 | 0.0124 |

## Boundary Diagnostics

- random clean rate: `0.8000`
- selection precision - random clean rate: `0.0807`
- noise excluded recall: `0.5218`
- topk neighborhood majority disagreement: `0.4673`
- topk neighborhood majority confidence: `0.6497`
- topk pairflip neighborhood consistency: `0.4416`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 0.40 | 0.76 | ns | 2/5 |
| sel_topk - ce | 1.03 | 2.45 | ns | 5/5 |
| sel_dcgb - sel_topk | -0.09 | -0.37 | ns | 2/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 2,
      "mean_delta": 0.003951907157897949,
      "mean_delta_pp": 0.3951907157897949,
      "std_delta": 0.011571660822287286,
      "t_value": 0.7636529605852306,
      "sig_label": "ns",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 3,
        "mean_delta": 0.0012027502059936523,
        "mean_delta_pp": 0.12027502059936523,
        "std_delta": 0.008862758288585983,
        "t_value": 0.30345307103966723,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 3,
        "mean_delta": 0.0024914264678955076,
        "mean_delta_pp": 0.24914264678955075,
        "std_delta": 0.008208862040061342,
        "t_value": 0.678656665926246,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.010309278964996338,
      "mean_delta_pp": 1.0309278964996338,
      "std_delta": 0.009415943811937054,
      "t_value": 2.4482143293501837,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 2,
      "mean_delta": -0.0008591055870056152,
      "mean_delta_pp": -0.08591055870056152,
      "std_delta": 0.005127726178029864,
      "t_value": -0.37463359502798077,
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
          "wins": 2,
          "mean_delta": 0.003951907157897949,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 2,
          "mean_delta": 0.003092801570892334,
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
              "wins": 3,
              "mean_delta": 0.0012027502059936523,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 3,
              "mean_delta": 0.0024914264678955076,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 4,
              "mean_delta": 0.006271493434906006,
              "passes": true
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
              "wins": 2,
              "mean_delta": 0.0003436446189880371,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.0016323208808898925,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 4,
              "mean_delta": 0.005412387847900391,
              "passes": true
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": 0.012113404273986817,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.706615149974823,
        "1": 0.7014604806900024,
        "2": 0.688144326210022,
        "3": 0.7126289010047913,
        "4": 0.7048969268798828
      }
    }
  }
}
```
