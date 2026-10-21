# Main-table setting: CiteSeer pair 0.4

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.5956 | 0.0113 |
| coteaching | 0.6521 | 0.0315 |
| sel_topk | 0.6293 | 0.0248 |
| sel_dcgb | 0.6082 | 0.0188 |
| sel_noball | 0.6536 | 0.0323 |
| sel_shuffled | 0.6058 | 0.0257 |
| sel_random | 0.6151 | 0.0266 |
| sel_clean_oracle | 0.7104 | 0.0038 |

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | -2.29 | -4.30 | SIG | 0/5 |
| sel_topk - ce | 3.37 | 4.21 | SIG | 5/5 |
| sel_topk - sel_noball | -2.43 | -4.04 | SIG | 0/5 |
| sel_topk - sel_shuffled | 2.35 | 1.92 | ns | 4/5 |
| sel_dcgb - sel_topk | -2.10 | -1.69 | ns | 2/5 |

## Gates

- Gate-1 topk > co-teaching: **FAIL**
- Gate-2 topk > no-ball and shuffled: **FAIL**

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.022852230072021484,
      "mean_delta_pp": -2.2852230072021484,
      "std_delta": 0.011886302136113166,
      "t_value": -4.298993858085996,
      "sig_label": "SIG",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 0,
        "mean_delta": -0.024312698841094972,
        "mean_delta_pp": -2.431269884109497,
        "std_delta": 0.013449217392580978,
        "t_value": -4.0422312866440935,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.02353954315185547,
        "mean_delta_pp": 2.353954315185547,
        "std_delta": 0.02737269988236971,
        "t_value": 1.9229385070904332,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.03367699384689331,
      "mean_delta_pp": 3.3676993846893306,
      "std_delta": 0.017904184953923433,
      "t_value": 4.205946694210944,
      "sig_label": "SIG"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 2,
      "mean_delta": -0.02104811668395996,
      "mean_delta_pp": -2.104811668395996,
      "std_delta": 0.02783989110282593,
      "t_value": -1.6905604813556372,
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
          "wins": 0,
          "mean_delta": -0.022852230072021484,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 0,
          "mean_delta": -0.043900346755981444,
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
              "wins": 0,
              "mean_delta": -0.024312698841094972,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.02353954315185547,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": 0.014175283908843993,
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
              "wins": 0,
              "mean_delta": -0.04536081552505493,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 3,
              "mean_delta": 0.0024914264678955076,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": -0.006872832775115967,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": true,
      "mean_delta": 0.036855685710906985,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.6387457251548767,
        "1": 0.6486254334449768,
        "2": 0.6361684203147888,
        "3": 0.599656343460083,
        "4": 0.639175295829773
      }
    }
  }
}
```
