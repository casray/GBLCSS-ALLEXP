# Main-table setting: Cora pair 0.4

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.5920 | 0.0355 |
| coteaching | 0.5225 | 0.0529 |
| sel_topk | 0.6082 | 0.0352 |
| sel_dcgb | 0.5834 | 0.0360 |
| sel_noball | 0.5218 | 0.0594 |
| sel_shuffled | 0.5430 | 0.0578 |
| sel_random | 0.5646 | 0.0672 |
| sel_clean_oracle | 0.8143 | 0.0052 |

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 8.57 | 4.74 | SIG | 5/5 |
| sel_topk - ce | 1.63 | 2.19 | ns | 4/5 |
| sel_topk - sel_noball | 8.65 | 4.24 | SIG | 5/5 |
| sel_topk - sel_shuffled | 6.53 | 4.50 | SIG | 5/5 |
| sel_dcgb - sel_topk | -2.48 | -1.54 | ns | 2/5 |

## Gates

- Gate-1 topk > co-teaching: **PASS**
- Gate-2 topk > no-ball and shuffled: **PASS**

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.08574445247650146,
      "mean_delta_pp": 8.574445247650146,
      "std_delta": 0.040439142856465736,
      "t_value": 4.741208910176049,
      "sig_label": "SIG",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.08648362755775452,
        "mean_delta_pp": 8.648362755775452,
        "std_delta": 0.04557108998053109,
        "t_value": 4.243551563996616,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.06525870561599731,
        "mean_delta_pp": 6.525870561599731,
        "std_delta": 0.032421768000065236,
        "t_value": 4.50076941765556,
        "sig_label": "SIG"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.01626187562942505,
      "mean_delta_pp": 1.6261875629425049,
      "std_delta": 0.016639407016816328,
      "t_value": 2.1853338470711305,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 2,
      "mean_delta": -0.02481520175933838,
      "mean_delta_pp": -2.481520175933838,
      "std_delta": 0.0360423355640615,
      "t_value": -1.5395361355156356,
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
          "mean_delta": 0.08574445247650146,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 4,
          "mean_delta": 0.06092925071716308,
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
              "mean_delta": 0.08648362755775452,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.06525870561599731,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.043611401319503786,
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
              "wins": 4,
              "mean_delta": 0.06166842579841614,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.04044350385665894,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": 0.018796199560165407,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": true,
      "mean_delta": 0.021119320392608644,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.5739176273345947,
        "1": 0.6087645292282104,
        "2": 0.6589229106903076,
        "3": 0.6293559074401855,
        "4": 0.5945090055465698
      }
    }
  }
}
```
