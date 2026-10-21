# Main-table setting: CiteSeer pair 0.6

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.3114 | 0.0935 |
| coteaching | 0.3284 | 0.0762 |
| sel_topk | 0.3088 | 0.0950 |
| sel_dcgb | 0.3088 | 0.0950 |
| sel_noball | 0.3088 | 0.0950 |
| sel_shuffled | 0.3088 | 0.0950 |
| sel_random | 0.3088 | 0.0950 |
| sel_clean_oracle | 0.7091 | 0.0043 |

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | -1.97 | -0.61 | ns | 2/5 |
| sel_topk - ce | -0.27 | -1.00 | ns | 0/5 |
| sel_topk - sel_noball | 0.00 | 0.00 | ns | 0/5 |
| sel_topk - sel_shuffled | 0.00 | 0.00 | ns | 0/5 |
| sel_dcgb - sel_topk | 0.00 | 0.00 | ns | 0/5 |

## Gates

- Gate-1 topk > co-teaching: **FAIL**
- Gate-2 topk > no-ball and shuffled: **FAIL**

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 2,
      "mean_delta": -0.019673538208007813,
      "mean_delta_pp": -1.9673538208007812,
      "std_delta": 0.07257443042202914,
      "t_value": -0.6061552055624785,
      "sig_label": "ns",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 0,
        "mean_delta": 0.0,
        "mean_delta_pp": 0.0,
        "std_delta": 0.0,
        "t_value": 0.0,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 0,
        "mean_delta": 0.0,
        "mean_delta_pp": 0.0,
        "std_delta": 0.0,
        "t_value": 0.0,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.0026632308959960937,
      "mean_delta_pp": -0.2663230895996094,
      "std_delta": 0.005955165323224938,
      "t_value": -1.0,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 0,
      "mean_delta": 0.0,
      "mean_delta_pp": 0.0,
      "std_delta": 0.0,
      "t_value": 0.0,
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
          "mean_delta": -0.019673538208007813,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 2,
          "mean_delta": -0.019673538208007813,
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
              "mean_delta": 0.0,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 0,
              "mean_delta": 0.0,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 0,
              "mean_delta": 0.0,
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
              "mean_delta": 0.0,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 0,
              "mean_delta": 0.0,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 0,
              "mean_delta": 0.0,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": -0.0026632308959960937,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.47680413722991943,
        "1": 0.2590206265449524,
        "2": 0.27963918447494507,
        "3": 0.24699312448501587,
        "4": 0.28135737776756287
      }
    }
  }
}
```
