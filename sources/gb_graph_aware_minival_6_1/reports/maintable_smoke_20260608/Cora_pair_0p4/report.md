# Main-table setting: Cora pair 0.4

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.5396 | 0.0000 |
| coteaching | 0.4456 | 0.0000 |
| sel_topk | 0.4324 | 0.0000 |
| sel_dcgb | 0.4414 | 0.0000 |
| sel_noball | 0.4467 | 0.0000 |
| sel_shuffled | 0.4451 | 0.0000 |
| sel_random | 0.4525 | 0.0000 |
| sel_clean_oracle | 0.8057 | 0.0000 |

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | -1.32 | -inf | SIG | 0/1 |
| sel_topk - ce | -10.72 | -inf | SIG | 0/1 |
| sel_topk - sel_noball | -1.43 | -inf | SIG | 0/1 |
| sel_topk - sel_shuffled | -1.27 | -inf | SIG | 0/1 |
| sel_dcgb - sel_topk | 0.90 | inf | SIG | 1/1 |

## Gates

- Gate-1 topk > co-teaching: **FAIL**
- Gate-2 topk > no-ball and shuffled: **FAIL**

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 1,
      "wins": 0,
      "mean_delta": -0.013199597597122192,
      "mean_delta_pp": -1.3199597597122192,
      "std_delta": 0.0,
      "t_value": -Infinity,
      "sig_label": "SIG",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 1,
        "wins": 0,
        "mean_delta": -0.014255553483963013,
        "mean_delta_pp": -1.4255553483963013,
        "std_delta": 0.0,
        "t_value": -Infinity,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 1,
        "wins": 0,
        "mean_delta": -0.012671619653701782,
        "mean_delta_pp": -1.2671619653701782,
        "std_delta": 0.0,
        "t_value": -Infinity,
        "sig_label": "SIG"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 1,
      "wins": 0,
      "mean_delta": -0.10718059539794922,
      "mean_delta_pp": -10.718059539794922,
      "std_delta": 0.0,
      "t_value": -Infinity,
      "sig_label": "SIG"
    },
    "dcgb_minus_topk": {
      "n": 1,
      "wins": 1,
      "mean_delta": 0.008975714445114136,
      "mean_delta_pp": 0.8975714445114136,
      "std_delta": 0.0,
      "t_value": Infinity,
      "sig_label": "SIG"
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
          "mean_delta": -0.013199597597122192,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 0,
          "mean_delta": -0.004223883152008057,
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
              "mean_delta": -0.014255553483963013,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 0,
              "mean_delta": -0.012671619653701782,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 0,
              "mean_delta": -0.0200633704662323,
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
              "mean_delta": -0.005279839038848877,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 0,
              "mean_delta": -0.0036959052085876465,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 0,
              "mean_delta": -0.011087656021118164,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": -0.09820488095283508,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.4413938820362091
      }
    }
  }
}
```
