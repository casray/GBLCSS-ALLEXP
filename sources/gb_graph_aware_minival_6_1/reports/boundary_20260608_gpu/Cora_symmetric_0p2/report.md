# Boundary setting: Cora symmetric 0.2

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.8113 | 0.0047 |
| coteaching | 0.7773 | 0.0056 |
| sel_topk | 0.8117 | 0.0069 |
| sel_dcgb | 0.8131 | 0.0083 |
| sel_noball | 0.7784 | 0.0071 |
| sel_shuffled | 0.7803 | 0.0125 |
| sel_random | 0.7748 | 0.0139 |
| sel_clean_oracle | 0.8143 | 0.0057 |

## Boundary Diagnostics

- random clean rate: `0.8000`
- selection precision - random clean rate: `0.1608`
- noise excluded recall: `0.8426`
- topk neighborhood majority disagreement: `0.3428`
- topk neighborhood majority confidence: `0.6966`
- topk pairflip neighborhood consistency: `0.5204`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 3.44 | 11.33 | SIG | 5/5 |
| sel_topk - ce | 0.04 | 0.36 | ns | 3/5 |
| sel_dcgb - sel_topk | 0.14 | 0.59 | ns | 3/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.03442451953887939,
      "mean_delta_pp": 3.4424519538879395,
      "std_delta": 0.006796440927161173,
      "t_value": 11.325864023047751,
      "sig_label": "SIG",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.03336853981018066,
        "mean_delta_pp": 3.336853981018066,
        "std_delta": 0.005728075574808777,
        "t_value": 13.026071732296021,
        "sig_label": "SIG"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 5,
        "mean_delta": 0.03146780729293823,
        "mean_delta_pp": 3.146780729293823,
        "std_delta": 0.0104908035459754,
        "t_value": 6.707222750050201,
        "sig_label": "SIG"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.0004223942756652832,
      "mean_delta_pp": 0.04223942756652832,
      "std_delta": 0.0026240391527084517,
      "t_value": 0.3599421573871993,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.0013727545738220215,
      "mean_delta_pp": 0.13727545738220215,
      "std_delta": 0.005234756130852617,
      "t_value": 0.5863831030061439,
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
          "mean_delta": 0.03442451953887939,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 5,
          "mean_delta": 0.035797274112701415,
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
              "mean_delta": 0.03336853981018066,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.03146780729293823,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.036958825588226316,
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
              "mean_delta": 0.03474129438400268,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.03284056186676025,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.03833158016204834,
              "passes": true
            }
          ],
          "passes": true
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": 0.002956712245941162,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.8257656097412109,
        "1": 0.8141499757766724,
        "2": 0.8109821081161499,
        "3": 0.8025343418121338,
        "4": 0.81784588098526
      }
    }
  }
}
```
