# Boundary setting: CiteSeer symmetric 0.6

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.6685 | 0.0046 |
| coteaching | 0.6668 | 0.0065 |
| sel_topk | 0.6743 | 0.0110 |
| sel_dcgb | 0.6634 | 0.0039 |
| sel_noball | 0.6659 | 0.0069 |
| sel_shuffled | 0.6692 | 0.0069 |
| sel_random | 0.6659 | 0.0069 |
| sel_clean_oracle | 0.7091 | 0.0043 |

## Boundary Diagnostics

- random clean rate: `0.4000`
- selection precision - random clean rate: `0.1223`
- noise excluded recall: `0.3220`
- topk neighborhood majority disagreement: `0.7826`
- topk neighborhood majority confidence: `0.4660`
- topk pairflip neighborhood consistency: `0.2024`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 0.75 | 1.31 | ns | 3/5 |
| sel_topk - ce | 0.58 | 0.84 | ns | 3/5 |
| sel_dcgb - sel_topk | -1.09 | -1.95 | ns | 0/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.007474231719970703,
      "mean_delta_pp": 0.7474231719970703,
      "std_delta": 0.012795352147876643,
      "t_value": 1.3061688347681097,
      "sig_label": "ns",
      "passes": false
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 2,
        "mean_delta": 0.008419251441955567,
        "mean_delta_pp": 0.8419251441955566,
        "std_delta": 0.013325172759758138,
        "t_value": 1.4128160950175537,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 1,
        "mean_delta": 0.005154645442962647,
        "mean_delta_pp": 0.5154645442962646,
        "std_delta": 0.014314525723649423,
        "t_value": 0.8052056933560393,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.0058419227600097655,
      "mean_delta_pp": 0.5841922760009766,
      "std_delta": 0.015543084752982066,
      "t_value": 0.840433969079323,
      "sig_label": "ns"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.010910654067993164,
      "mean_delta_pp": -1.0910654067993164,
      "std_delta": 0.0124962514948739,
      "t_value": -1.9523426033018967,
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
          "wins": 3,
          "mean_delta": 0.007474231719970703,
          "passes": false
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 2,
          "mean_delta": -0.003436422348022461,
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
              "wins": 2,
              "mean_delta": 0.008419251441955567,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 1,
              "mean_delta": 0.005154645442962647,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 2,
              "mean_delta": 0.008419251441955567,
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
              "mean_delta": -0.002491402626037598,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 0,
              "mean_delta": -0.005756008625030518,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 0,
              "mean_delta": -0.002491402626037598,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": 0.0058419227600097655,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.6696735620498657,
        "1": 0.6709622144699097,
        "2": 0.6761168241500854,
        "3": 0.6920103430747986,
        "4": 0.662800669670105
      }
    }
  }
}
```
