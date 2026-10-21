# Deathmatch report: CiteSeer symmetric 0.4

Setting: `CiteSeer` / `ratio_train0.2_val0.1` / `symmetric` `0.4` / seeds `[0]` / frozen GRACE minival.

## Mean Test Accuracy

| method | mean test |
|---|---:|
| ce | 0.6894 |
| coteaching | 0.6813 |
| sel_topk | 0.6894 |
| sel_dcgb | 0.6894 |
| sel_noball | 0.6894 |
| sel_shuffled | 0.6946 |
| sel_random | 0.6894 |
| sel_clean_oracle | 0.7131 |

## Pre-registered Gates

- Gate A beats co-teaching: **FAIL**
- Gate B beats no-ball/shuffled/random controls: **FAIL**
- Gate C beats CE with required margin: **FAIL** (mean delta=0.00pp, threshold=1.00pp)

## Gate Details

```json
{
  "gate_A_beats_coteaching": {
    "passes": false,
    "candidates": [
      {
        "method": "sel_topk",
        "baseline": "coteaching",
        "wins": 1,
        "mean_delta": 0.008161485195159912,
        "passes": false
      },
      {
        "method": "sel_dcgb",
        "baseline": "coteaching",
        "wins": 1,
        "mean_delta": 0.008161485195159912,
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
            "mean_delta": -0.005154669284820557,
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
            "mean_delta": -0.005154669284820557,
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
    "mean_delta": 0.0,
    "threshold": 0.01,
    "best_structured_by_seed": {
      "0": 0.6894329786300659
    }
  }
}
```
