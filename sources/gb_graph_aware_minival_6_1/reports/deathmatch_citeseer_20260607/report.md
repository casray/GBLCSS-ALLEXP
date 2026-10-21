# Deathmatch report: CiteSeer symmetric 0.4

Setting: `CiteSeer` / `ratio_train0.2_val0.1` / `symmetric` `0.4` / seeds `[0, 1, 2, 3, 4]` / frozen GRACE minival.

## Mean Test Accuracy

| method | mean test |
|---|---:|
| ce | 0.6979 |
| coteaching | 0.6958 |
| sel_topk | 0.7082 |
| sel_dcgb | 0.7000 |
| sel_noball | 0.6977 |
| sel_shuffled | 0.7040 |
| sel_random | 0.6944 |
| sel_clean_oracle | 0.7104 |

## Pre-registered Gates

- Gate A beats co-teaching: **PASS**
- Gate B beats no-ball/shuffled/random controls: **PASS**
- Gate C beats CE with required margin: **PASS** (mean delta=1.03pp, threshold=1.00pp)

## Gate Details

```json
{
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
```
