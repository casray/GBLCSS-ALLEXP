# Deathmatch report: Cora pair 0.4

Setting: `Cora` / `ratio_train0.2_val0.1` / `pair` `0.4` / seeds `[0, 1, 2, 3, 4]` / frozen GRACE minival.

## Mean Test Accuracy

| method | mean test |
|---|---:|
| ce | 0.5920 |
| coteaching | 0.5225 |
| sel_topk | 0.6082 |
| sel_dcgb | 0.5834 |
| sel_noball | 0.5218 |
| sel_shuffled | 0.5430 |
| sel_random | 0.5646 |
| sel_clean_oracle | 0.8143 |

## Pre-registered Gates

- Gate A beats co-teaching: **PASS**
- Gate B beats no-ball/shuffled/random controls: **PASS**
- Gate C beats CE with required margin: **PASS** (mean delta=2.11pp, threshold=1.50pp)

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
```
