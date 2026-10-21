# Deathmatch report: Cora pair 0.4

Setting: `Cora` / `ratio_train0.2_val0.1` / `pair` `0.4` / seeds `[0]` / frozen GRACE minival.

## Mean Test Accuracy

| method | mean test |
|---|---:|
| ce | 0.5396 |
| coteaching | 0.4456 |
| sel_topk | 0.4324 |
| sel_dcgb | 0.4414 |
| sel_noball | 0.4467 |
| sel_shuffled | 0.4451 |
| sel_random | 0.4525 |
| sel_clean_oracle | 0.8057 |

## Pre-registered Gates

- Gate A beats co-teaching: **FAIL**
- Gate B beats no-ball/shuffled/random controls: **FAIL**
- Gate C beats CE with required margin: **FAIL** (mean delta=-9.82pp, threshold=1.50pp)

## Gate Details

```json
{
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
```
