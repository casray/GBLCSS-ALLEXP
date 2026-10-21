# Boundary setting: Cora symmetric 0.6

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.7136 | 0.0286 |
| coteaching | 0.6646 | 0.0451 |
| sel_topk | 0.6725 | 0.0375 |
| sel_dcgb | 0.6785 | 0.0391 |
| sel_noball | 0.6673 | 0.0488 |
| sel_shuffled | 0.6601 | 0.0629 |
| sel_random | 0.6577 | 0.0682 |
| sel_clean_oracle | 0.8048 | 0.0057 |

## Boundary Diagnostics

- random clean rate: `0.4000`
- selection precision - random clean rate: `0.0674`
- noise excluded recall: `0.1729`
- topk neighborhood majority disagreement: `0.7554`
- topk neighborhood majority confidence: `0.4604`
- topk pairflip neighborhood consistency: `0.1997`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 0.79 | 2.00 | ns | 4/5 |
| sel_topk - ce | -4.11 | -3.45 | SIG | 0/5 |
| sel_dcgb - sel_topk | 0.59 | 1.42 | ns | 3/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.007919752597808838,
      "mean_delta_pp": 0.7919752597808838,
      "std_delta": 0.008842739774401557,
      "t_value": 2.0026717539451284,
      "sig_label": "ns",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": false,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 1,
        "mean_delta": 0.005279839038848877,
        "mean_delta_pp": 0.5279839038848877,
        "std_delta": 0.012103387051913937,
        "t_value": 0.9754359627172561,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 2,
        "mean_delta": 0.01246039867401123,
        "mean_delta_pp": 1.246039867401123,
        "std_delta": 0.026411304182289992,
        "t_value": 1.054938380533299,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 0,
      "mean_delta": -0.04107707738876343,
      "mean_delta_pp": -4.107707738876343,
      "std_delta": 0.02663411367996969,
      "t_value": -3.4486275181505914,
      "sig_label": "SIG"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 3,
      "mean_delta": 0.005913400650024414,
      "mean_delta_pp": 0.5913400650024414,
      "std_delta": 0.009306573406221673,
      "t_value": 1.4207985317997158,
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
          "wins": 4,
          "mean_delta": 0.007919752597808838,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 4,
          "mean_delta": 0.013833153247833251,
          "passes": true
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
              "wins": 1,
              "mean_delta": 0.005279839038848877,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 2,
              "mean_delta": 0.01246039867401123,
              "passes": false
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 2,
              "mean_delta": 0.014889121055603027,
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
              "wins": 3,
              "mean_delta": 0.01119323968887329,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 3,
              "mean_delta": 0.018373799324035645,
              "passes": false
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 3,
              "mean_delta": 0.020802521705627443,
              "passes": false
            }
          ],
          "passes": false
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": false,
      "mean_delta": -0.03516367673873901,
      "threshold": 0.01,
      "best_structured_by_seed": {
        "0": 0.712249219417572,
        "1": 0.6663147211074829,
        "2": 0.6953537464141846,
        "3": 0.7027455568313599,
        "4": 0.6156283020973206
      }
    }
  }
}
```
