# Boundary setting: Cora pair 0.2

## Mean +- Std Test

| method | mean | std |
|---|---:|---:|
| ce | 0.7696 | 0.0153 |
| coteaching | 0.7414 | 0.0418 |
| sel_topk | 0.7824 | 0.0196 |
| sel_dcgb | 0.7914 | 0.0273 |
| sel_noball | 0.7412 | 0.0434 |
| sel_shuffled | 0.7653 | 0.0102 |
| sel_random | 0.7442 | 0.0336 |
| sel_clean_oracle | 0.8143 | 0.0057 |

## Boundary Diagnostics

- random clean rate: `0.8000`
- selection precision - random clean rate: `0.1138`
- noise excluded recall: `0.6537`
- topk neighborhood majority disagreement: `0.3446`
- topk neighborhood majority confidence: `0.7142`
- topk pairflip neighborhood consistency: `0.5544`

## Paired Deltas

| contrast | delta pp | t | sig | wins |
|---|---:|---:|---|---:|
| sel_topk - coteaching | 4.10 | 1.78 | ns | 4/5 |
| sel_topk - ce | 1.28 | 3.98 | SIG | 5/5 |
| sel_dcgb - sel_topk | 0.91 | 0.99 | ns | 4/5 |

## Raw Gate JSON

```json
{
  "main_gate": {
    "gate_1_topk_beats_coteaching": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.0409714937210083,
      "mean_delta_pp": 4.09714937210083,
      "std_delta": 0.05160286383377121,
      "t_value": 1.77538683502142,
      "sig_label": "ns",
      "passes": true
    },
    "gate_2_topk_beats_controls": {
      "passes": true,
      "sel_topk_minus_sel_noball": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.04118267297744751,
        "mean_delta_pp": 4.118267297744751,
        "std_delta": 0.05336156997590076,
        "t_value": 1.7257223937433794,
        "sig_label": "ns"
      },
      "sel_topk_minus_sel_shuffled": {
        "n": 5,
        "wins": 4,
        "mean_delta": 0.017106640338897704,
        "mean_delta_pp": 1.7106640338897705,
        "std_delta": 0.016116892171407753,
        "t_value": 2.373386274326247,
        "sig_label": "ns"
      }
    },
    "diagnostic_topk_minus_ce": {
      "n": 5,
      "wins": 5,
      "mean_delta": 0.012777197360992431,
      "mean_delta_pp": 1.2777197360992432,
      "std_delta": 0.007175523553789165,
      "t_value": 3.981686025686966,
      "sig_label": "SIG"
    },
    "dcgb_minus_topk": {
      "n": 5,
      "wins": 4,
      "mean_delta": 0.00908130407333374,
      "mean_delta_pp": 0.908130407333374,
      "std_delta": 0.02050451280521836,
      "t_value": 0.990338733000866,
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
          "mean_delta": 0.0409714937210083,
          "passes": true
        },
        {
          "method": "sel_dcgb",
          "baseline": "coteaching",
          "wins": 4,
          "mean_delta": 0.05005279779434204,
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
              "wins": 4,
              "mean_delta": 0.04118267297744751,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_shuffled",
              "wins": 4,
              "mean_delta": 0.017106640338897704,
              "passes": true
            },
            {
              "method": "sel_topk",
              "baseline": "sel_random",
              "wins": 4,
              "mean_delta": 0.038120377063751223,
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
              "mean_delta": 0.05026397705078125,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_shuffled",
              "wins": 5,
              "mean_delta": 0.026187944412231445,
              "passes": true
            },
            {
              "method": "sel_dcgb",
              "baseline": "sel_random",
              "wins": 5,
              "mean_delta": 0.047201681137084964,
              "passes": true
            }
          ],
          "passes": true
        }
      ]
    },
    "gate_C_beats_ce": {
      "passes": true,
      "mean_delta": 0.026610350608825682,
      "threshold": 0.015,
      "best_structured_by_seed": {
        "0": 0.8093981146812439,
        "1": 0.8115100860595703,
        "2": 0.8130940198898315,
        "3": 0.7851108908653259,
        "4": 0.7618796229362488
      }
    }
  }
}
```
