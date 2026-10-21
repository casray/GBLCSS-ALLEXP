# MLP-46 Tie-Safe LOO Clean-Label Control

All values use five paired seeds and the final `distribution_only + leave_one_out=true` target.

| Dataset | LOO-GB-LCSS | LOO-shuffled | MLP+CE | LOO-CE (pp) | LOO-shuffled (pp) |
|---|---:|---:|---:|---:|---:|
| cora | 66.25±1.28 | 62.22±0.80 | 66.49±1.19 | -0.24 | +4.03 |
| citeseer | 66.55±0.64 | 63.18±0.94 | 64.55±0.75 | +2.00 | +3.38 |
| corafull | 47.13±0.67 | 48.34±0.29 | 47.50±0.59 | -0.38 | -1.21 |
| coauthorcs | 92.09±0.39 | 91.50±0.54 | 92.96±0.27 | -0.87 | +0.58 |
| amazonpho | 87.07±0.53 | 86.04±1.21 | 88.82±0.77 | -1.75 | +1.03 |

This is a descriptive clean-label control. It is not used to tune the method or to alter the MLP-44 promotion gate.
