# Engineering trace

2026-10-22: initial nine unit tests failed because kernels.py had not yet been implemented, then passed 9/9 after implementation.

First preparation stopped before training/freeze at the exact seed0 historical-target check. The proposed GPU target-construction path differed from the previous CPU builder by 5.960464477539063e-08 on Cora clean. A direct comparison showed CPU vs historical max difference 0, GPU vs historical max difference 5.960464477539063e-08; observed labels unchanged. Changed the new preparation path to the exact historical CPU construction. Training remains GPU. No tolerance was relaxed and no historical file was edited. failure_prepare.json preserves the original failure. Partially prepared new inputs are regenerated before the first freeze; no completed model exists at this point.

## Formal numerical failure and v2 recovery

339 formal fits completed; CoraFull seed0 clean R0_SOFT_GCE failed at update53. Loss remained finite at update52 but validation collapsed; next objective nonfinite. Raw trajectory/failure_run.json retained. A minimal float32 example logits[0,-200,-1000], target[.6,.3,.1] produced finite loss0.5714285969734192 but all-NaN gradient under exp(log_p).pow(.7). Evaluating exp(.7*log_p) gave the same scalar loss and finite gradient. Cause: intermediate probability underflow to zero and the derivative of power(q<1) at zero.

Frozen run.py/kernels.py/config/PLAN are not edited. stable_losses.py and resume_v2.py implement only the algebraically identical log-space GCE expressions. All completed and failed GCE/SOFT_GCE arms are preserved under engineering_v1_runs and uniformly rerun with v2. Other arms use exactly the old objective function and successful artifacts unchanged. No observed test metric, oracle-based arm selection, hyperparameter or success gate changes. Added three numerical regression tests; initial missing-module RED preserved in execution history, then GREEN. Recovery source hashes separately frozen in numerical_recovery_v2.json. Formal runtime must include rerun cost and retain the failed attempt.
