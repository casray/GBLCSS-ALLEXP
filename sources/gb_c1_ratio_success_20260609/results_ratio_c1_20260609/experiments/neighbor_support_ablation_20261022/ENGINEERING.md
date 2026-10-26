# Preparation metadata compatibility

The first prepare attempt stopped before any new fitting. Historical fusion CE
records hash the loaded LCSS payload as target_hash, while objective(rule='CE')
replaces it with observed one-hot labels. This is confirmed in the unchanged
fusion kernels.py. Initial check incorrectly compared that metadata to one-hot.

Corrected preparation to compare CE metadata to the historical loaded target,
verify its input file and training duration, and independently reconstruct CE
one-hot targets. Three-epoch historical CE/LCSS bridges remain mandatory for all
seeds/routes before full fitting. No optimization or target rule was changed.
First traceback is preserved as failure_prepare_v1.json and preparation_v1.log.

Before any fit, independent review also requested a mandatory final source freeze
(analysis was implemented in parallel), actual-target metadata for reused CE,
and a smoke-audit manifest binding. Added a freeze phase before smoke; it
preserves preparation_manifest.json, checks inputs, and hashes all required
sources. Reused records retain historical_target_hash while target_hash always
describes the actual new-arm target. Full fitting requires a successful smoke
audit bound to this final manifest. No fits preceded these changes.

## Post-grid audit normalization check

The first full audit verified all384 Cora/CiteSeer cells, then stopped at the
historical CoraFull clean R0 CE softmax sum. A read-only scan of576 saved
probability tensors found79 models above the original3e-7 float32 sum threshold;
maximum float32 sum error3.5762786865234375e-7 (three float32 machine eps).
Float64 accumulation maximum was3.426994226174429e-7. Historical CE and other
old anchors show the same rounding; this is not a new training failure.

Preserved frozen analyze.py and analysis_execution_v1.log. Mechanical copy
analysis_recovery_v2.py changes only probability normalization checking to
float64 accumulation with8*float32_eps=9.5367431640625e-7 tolerance, and adds a
separate source-hash manifest for the recovery. No tensor is renormalized,
no prediction/class/validation selection/gate changes, and no fit reruns.
Exact checkpoint-reloaded probability equality is still required.
