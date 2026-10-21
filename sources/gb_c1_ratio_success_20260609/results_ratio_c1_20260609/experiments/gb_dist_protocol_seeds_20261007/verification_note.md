# Teacher reload numerical tolerance

The first post-training audit failed strict bit equality for Cora uniform .4
seed1 teacher: max absolute probability difference 1.1920928955078125e-7.
The original failure log is retained as analysis.log. No training code,
checkpoint, saved probability or experiment gate was changed.

The verification now requires teacher probability maximum absolute error <=1e-6
AND identical argmax predictions for ALL nodes, plus matching selected epoch
and validation count. All eighty student checkpoints still require exact
bit equality. This is a numerical verification amendment, not model selection.
The precise floating-point kernel responsible has not been isolated.
Detailed per-check differences are recorded in reload_audit.json.
