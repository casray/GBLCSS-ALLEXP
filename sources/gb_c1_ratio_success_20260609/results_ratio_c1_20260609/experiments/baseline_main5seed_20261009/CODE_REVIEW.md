# Code and protocol review

## Blocking issues fixed in new files only

1. Old common.py hardcoded seed0 for split, noise and final metrics, plus rate .4.
   New protocol.py passes seed/rate through all three stages and keeps original
   random-choice index order. All 25 .4 split/noise fingerprints match replay.
   All 50 .4/.6 cells also match independently AST-loaded original NoisyGL noise
   functions, and have zero val/test label changes. Raw assets additionally hashed.
2. Old TINED config and three internal RNG resets forced zero. New tined_formal.py
   records/uses the actual seed in teacher initialization, student initialization
   and student training. Positions of resets are unchanged.
3. Official TINED train_and_eval.py distill_run_transductive sorts idx_l before
   CE minibatches. Prior gcn_adapter.py omitted this. New version sorts only for
   minibatching after ordered noise generation; labels/splits remain unchanged.
   Previous TINED student seed0 is invalid for this corrected batch policy and
   is not reused. This is independent of whether its old accuracy was high/low.
4. Old output paths lacked seed/rate identity. New paths isolate all cells;
   resume checks source/protocol/artifact hashes and refuses partial overwrite.

## SimMLP core: conditional pass

Author Bootstrap stop-gradient applies to graph target h, not reconstruction.
Alignment gradients reach encoder/predictor; reconstruction reaches encoder,
graph aggregator and decoder. Labels/masks are absent from SSL input.
Author Tensor-edge sparse propagation preserves self-loops, normalization and
gradient semantics. Common deployed first block matches ReLU/BN/dropout exactly.
Head is registered/saved separately and excluded from SSL; probe freezes encoder
parameters, BN state and dropout. Bootstrap still performs its own normalization.
Coauthor-CS has recon_lambda=0 by the retained author preset. Do not report
reconstruction as active on every dataset.

This is a matched-backbone adaptation, not the original official AdamW/cosine/
three-FC deployment. No bug found in the reviewed core. Fixed pretrained encoder
across downstream seeds is explicitly part of this experiment; it does not sample
SSL pretraining variability. Classification heads are new per downstream seed.

## TINED core: conditional pass with native-student exception

Common PyG GCN forward equals extracted FT/GP composition. For CoraFull70 >64,
sequence is FT-GP-GP-FT; others FT-GP-FT-GP. Deferred teacher bias is excluded
from traces as upstream. Rowwise normalized Dirichlet energy has detached norms;
its value and gradient match direct edge differences. Teacher weight/bias injection
uses copy_ and preserves parameter registration; teacher/student storage does not
alias. Native GP identity initialization, separate CE/KL/DE Adam steps (including
zero-weight CE), dropout/norm and predefined method-knob presets are preserved.

Keep the native four-FC student. It cannot be described as the common two-FC MLP.
Teacher and student budget200, raw inputs, split/noise, Adam base settings and
validation selection are matched. Method-specific depth/dropout/minibatching and
injected-layer LR ratios are explicit exceptions, not hidden equivalence claims.
No GCN-specific author best preset exists in the vendored best_records file;
predeclared SAGE knobs are reused and disclosed. Complete DGL runtime parity has
not been established. Late Cora/Photo degradation under fixed parameters is not
proof of an implementation bug; it is retained, not tuned away based on test.

## GSSC: do not expand present port as a strict common-backbone main row

NeighContrast has an input Linear, an additional hidden Linear, and output Linear:
three deployed FCs for n_layers=2. There is also an auxiliary training-only inf
head. Classifier warmup100 plus main200 gives300 supervised epochs, excluding
edge pretraining100. Validation is last-tie, not main first-tie. Metadata labels
all model parameters (including unused inf head) as inference parameters and
return value also includes mixup/sampler. Seed/config entry is hardcoded zero.

Thirteen CPU tests pass (one opt-in GPU test skipped): author architecture parity,
mixup, sparse dot-product gradients, straight-through sampler, homophily gradient,
contrast objective, train-only class counts and validation label isolation.
These support port correctness but do not make its protocol equivalent. A future
adaptation must explicitly choose a common two-FC deployment/total supervised
budget or disclose native exceptions; neither is silently chosen this round.

## Verification and claim limits

SimMLP4 + TINED5 + formal4 =13 tests passed. GSSC13 additional CPU tests passed,
one GPU test skipped. Two requested-method GPU two-epoch smoke runs passed.
The preflight covers50 protocol cells without full training. Training outputs
will receive checkpoint/probability/selection audits by report_results.py.
No test-driven hyperparameter search, no seed exclusion, no manuscript edits.
Passing tests cannot prove an arbitrary program bug-free or validate every
possible setting; conclusions apply to the audited configured adapters only.
