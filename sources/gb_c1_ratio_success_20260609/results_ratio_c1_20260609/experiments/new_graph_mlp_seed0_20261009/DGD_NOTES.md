# DGD implementation prerequisite audit

Author repository: https://github.com/KangL-G/Dynamic-Gradient-Distillation
Pinned commit: cdffc16861193ed7ea64054388b59f6ef6ef07d3
Paper DOI: 10.1016/j.inffus.2026.104163

Status: blocked before training, not an accuracy result or a failure of the method.

## Verified source gaps

1. `train_student.py` and `DGD.py` call `dtw_dsasm`; no definition or import exists in the repository. A differently named `gradient_fusion` is available, but equivalence is unverified.
2. Cora/CiteSeer/PubMed GCN-to-MLP presets set alpha=18. The implemented objective is alpha*CE+(1-alpha)*KL, yielding 18*CE-17*KL. No evidence justifies silently changing 18 to 0.18 or changing the formula.
3. Generator loss evaluates sampled actions under their own distribution with a fixed decaying coefficient; it does not consume discriminator output or teacher-gradient targets. The required imitation reward is not documented in available source.
4. Generator checkpoint snapshots retain live parameter references; deterministic deep copies are an identifiable engineering fix.
5. Student Hessian indexing `second_grad[2*i+4]` does not match the optimizer's MLP parameter list. Parameter-identity mapping is fixable, but the intended second-order fusion formula remains uncertain.
6. The attention helper resets its accumulator inside the input loop and retains only the last input contribution. The intended combination rule is not established.
7. Student CE is evaluated on all labels; the common protocol must instead restrict supervision to noisy training labels. Evaluation also references undefined variables.

## Evidence limits

Publisher search snippets confirm GAIL gradient mapping and two-stage fusion but do not provide the method equations. Accessible Elsevier metadata contains no full text. Public repository history contains no corrected implementation of the relevant files. A full paper or author clarification is required to resolve the algorithmic ambiguities.

Do not run plain KD, guessed rewards, disabled gradient modules, or guessed fusion as DGD. No CPU fallback, test-label supervision, or fabricated score is permitted. Continue GSSC and TINED while awaiting the missing evidence.
