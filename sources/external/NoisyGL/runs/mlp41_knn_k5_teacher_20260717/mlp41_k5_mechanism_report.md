# MLP-41 k=5 Mechanism Diagnostic

## Question

Why can a fixed `k=5` kNN teacher match or exceed the original vote-matched kNN teacher in several Table-4 settings?

## Scope

This is a clean-label-dependent post-hoc diagnostic. Clean labels are used only to explain already completed experiments and never enter training, validation selection, or target construction.

## First correction: the apparent gain is small

Relative to the original vote-matched kNN teacher, fixed k=5 changes student accuracy by:

| Dataset | u0.4 | u0.6 |
|---|---:|---:|
| Cora | +0.24pp | +0.06pp |
| CiteSeer | -0.75pp | +0.45pp |
| CoraFull | +0.49pp | +0.52pp |

The largest apparent gains in the temporary Table 4 were measured against the ball teacher, not against the original kNN teacher. Only CoraFull gives a consistent 4/5-seed advantage over vote-matched kNN.

## Diagnostic evidence

| Dataset/rate | k | External clean agreement | Mean alpha | Pseudo accuracy | Brier | NCR | CDR | Own-label lower-bound weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora u0.4 k5 / matched | 5 / 10 | .786 / .759 | .486 / .434 | .720 / .789 | .545 / .520 | .575 / .722 | .181 / .167 | .611 / .609 |
| Cora u0.6 k5 / matched | 5 / 10 | .786 / .759 | .368 / .288 | .472 / .577 | .906 / .915 | .326 / .499 | .306 / .304 | .706 / .741 |
| CiteSeer u0.4 k5 / matched | 5 / 10.2 | .633 / .612 | .435 / .372 | .596 / .658 | .598 / .584 | .426 / .575 | .288 / .285 | .652 / .665 |
| CiteSeer u0.6 k5 / matched | 5 / 10.2 | .633 / .612 | .360 / .276 | .422 / .480 | .919 / .937 | .286 / .386 | .373 / .377 | .712 / .751 |
| CoraFull u0.4 k5 / matched | 5 / 62 | .485 / .317 | .419 / .254 | .546 / .513 | .602 / .619 | .396 / .495 | .353 / .475 | .665 / .750 |
| CoraFull u0.6 k5 / matched | 5 / 62 | .485 / .317 | .314 / .168 | .381 / .498 | .954 / 1.013 | .253 / .483 | .422 / .480 | .749 / .835 |

## Interpretation

1. **The current kNN target includes the node itself.** In at least 97.44% of training cases, top-1 is the query node. Small k therefore changes both neighborhood locality and self-label feedback.
2. **CoraFull is the clearest mechanism case.** With 70 classes, expanding from k=5 to about 62 sharply reduces external clean-label agreement (`.485→.317`). Fixed k=5 lowers CDR and Brier at both noise rates, consistent with its stable downstream advantage.
3. **Small k raises the discrete vote margin and alpha.** It is not simply more conservative. The higher alpha can move probability mass more strongly, while tighter neighbors reduce cross-class contamination. The net effect depends on the dataset.
4. **Cora and CiteSeer do not show a universal k=5 advantage.** Their target-quality metrics and per-seed accuracy directions are mixed. Most differences are below 1pp.
5. **Distillation changes the scale behavior.** In MLP-11 direct target training, k=5 was lower than k=20 on all four tested datasets. In Table 4, a GCN[X||H] teacher is selected using clean validation and its predictions over all nodes are distilled to the student. This pipeline can absorb local-target variance, but the direct and teacher experiments also use different target-beta settings, so this is a supported explanation rather than a causal proof.

## Decision

Do not claim that k=5 is generally better. Use fixed k=5 as a fixed-budget method comparison and retain vote-matched kNN as a vote-count-controlled diagnostic. Report both roles explicitly.
