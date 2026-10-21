# MLP-45 LOO Grouping Accuracy-Cost Pareto Report

All accuracy rows use uniform 0.4, train-only noise, five paired seeds, tie-safe targets, and the same frozen GRACE representation. GB and k-means use leave-one-out votes; kNN excludes the query node.

| Dataset | LOO-GB | Exact kNN | LOO-k-means | GB-best alternative |
|---|---:|---:|---:|---:|
| Cora | 57.49+/-0.76 | 57.21+/-1.19 | 56.58+/-1.23 | +0.27pp |
| CiteSeer | 55.24+/-2.04 | 56.17+/-3.24 | 56.18+/-1.78 | -0.94pp |
| CoraFull | 37.39+/-0.51 | 40.50+/-0.51 | 39.26+/-0.48 | -3.11pp |
| Coauthor-CS | 90.71+/-0.59 | 90.83+/-0.57 | 91.01+/-0.64 | -0.29pp |
| Amazon-Photo | 82.12+/-1.08 | 82.23+/-1.20 | 81.68+/-0.66 | -0.11pp |

## Cost summary against exact kNN

- GB has lower end-to-end construction time on 3/5 datasets.
- GB has lower measured peak RSS on 3/5 datasets.
- GB has lower grouping-structure storage on 3/5 datasets.
- No maintained approximate-kNN package was available in the frozen environment, so no approximate-kNN timing is claimed.

## Preregistered decision

- Decision: `implementation_profile_only`.
- Mean GB gap to the best alternative: -0.84pp.
- Datasets within 1pp of the best alternative: 4/5.

Granular balls remain an auditable fixed-budget compression of local consensus, but this experiment does not establish a general accuracy or runtime advantage over exact kNN.
