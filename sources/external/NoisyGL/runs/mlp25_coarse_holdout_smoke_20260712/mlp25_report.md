# MLP-25 Coarse-Ball Holdout Smoke

| Dataset | Arm | Test (%) | Balls | Train votes/ball |
|---|---|---:|---:|---:|
| attribwiki | candidate | 63.64 | 22 | 21.86 |
| attribwiki | candidate_shuffled | 60.67 | 22 | 21.86 |
| attribwiki | fixed | 65.60 | 41 | 11.73 |
| attribwiki | vote_target | 60.31 | 25 | 19.24 |
| attribwiki | vote_target_shuffled | 59.60 | 25 | 19.24 |
| citeseerfull | candidate | 60.69 | 38 | 22.26 |
| citeseerfull | candidate_shuffled | 47.42 | 38 | 22.26 |
| citeseerfull | fixed | 59.81 | 64 | 13.22 |
| citeseerfull | vote_target | 60.08 | 43 | 19.67 |
| citeseerfull | vote_target_shuffled | 46.71 | 43 | 19.67 |
| coraml | candidate | 52.67 | 20 | 29.95 |
| coraml | candidate_shuffled | 46.80 | 20 | 29.95 |
| coraml | fixed | 51.91 | 62 | 9.66 |
| coraml | vote_target | 53.15 | 30 | 19.97 |
| coraml | vote_target_shuffled | 46.66 | 30 | 19.97 |

## Gate decisions

| Candidate | Pass | Mean delta vs fixed | Reasons |
|---|---:|---:|---|
| vote_target | 0 | -1.26pp | fewer_than_2_of_3_gain_1pp;regression_over_0p5pp |
| candidate | 0 | -0.11pp | fewer_than_2_of_3_gain_1pp;regression_over_0p5pp |

## Final decision

Neither candidate passes the preregistered seed-0 gate. Do not run five seeds, do not change the fixed-64 default, and close the granularity axis. Both real candidates remain above their own shuffled controls on most/all holdouts, so the negative result concerns transferable accuracy gain rather than the existence of local structure.
