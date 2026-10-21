# Appendix D2. Beta validation audit

| dataset | noise_type | noise_rate | val_best_method | val_best_beta | val_best_valid | val_best_test | test_best_method | test_best_beta | test_best_valid | test_best_test | beta05_valid | beta05_test | val_best_minus_beta05_valid_pp | test_best_minus_beta05_test_pp | val_and_test_choose_same_beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| citeseer | uniform | 0.2 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.672892 | 0.628866 | frozengrace_gb_mlp_softdist_beta05 | 0.5 | 0.672289 | 0.633849 | 0.672289 | 0.633849 | 0.060241 | 0.0 | False |
| citeseer | uniform | 0.4 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.586145 | 0.545619 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.586145 | 0.545619 | 0.584337 | 0.538316 | 0.180723 | 0.730241 | True |
| cora | uniform | 0.2 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.66 | 0.636834 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.66 | 0.636834 | 0.648889 | 0.629446 | 1.111111 | 0.738786 | True |
| cora | uniform | 0.4 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.582222 | 0.567916 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.582222 | 0.567916 | 0.576296 | 0.555884 | 0.592593 | 1.203166 | True |
| pubmed | uniform | 0.2 | frozengrace_gb_mlp_softdist_beta05 | 0.5 | 0.82618 | 0.825114 | frozengrace_gb_mlp_softdist_beta05 | 0.5 | 0.82618 | 0.825114 | 0.82618 | 0.825114 | 0.0 | 0.0 | True |
| pubmed | uniform | 0.4 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.756773 | 0.742366 | frozengrace_gb_mlp_softdist_beta1 | 1.0 | 0.756773 | 0.742366 | 0.733536 | 0.718687 | 2.323694 | 2.367944 | True |
