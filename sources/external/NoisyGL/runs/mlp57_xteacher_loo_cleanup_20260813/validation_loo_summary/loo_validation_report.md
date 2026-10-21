# Formal LOO validation-protocol sensitivity

Both real and shuffled GB targets use distribution_only and leave_one_out=true.
Completeness: 180/180 rows. Test-label contamination: 0. Gate passed: True.

## LOO paired results

- clean_validation citeseer vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +13.16pp, 5/5 positive seeds.
- clean_validation citeseer vs gsdn: +4.88pp, 5/5 positive seeds.
- clean_validation citeseer vs mlp_ce: +11.74pp, 5/5 positive seeds.
- clean_validation citeseer vs mlp_jocor: +5.34pp, 5/5 positive seeds.
- clean_validation citeseer vs postel_mlp_noipl_rm: +9.40pp, 5/5 positive seeds.
- clean_validation cora vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +14.17pp, 5/5 positive seeds.
- clean_validation cora vs gsdn: +1.52pp, 3/5 positive seeds.
- clean_validation cora vs mlp_ce: +13.25pp, 5/5 positive seeds.
- clean_validation cora vs mlp_jocor: +5.77pp, 5/5 positive seeds.
- clean_validation cora vs postel_mlp_noipl_rm: +10.40pp, 5/5 positive seeds.
- clean_validation corafull vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +5.09pp, 5/5 positive seeds.
- clean_validation corafull vs gsdn: +1.59pp, 5/5 positive seeds.
- clean_validation corafull vs mlp_ce: +5.41pp, 5/5 positive seeds.
- clean_validation corafull vs mlp_jocor: +1.69pp, 5/5 positive seeds.
- clean_validation corafull vs postel_mlp_noipl_rm: +0.88pp, 5/5 positive seeds.
- noisy_validation citeseer vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +12.35pp, 5/5 positive seeds.
- noisy_validation citeseer vs gsdn: +4.39pp, 5/5 positive seeds.
- noisy_validation citeseer vs mlp_ce: +11.93pp, 5/5 positive seeds.
- noisy_validation citeseer vs mlp_jocor: +4.32pp, 5/5 positive seeds.
- noisy_validation citeseer vs postel_mlp_noipl_rm: +8.72pp, 5/5 positive seeds.
- noisy_validation cora vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +13.69pp, 5/5 positive seeds.
- noisy_validation cora vs gsdn: +2.01pp, 3/5 positive seeds.
- noisy_validation cora vs mlp_ce: +13.51pp, 5/5 positive seeds.
- noisy_validation cora vs mlp_jocor: +6.05pp, 5/5 positive seeds.
- noisy_validation cora vs postel_mlp_noipl_rm: +10.01pp, 5/5 positive seeds.
- noisy_validation corafull vs frozengrace_gb_mlp_softdist_beta05_loo_shuffled: +4.87pp, 5/5 positive seeds.
- noisy_validation corafull vs gsdn: +0.77pp, 4/5 positive seeds.
- noisy_validation corafull vs mlp_ce: +4.91pp, 5/5 positive seeds.
- noisy_validation corafull vs mlp_jocor: +1.33pp, 5/5 positive seeds.
- noisy_validation corafull vs postel_mlp_noipl_rm: +1.16pp, 5/5 positive seeds.

## Ranking sensitivity

- cora: Kendall tau=1.000, top-2 overlap=2/2.
- citeseer: Kendall tau=0.867, top-2 overlap=1/2.
- corafull: Kendall tau=0.867, top-2 overlap=1/2.
