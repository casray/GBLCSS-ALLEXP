# MLP-43 Corrected kNN-Teacher Audit

Protocol: Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only label noise, percent split 0.2/0.1/0.7, seeds 0--4. The corrected kNN teacher excludes the query node. The other four teacher arms are reused only after split/noise-hash equality is verified.

| Dataset | Noise | Teacher | Student test (%) |
|---|---:|---|---:|
| citeseer | 0.4 | GCN(X)+CE (GLNN-style) | 66.09 +/- 1.38 |
| citeseer | 0.4 | GCN[X||H]+CE | 68.81 +/- 1.12 |
| citeseer | 0.4 | kNN-consensus teacher (exclude self) | 69.65 +/- 1.61 |
| citeseer | 0.4 | ball-consensus teacher | 69.40 +/- 1.08 |
| citeseer | 0.4 | shuffled-consensus teacher | 68.60 +/- 1.14 |
| citeseer | 0.6 | GCN(X)+CE (GLNN-style) | 53.30 +/- 4.42 |
| citeseer | 0.6 | GCN[X||H]+CE | 61.04 +/- 3.08 |
| citeseer | 0.6 | kNN-consensus teacher (exclude self) | 62.90 +/- 2.26 |
| citeseer | 0.6 | ball-consensus teacher | 62.28 +/- 2.30 |
| citeseer | 0.6 | shuffled-consensus teacher | 61.07 +/- 2.76 |
| cora | 0.4 | GCN(X)+CE (GLNN-style) | 77.95 +/- 2.62 |
| cora | 0.4 | GCN[X||H]+CE | 80.23 +/- 1.57 |
| cora | 0.4 | kNN-consensus teacher (exclude self) | 81.61 +/- 0.74 |
| cora | 0.4 | ball-consensus teacher | 80.60 +/- 0.69 |
| cora | 0.4 | shuffled-consensus teacher | 78.81 +/- 1.63 |
| cora | 0.6 | GCN(X)+CE (GLNN-style) | 65.50 +/- 3.28 |
| cora | 0.6 | GCN[X||H]+CE | 69.44 +/- 2.18 |
| cora | 0.6 | kNN-consensus teacher (exclude self) | 71.80 +/- 2.63 |
| cora | 0.6 | ball-consensus teacher | 71.27 +/- 2.46 |
| cora | 0.6 | shuffled-consensus teacher | 68.51 +/- 1.82 |
| corafull | 0.4 | GCN(X)+CE (GLNN-style) | 58.74 +/- 0.82 |
| corafull | 0.4 | GCN[X||H]+CE | 59.45 +/- 0.63 |
| corafull | 0.4 | kNN-consensus teacher (exclude self) | 58.58 +/- 0.55 |
| corafull | 0.4 | ball-consensus teacher | 58.22 +/- 0.69 |
| corafull | 0.4 | shuffled-consensus teacher | 58.98 +/- 0.41 |
| corafull | 0.6 | GCN(X)+CE (GLNN-style) | 51.63 +/- 2.02 |
| corafull | 0.6 | GCN[X||H]+CE | 52.75 +/- 0.88 |
| corafull | 0.6 | kNN-consensus teacher (exclude self) | 52.65 +/- 0.93 |
| corafull | 0.6 | ball-consensus teacher | 51.82 +/- 0.86 |
| corafull | 0.6 | shuffled-consensus teacher | 52.20 +/- 0.72 |

## Paired deltas

- citeseer/u0.4/ball_minus_knn: -0.25pp, 2/5 positive seeds.
- citeseer/u0.4/ball_minus_gcn_xh: +0.59pp, 3/5 positive seeds.
- citeseer/u0.4/ball_minus_shuffled: +0.80pp, 4/5 positive seeds.
- citeseer/u0.4/knn_minus_gcn_xh: +0.84pp, 4/5 positive seeds.
- citeseer/u0.6/ball_minus_knn: -0.62pp, 1/5 positive seeds.
- citeseer/u0.6/ball_minus_gcn_xh: +1.24pp, 4/5 positive seeds.
- citeseer/u0.6/ball_minus_shuffled: +1.21pp, 4/5 positive seeds.
- citeseer/u0.6/knn_minus_gcn_xh: +1.86pp, 5/5 positive seeds.
- cora/u0.4/ball_minus_knn: -1.01pp, 0/5 positive seeds.
- cora/u0.4/ball_minus_gcn_xh: +0.37pp, 4/5 positive seeds.
- cora/u0.4/ball_minus_shuffled: +1.79pp, 4/5 positive seeds.
- cora/u0.4/knn_minus_gcn_xh: +1.38pp, 4/5 positive seeds.
- cora/u0.6/ball_minus_knn: -0.53pp, 2/5 positive seeds.
- cora/u0.6/ball_minus_gcn_xh: +1.84pp, 5/5 positive seeds.
- cora/u0.6/ball_minus_shuffled: +2.77pp, 5/5 positive seeds.
- cora/u0.6/knn_minus_gcn_xh: +2.36pp, 3/5 positive seeds.
- corafull/u0.4/ball_minus_knn: -0.36pp, 1/5 positive seeds.
- corafull/u0.4/ball_minus_gcn_xh: -1.23pp, 0/5 positive seeds.
- corafull/u0.4/ball_minus_shuffled: -0.75pp, 1/5 positive seeds.
- corafull/u0.4/knn_minus_gcn_xh: -0.87pp, 0/5 positive seeds.
- corafull/u0.6/ball_minus_knn: -0.83pp, 0/5 positive seeds.
- corafull/u0.6/ball_minus_gcn_xh: -0.93pp, 0/5 positive seeds.
- corafull/u0.6/ball_minus_shuffled: -0.38pp, 1/5 positive seeds.
- corafull/u0.6/knn_minus_gcn_xh: -0.10pp, 2/5 positive seeds.

## Matched-k and protocol audit

- citeseer/u0.4/seed0: k=10, exclude_self=1, split=98b2579310f1, noise=23c3e3d7b44f.
- citeseer/u0.4/seed1: k=10, exclude_self=1, split=fab73b852f0c, noise=0e275078053b.
- citeseer/u0.4/seed2: k=10, exclude_self=1, split=dc4a52d01c14, noise=df5a14140293.
- citeseer/u0.4/seed3: k=10, exclude_self=1, split=82594f6c32c3, noise=935249477846.
- citeseer/u0.4/seed4: k=11, exclude_self=1, split=1d7c2ceabff1, noise=2fe6fd009b87.
- cora/u0.4/seed0: k=10, exclude_self=1, split=fe241a5bd04c, noise=eb786f64ad9b.
- cora/u0.4/seed1: k=10, exclude_self=1, split=f2c5eac42097, noise=2cd3b3bd23f3.
- cora/u0.4/seed2: k=10, exclude_self=1, split=49fa3991fafb, noise=3ef49b8e22f7.
- cora/u0.4/seed3: k=10, exclude_self=1, split=103fdc865c84, noise=487f3554df46.
- cora/u0.4/seed4: k=10, exclude_self=1, split=1d6d007e584c, noise=26ce446eecf3.
- corafull/u0.4/seed0: k=62, exclude_self=1, split=947dc6d3f86b, noise=c8b2bcd32274.
- corafull/u0.4/seed1: k=62, exclude_self=1, split=6cace3e4e28c, noise=5d99888409af.
- corafull/u0.4/seed2: k=62, exclude_self=1, split=8637097ec766, noise=30f5b138be57.
- corafull/u0.4/seed3: k=62, exclude_self=1, split=011e5c9484bf, noise=d27f06e98bea.
- corafull/u0.4/seed4: k=62, exclude_self=1, split=c03af05c9bff, noise=79a92903f0c9.
- citeseer/u0.6/seed0: k=10, exclude_self=1, split=98b2579310f1, noise=02ca1a71e920.
- citeseer/u0.6/seed1: k=10, exclude_self=1, split=fab73b852f0c, noise=d4b7546aa11b.
- citeseer/u0.6/seed2: k=10, exclude_self=1, split=dc4a52d01c14, noise=76c8dcdfa3ac.
- citeseer/u0.6/seed3: k=10, exclude_self=1, split=82594f6c32c3, noise=a3e99b3fcd37.
- citeseer/u0.6/seed4: k=11, exclude_self=1, split=1d7c2ceabff1, noise=4f4b79017a2b.
- cora/u0.6/seed0: k=10, exclude_self=1, split=fe241a5bd04c, noise=1840b1e631d8.
- cora/u0.6/seed1: k=10, exclude_self=1, split=f2c5eac42097, noise=ecf82568f9e6.
- cora/u0.6/seed2: k=10, exclude_self=1, split=49fa3991fafb, noise=732a08931d61.
- cora/u0.6/seed3: k=10, exclude_self=1, split=103fdc865c84, noise=89482a74433f.
- cora/u0.6/seed4: k=10, exclude_self=1, split=1d6d007e584c, noise=7997e98b6bc2.
- corafull/u0.6/seed0: k=62, exclude_self=1, split=947dc6d3f86b, noise=d8826084bac3.
- corafull/u0.6/seed1: k=62, exclude_self=1, split=6cace3e4e28c, noise=756c33fdfec8.
- corafull/u0.6/seed2: k=62, exclude_self=1, split=8637097ec766, noise=fa5b7f148655.
- corafull/u0.6/seed3: k=62, exclude_self=1, split=011e5c9484bf, noise=9d3b488dd358.
- corafull/u0.6/seed4: k=62, exclude_self=1, split=c03af05c9bff, noise=5c81f0d07290.

## Interpretation

- The corrected kNN arm is a pointwise local-consensus alternative, not evidence that the granular-ball partition is uniquely optimal.
- All five arms use identical per-seed splits and noisy labels. Only the teacher construction differs.
