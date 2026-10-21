# Offline Clustering Smoke v2: SAGBC-style Soft Affiliation

Decision: **PARTIAL**

## Protocol

- Evaluated datasets in this report: `amazonpho,citeseer,coauthorcs,cora,pubmed`.
- Last command datasets: `coauthorcs,amazonpho`; seeds: `0,1,2,3,4`.
- Frozen GRACE-H is used as clustering space. Labels are evaluation-only.
- Ordinary GB anchors: default max_balls=64, min_ball_size=32, split_improvement=0.05; PubMed override=True.
- Soft affiliation: m=5; similarity S=0.5*center_cosine + 0.3*A^T A + 0.2*edge_density.

## Gate Summary

| Dataset | SAGBC ACC | KMeans trueK ACC | ACC gap | SAGBC NMI | KMeans NMI | NMI ratio | Old GB ACC | Shuffled ACC | Wrong-K adv | G1 | G2 | G3 | G4 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazonpho | 0.4555 | 0.6707 | -0.2152 | 0.4186 | 0.5961 | 0.7022 | 0.4721 | 0.3913 | -0.1518 | 0 | 0 | 0 | 0 | FAIL |
| citeseer | 0.5702 | 0.6395 | -0.0693 | 0.3245 | 0.3782 | 0.8579 | 0.5342 | 0.4097 | +0.0600 | 0 | 1 | 0 | 1 | FAIL |
| coauthorcs | 0.4840 | 0.6603 | -0.1763 | 0.5539 | 0.7378 | 0.7508 | 0.4081 | 0.3746 | -0.1665 | 0 | 1 | 0 | 0 | FAIL |
| cora | 0.5513 | 0.6317 | -0.0804 | 0.4398 | 0.5193 | 0.8469 | 0.4823 | 0.4459 | -0.0518 | 0 | 1 | 0 | 0 | FAIL |
| pubmed | 0.5388 | 0.6766 | -0.1378 | 0.2168 | 0.3108 | 0.6976 | 0.5941 | 0.5193 | +0.0235 | 0 | 0 | 0 | 0 | FAIL |

## Aggregate Metrics

| Dataset | Method | ACC | NMI | ARI | K auto | Balls | Warnings |
|---|---|---:|---:|---:|---:|---:|---|
| amazonpho | kmeans_H_trueK | 0.6707±0.0008 | 0.5961±0.0046 | 0.4739±0.0056 | 3.40 | 64.00 | ok |
| amazonpho | spectral_knn_H_trueK | 0.7565±0.0000 | 0.6468±0.0000 | 0.5588±0.0000 | 3.40 | 64.00 | ok |
| amazonpho | kmeans_H_autoK | 0.4746±0.0412 | 0.4317±0.0585 | 0.3069±0.0500 | 3.40 | 64.00 | ok |
| amazonpho | old_gb_modularity | 0.4721±0.0026 | 0.4037±0.0338 | 0.2442±0.0353 | 3.00 | 64.00 | ok |
| amazonpho | sagbc_soft_affiliation | 0.4555±0.0060 | 0.4186±0.0168 | 0.2421±0.0159 | 3.40 | 64.00 | ok |
| amazonpho | sagbc_shuffled_ball | 0.3913±0.0215 | 0.2940±0.0469 | 0.1623±0.0190 | 3.00 | 64.00 | ok |
| citeseer | kmeans_H_trueK | 0.6395±0.0009 | 0.3782±0.0009 | 0.3808±0.0013 | 4.40 | 63.80 | ok |
| citeseer | spectral_knn_H_trueK | 0.6658±0.0000 | 0.4113±0.0000 | 0.4205±0.0000 | 4.40 | 63.80 | ok |
| citeseer | kmeans_H_autoK | 0.6200±0.0444 | 0.3516±0.0366 | 0.3563±0.0377 | 4.40 | 63.80 | ok |
| citeseer | old_gb_modularity | 0.5342±0.0285 | 0.2942±0.0319 | 0.2770±0.0371 | 4.40 | 63.80 | ok |
| citeseer | sagbc_soft_affiliation | 0.5702±0.0273 | 0.3245±0.0220 | 0.2928±0.0255 | 4.40 | 63.80 | ok |
| citeseer | sagbc_shuffled_ball | 0.4097±0.0353 | 0.2014±0.0458 | 0.1598±0.0356 | 3.60 | 63.80 | ok |
| coauthorcs | kmeans_H_trueK | 0.6603±0.0206 | 0.7378±0.0056 | 0.5700±0.0096 | 3.80 | 64.00 | ok |
| coauthorcs | spectral_knn_H_trueK | 0.7639±0.0002 | 0.7457±0.0001 | 0.6249±0.0005 | 3.80 | 64.00 | ok |
| coauthorcs | kmeans_H_autoK | 0.4460±0.0211 | 0.5300±0.0274 | 0.3602±0.0235 | 3.80 | 64.00 | ok |
| coauthorcs | old_gb_modularity | 0.4081±0.0177 | 0.4673±0.0214 | 0.2307±0.0509 | 3.20 | 64.00 | ok |
| coauthorcs | sagbc_soft_affiliation | 0.4840±0.0313 | 0.5539±0.0353 | 0.3737±0.0562 | 3.80 | 64.00 | ok |
| coauthorcs | sagbc_shuffled_ball | 0.3746±0.0221 | 0.3488±0.0305 | 0.2168±0.0358 | 3.60 | 64.00 | ok |
| cora | kmeans_H_trueK | 0.6317±0.0014 | 0.5193±0.0018 | 0.4066±0.0016 | 3.80 | 52.40 | ok |
| cora | spectral_knn_H_trueK | 0.6041±0.0000 | 0.4733±0.0000 | 0.2836±0.0000 | 3.80 | 52.40 | ok |
| cora | kmeans_H_autoK | 0.5734±0.0454 | 0.4314±0.0395 | 0.3522±0.0327 | 3.80 | 52.40 | ok |
| cora | old_gb_modularity | 0.4823±0.0119 | 0.3404±0.0059 | 0.2889±0.0102 | 3.00 | 52.40 | ok |
| cora | sagbc_soft_affiliation | 0.5513±0.0384 | 0.4398±0.0234 | 0.3514±0.0406 | 3.80 | 52.40 | ok |
| cora | sagbc_shuffled_ball | 0.4459±0.0409 | 0.3205±0.0404 | 0.2379±0.0826 | 3.40 | 52.40 | ok |
| pubmed | kmeans_H_trueK | 0.6766±0.0000 | 0.3108±0.0000 | 0.2881±0.0000 | 4.40 | 256.00 | ok |
| pubmed | spectral_knn_H_trueK | 0.6481±0.0000 | 0.2886±0.0000 | 0.2440±0.0000 | 4.40 | 256.00 | ok |
| pubmed | kmeans_H_autoK | 0.4987±0.0139 | 0.2769±0.0372 | 0.2136±0.0449 | 4.40 | 256.00 | ok |
| pubmed | old_gb_modularity | 0.5941±0.0102 | 0.2221±0.0200 | 0.1991±0.0144 | 4.00 | 256.00 | ok |
| pubmed | sagbc_soft_affiliation | 0.5388±0.0358 | 0.2168±0.0259 | 0.1952±0.0243 | 4.40 | 256.00 | ok |
| pubmed | sagbc_shuffled_ball | 0.5193±0.0460 | 0.1735±0.0438 | 0.1436±0.0464 | 4.60 | 256.00 | ok |

## Wrong-K Robustness

| Dataset | SAGBC ACC | KMeans wrong-K ACC | ACC advantage | NMI advantage |
|---|---:|---:|---:|---:|
| amazonpho | 0.4555 | 0.6073 | -0.1518±0.0729 | -0.1509 |
| citeseer | 0.5702 | 0.5103 | +0.0600±0.0603 | -0.0001 |
| coauthorcs | 0.4840 | 0.6505 | -0.1665±0.0622 | -0.1726 |
| cora | 0.5513 | 0.6031 | -0.0518±0.0381 | -0.0636 |
| pubmed | 0.5388 | 0.5153 | +0.0235±0.0406 | -0.0648 |

## Decision Notes

- Only G2 passed broadly; this supports an improvement over old hard balls but not a clustering mainline.
- Gate counts: G1=0/5, G2=3/5, G3=0/5, G4=1/5.
- G1: NMI >= 0.9 * kmeans_H_trueK NMI and ACC gap >= -5pp.
- G2: SAGBC beats both old hard GB modularity and shuffled-ball control by >= +3pp ACC.
- G3: SAGBC beats kmeans_H_trueK or spectral_knn_H_trueK.
- G4: SAGBC beats mis-specified-K k-means by >= +5pp average ACC.
