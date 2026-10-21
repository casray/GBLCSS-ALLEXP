# Ellipsoid selected downstream seed0 sanity report

Scope: seed0 downstream sanity only. This is not a 5-seed result and not a paper claim.

## Protocol

- Datasets: DBLP, PubMed, AmazonPhoto, AmazonCom
- Seed: 0 only
- Noise: uniform 0.4, train scope
- Split: percent 0.2 / 0.1 / 0.7
- Device: cuda:0
- Default controls: gcn_xh_ce, current SoftDist, current shuffled
- Selected methods: ellipsoid and ellipsoid shuffled with dataset-specific max_balls/split_improvement

## Selected settings

| Dataset | max_balls | split_improvement |
|---|---:|---:|
| dblp | 128 | 0.001 |
| pubmed | 512 | 0.01 |
| amazonpho | 64 | 0.05 |
| amazoncom | 64 | 0.001 |

## Seed0 results

| Dataset | GCN[XH] | Current SoftDist | Current shuffled | Selected ellipsoid | Selected shuffled | Sel - sel shuf | Sel - current | Sel - current shuf | Sel - GCN | Sanity decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazoncom | 0.8564 | 0.8558 | 0.8578 | 0.8330 | 0.8579 | -2.49pp | -2.29pp | -2.48pp | -2.35pp | STOP for now |
| amazonpho | 0.9109 | 0.8964 | 0.9001 | 0.8986 | 0.8949 | +0.37pp | +0.22pp | -0.15pp | -1.23pp | borderline GO candidate |
| dblp | 0.8429 | 0.8426 | 0.8426 | 0.8424 | 0.8320 | +1.04pp | -0.02pp | -0.02pp | -0.05pp | GO 5-seed candidate |
| pubmed | 0.8249 | 0.8252 | 0.8241 | 0.8251 | 0.8294 | -0.43pp | -0.01pp | +0.09pp | +0.01pp | inconclusive / low priority |

## Integrity checks

- amazoncom: status_ok=True, same_split_hash=True, same_noise_hash=True
- amazonpho: status_ok=True, same_split_hash=True, same_noise_hash=True
- dblp: status_ok=True, same_split_hash=True, same_noise_hash=True
- pubmed: status_ok=True, same_split_hash=True, same_noise_hash=True

## Interpretation

- DBLP is the cleanest seed0 pass: selected ellipsoid beats selected shuffled by +1.04pp and is essentially tied with current SoftDist.
- PubMed fails the real-vs-shuffled sanity check on seed0: selected shuffled is +0.43pp above selected ellipsoid.
- AmazonPhoto is mixed: selected ellipsoid beats selected shuffled and current SoftDist, but remains below GCN[XH] and current shuffled.
- AmazonCom is a clear seed0 stop: selected ellipsoid is far below selected shuffled and current controls.

Recommended next action: only DBLP is a strong 5-seed candidate. AmazonPhoto can be run as a low-priority/borderline check if time is cheap; PubMed and AmazonCom should not enter 5-seed from this seed0 evidence without a revised reason.
