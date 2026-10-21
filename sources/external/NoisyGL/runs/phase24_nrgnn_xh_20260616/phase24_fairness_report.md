# Phase 24: NRGNN [X||H] Fairness Control

## Protocol

- New run CSV: `runs/phase24_nrgnn_xh_20260616/nrgnn_xh_5seed.csv`
- Existing NoisyGL anchors: `runs/softdist_5seed_rocm_20260610/softdist_5seed_rocm_raw_summary_20260610.csv`
- Existing XH anchors: `runs/xh_5seed_rocm_20260611/xh_5seed_raw_summary_20260611.csv`
- Split: `percent`, train/val/test = `0.2/0.1/0.7`
- Noise: `uniform`, noise rate = `0.4`
- Seeds: `0,1,2,3,4`

## Fairness Matrix

| Dataset | Method | H? | Ball? | Test acc | Role |
|---|---|---|---|---:|---|
| citeseer | NRGNN | No | No | 0.7027 ± 0.0108 | strong NoisyGL baseline |
| citeseer | NRGNN [X||H] | Yes | No | 0.7090 ± 0.0187 | H contribution to NRGNN |
| citeseer | GCN+CE [X||H] | Yes | No | 0.6761 ± 0.0181 | H contribution to plain CE |
| citeseer | SoftDist [X||H] | Yes | Yes | 0.6891 ± 0.0159 | full method |
| cora | NRGNN | No | No | 0.8107 ± 0.0230 | strong NoisyGL baseline |
| cora | NRGNN [X||H] | Yes | No | 0.8139 ± 0.0170 | H contribution to NRGNN |
| cora | GCN+CE [X||H] | Yes | No | 0.8027 ± 0.0069 | H contribution to plain CE |
| cora | SoftDist [X||H] | Yes | Yes | 0.8137 ± 0.0067 | full method |

## Paired Deltas

| Dataset | Comparison | Mean delta | Positive seeds | Per-seed deltas |
|---|---|---:|---:|---|
| citeseer | NRGNN [X||H] - NRGNN | +0.64pp | 4/5 | 0.0005 -0.0391 0.0108 0.0292 0.0305 |
| citeseer | SoftDist [X||H] - NRGNN [X||H] | -1.99pp | 1/5 | -0.0159 0.0077 -0.0438 -0.0095 -0.0382 |
| citeseer | SoftDist [X||H] - GCN+CE [X||H] | +1.30pp | 5/5 | 0.0069 0.0176 0.0181 0.0159 0.0065 |
| citeseer | SoftDist [X||H] - SoftDist [X||H] shuffled | +1.95pp | 5/5 | 0.0206 0.0094 0.0353 0.0141 0.0181 |
| citeseer | GCN+CE [X||H] - GCN+CE | +2.31pp | 4/5 | 0.0455 0.0077 -0.0004 0.0249 0.0378 |
| cora | NRGNN [X||H] - NRGNN | +0.33pp | 3/5 | 0.0032 0.0127 0.0327 -0.0068 -0.0254 |
| cora | SoftDist [X||H] - NRGNN [X||H] | -0.02pp | 3/5 | -0.0285 -0.0064 0.0095 0.0137 0.0106 |
| cora | SoftDist [X||H] - GCN+CE [X||H] | +1.10pp | 4/5 | -0.0048 0.0132 0.0201 0.0121 0.0143 |
| cora | SoftDist [X||H] - SoftDist [X||H] shuffled | +2.16pp | 5/5 | 0.0253 0.0269 0.0333 0.0058 0.0169 |
| cora | GCN+CE [X||H] - GCN+CE | +1.86pp | 5/5 | 0.0132 0.0105 0.0343 0.0148 0.0200 |

## Decision

- Cora: SoftDist [X||H] is effectively tied with NRGNN [X||H]; the mean paired delta is -0.02pp with 3/5 positive seeds.
- CiteSeer: NRGNN [X||H] is stronger; the mean paired delta for SoftDist [X||H] minus NRGNN [X||H] is -1.99pp.
- Interpretation: Phase 24 does not support a strong claim that SoftDist beats a fully H-augmented NRGNN. It does support the narrower, honest claim that SoftDist remains competitive on Cora while being much simpler, and that H alone explains a substantial part of the CiteSeer gap.

## Artifacts

- Aggregate: `runs\phase24_nrgnn_xh_20260616\phase24_fairness_aggregate.csv`
- Paired deltas: `runs\phase24_nrgnn_xh_20260616\phase24_fairness_paired_deltas.csv`
- Combined raw rows: `runs\phase24_nrgnn_xh_20260616\phase24_fairness_combined_raw.csv`
