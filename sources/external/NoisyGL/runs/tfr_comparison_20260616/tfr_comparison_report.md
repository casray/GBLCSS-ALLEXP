# Phase 25: TFR Same-Protocol Comparison

## Protocol

- External repository: `https://github.com/eaglelab-zju/TFR.git`
- Cloned commit: `d7e9ff8a58d457b519f068c3d170cd37579c8bd8`
- Method integrated: `tfr` / TFR-GCN.
- Important caveat: the official TFR repo does not provide Cora/CiteSeer configs; this run transfers the official `tfr_gcn_dblp.yaml` hyperparameter template to Cora/CiteSeer without tuning.
- Split: `percent`, train/val/test = `0.2/0.1/0.7`.
- Noise: `uniform`, noise rate = `0.4`; seeds `0,1,2,3,4`.

## Mean ± Std

| Dataset | Method | Test acc |
|---|---|---:|
| citeseer | NRGNN | 0.7027 ± 0.0108 |
| citeseer | NRGNN [X||H] | 0.7090 ± 0.0187 |
| citeseer | SoftDist [X||H] | 0.6891 ± 0.0159 |
| citeseer | TFR (CIKM 2025) | 0.6979 ± 0.0129 |
| cora | NRGNN | 0.8107 ± 0.0230 |
| cora | NRGNN [X||H] | 0.8139 ± 0.0170 |
| cora | SoftDist [X||H] | 0.8137 ± 0.0067 |
| cora | TFR (CIKM 2025) | 0.7758 ± 0.0185 |

## Paired Deltas

| Dataset | Comparison | Mean delta | Positive seeds | Per-seed deltas |
|---|---|---:|---:|---|
| citeseer | TFR (CIKM 2025) - SoftDist [X||H] | +0.88pp | 3/5 | -0.0030 0.0176 0.0098 -0.0047 0.0240 |
| citeseer | TFR (CIKM 2025) - NRGNN [X||H] | -1.12pp | 1/5 | -0.0189 0.0253 -0.0339 -0.0142 -0.0142 |
| citeseer | TFR (CIKM 2025) - NRGNN | -0.48pp | 2/5 | -0.0184 -0.0138 -0.0232 0.0150 0.0163 |
| citeseer | SoftDist [X||H] - TFR (CIKM 2025) | -0.88pp | 2/5 | 0.0030 -0.0176 -0.0098 0.0047 -0.0240 |
| cora | TFR (CIKM 2025) - SoftDist [X||H] | -3.79pp | 0/5 | -0.0174 -0.0533 -0.0158 -0.0459 -0.0570 |
| cora | TFR (CIKM 2025) - NRGNN [X||H] | -3.81pp | 0/5 | -0.0459 -0.0596 -0.0063 -0.0322 -0.0464 |
| cora | TFR (CIKM 2025) - NRGNN | -3.48pp | 1/5 | -0.0427 -0.0470 0.0264 -0.0390 -0.0718 |
| cora | SoftDist [X||H] - TFR (CIKM 2025) | +3.79pp | 5/5 | 0.0174 0.0533 0.0158 0.0459 0.0570 |

## Decision

- TFR is runnable under the local NoisyGL percent-split protocol after a minimal GCN-version integration.
- Under the transferred official DBLP template, TFR does not beat SoftDist [X||H] on Cora and is slightly above SoftDist [X||H] on CiteSeer, but remains below NRGNN [X||H].
- Because the official repo lacks Cora/CiteSeer tuned configs, these numbers should be reported as a same-protocol transferred-config comparison, not as definitive tuned TFR performance.

## Artifacts

- Raw TFR runs: `runs/tfr_comparison_20260616/tfr_5seed.csv`
- Aggregate: `runs\tfr_comparison_20260616\tfr_comparison_aggregate.csv`
- Paired deltas: `runs\tfr_comparison_20260616\tfr_comparison_paired_deltas.csv`
