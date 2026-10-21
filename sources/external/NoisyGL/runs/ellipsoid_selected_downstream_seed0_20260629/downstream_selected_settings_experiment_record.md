# Selected ellipsoid settings downstream experiment record

## Research Hypothesis

离线 target diagnostic 选出的 ellipsoid 构球参数，可能在 downstream 噪声标签训练中提供比 shuffled 粒球关系更稳定的结构先验，尤其是 DBLP 上由 `split_improvement=0.001` 修复单球坍塌后，可能转化为可用的精度收益。

## Granular Ball Role

本轮仍将粒球视为 soft structural constraint / weak view relation teacher，而不是 hard positive label 或强制 prototype pulling 模块。clean label 只用于前置离线诊断，不参与 downstream 构球或训练选择。

## Method Motivation

前置离线诊断显示不同数据集对 `max_balls` 和 `split_improvement` 敏感，因此先选每个数据集最值得进入 downstream sanity 的参数：

| Dataset | max_balls | split_improvement |
|---|---:|---:|
| DBLP | 128 | 0.001 |
| PubMed | 512 | 0.01 |
| AmazonPhoto | 64 | 0.05 |
| AmazonCom | 64 | 0.001 |

## Code And Config Changes

- Source code changes: none.
- YAML changes: none.
- Runner: `scripts/run_standard_downstream.py`.
- Parameter override mechanism: `--config_overrides` in memory.
- Output directory: `runs/ellipsoid_selected_downstream_seed0_20260629`.

## Protocol

- Noise: uniform, rate 0.4, train scope.
- Split: percent 0.2 / 0.1 / 0.7.
- Initial sanity: seed 0 for DBLP, PubMed, AmazonPhoto, AmazonCom.
- Methods in seed0 sanity:
  - `gcn_xh_ce`
  - `frozengrace_gb_gcn_softdist_xh`
  - `frozengrace_gb_gcn_softdist_xh_shuffled`
  - `frozengrace_gb_gcn_softdist_xh_ellipsoid`
  - `frozengrace_gb_gcn_softdist_xh_ellipsoid_shuffled`
- Follow-up 5-seed: only DBLP advanced to seeds 0-4 because it was the only strong seed0 candidate.

## Success Standard

进入多 seed 的最低标准：selected ellipsoid 在 seed0 至少不输给真实对照，并且明显优于 selected shuffled。  
保留为 downstream 候选的标准：在 5 个 seed 上 selected ellipsoid 应稳定优于 selected shuffled，最好正向 seed 数不少于 4/5，并且不明显弱于 current SoftDist / GCN[XH]。

## Failure Standard

若 selected ellipsoid 不优于 shuffled，或优势只来自单个 seed，或均值收益远小于 seed 间波动，则不能作为论文主 claim，只能作为负结果或后续机制修正依据。

## Seed0 Results

| Dataset | GCN[XH] | Current SoftDist | Current shuffled | Selected ellipsoid | Selected shuffled | Sel - sel shuf | Sel - current | Sel - GCN | Sanity decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AmazonCom | 0.8564 | 0.8558 | 0.8578 | 0.8330 | 0.8579 | -2.49pp | -2.29pp | -2.35pp | stop |
| AmazonPhoto | 0.9109 | 0.8964 | 0.9001 | 0.8986 | 0.8949 | +0.37pp | +0.22pp | -1.23pp | borderline |
| DBLP | 0.8429 | 0.8426 | 0.8426 | 0.8424 | 0.8320 | +1.04pp | -0.02pp | -0.05pp | advance to 5-seed |
| PubMed | 0.8249 | 0.8252 | 0.8241 | 0.8251 | 0.8294 | -0.43pp | -0.01pp | +0.01pp | low priority / stop |

Seed0 integrity: all 20 rows finished with `status=ok`; every dataset kept the same split hash and noise hash across compared methods.

## DBLP 5-Seed Results

| Method | Mean | Std | ok seeds |
|---|---:|---:|---:|
| GCN[XH] | 0.8391 | 0.0032 | 5 |
| Current SoftDist | 0.8391 | 0.0036 | 5 |
| Current shuffled | 0.8391 | 0.0036 | 5 |
| Selected ellipsoid | 0.8392 | 0.0031 | 5 |
| Selected shuffled | 0.8374 | 0.0041 | 5 |

Paired deltas:

| Comparison | Mean delta | Std | Positive seeds |
|---|---:|---:|---:|
| selected minus selected shuffled | +0.19pp | +0.49pp | 3 / 5 |
| selected minus current SoftDist | +0.01pp | +0.09pp | 2 / 5 |
| selected minus current shuffled | +0.01pp | +0.09pp | 2 / 5 |
| selected minus GCN[XH] | +0.01pp | +0.19pp | 2 / 5 |

Selected ellipsoid vs selected shuffled by seed:

| Seed | Selected ellipsoid | Selected shuffled | Delta |
|---:|---:|---:|---:|
| 0 | 0.8424 | 0.8320 | +1.04pp |
| 1 | 0.8416 | 0.8414 | +0.02pp |
| 2 | 0.8397 | 0.8403 | -0.06pp |
| 3 | 0.8376 | 0.8391 | -0.15pp |
| 4 | 0.8349 | 0.8340 | +0.08pp |

DBLP integrity: 25 / 25 rows finished with `status=ok`; each seed has 5 rows; each seed kept the same split hash and noise hash across methods.

## Final Decision

Stop this selected downstream path as a paper-claim candidate for now.

The strongest seed0 signal was DBLP, but its apparent +1.04pp selected-vs-shuffled advantage was not stable across seeds. Over seeds 0-4, selected ellipsoid is only +0.19pp over selected shuffled on average, with 3/5 positive seeds and a paired-delta std of 0.49pp. Against current SoftDist, current shuffled, and GCN[XH], the mean difference is about +0.01pp, which is too small to interpret as a real improvement.

PubMed and AmazonCom should not enter 5-seed under the current selected settings because seed0 already failed the real-vs-shuffled sanity check. AmazonPhoto is the only borderline case, but it is still below GCN[XH] on seed0 and should be lower priority than analyzing why the offline diagnostic signal does not transfer to downstream accuracy.

## Output Files

- `seed0_combined_raw.csv`
- `seed0_sanity_summary.csv`
- `seed0_sanity_report.md`
- `dblp_5seed_combined_raw.csv`
- `dblp_5seed_summary.csv`
- `dblp_5seed_paired_deltas.csv`
- `dblp_5seed_report.md`
