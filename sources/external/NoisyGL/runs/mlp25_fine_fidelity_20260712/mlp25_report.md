# MLP-25 FINE Fidelity Report

## Protocol and identity

- Core five datasets, uniform 0.4, train-only noise, percent 0.2/0.1/0.7, seeds 0--4.
- Same 2-layer X-only MLP backbone as the main comparison.
- Exact FINE detector core: class-wise uncentered SVD, normalized absolute projection, per-class two-component GMM, posterior threshold 0.5.
- Ten-epoch CE warmup; selector recomputed dynamically each epoch.
- This is `FINE (MLP-adapted)`, not a reproduction of the original image architecture.

## Results

| Dataset | FINE (MLP-adapted) | MLP+CE anchor | LCSS beta=0.5 |
|---|---:|---:|---:|
| Cora | 44.83+-2.19 | 44.24+-2.62 | 55.59+-1.50 |
| CiteSeer | 45.27+-2.82 | 43.50+-3.02 | 53.83+-2.75 |
| CoraFull | 32.01+-0.96 | 31.98+-0.98 | 37.33+-0.81 |
| Coauthor-CS | 88.51+-0.66 | 88.42+-0.62 | 90.62+-0.57 |
| Amazon-Photo | 72.69+-3.15 | 73.90+-1.80 | 81.64+-0.92 |

## Diagnostics

- Completeness: 25/25 rows, all `status=ok`.
- Mean final selected ratio: Cora 0.43, CiteSeer 0.39, CoraFull 0.55, Coauthor-CS 0.50, Amazon-Photo 0.28.
- Every run retained at least one sample from every observed noisy class.
- Mean runtime: 15--50 seconds on four datasets; CoraFull 220 seconds because 70 class-wise GMMs are recomputed each epoch.

## Conclusion

The faithful detector is stronger than the old proxy on Cora/CiteSeer, but does not alter the paper's ranking. Its largest CE gain is +1.77 pp on CiteSeer; it is effectively tied with CE on CoraFull/Coauthor-CS and lower on Amazon-Photo. LCSS remains higher by 2.11--11.15 pp across all five datasets.

The formal FINE row must use these values. The historical `mlp_fine` row may only be retained as `eigenvector-alignment proxy` and must not be counted as FINE.
