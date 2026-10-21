# MLP-24 Faithful ANL-CE / GMM+CT Report

## Protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo
- Split: percent 0.2/0.1/0.7
- Noise: train-only uniform 0.4
- Seeds: 0--4, paired by split/noise seed
- Backbone: 2-layer X-only MLP, hidden 64, dropout 0.5
- `mlp_anl_ce`: exact NCE+NNCE+L1 objective; MLP-adapted optimizer/backbone
- `mlp_gmm_ct`: exact confidence-gap Mann-Kendall statistic unioned with low-loss GMM; full-batch epoch used as the observation step

## Five-seed results

| Dataset | ANL-CE | GMM+CT | Existing MLP+CE anchor |
|---|---:|---:|---:|
| Cora | 43.35+-3.77 | 43.53+-1.41 | 44.24+-2.62 |
| CiteSeer | 44.27+-4.28 | 43.93+-2.72 | 43.50+-3.02 |
| CoraFull | 31.09+-1.46 | 31.98+-0.98 | 31.98+-0.98 |
| Coauthor-CS | 84.46+-0.91 | 88.42+-0.62 | 88.42+-0.62 |
| Amazon-Photo | 67.27+-3.40 | 73.90+-1.80 | 73.90+-1.80 |

## Findings

1. Both faithful implementations ran for all 50 required rows with finite outputs and no leakage/OOM failures.
2. GMM+CT collapses to the CE anchor on CoraFull, Coauthor-CS, and Amazon-Photo. Its final GMM selector retains about 98--100% of train nodes, so the method supplies almost no effective filtering under this full-batch MLP protocol. Cora/CiteSeer differences from CE are within ordinary seed variation.
3. ANL-CE is not consistently competitive after faithful migration: it is approximately tied with CE on Cora/CiteSeer, but lower by 0.89 pp on CoraFull, 3.96 pp on Coauthor-CS, and 6.63 pp on Amazon-Photo.
4. The old `mlp_anl` and `mlp_ct` rows were not faithful implementations. They must remain labelled `ANL-inspired` and `CT-inspired` if retained; the published-method rows must use the values above.
5. The negative result is informative: the paper's LCSS advantage is not an artifact of comparing only against weak proxy implementations. Replacing two proxies with their faithful core mechanisms does not create a stronger same-backbone competitor in this protocol.

## Decision

- Use the new ANL-CE and GMM+CT five-seed rows in the revised comparison table.
- Keep proxy rows out of published-method rankings.
- Continue the fidelity repair with FINE. DivideMix remains explicitly `DivideMix-inspired` unless the full dual-network MixMatch pipeline is implemented.
