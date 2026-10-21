# Quantitative figure reuse audit

## Decision

The previous label-rate and test-feature-noise figures are not direct evidence for the current formal GB-LCSS method. Both were generated before the leave-one-out, tie-safe target became the sole formal definition. They are retained as historical diagnostics and are not inserted into the submission manuscript.

## Label-rate curves

- Existing evidence: Cora and CiteSeer, uniform label noise 0.4, six train ratios, five seeds.
- Historical method: `frozengrace_gb_mlp_softdist_beta05` with self-vote; an earlier sensitivity set also uses `[X||H]` as classifier input.
- Current formal method mismatch: no leave-one-out target and no complete tie-safe lineage across all ratios.
- Submission decision: do not reuse as GB-LCSS evidence. A future inclusion requires rerunning the current leave-one-out, distribution-only target at every label ratio with paired splits and a matched shuffled control.

## Test-feature-noise curves

- Existing evidence: five datasets, uniform training-label noise 0.4, six test-feature perturbation levels, five seeds.
- Historical method: self-vote SoftDist target rather than the formal leave-one-out target.
- Scientific scope: the curves test sensitivity to corrupted node attributes at inference, not robustness to noisy training labels. They also show that absolute degradation can be steeper than MLP+CE on some datasets.
- Submission decision: do not include in the main manuscript. Reusing them would open a feature-noise claim that the formal method has not been evaluated for.

## Added figure

The manuscript instead adds a five-panel uniform label-noise-rate figure generated from the current formal aggregate:

- target: leave-one-out and tie-safe GB-LCSS;
- datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo;
- rates: uniform 0.2, 0.4, and 0.6;
- comparisons: JoCoR, MLP+CE, and size-matched random grouping;
- statistics: mean and sample standard deviation over five paired seeds;
- source data: `figures/source_data/fig3_uniform_noise_rate_curves_source.csv`.

This figure complements the gain heatmap: the heatmap shows relative gain over the strongest common-backbone baseline, while the curve figure shows absolute performance degradation as label noise increases.
