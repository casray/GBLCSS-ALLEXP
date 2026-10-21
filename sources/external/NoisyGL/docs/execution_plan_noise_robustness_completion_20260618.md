# Noise Robustness Completion Plan (2026-06-18)

## Material Passport

- Artifact type: code experiment plan
- Scope: noise-type robustness and uniform-noise rate sensitivity
- Status: in progress
- Repository: `external/NoisyGL`
- Runtime: ROCm PyTorch on `cuda:0`

## Research hypothesis

The frozen-GRACE granular-ball signal should retain a measurable contribution over
`GCN[X||H]` and shuffled balls under moderate label noise, while its offline
ball-vote mechanism may degrade when uniform noise approaches the point where the
within-ball noisy-label majority becomes unreliable. Uniform noise rate 0.5 is
added only to locate that transition between the existing 0.4 and 0.6 results.

## Granular-ball role

Granular balls remain a label-noise correction unit built from frozen, label-free
GRACE embeddings. They produce the existing multiclass-aware soft distribution
targets. No ball construction, target formula, or classifier behavior is changed.

## Motivation

1. Complete the noise-type comparison on all three successful main datasets.
2. Add the missing 0.5 point needed to interpret the sharp 0.4-to-0.6 transition.
3. Test whether Coauthor-CS preserves the method's contribution across moderate
   pair/instance noise and low/high uniform-noise regimes.
4. Report high-noise failures as an applicability boundary rather than selecting
   only favorable regimes.

## Protocol

- Datasets: Cora, CiteSeer, Coauthor-CS.
- Split: `percent`, train/validation/test = 0.2/0.1/0.7.
- Seeds: 0, 1, 2, 3, 4.
- Device: `cuda:0` using the established ROCm Python environment.
- Methods:
  - `gcn`
  - `nrgnn`
  - `gcn_xh_ce`
  - `frozengrace_gb_gcn_softdist_xh`
  - `frozengrace_gb_gcn_softdist_xh_shuffled`
- New regimes:
  - Cora/CiteSeer: uniform 0.5.
  - Coauthor-CS: uniform 0.2 and 0.6; pair 0.4; instance 0.4.
- Existing anchors are reused for uniform 0.2/0.4/0.6 and pair/instance results
  already completed on Cora/CiteSeer, and uniform 0.4 on Coauthor-CS.
- GRACE embeddings are reused without retraining because SSL pretraining is
  label-free and independent of the synthetic label-noise realization.

## Code and configuration changes

- No predictor, loss, data split, or model configuration changes.
- Add only a reproducible batch runner, result aggregation, and this plan/report.
- The batch runner relies on the verified CLI defaults for the same explicit
  percent split (0.2/0.1/0.7); result aggregation must verify split counts from
  the raw logs before accepting a run.

## Expected mechanism

At moderate noise, a ball aggregates several noisy train labels and can provide a
more stable distribution than an individual corrupted target. At extreme uniform
noise, the noisy majority can cease to reflect the latent class, so real balls may
lose their advantage over shuffled balls. Pair noise is interpreted separately:
above 0.5, the paired wrong class becomes the population majority, so pair 0.6 is
not used as a generic high-noise robustness point.

## Success criteria

- All requested runs complete with 5 valid seeds under the matched protocol.
- Moderate-noise contribution is supported when SoftDist `[X||H]` has positive
  mean paired delta over both `GCN[X||H]` and shuffled balls, with at least 4/5
  positive seed-wise deltas.
- Cross-method claims use mean, sample standard deviation, seed-wise paired
  deltas, and win counts; a mean difference smaller than the observed variability
  is not described as significant.

## Failure criteria

- Real balls do not beat shuffled balls, or SoftDist does not beat `GCN[X||H]`,
  in a regime.
- Fewer than five valid seeds, protocol mismatch, missing embedding alignment, or
  runtime failure makes the corresponding comparison incomplete.
- Uniform 0.5/0.6 degradation is retained and reported as a mechanism boundary;
  it does not trigger parameter tuning.

## Required outputs

- Raw per-seed CSVs and logs.
- Noise-type robustness table at noise rate 0.4.
- Uniform-noise rate table, including 0.5 for Cora/CiteSeer.
- Paired-delta table and a concise Markdown report.
- Final decision for each regime: retain, boundary evidence, incomplete, or drop.

## Final decision

Completed with all 150 requested new per-seed runs and no failed rows.

- Uniform 0.5 retains a controlled granular-ball increment on Cora and CiteSeer,
  locating the practical degradation transition between 0.5 and 0.6 rather than
  justifying an additional broad 0.3/0.5 sweep.
- Coauthor-CS retains a stable ball increment under instance 0.4 and uniform 0.2.
- Coauthor-CS pair 0.4 does not beat shuffled balls and has high seed variance;
  it is boundary evidence, not a robustness claim.
- Coauthor-CS uniform 0.6 has only a negligible ball increment and remains below
  NRGNN; Cora uniform 0.6 also fails the real-vs-shuffled control.
- Keep uniform 0.4 as the main table. Use the completed noise-type table and
  uniform-rate table as robustness and applicability analyses, respectively.
