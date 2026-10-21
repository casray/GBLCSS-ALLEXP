# Five-dataset extension

## Question and hypothesis
Extend the existing representation and single-parameter diagnostics to Coauthor-CS
and Amazon-Photo. Test whether observations from the three citation datasets also
hold on coauthorship and co-purchase graphs. No direction of improvement is assumed.

## Method and role
Granular balls organize local label evidence. The existing formal tie-safe
leave-one-out targets and X-only MLP remain unchanged. Only dataset loading is added
through a wrapper around the September 19 driver. Original sources and outputs are
read-only. Source snapshots must match the previous experiment.

## Fixed protocol
Uniform 0.4, ordered 20/10/70 split, seeds 0--4, clean validation, frozen archived
GRACE embeddings, 200 epochs, unchanged model/optimizer/selection. Target computation
runs on CPU as in the previous experiment; all MLP training runs on cuda:0.
Four training threads and six grouping threads match the completed experiment.
Eighteen arms per dataset and seed give 180 logical evaluations. Exact target reuse
is recorded and is not counted as an independent training run.

## Expected mechanisms
Representation changes can change the partition and therefore supervision.
Group-size and split controls can affect locality; support thresholds can affect
coverage; beta can affect target concentration. Raw X and A2X are not assumed to
produce one ball on these datasets. Clean labels are used only for diagnostics.

## Checks and decision rules
First run all ten default preparations and compare archived target Brier values
at the existing tolerance of 2e-5. A mismatch is recorded rather than hidden when
the current scikit-learn/KMeans environment produces a different partition; split,
noise, representation, and current assignment hashes remain in every record.
Completion requires the full matrix, finite normalized targets, 200 finite losses
for each actual run, GPU execution, identical split/noise fingerprints within pairs,
and the exact Brier gain decomposition. Report all arms with five-seed mean/SD and
paired differences from fresh defaults, including negative results. Do not replace
published defaults or infer statistical significance from a small mean difference.

## Results and decision
Completed. See `REPORT_zh.md`, `verified/`, and `combined_five_datasets/`.

The two added datasets retain the main qualitative pattern: the default grouping
beats MLP+CE and shuffled size-matched grouping, while the two-step propagated
representation is weaker than GRACE. Parameter variations are exploratory and do
not change the paper defaults. The three-dataset interpretation therefore extends
to these two datasets for the tested uniform-0.4 protocol, with dataset-specific
variation in the size of the gains.
