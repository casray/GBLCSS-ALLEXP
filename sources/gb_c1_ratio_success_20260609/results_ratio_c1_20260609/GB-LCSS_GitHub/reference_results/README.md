# Reference results

`matched_per_seed.csv` is an unchanged copy of `results/main_tables/table2_per_seed.csv` from the original submission code archive. The historical filename predates the final paper's table numbering.

The file contains 1,000 recorded runs: five main datasets, five noise settings, eight methods (GB-LCSS and seven matched-MLP baselines), and seeds 0-4. These records cover the corresponding rows in Tables A.1-A.5; Table 3 uses the uniform-noise results. Random size-matched grouping and the other experiment families are not included in this reference file.

- `test_accuracy` and `validation_accuracy` are fractions in [0, 1]. Multiply by 100 for percentages.
- Aggregate each dataset, noise type, noise rate and method over seeds, using the sample standard deviation (`ddof=1`).
- The historical dataset identifier `amazonpho` means Amazon-Photo; use `amazonphoto` in current commands.
- `noise_scope=train` denotes corruption of training labels. For GB-LCSS, `leave_one_out=1` excludes the current node's vote, and `tie_mode=distribution_only` retains the empirical distribution when the largest vote count is tied.

This is a reference archive, not the output of a new experiment. Training, evaluation and table-generation commands do not read it. New results are written to the selected `--output` directory.
