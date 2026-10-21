# GPU single-seed baseline screening

Five main datasets, uniform training-label noise 0.4, seed 0, historical ordered 20/10/70 splits. Official-core GSSC, TINED and DGD; preserve core methods and record blocked components. GPU only, no search, no paper changes, no follow-up seeds. Common requested supervised budget: 200 epochs, hidden 64, dropout 0.5, Adam lr 0.01, weight decay 0.0005. Method-specific core phases/layers and fixed author coefficients are recorded separately.

Granular balls are not used by these baselines. The hypothesis is to measure their behavior under the existing protocol, with no assumed accuracy ordering. Engineering success requires finite training, unchanged split/noise fingerprints, validation-only checkpoint selection and valid saved predictions. Failures and resource limitations are preserved, not replaced with simpler algorithms. Single-seed differences are not stable rankings or significance evidence.

Save author commits, configuration, epoch/console logs, checkpoints, probabilities, timing, memory and results. Test labels are never passed to training adapters. Final decision: report the screening and wait for authorization before any multi-seed experiment.
