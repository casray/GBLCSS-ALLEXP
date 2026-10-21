# GSSC common-backbone adaptation: single-seed screening

## Protocol and changes

Five main datasets; uniform noise 0.4; split/noise/downstream seed 0;20% train,10% validation,70% test. Raw features and original graph. Ordered splits/noise match historical main fingerprints and original NoisyGL noise generation. Clean test labels enter only final evaluation.

Deployed backbone: exact common two-FC MLP, hidden64, ReLU-BN-dropout0.5. Native separate auxiliary BN-LeakyReLU-dropout-linear head remains training-only. Adam lr0.01, weight decay0.0005; native mixup lr0.0001.

100 label-free edge pretraining +100 classifier warmup +100 main epochs. First clean-validation maximum over the 200 supervised epochs, including warmup. Native per-edge minibatches, class-center extra steps and sampler/mixup training remain: matched supervised epochs/backbone is not matched total optimizer steps or training cost.

Preserved: edge sampler, straight-through gradients, virtual-node mixup, positive/negative self-contrast, noisy-train-only CE and class weights, class-center loss and first-order homophily updates. Preset mapping remains Cora/CiteSeer/Coauthor-CS native; CoraFull/Amazon-Photo Cora fallback. No test-based search.

Frozen historical core is unchanged. An isolated AST copy changes only last-tie to first-tie and two inference-parameter counts; adapter records the source hash and asserts transformation counts. Emitted adapted_training_core.py is saved per run for inspection.

## Results

| Dataset | Test accuracy (%) | Best clean validation (%) | Selected supervised epoch | Phase | Time (s) |
|---|---:|---:|---:|---|---:|
| Cora | 61.85 | 65.19 | 172 | main | 32.3 |
| CiteSeer | 54.51 | 57.53 | 3 | classifier_pretrain | 23.5 |
| CoraFull | 39.96 | 43.00 | 152 | main | 503.1 |
| Coauthor-CS | 84.87 | 85.05 | 1 | classifier_pretrain | 350.4 |
| Amazon-Photo | 82.35 | 83.27 | 65 | classifier_pretrain | 677.1 |

## Verification and conclusion

Four new adaptation tests and thirteen prior GSSC core tests passed. The optional old synthetic GPU test was skipped; the new real-Cora short GPU smoke passed. Every formal run completed all300 phase epochs (200 supervised), selected the first validation maximum, and reloaded its deployed checkpoint with zero absolute probability difference. Artifact/source hashes and split/noise identities were rechecked after training. Non-training optimization labels are masked to -1.

Conclusion: GSSC now runs under the approved common deployed backbone and supervised-epoch budget, retaining its training mechanisms. These five numbers are single-seed screening results, not five-seed mean/std; they should not directly replace a main-table five-seed row. This experiment verifies the configured port and integration, not universal bug-freedom or bitwise parity with the old official DGL/Pyro environment. No other methods or GB-LCSS runs were repeated; no manuscript edits.

Decision: screening complete; retain all outcomes. A main-table five-seed extension requires explicit authorization.
