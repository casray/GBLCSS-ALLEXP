# SimMLP screening and TINED diagnosis

Scope: five main datasets, uniform train-label noise 0.4, seed 0, GPU.
No original experiment is overwritten; no paper table changes or additional seeds.

Hypothesis: unsupervised MLP/graph alignment can provide useful attribute-only
representations before fitting a noisy-label classifier. Granular balls are not
part of these baseline methods.

SimMLP executes unchanged author MLP, GNN orchestration, predictor, decoder,
augmentation, Bootstrap loss and full-graph train definitions. Only nonparametric
graph propagation is implemented using PyTorch sparse.mm instead of unavailable
torch_scatter/torch_sparse binaries. Official source stays unchanged in vendor.

Main-protocol adaptations: exact existing 20/10/70 ordered split and train-only
uniform 0.4 corruption with SHA256 checks; raw input features; hidden/predictor
width 64, encoder dropout 0.5, learning rate 0.01, weight decay 0.0005.
Retain author AdamW and cosine scheduler. Run 200 label-free pretraining epochs,
freeze final encoder, then 200 linear-probe epochs, selected by clean validation
accuracy (last tie). Author architecture/augmentation/loss knobs use corresponding
public presets; CoraFull has no preset and uses Cora, declared before training.
Author 1000 pretraining and 5000 probe budgets are not used in this screening.
Official outer pretraining test-accuracy selection is not used; no test labels
enter training. This is an adapted screening, not the author's published setting.

Expected mechanism: graph alignment and reconstruction shape the MLP; only the
linear probe receives observed noisy labels. Success criterion: finite losses,
valid probability rows, saved-checkpoint validation match and all five protocol
fingerprints match. Accuracy is reported regardless of strength. A single seed
does not establish stable superiority. Failure criterion: core mismatch, leakage,
nonfinite training, GPU error or fingerprint mismatch; failed runs are preserved.

TINED: first compare saved logs/configuration and unchanged official source.
Only specific single-factor validation-only diagnostic experiments are allowed;
test accuracy must never guide configuration changes. Record conclusions and
remaining uncertainty separately from the original screening result.
