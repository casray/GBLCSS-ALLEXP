# MLP-61: KRD and VQGraph faithful protocol ports on five datasets

## Research hypothesis

The faithful KRD and VQGraph protocol ports can materially change the rankings
obtained with the earlier mechanism adaptations. Paired five-seed evaluation is
required before either row can be retained or replaced in the manuscript.

Accuracy relative to the granular-ball consensus teacher is an outcome, not a
success criterion. All completed runs are retained regardless of direction.

## Role of granular balls

Granular balls are absent from KRD and VQGraph. The existing granular-ball
consensus teacher is a reference under the same data split, noise realization,
validation rule, and attribute-only student inference protocol.

## Methods and parameter sources

- KRD retains the official GCN-to-MLP architecture, neighborhood knowledge
  sampling, reliability update, and optimization settings from
  `LirongWu/KRD` at commit
  `32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21`.
- VQGraph retains the official vector quantizer, graph reconstruction terms,
  class distillation, code-assignment distillation, and the paper settings in
  ICLR 2024 Tables 10 and 12 from `YangLing0818/VQGraph` at commit
  `60aa1c99782d542c316a37f0394eba934bcf97b0`.
- Official KRD settings exist for Cora, CiteSeer, Coauthor-CS, and Amazon-Photo.
  CoraFull uses the Coauthor-CS setting, selected before execution by nearest
  node count among official KRD datasets.
- Official VQGraph settings exist for Cora, CiteSeer, and Amazon-Photo.
  CoraFull and Coauthor-CS use the PubMed setting, selected before execution by
  nearest node count among the five small datasets in the VQGraph paper.

Mapped settings are protocol extensions, not official dataset configurations.
No parameter is changed after observing MLP-61 accuracy.

## Experimental protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Seeds: 0, 1, 2, 3, 4.
- Split: fixed percent 0.2/0.1/0.7.
- Noise: uniform 0.4 and uniform 0.6, injected into training labels only.
- Model selection: clean validation labels, matching the manuscript protocol.
- Inference: the deployed student receives node attributes only.
- Execution order: configuration tests, one seed-0 feasibility run per dataset,
  then the remaining seeds for each noise rate.

## Expected mechanism

KRD may be more robust than its earlier proxy because it samples neighborhood
knowledge according to learned teacher reliability. VQGraph may remain
sensitive to label noise because its graph tokenizer includes a supervised
classification term, despite the richer code-assignment target.

## Success and failure criteria

Technical success requires finite outputs, complete method diagnostics, five
seeds per dataset and noise rate, identical split/noise hashes across methods
within each dataset-seed cell, graph use only during training, and attribute-only
student inference.

Technical failure includes OOM, NaN, incomplete cells, hash mismatch, missing
active VQ codes, absent KRD reliability updates, or test-label checkpoint
selection. Accuracy is not a technical pass criterion.

## Reporting contract

Report per-dataset mean and sample standard deviation, paired seed differences
to the existing granular-ball teacher when hashes match, implementation status,
parameter source, failures, and runtime. Do not modify the manuscript until the
complete result matrix and protocol audit are available.

## Execution status

- KRD completed all 50 planned GPU runs: five datasets, two noise rates, and
  five seeds.
- VQGraph completed 30 GPU runs on Cora, CiteSeer, and Amazon-Photo.
- The full-batch VQGraph protocol port did not complete on CoraFull or
  Coauthor-CS within the current Windows ROCm environment. The official DGL
  neighbor-sampled runtime is not available with the installed PyTorch/ROCm
  stack. No CPU fallback, feature compression, or reduced-codebook substitute
  was accepted.
- All 80 retained rows use graph information during training, use no graph at
  student inference, select checkpoints on the clean validation split, and
  match the granular-ball teacher split/noise hashes.
- Final verification passed 26 tests and 4 subtests, Python compilation, and
  all CSV integrity checks. No experiment process remains active.
- The manuscript was not modified. Results and the evidence decision are
  recorded in `runs/mlp61_krd_vqgraph_faithful_5x5_20260906/mlp61_report.md`.
