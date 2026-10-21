# MLP-7 Baselines Report

## Data Completeness
- mlp7_ct_new: observed 75/75, missing=0
- combined_with_mlp5_reuse: observed 825/825, missing=0

## 7a: CT-MLP Adapted Baseline
Verdict: **beta05_stronger_than_ct_on_core**

DSS was pre-registered as optional and was not implemented in this run; CT is the only new latest-style MLP-adapted baseline added here.

### cora
- uniform0.2: beta05=0.6294 +/- 0.0160; CT=0.5987 +/- 0.0105; delta=+0.0307, 5/5 positive
- uniform0.4: beta05=0.5559 +/- 0.0150; CT=0.5145 +/- 0.0220; delta=+0.0414, 5/5 positive
- uniform0.6: beta05=0.4104 +/- 0.0237; CT=0.3448 +/- 0.0124; delta=+0.0656, 5/5 positive
- pair0.4: beta05=0.5029 +/- 0.0276; CT=0.4569 +/- 0.0283; delta=+0.0460, 5/5 positive
- instance0.4: beta05=0.5567 +/- 0.0081; CT=0.5043 +/- 0.0226; delta=+0.0525, 5/5 positive

### citeseer
- uniform0.2: beta05=0.6338 +/- 0.0120; CT=0.5961 +/- 0.0145; delta=+0.0377, 5/5 positive
- uniform0.4: beta05=0.5383 +/- 0.0275; CT=0.4945 +/- 0.0257; delta=+0.0438, 5/5 positive
- uniform0.6: beta05=0.4017 +/- 0.0275; CT=0.3729 +/- 0.0285; delta=+0.0289, 5/5 positive
- pair0.4: beta05=0.5009 +/- 0.0240; CT=0.4881 +/- 0.0250; delta=+0.0127, 5/5 positive
- instance0.4: beta05=0.5462 +/- 0.0225; CT=0.4991 +/- 0.0250; delta=+0.0471, 5/5 positive

### pubmed
- uniform0.2: beta05=0.8244 +/- 0.0042; CT=0.8152 +/- 0.0053; delta=+0.0091, 5/5 positive
- uniform0.4: beta05=0.7190 +/- 0.0088; CT=0.7723 +/- 0.0033; delta=-0.0533, 0/5 positive
- uniform0.6: beta05=0.4753 +/- 0.0242; CT=0.5332 +/- 0.0435; delta=-0.0579, 0/5 positive
- pair0.4: beta05=0.6849 +/- 0.0127; CT=0.7097 +/- 0.0234; delta=-0.0248, 0/5 positive
- instance0.4: beta05=0.7201 +/- 0.0123; CT=0.7625 +/- 0.0122; delta=-0.0424, 0/5 positive

## 7c-step0: GRACE Encoder Checkpoint Scan
- Candidate encoder/model checkpoint files found: 112
- First candidates:
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p1_val0p1_split0_pretrain0\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p1_val0p1_split0_pretrain1\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p1_val0p1_split0_pretrain2\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p1_val0p1_split0_pretrain3\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p1_val0p1_split0_pretrain4\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p2_val0p1_split0_pretrain0\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p2_val0p1_split0_pretrain1\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p2_val0p1_split0_pretrain2\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p2_val0p1_split0_pretrain3\checkpoint.pt`
  - `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\grace\CiteSeer\ratio_train0p2_val0p1_split0_pretrain4\checkpoint.pt`

## 7c: H-input Structure Robustness (seed0)
Verdict: **weak_structure_tradeoff_signal_seed0**
### cora
- frozengrace_gb_mlp_softdist_beta05: clean=0.5546
  - edge_drop p=0.25: test=0.5546, delta=+0.0000, H_recomputed=0
  - edge_drop p=0.50: test=0.5546, delta=+0.0000, H_recomputed=0
  - edge_drop p=0.75: test=0.5546, delta=+0.0000, H_recomputed=0
  - isolated p=0.25: test=0.5546, delta=+0.0000, H_recomputed=0
  - isolated p=0.50: test=0.5546, delta=+0.0000, H_recomputed=0
  - isolated p=0.75: test=0.5546, delta=+0.0000, H_recomputed=0
- mlp_small_loss_xh: clean=0.5799
  - edge_drop p=0.25: test=0.5778, delta=-0.0021, H_recomputed=1
  - edge_drop p=0.50: test=0.5794, delta=-0.0005, H_recomputed=1
  - edge_drop p=0.75: test=0.5609, delta=-0.0190, H_recomputed=1
  - isolated p=0.25: test=0.5715, delta=-0.0084, H_recomputed=1
  - isolated p=0.50: test=0.5641, delta=-0.0158, H_recomputed=1
  - isolated p=0.75: test=0.5551, delta=-0.0248, H_recomputed=1
- mlp_jocor_xh: clean=0.5979
  - edge_drop p=0.25: test=0.5974, delta=-0.0005, H_recomputed=1
  - edge_drop p=0.50: test=0.5931, delta=-0.0047, H_recomputed=1
  - edge_drop p=0.75: test=0.5810, delta=-0.0169, H_recomputed=1
  - isolated p=0.25: test=0.5947, delta=-0.0032, H_recomputed=1
  - isolated p=0.50: test=0.5889, delta=-0.0090, H_recomputed=1
  - isolated p=0.75: test=0.5794, delta=-0.0185, H_recomputed=1
### citeseer
- frozengrace_gb_mlp_softdist_beta05: clean=0.5533
  - edge_drop p=0.25: test=0.5533, delta=+0.0000, H_recomputed=0
  - edge_drop p=0.50: test=0.5533, delta=+0.0000, H_recomputed=0
  - edge_drop p=0.75: test=0.5533, delta=+0.0000, H_recomputed=0
  - isolated p=0.25: test=0.5533, delta=+0.0000, H_recomputed=0
  - isolated p=0.50: test=0.5533, delta=+0.0000, H_recomputed=0
  - isolated p=0.75: test=0.5533, delta=+0.0000, H_recomputed=0
- mlp_small_loss_xh: clean=0.5253
  - edge_drop p=0.25: test=0.5284, delta=+0.0030, H_recomputed=1
  - edge_drop p=0.50: test=0.5241, delta=-0.0013, H_recomputed=1
  - edge_drop p=0.75: test=0.5253, delta=+0.0000, H_recomputed=1
  - isolated p=0.25: test=0.5241, delta=-0.0013, H_recomputed=1
  - isolated p=0.50: test=0.5193, delta=-0.0060, H_recomputed=1
  - isolated p=0.75: test=0.5180, delta=-0.0073, H_recomputed=1
- mlp_jocor_xh: clean=0.5558
  - edge_drop p=0.25: test=0.5576, delta=+0.0017, H_recomputed=1
  - edge_drop p=0.50: test=0.5563, delta=+0.0004, H_recomputed=1
  - edge_drop p=0.75: test=0.5515, delta=-0.0043, H_recomputed=1
  - isolated p=0.25: test=0.5537, delta=-0.0021, H_recomputed=1
  - isolated p=0.50: test=0.5546, delta=-0.0013, H_recomputed=1
  - isolated p=0.75: test=0.5485, delta=-0.0073, H_recomputed=1

## 7c-ext: Cold-start Isolated p=1.0 Probe (seed0)
Verdict: **coldstart_structure_headline_not_supported_seed0**

Output: `runs/mlp7_baselines_20260701/hinput_structure_coldstart_p1_seed0.csv`

Protocol: test nodes are fully isolated at evaluation time; H-input baselines recompute GRACE-H on the perturbed graph. beta05 remains X-only at inference.

### cora
- frozengrace_gb_mlp_softdist_beta05: clean=0.5546; isolated p=1.0 test=0.5546, delta=+0.0000, H_recomputed=0
- mlp_small_loss_xh: clean=0.5799; isolated p=1.0 test=0.5515, delta=-0.0285, H_recomputed=1
- mlp_jocor_xh: clean=0.5979; isolated p=1.0 test=0.5710, delta=-0.0269, H_recomputed=1

### citeseer
- frozengrace_gb_mlp_softdist_beta05: clean=0.5533; isolated p=1.0 test=0.5533, delta=+0.0000, H_recomputed=0
- mlp_small_loss_xh: clean=0.5253; isolated p=1.0 test=0.5185, delta=-0.0069, H_recomputed=1
- mlp_jocor_xh: clean=0.5558; isolated p=1.0 test=0.5468, delta=-0.0090, H_recomputed=1

Interpretation:
- Fully isolated test nodes amplify the Cora H-input drop from the p<=0.75 setting, but the drop is still only about 2.7-2.9pp, below the pre-registered >5pp strong-headline threshold.
- CiteSeer remains weakly affected, with drops below 1pp.
- This closes the H-input structure-robustness branch for now: beta05 is graph-free and unchanged, but H-input baselines do not collapse enough to support a headline claim. Do not extend this branch to 5 seeds unless a reviewer specifically asks for a cold-start appendix.

## Claim Boundary
- CT is an MLP-adapted strategy, not an official reproduction of the cited CT method.
- Graph-native 7b methods were not run in this phase; they remain related-work or optional positioning-table work.
- MLP-6 reframe remains active: do not claim superiority under H-input robust baselines.
