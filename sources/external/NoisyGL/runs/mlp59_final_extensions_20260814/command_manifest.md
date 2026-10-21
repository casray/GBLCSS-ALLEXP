# MLP-59 command manifest

Working directory: `external/NoisyGL`

Interpreter: `GRACE-master/.venv-rocm/Scripts/python.exe`

Common protocol: percent split `0.2/0.1/0.7`, train-only noise, seeds `0,1,2,3,4`, `cuda:0`.

## Teacher extension

```text
python scripts/run_standard_downstream.py --datasets coauthorcs,amazonpho --methods mlp_distill_formal_lcss_x,mlp_distill_formal_lcss_x_shuffled,mlp_distill_gcn --seeds 0,1,2,3,4 --noise_type uniform --noise_rate {0.4|0.6} --noise_scope train --device cuda:0 --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7 --output runs/mlp59_final_extensions_20260814/teacher_u0p{4|6}_seeds0_4.csv --stop_on_error
```

## GNN-to-MLP follow-up

```text
python scripts/run_standard_downstream.py --datasets coauthorcs,amazonpho --methods ffg2m_reimplemented,krd_adapted,vqgraph_adapted --seeds 0,1,2,3,4 --noise_type uniform --noise_rate {0.4|0.6} --noise_scope train --device cuda:0 --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7 --output runs/mlp59_final_extensions_20260814/g2m_u0p{4|6}_seeds0_4.csv --stop_on_error
```

## Boundary entry

```text
python scripts/run_standard_downstream.py --datasets pubmed,amazoncom,coraml,attribwiki,citeseerfull,coauthorph --methods frozengrace_gb_mlp_softdist_beta05_loo,frozengrace_gb_mlp_softdist_beta05_loo_shuffled --seeds 0,1,2,3,4 --noise_type uniform --noise_rate 0.4 --noise_scope train --device cuda:0 --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7 --output runs/mlp59_final_extensions_20260814/boundary_u0p4_seeds0_4.csv --stop_on_error
```

## Conditional CiteSeerFull spectrum

The same real/shuffled methods were run for uniform `0.2/0.6`, pair `0.4`, and instance `0.4`. The seven same-backbone baselines were run for pair `0.4` and instance `0.4`; exact-hash rows for uniform `0.2/0.4/0.6` were reused from MLP-23.

## Audit note

The first GNN-to-MLP `uniform 0.4` shell call timed out while leaving a child process. That older duplicate process was terminated before it wrote the result file. The retained formal file has 30 unique dataset-method-seed rows, and all downstream uniqueness and completeness audits pass.
