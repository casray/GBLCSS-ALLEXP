# GB-LCSS

Code for **Granular-Ball Local-Consensus Soft Supervision** for node classification with noisy training labels.

GB-LCSS partitions frozen graph-contrastive representations into granular balls, builds a leave-one-out label distribution inside each ball, and trains an attribute-only MLP on the resulting soft targets. The graph is used during training, not during MLP inference.

## Setup

Use Python 3.10 or later. Install a PyTorch build suitable for your CPU or GPU, then:

```bash
pip install -r requirements.txt
```

The frozen GRACE representations for all eleven datasets are included in `representations/`. Public graph datasets are downloaded by PyTorch Geometric on first use and cached in `data/`. See [DATASETS.md](DATASETS.md) for the dataset names, loaders and preparation details.

## Run experiments

One entry point runs experiments and writes tables from the new results:

```bash
# Quick execution check on Cora, using two training epochs
python run.py --suite target_controls --datasets cora --seeds 0 --epochs 2 --output outputs/quick

# Full five-seed main comparison, including all five noise settings
python run.py --suite matched --device cuda --output outputs/matched

# All main and supplementary experiments
python run.py --suite all --device cuda --output outputs/paper
```

Use `--device cpu` for CPU execution. `--device cuda` also applies to ROCm-enabled PyTorch. The default training budget is 200 epochs, with model selection by validation accuracy. A two-epoch check tests execution only; use the default budget for the paper experiments.

| `--suite` | Paper results |
|---|---|
| `matched` | Table 3, Tables A.1-A.5, paired tests in Table B.1 |
| `graph_assisted` | Table 4: graph-assisted training methods |
| `target_controls` | Table 5: random grouping and label-smoothing controls |
| `grouping` | Table 6: granular balls, kNN and k-means |
| `representations` | Table 7: GRACE, raw attributes and propagated attributes |
| `distillation` | Table 8: graph-to-MLP transfer |
| `validation` | Table C.1: clean and noisy validation labels |
| `leave_one_out,clean` | Table D.1: self-vote and clean-label controls |
| `soft_targets` | Table E.1: graph-based target methods |
| `extra` | Table G.2: six additional datasets |
| `citeseerfull` | Table G.3: additional noise settings |
| `sensitivity` | Table I.1: parameter variants; controls are in `target_controls` and `representations` |
| `targets` | Table I.2, Figure 4, voting coverage and tie statistics |
| `statistics` | Tables 2 and G.1: dataset statistics |
| `cost` | Figure 5: grouping time and retained storage |

`configs/experiments.yaml` defines the dataset, method, noise and parameter combinations. The default seeds are `0,1,2,3,4`. Select a subset with `--datasets cora,citeseer --seeds 0,1`. Use `--list` to list suites and `--dry-run` to inspect the run count before execution.

When several tables share an experiment, `--suite all` reuses its computed result within that invocation.

Each output directory contains:

- `per_seed.csv`: individual experimental results.
- `tables/`: mean accuracy, sample standard deviation, paired comparisons and readable comparison tables.
- `targets_summary.csv`, `statistics_summary.csv`, `cost_summary.csv`: summaries for the corresponding selected suites.
- `runs.jsonl` and `config.json`: completed runs and their settings, used to resume an interrupted experiment.

Repeat the same command with `--resume` to continue. To rebuild tables without training again:

```bash
python run.py --summarize --output outputs/paper
```

The [reference results](reference_results/README.md) contain the archived per-seed results for GB-LCSS and the seven matched-MLP baselines in Tables A.1-A.5. They are provided for comparison only; experiment commands compute new results and do not load this archive.

## Algorithm and structure

```text
run.py                       Experiment entry point
configs/                     Dataset, method, GRACE and experiment settings
representations/             Frozen encoder outputs
src/gblcss/
  granular_ball.py           Label-free geometric splitting (Algorithm 1)
  targets.py                 Leave-one-out, tie-safe soft targets (Algorithm 2)
  models.py, training.py     Attribute-only MLP and training objective
  distillation.py            GB-LCSS teacher extension
  baselines/                 Comparison methods used in the experiments
  ablation_targets.py        Target, grouping and representation controls
  data.py                    Public datasets, paired splits and label noise
  diagnostics.py             Brier decomposition and local-vote statistics
  reporting.py               Tables and paired statistical tests
scripts/prepare_representation.py  Optional GRACE pretraining
tests/                       Algorithm and experiment tests
```

For each training node, `targets.py` excludes that node's label from the ball's class counts. A unique majority adjusts the empirical distribution; a tied vote retains it. The consensus weight uses the original vote proportions. If too few other training nodes remain, the observed one-hot label is retained. Only training labels contribute to targets.

All runs use the same 20%/10%/70% split procedure and paired seeds. The main experiments corrupt training labels only. Test labels are used for evaluation and retrospective target diagnostics, not for target construction or checkpoint selection. Accuracy tables report percentages and sample standard deviations over seeds; Brier scores sum over classes before averaging over training nodes. Tables F.1-F.3 correspond to the implementations and settings in `baselines/` and `configs/`.

Comparison methods retain the experiment-specific implementations and settings in `configs/main/protocol_adaptations.yaml`. Graph-MLP uses the shared-MLP, sampled edge-contrastive adaptation used to produce the reported comparison. Other method adaptations are described in the paper and in their source modules.

## Optional pretraining

To train a new GRACE representation instead of using the included frozen output:

```bash
python scripts/prepare_representation.py --dataset cora --seed 0 --device cuda
python run.py --suite target_controls --datasets cora --device cuda --representation representations_generated/cora/grace/H.pt --output outputs/new_grace
```

Pretraining uses `configs/representations/grace.yaml`, which contains settings for the five main datasets, and does not use class labels. Frozen representations for the six additional datasets are supplied directly. Downstream experiments otherwise retain their settings.

## Tests and licenses

```bash
python -m pytest -q
```

See `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/` for licensing and source attribution.

`SHA256SUMS.txt` identifies the files in this submission snapshot, excluding the checksum list itself and generated files such as local caches and experiment outputs.
