# Reproducing the paper

Code for Gupta, Neveu & Byrne, *A graph neural network approach for inferring neuronal
connectivity*. Not part of the `circuit-gnn` package.

## Setup

```bash
git clone https://github.com/Byrne-Lab/circuitGNN.git
cd circuitGNN
pip install -e ".[paper]"
pip install -r paper/requirements.txt    # exact versions used, incl. spycon for the baselines
```

Download the data archive and place its `datasets/` and `results/` folders at the
repository root. `results/` holds the trained replicates, so every evaluation and figure
below runs without retraining. Expected layout:

```
datasets/
  small_circuit_A/, small_circuit_B/       spikes.out, spikes.head
  arbitrary/, arbitrary_modification1/,
  arbitrary_modification3/                 spikes.out
  OC_1/                                    spikes.out, spikes.head, true_connectivity_0.20.npy
  OC_1/OCx1_ablation_<pre>_to_<post>/      smu/BMP.smu.out (single-synapse ablations, Fig. 5)
  pdcm/, pdcm_0p002/, pdcm_0p010/          spkt.npy, spkid.npy, weight_matrix.npy,
                                           pop_labels.npy, sim_duration_ms.npy
  OC_1/baseline/, pdcm/baseline/           baseline matrices (see Baselines)
results/<dataset>/                         gnn_<i>.npy, best_model_run_<i>.pt, ablation.csv
```

Run every command as `python -m ...`. Settings for every dataset (replicates, epochs, bin)
are in [`config.py`](config.py).

## Datasets

| Code name | Manuscript | Neurons | Recording | Native resolution | Bin | Replicates × epochs |
|---|---|---|---|---|---|---|
| `small_circuit_A` | Network A | 3 | 200 s | 0.5 ms | auto (9 ms) | 50 × 20 |
| `small_circuit_B` | Network B | 3 | 200 s | 0.5 ms | auto (4 ms) | 50 × 20 |
| `arbitrary` | Network C | 11 | 300 s | 1 ms | auto (4 ms) | 50 × 20 |
| `arbitrary_modification1` | Network C, N3→N10 weakened | 11 | 300 s | 1 ms | auto | 50 × 20 |
| `arbitrary_modification3` | Network C, N6→N2 strengthened | 11 | 300 s | 1 ms | auto | 50 × 20 |
| `OC_1` | *Aplysia* feeding circuit | 14 | 240 s | 0.5 ms | auto (4 ms) | 50 × 20 |
| `pdcm` | PDCM | 149 | 200 s | 1 ms | auto (3 ms) | 50 × 50 |
| `pdcm_hide20`, `pdcm_hide40` | PDCM, 20% / 40% hidden | 119 / 89 observed | 200 s | 1 ms | 3 ms | 20 × 50 |
| `pdcm_0p002_<T>s` | PDCM, 149 neurons, first *T* s | 149 | *T* = 60, 120, 240, 480 s | 1 ms | 3 ms | 20 × 50 |
| `pdcm_0p010_<T>s` | PDCM, 767 neurons, first *T* s | 767 | *T* = 60, 120, 240, 480 s | 1 ms | 3 ms | 20 × 50 |

`auto` is circuit_gnn's bin selection; the value it picks is in parentheses. The `pdcm_0p*`
sweeps are cut from 2000 s simulations.

## Pipeline

1. **Train** the replicates (skip if you downloaded `results/`):
   `python -m paper.train <dataset>`, for every dataset above.
2. **Evaluate**: `python -m paper.evaluate <dataset>` writes `results/<dataset>/mean.npy`
   and `mean_consistent.npy` (the consensus, with heatmaps) and prints the metrics.
3. **Figures**:

| Figure | Command | Output |
|---|---|---|
| 2A1, 2B1, 2C1 (rasters), 4A | `python -m paper.figures.plot_spikes` | `paper/figures/<dataset>_raster.pdf` |
| 2A2, 2B2, 2C2–C3 (connectivity) | `python -m paper.evaluate <dataset>` | `results/<dataset>/mean_consistent.pdf`, `datasets/arbitrary/ground_truth.pdf` |
| 3 | `python -m paper.figures.plot_arb_exp` | `paper/figures/arb_exp/` |
| 4B–C | `python -m paper.figures.plot_oc_1` | `paper/figures/oc_1/` |
| 4D, 6D | `python -m paper.figures.plot_baselines` | `paper/figures/baselines/` |
| 5 | `python -m paper.figures.plot_ablation_spikes` | `paper/figures/OC_1_ablation_rasters.pdf` |
| 6A–C | `python -m paper.figures.plot_pdcm` | `paper/figures/pdcm/` |
| 7A | `python -m paper.experiments.scaling` | `paper/figures/scaling/`, `paper/figures/metrics_scaling.csv` |
| 7B–C | `python -m paper.experiments.hiding` | `paper/figures/hiding/`, `paper/figures/metrics_hiding.csv` |

The figure scripts read the consensus written by step 2; `scaling` and `hiding` compute
it themselves. 4D and 6D need the baseline scores below. Fig. 5 reads the raw voltage
traces of the default and ablated *Aplysia* simulations, not the consensus.

## Baselines

sCCG, dSTTC, GLMCC and GLMPP run through [spycon](https://github.com/christiando/spycon)
(Donner et al., 2024), over the parameter grid in the Methods:

```bash
python -m paper.baselines.run_spycon     # datasets/{OC_1,pdcm}/baseline/spycon_*.npy
```

CTM-DI (Cai et al., 2017) and CoNNECT (Endo et al., 2021) were run with their authors'
code, which is not redistributed here.

```bash
python -m paper.baselines.compare        # baseline_comparison.csv, baseline_best_per_class.csv
```

scores every matrix against ground truth and, per method, keeps the setting with the best F1. Settings are thus tuned on the ground truth, which favours the baselines.
