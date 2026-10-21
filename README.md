# CircuitGNN

[![CI](https://github.com/Byrne-Lab/circuitGNN/actions/workflows/ci.yml/badge.svg)](https://github.com/Byrne-Lab/circuitGNN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Inference of **signed synaptic connectivity** from multi-neuron spike trains, with a graph structure learning approach using graph neural networks.

This repository holds the `circuit_gnn` Python package and the code that reproduces the accompanying manuscript ([`paper/`](paper/README.md)).

## Install

Python ≥ 3.10. If you have a GPU, first install the [PyTorch build](https://pytorch.org/get-started/locally/)
for your CUDA version.

```bash
pip install "circuit-gnn @ git+https://github.com/Byrne-Lab/circuitGNN.git"
```

## Quickstart

```python
import numpy as np
from circuit_gnn import infer_connectivity, spikes_from_times, calculate_fit

# (n_neurons, n_bins) spike counts at the recording's native resolution ...
spikes = np.load("my_recording.npy")
# ... or binned from spike times in seconds, one array per neuron:
# spikes = spikes_from_times(spike_times, bin_ms=1.0)

result = infer_connectivity(spikes, native_bin_ms=1.0, n_runs=10, work_dir="runs/")
result.A_consistent           # (N, N) signed connectivity, the estimate to use
result.save("connectivity.npz")

# If ground-truth connectivity is available, score the result:
true_conn = np.load("true_connectivity.npy")
res = calculate_fit(true_conn, result.A_consistent)
print(f"Signed F1={res.signed_f1:.3f}  NIE={res.nie:.3f}  AUROC={res.auroc:.3f}")
```

`A_consistent[i, j]` is the connection from neuron `i` (presynaptic) to neuron `j`
(postsynaptic): positive is excitatory, negative inhibitory, 0 none. Weights are in arbitrary units.

| `result.` | |
|---|---|
| `A_consistent` | consensus connectivity, keeping only edges whose sign is stable across replicates |
| `A`, `consistency` | consensus before that mask, and the 0/1 mask |
| `runs`, `run_ids`, `flipped` | the kept replicates (sign-aligned), their ids, and which were sign-flipped |
| `ablation` | per replicate: validation loss with the learned connectivity minus without it |
| `bin_ms` | analysis bin used |

| `res.` | description |
|---|---|
| `signed_f1` | F1 over {excitatory, inhibitory} classes (main summary metric) |
| `signed_precision`, `signed_recall` | precision / recall for signed edges |
| `overall_f1` | F1 ignoring sign (any non-zero vs zero) |
| `overall_precision`, `overall_recall` | unsigned precision / recall |
| `nie` | normalized inference error (‖true − s·learned‖ / ‖true‖, optimal scale `s`) |
| `cosine` | cosine similarity of off-diagonal entries |
| `pearson`, `spearman` | linear / rank correlation of off-diagonal entries |
| `auroc` | AUROC using \|learned\| as a score for presence of any connection |

## How it works

1. **Bin size.** `bin_ms="auto"` picks the bin (1–10 or 20 ms) at which signed lag-1
   coupling is most reproducible across four segments of the recording, or 4 ms if there is
   no clear peak. Pass a number to fix it.
2. **Replicates.** `n_runs` models are trained independently to predict whether each
   neuron spikes in the next bin from the last 100 bins of the whole population. Each
   learns its own connectivity matrix. There are no pretrained weights. Use a GPU
   (`device="cuda"`; picked automatically when available).
3. **Ablation test.** A replicate is kept only if zeroing its connectivity raises its
   validation loss by at least `ablation_threshold` (0.10). This is an absolute loss difference, so it depends on the data; if every replicate fails, all are kept with a warning. Check `result.ablation`.
4. **Consensus.** Each replicate is correct only up to a global sign, so replicates are
   aligned to one another and to a partial-correlation sign estimate (20 ms bins),
   averaged and rounded to 3 decimals (`round_decimals`). `A_consistent` keeps the edges
   whose sign agrees in at least 80% of replicates (`consistency_threshold`).
The steps are also available separately: `fit_runs` trains replicates (and with
`out_dir` saves them), `load_runs` reads them back, `consensus` combines them, `Config`
holds the hyperparameters, and `calculate_fit` scores a matrix against known connectivity
(signed F1, precision, recall, normalized inference error).

## Reproducing the paper

See [`paper/README.md`](paper/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
