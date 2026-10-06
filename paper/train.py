"""Train the CircuitGNN replicates for one of the paper's datasets.

    python -m paper.train OC_1
    python -m paper.train pdcm_hide20 --run-ids 5 6     # (re)train selected replicates only

Writes to results/<dataset>/, per replicate: gnn_<i>.npy (learned connectivity,
[pre, post]), best_model_run_<i>.pt (checkpoint), gnn_<i>.pdf (heatmap) and a row of
ablation.csv (validation loss with and without the learned connectivity). Datasets,
replicate counts, epochs and bin sizes are listed in paper/config.py.
"""
import argparse

import numpy as np

from circuit_gnn import Config, fit_runs
from circuit_gnn.plotting import plot_connectivity_matrix
from paper import data
from paper.config import DATASETS, DATASETS_DIR, RESULTS_DIR
from paper.figures.plotting import embed, plot_pdcm_matrix


def plotter(name, neuron_names, out_dir):
    """``plot(matrix, label, **kw)``: heatmap of one matrix in the dataset's style, saved
    as out_dir/<label>.pdf (lower case, spaces as underscores)."""
    spec = DATASETS[name]
    if spec.kind == "small":
        return lambda m, label, **kw: plot_connectivity_matrix(
            m, neuron_names, method_name=label, save_dir=out_dir, **kw)

    def path(label):
        return out_dir / f"{label.replace(' ', '_').lower()}.pdf"

    if spec.kwargs.get("hide_frac"):   # hidden neurons: grey rows/columns of the full network
        pop_labels = np.load(DATASETS_DIR / "pdcm" / "pop_labels.npy")
        kept, hidden = data.pdcm_hidden_indices(len(pop_labels), spec.kwargs["hide_frac"],
                                                spec.kwargs["hide_seed"])
        return lambda m, label, **kw: plot_pdcm_matrix(
            embed(m, kept, len(pop_labels)), pop_labels, path(label),
            f"{label} Connectivity Matrix", blackout=hidden, **kw)
    pop_labels = np.asarray(neuron_names)
    return lambda m, label, **kw: plot_pdcm_matrix(
        m, pop_labels, path(label), f"{label} Connectivity Matrix", **kw)


def train(name, run_ids=None):
    spec = DATASETS[name]
    binned_spikes, _, neuron_names = data.load(name)
    out_dir = RESULTS_DIR / name
    plot = plotter(name, neuron_names, out_dir)
    fit_runs(binned_spikes, n_runs=spec.num_runs, run_ids=run_ids,
             cfg=Config(num_epochs=spec.num_epochs), out_dir=out_dir,
             on_run=lambda run_id, A: plot(A, f"GNN {run_id}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--run-ids", type=int, nargs="+",
                        help="replicates to train (default: 1 to the dataset's num_runs)")
    args = parser.parse_args()
    train(args.dataset, args.run_ids)


if __name__ == "__main__":
    main()
