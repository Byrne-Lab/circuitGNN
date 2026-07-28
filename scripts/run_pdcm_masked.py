import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_pdcm_network, pdcm_hidden_indices
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_pdcm_matrix

HIDE_FRACS = [0.2, 0.4, 0.6, 0.8]
HIDE_SEED = 0
BIN_MS = 3
NUM_RUNS = 20


def main():
    pop_labels = np.load("datasets/pdcm/pop_labels.npy")   # full population
    n_full = len(pop_labels)

    for frac in HIDE_FRACS:
        kept, hidden = pdcm_hidden_indices(n_full, frac, HIDE_SEED)

        def plot_fn(learned_A, run_idx, save_dir, neuron_names, kept=kept, hidden=hidden):
            full = np.zeros((n_full, n_full))
            full[np.ix_(kept, kept)] = learned_A       # observed-only A back into full N x N
            plot_pdcm_matrix(full, pop_labels, method_name=f"gnn_{run_idx}", save_dir=save_dir, blackout=hidden)

        cfg = RunConfig(num_runs=NUM_RUNS, num_epochs=50, downsample_factor=BIN_MS)
        run_experiment(
            f"pdcm_hide{int(frac * 100)}",
            load_pdcm_network,
            cfg,
            plot_fn,
            loader_kwargs={"hide_frac": frac, "hide_seed": HIDE_SEED, "plot": False},
        )


if __name__ == "__main__":
    main()
