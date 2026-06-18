import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_pdcm_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_pdcm_matrix


def main():
    cfg = RunConfig(num_runs=50, num_epochs=50)

    pop_labels = np.load("datasets/pdcm/pop_labels.npy")

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_pdcm_matrix(learned_A, pop_labels, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    run_experiment("pdcm", load_pdcm_network, cfg, plot_fn)


if __name__ == "__main__":
    main()
