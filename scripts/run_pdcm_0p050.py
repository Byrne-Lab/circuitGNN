import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_pdcm_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_pdcm_matrix

SIZE = "0p050"
DURATIONS_S = [60, 120, 240, 480, 960, 1920]
BIN_MS = 3
NUM_RUNS = 20


def main():
    pop_labels = np.load(f"datasets/pdcm_{SIZE}/pop_labels.npy")

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_pdcm_matrix(learned_A, pop_labels, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    for dur_s in DURATIONS_S:
        cfg = RunConfig(num_runs=NUM_RUNS, num_epochs=50, downsample_factor=BIN_MS)
        run_experiment(
            f"pdcm_{SIZE}_{dur_s}s",
            load_pdcm_network,
            cfg,
            plot_fn,
            loader_kwargs={"size": SIZE, "max_time_s": dur_s, "plot": False},
        )


if __name__ == "__main__":
    main()
