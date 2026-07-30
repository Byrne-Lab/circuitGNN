import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_OC_2_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_connectivity_matrix


def main():
    cfg = RunConfig(num_runs=50, num_epochs=20)

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_connectivity_matrix(learned_A, neuron_names=neuron_names, method_name=f"GNN {run_idx}", save_dir=save_dir)

    run_experiment("OC_2", load_OC_2_network, cfg, plot_fn)


if __name__ == "__main__":
    main()
