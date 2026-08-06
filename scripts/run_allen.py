import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_allen_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_region_matrix


def main():
    cfg = RunConfig(num_epochs=50) # num_runs=5, 
    regions = ["DG", "CA3", "CA1", "SUB"]

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_region_matrix(learned_A, [n.split("_")[0] for n in neuron_names], regions,
                           method_name=f"gnn_{run_idx}", save_dir=save_dir)

    run_experiment("allen", load_allen_network, cfg, plot_fn,
                   loader_kwargs={"data_dir": "datasets/allen/prep_793224716",
                                  "max_time_s": 600},
                   run_ids=range(8, 21))


if __name__ == "__main__":
    main()
