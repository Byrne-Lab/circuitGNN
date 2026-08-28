import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_allen_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_region_matrix


def main():
    cfg = RunConfig(num_epochs=50)
    # Feedforward order; must match prepare_allen.REGIONS so the blocks line up.
    regions = ["LGd", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "LP"]

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_region_matrix(learned_A, [n.split("_")[0] for n in neuron_names], regions,
                           method_name=f"gnn_{run_idx}", save_dir=save_dir)

    # Own results dir, so the hippocampal runs in results/allen stay intact.
    run_experiment("allen_750749662", load_allen_network, cfg, plot_fn,
                   # num_runs=20,
                   loader_kwargs={"data_dir": "datasets/allen/prep_750749662_natural_movie_three",
                                  "start_time_s": 0,
                                  "max_time_s": 600},
                   run_ids=range(11, 21))


if __name__ == "__main__":
    main()
