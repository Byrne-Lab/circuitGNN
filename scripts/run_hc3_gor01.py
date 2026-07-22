import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_hc3_gor01
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_hc3_matrix


def main():
    cfg = RunConfig(num_runs=5, num_epochs=20)

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_hc3_matrix(learned_A, neuron_names, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    # gor01-6-7 / 2006-6-7_11-26-53: ~60 CA3 cells in one recurrent region --
    # dense single-population sampling to limit unrecorded-neuron confounding.
    # Session config is baked into load_hc3_gor01; override via loader_kwargs if needed.
    run_experiment("hc3_gor01", load_hc3_gor01, cfg, plot_fn)


if __name__ == "__main__":
    main()
