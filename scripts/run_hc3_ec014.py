import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_hc3_ec014
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_hc3_matrix


def main():
    cfg = RunConfig(num_runs=5, num_epochs=30)

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_hc3_matrix(learned_A, neuron_names, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    # ec014.333 / topdir ec014.24 (bigSquare open field): the CA1(+EC) session used
    # for Fig. 5 of Kobayashi et al. 2019. Config (regions CA1+EC, min_rate_hz=0.5,
    # ~93 min) is baked into load_hc3_ec014; override via loader_kwargs if needed.
    run_experiment("hc3_ec014", load_hc3_ec014, cfg, plot_fn)


if __name__ == "__main__":
    main()
