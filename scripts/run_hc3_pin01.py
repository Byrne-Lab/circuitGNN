import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_hc3_pin01
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_hc3_matrix


def main():
    cfg = RunConfig(num_runs=5, num_epochs=20)

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_hc3_matrix(learned_A, neuron_names, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    # pin01-11-04 / 11-04_22-31-40: CA1+CA3 (20+27 cells) with monosynaptic labels
    # in both regions -- a cleaner E/I validation reference than the CA3-only session.
    # Session config is baked into load_hc3_pin01; override via loader_kwargs if needed.
    run_experiment("hc3_pin01", load_hc3_pin01, cfg, plot_fn)


if __name__ == "__main__":
    main()
