import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_hc3_session
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_hc3_matrix


def main():
    cfg = RunConfig(num_runs=5, num_epochs=20)

    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_hc3_matrix(learned_A, neuron_names, method_name=f"gnn_{run_idx}", save_dir=save_dir)

    # CRCNS hc-3 in-vivo session ec016.1016 (topdir ec016.58). No ground-truth
    # connectivity exists; the E/I labels in neuron_names are the reference.
    run_experiment(
        "hc3",
        load_hc3_session,
        cfg,
        plot_fn,
        loader_kwargs={
            "tar_path": "datasets/hc3/ec016.1016.tar.gz",
            "topdir": "ec016.58",
            "cell_csv": "datasets/hc3/hc3-cell.csv",
            "regions": ["CA3", "DG", "EC2", "EC3"],
            "quality_filter": True,
            "max_time_s": 2000,   # use only the first X s of the session
        },
    )


if __name__ == "__main__":
    main()
