import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from circuit_gnn.data import load_th2_network
from circuit_gnn.experiment import RunConfig, run_experiment
from circuit_gnn.utils import plot_region_matrix

GROUPS = ["HD", "FS", "EX"]     

SESSIONS = [
    "A3716-201015b",    # 147 units
    # "A3711-200810b",    # 142 units
]
EPOCH = "wake1"         # square open field; wake2 = triangle

NUM_RUNS = 20
NUM_EPOCHS = 50
MAX_TIME_S = 600       


def plot_fn_for(save_groups):
    def plot_fn(learned_A, run_idx, save_dir, neuron_names):
        plot_region_matrix(learned_A, [n.split("_")[0] for n in neuron_names],
                           save_groups, method_name=f"gnn_{run_idx}",
                           save_dir=save_dir)
    return plot_fn


def main():
    for session in SESSIONS:
        data_dir = f"datasets/th2/prep_{session}_{EPOCH}"

        cfg = RunConfig(downsample_factor="auto", num_runs=NUM_RUNS,
                        num_epochs=NUM_EPOCHS)

        data = f"th2_{session}_{EPOCH}"    
        print(f"\n{'=' * 60}\n{data}: {MAX_TIME_S} s, bin auto, "
              f"{NUM_RUNS} runs\n{'=' * 60}")

        run_experiment(data, load_th2_network, cfg, plot_fn_for(GROUPS),
                       loader_kwargs={"data_dir": data_dir,
                                      "start_time_s": 0,
                                      "max_time_s": MAX_TIME_S},
                       run_ids=range(11, NUM_RUNS + 1))


if __name__ == "__main__":
    main()
