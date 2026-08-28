# plot_connectivity_matrix.py
# Convert CoNNECT estimated.csv files into connectivity matrices and plots,

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from circuit_gnn.utils import plot_pdcm_matrix, plot_connectivity_matrix, plot_region_matrix

HERE = Path(__file__).resolve().parent          # benchmarks/CoNNECT
REPO = HERE.parents[1]                           # repo root

DATASETS = ["OC_1", "OC_2", "control", "pdcm", "arbitrary", "allen"]

# allen units are '<region>_<probe>_u<id>' and already grouped by region
ALLEN_REGIONS = ["DG", "CA3", "CA1", "SUB"]


def convert(dataset):
    res_dir = HERE / "results" / dataset        # CoNNECT scratch (estimated.csv, spiketrain.txt)
    out_dir = REPO / "datasets" / dataset / "baseline"
    csv_path = res_dir / "estimated.csv"
    if not csv_path.exists():
        print(f"[{dataset}] skipped (no {csv_path.relative_to(HERE)})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.loadtxt(csv_path, delimiter=",").T
    np.save(out_dir / "CoNNECT.npy", matrix)
    print(f"[{dataset}] {matrix.shape}, non-zero connections: {np.count_nonzero(matrix)} -> {out_dir}")

    if dataset == "pdcm":
        pop_labels = np.load(REPO / "datasets" / "pdcm" / "pop_labels.npy")
        plot_pdcm_matrix(matrix, pop_labels, method_name="CoNNECT", save_dir=str(out_dir))
    elif dataset == "allen":
        names_file = res_dir / "neuron_names.txt"
        regions = [n.split("_")[0] for n in names_file.read_text().splitlines()]
        plot_region_matrix(matrix, regions, ALLEN_REGIONS, method_name="CoNNECT",
                           save_dir=str(out_dir))
    else:
        names_file = res_dir / "neuron_names.txt"
        neuron_names = names_file.read_text().splitlines() if names_file.exists() \
            else list(range(matrix.shape[0]))
        plot_connectivity_matrix(matrix, neuron_names, method_name="CoNNECT",
                                 save_dir=str(out_dir), plot_val=False, signed=True)


if __name__ == "__main__":
    for dataset in DATASETS:
        convert(dataset)
