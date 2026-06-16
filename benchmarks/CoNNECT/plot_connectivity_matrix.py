# plot_connectivity_matrix.py
# Convert CoNNECT estimated.csv files into connectivity matrices and plots,

import os
import copy
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from circuit_gnn.utils import plot_pdcm_matrix

HERE = Path(__file__).resolve().parent          # benchmarks/CoNNECT
REPO = HERE.parents[1]                           # repo root

# datasets to convert (every dataset CoNNECT was run on, except v1)
DATASETS = ["arbitrary", "yuto", "small_circuit_A", "small_circuit_B", "pdcm"]


def plot_signed_matrix(matrix, neuron_names, method_name="CoNNECT", save_dir="results"):
    """Heatmap that always uses a diverging red/blue colormap centered at 0.

    Unlike circuit_gnn.utils.plot_connectivity_matrix (which falls back to a
    grey sequential map when no negative values are present), this keeps the
    excitatory/inhibitory coloring even when CoNNECT detects only excitatory
    connections -- because CoNNECT genuinely infers the sign of the PSP
    (positive = excitatory, negative = inhibitory), so an all-positive result
    means inhibitory connections were not detected, not that sign is unknown.
    """
    os.makedirs(save_dir, exist_ok=True)

    # force a symmetric-about-zero diverging norm even if all values are >= 0
    vmin = min(matrix.min(), -1e-6)
    vmax = max(matrix.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = copy.copy(sns.diverging_palette(240, 10, as_cmap=True))  # blue=inh, red=exc

    # render exact zeros (no connection) as white
    cmap.set_bad(color="white")
    masked = np.ma.masked_where(matrix == 0, matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(masked, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm, cbar_kws={"label": "Estimated PSP (mV)"},
                square=True)
    plt.title(f"{method_name} Connectivity Matrix")
    plt.xlabel("Postsynaptic neuron")
    plt.ylabel("Presynaptic neuron")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close()


def convert(dataset):
    res_dir = HERE / "results" / dataset
    csv_path = res_dir / "estimated.csv"
    if not csv_path.exists():
        print(f"[{dataset}] skipped (no {csv_path.relative_to(HERE)})")
        return

    # [post, pre] -> [pre, post]
    matrix = np.loadtxt(csv_path, delimiter=",").T
    np.save(res_dir / "CoNNECT.npy", matrix)
    print(f"[{dataset}] {matrix.shape}, non-zero connections: {np.count_nonzero(matrix)}")

    if dataset == "pdcm":
        pop_labels = np.load(REPO / "datasets" / "pdcm" / "pop_labels.npy")
        plot_pdcm_matrix(matrix, pop_labels, method_name="CoNNECT", save_dir=str(res_dir))
    else:
        names_file = res_dir / "neuron_names.txt"
        neuron_names = names_file.read_text().splitlines() if names_file.exists() \
            else list(range(matrix.shape[0]))
        plot_signed_matrix(matrix, neuron_names, method_name="CoNNECT",
                           save_dir=str(res_dir))


if __name__ == "__main__":
    for dataset in DATASETS:
        convert(dataset)
