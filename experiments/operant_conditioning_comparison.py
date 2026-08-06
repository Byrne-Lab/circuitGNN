import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import load_OC_1_network, load_OC_2_network
from circuit_gnn.utils import align_matrices, get_consistent_connections

DATASETS = ["OC_1", "OC_2"]
LOADERS = {"OC_1": load_OC_1_network, "OC_2": load_OC_2_network}

ABLATION_THRESHOLD = 0.10   # runs with difference > -threshold are non-optimal
CONSISTENCY_THRESHOLD = 0.8
GDI_BIN_MS = 20
NATIVE_MS = {"OC_1": 0.5, "OC_2": 0.5}
PLOT_CLIM = (-0.015, 0.010)     # GNN estimates
TRUTH_CLIM = (-6.0, 6.0)        # ground truth, in mV
OUT_DIR = "figures/operant_conditioning"


def load_passing_runs(results_loc):
    """(mats, passed_run_ids) for the runs in ablation.csv that pass ablation."""
    ablation_csv = pd.read_csv(f"{results_loc}/ablation.csv")
    mats, passed_run_ids = [], []
    for run_idx in ablation_csv["run_id"]:
        row = ablation_csv[ablation_csv["run_id"] == run_idx]
        fname = f"{results_loc}/gnn_{run_idx}.npy"
        if row.difference.values[0] > -ABLATION_THRESHOLD:
            print(f"  Skipping run {run_idx} for failing the ablation test")
            continue
        if not os.path.exists(fname):
            print(f"  Skipping run {run_idx}: {fname} not found")
            continue
        mats.append(np.load(fname))
        passed_run_ids.append(run_idx)
    print(f"  Kept {len(mats)}/{len(ablation_csv)} runs")
    return mats, passed_run_ids


def aligned_runs(data):
    """Sign-aligned adjacency matrices for one dataset, plus its true connectivity."""
    print(f"\n{data}")
    sign_bin = round(GDI_BIN_MS / NATIVE_MS.get(data, 1.0))
    binned_spikes, true_conn, neuron_names = LOADERS[data](downsample_factor=sign_bin)

    mats, passed_run_ids = load_passing_runs(f"results/{data}")

    before = [np.sign(A.sum()) for A in mats]
    mats = align_matrices(mats, bias=None, binned_spikes=binned_spikes)
    after = [np.sign(A.sum()) for A in mats]
    print(f"  Flipped runs: {[passed_run_ids[i] for i, (b, a) in enumerate(zip(before, after)) if b != a]}")

    return mats, true_conn, neuron_names


def consistent_mean(mats):
    """Consistency-masked mean over a stack of aligned matrices."""
    stack = np.stack(mats, axis=0)
    mask = get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)
    return np.round(stack.mean(axis=0), 3) * mask


def draw_matrix(ax, matrix, neuron_names, title, vmin, vmax,
                cbar_label='Connectivity Strength', cbar=True, annot=True):
    vmin, vmax = min(vmin, -1e-6), max(vmax, 1e-6)   # TwoSlopeNorm needs vmin < 0 < vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = copy.copy(sns.diverging_palette(240, 10, as_cmap=True))
    cmap.set_bad(color='white')                      # mask exact zeros so they render as white
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    annot_labels = (np.where(matrix == 0, '', np.vectorize(lambda v: f"{v:.3f}")(matrix))
                    if annot else None)

    sns.heatmap(masked_matrix, ax=ax, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm, cbar=cbar, cbar_kws={'label': cbar_label},
                square=True, annot=annot_labels, fmt='', annot_kws={'fontsize': 6})
    ax.set_title(title)
    ax.set_xlabel("Postsynaptic neuron")
    ax.set_ylabel("Presynaptic neuron")
    return cmap, norm


def plot_truth_vs_estimate(truths, estimates, names, save_dir):
    rows = [(truths, "Ground Truth", TRUTH_CLIM),
            (estimates, "GNN Consistent Mean", PLOT_CLIM)]

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(9 * len(DATASETS), 15),
                             layout='constrained')
    for row, (matrices, label, clim) in enumerate(rows):
        for col, data in enumerate(DATASETS):
            matrix = matrices[data]
            if matrix.min() < clim[0] or matrix.max() > clim[1]:
                print(f"  Note: {data} {label} has values outside {clim}; "
                      f"they saturate at the colourbar ends")
            cmap, norm = draw_matrix(axes[row, col], matrix, names[data],
                                     f"{data} {label}", clim[0], clim[1],
                                     cbar=False, annot=False)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes[row, :]),
                     label='Connectivity Strength', fraction=0.03, pad=0.02)
    save_path = f"{save_dir}/truth_vs_consistent_mean.pdf"
    fig.savefig(save_path, format='pdf')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_difference(truths, estimates, names, save_dir):
    if names["OC_2"] != names["OC_1"]:
        print("\nWARNING: OC_2 neuron order differs from OC_1; the difference maps are not meaningful")

    diffs = {"ground_truth": (truths["OC_2"] - truths["OC_1"], "Ground Truth"),
             "consistent_mean": (estimates["OC_2"] - estimates["OC_1"], "GNN Consistent Mean")}
    if not diffs["ground_truth"][0].any():
        print("\nNote: OC_1 and OC_2 ground truths are identical; that panel is empty")

    # separate scales per panel: ground truth is in mV, the estimates are ~100x smaller
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (diff, label) in zip(axes, diffs.values()):
        lim = np.abs(diff).max()
        draw_matrix(ax, np.round(diff, 3), names["OC_2"], f"OC_2 - OC_1, {label}", -lim, lim,
                    cbar_label='Difference in Connectivity Strength')
    fig.tight_layout()
    save_path = f"{save_dir}/oc2_minus_oc1.pdf"
    fig.savefig(save_path, format='pdf')
    plt.close(fig)
    for name, (diff, _) in diffs.items():
        np.save(f"{save_dir}/oc2_minus_oc1_{name}.npy", diff)
    print(f"Saved {save_path} and {save_dir}/oc2_minus_oc1_{{ground_truth,consistent_mean}}.npy")


def main():
    print(f"\nOC Comparison ({' + '.join(DATASETS)})")
    print(f"Ablation threshold: {ABLATION_THRESHOLD}")
    print(f"Consistency threshold: {CONSISTENCY_THRESHOLD}")

    estimates, truths, names = {}, {}, {}
    for data in DATASETS:
        mats, true_conn, neuron_names = aligned_runs(data)
        estimates[data] = consistent_mean(mats)
        truths[data] = true_conn
        names[data] = neuron_names

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_truth_vs_estimate(truths, estimates, names, OUT_DIR)
    plot_difference(truths, estimates, names, OUT_DIR)


if __name__ == "__main__":
    main()
