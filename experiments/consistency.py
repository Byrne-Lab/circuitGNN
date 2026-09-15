"""
Inference quality vs the consistency threshold (0% = no threshold, 100% = the
connection must appear with the same sign in every replicate), for PDCM and OC_1.

The pipeline is exactly the one in src/circuit_gnn/eval.py (same ablation
filter, alignment, rounding), so the curve at 80% reproduces the "Consistent
Mean" line that eval.py prints for each dataset.

Outputs, per dataset, under figures/consistency/:
    consistency_{dataset}.pdf           metrics vs threshold
    consistency_matrices_{dataset}.pdf  ground truth + inferred matrices
    metrics_consistency_{dataset}.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import load_pdcm_network, load_OC_1_network
from circuit_gnn.utils import (align_matrices, get_consistent_connections, calculate_fit,
                               PDCM_POPS_ORDERED)

DATASETS = ["pdcm", "OC_1"]
LOADERS = {"pdcm": load_pdcm_network, "OC_1": load_OC_1_network}
TITLES = {"pdcm": "PDCM", "OC_1": "OC_1"}
# native bin width of each recording, used to hit the same GDI_BIN_MS as eval.py
NATIVE_MS = {"pdcm": 1.0, "OC_1": 0.5}

NUM_RUNS = 50
ABLATION_THRESHOLD = 0.20
GDI_BIN_MS = 20
THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
REFERENCE_THRESHOLD = 0.8
MATRIX_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

OUT_DIR = "figures/consistency"

# NIE (normalized inference error) is lower-is-better, unlike the other three.
CURVES = [
    ("F1",                 "signed_f1",        "#1a1a1a", "-",  1.8),
    ("precision",          "signed_precision", "#595959", "--", 1.2),
    ("recall",             "signed_recall",    "#8c8c8c", ":",  1.2),
    # ("NIE",                "nie",              "#b3b3b3", "-.", 1.4),
]


def load_passing_runs(results_loc):
    """Adjacency matrices of the runs that passed ablation and have a saved .npy."""
    ablation = pd.read_csv(f"{results_loc}/ablation.csv")
    mats = []
    for _, row in ablation.iterrows():
        if int(row.run_id) > NUM_RUNS:
            continue
        fname = f"{results_loc}/gnn_{int(row.run_id)}.npy"
        if row.difference <= -ABLATION_THRESHOLD and os.path.exists(fname):
            mats.append(np.load(fname))
    return mats


def prepare(dataset):
    """Aligned stack of passing runs, its mean, and the ground truth."""
    results_loc = f"results/{dataset}"
    downsample = round(GDI_BIN_MS / NATIVE_MS[dataset])

    binned_spikes, true_conn, neuron_names = LOADERS[dataset](downsample_factor=downsample)
    mats = load_passing_runs(results_loc)
    print(f"{len(mats)}/{NUM_RUNS} runs passed the ablation test")

    stack = np.stack(align_matrices(mats, binned_spikes=binned_spikes), axis=0)
    mean = np.round(stack.mean(axis=0), 3)
    return true_conn, neuron_names, stack, mean


def consistent_mean(stack, mean, threshold):
    """The inferred matrix at one consistency threshold, as eval.py builds it."""
    return mean * get_consistent_connections(stack, threshold=threshold)


def sweep(true_conn, stack, mean):
    """Metrics at every consistency threshold."""
    rows = []
    for t in THRESHOLDS:
        res = calculate_fit(true_conn, consistent_mean(stack, mean, t))
        rows.append({
            "threshold": t,
            "signed_f1": res.signed_f1,
            "signed_precision": res.signed_precision,
            "signed_recall": res.signed_recall,
            "overall_f1": res.overall_f1,
            "overall_precision": res.overall_precision,
            "overall_recall": res.overall_recall,
            "nie": res.nie,
        })
        if np.isclose(t, REFERENCE_THRESHOLD):
            print(f"threshold={t:.2f} (eval.py setting): F1={res.signed_f1:.2f}, "
                  f"Precision={res.signed_precision:.2f}, Recall={res.signed_recall:.2f}, "
                  f"NIE={res.nie:.2f}")
    return pd.DataFrame(rows)


def plot_curves(dataset, df):
    x = df["threshold"] * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, col, color, ls, lw in CURVES:
        ax.plot(x, df[col], color=color, lw=lw, ls=ls, label=label)
    ax.axvline(REFERENCE_THRESHOLD * 100, color="grey", lw=0.8, ls="-", alpha=0.6)

    ax.set_xlabel("consistency threshold (%)")
    ax.set_ylabel("score")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_title(TITLES[dataset])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()

    pdf_path = f"{OUT_DIR}/consistency_{dataset}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def _annotate_axes(ax, dataset, neuron_names, n):
    """Population blocks for PDCM, neuron names for the small OC_1 matrix."""
    if dataset == "pdcm":
        pop_labels = np.load("datasets/pdcm/pop_labels.npy")
        boundaries, centers, labels = [], [], []
        cursor = 0
        for pop in PDCM_POPS_ORDERED:
            count = int(np.sum(pop_labels == pop))
            if count == 0:
                continue
            boundaries.append(cursor)
            centers.append(cursor + count / 2)
            labels.append(pop)
            cursor += count
        boundaries.append(cursor)
        for b in boundaries:
            ax.axhline(b - 0.5, color="k", lw=0.4)
            ax.axvline(b - 0.5, color="k", lw=0.4)
        ticks, ticklabels, size = np.array(centers) - 0.5, labels, 5
    else:
        ticks = np.arange(n)
        ticklabels = list(neuron_names) if neuron_names is not None else list(range(n))
        size = 4

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=size, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=size)
    ax.tick_params(length=0)


def _diverging_norm(*mats):
    """TwoSlopeNorm centred at 0 spanning every value in the given matrices."""
    vals = np.concatenate([np.asarray(m, dtype=float).ravel() for m in mats])
    return TwoSlopeNorm(vmin=min(float(vals.min()), -1e-6),
                        vcenter=0,
                        vmax=max(float(vals.max()), 1e-6))


def plot_matrices(dataset, true_conn, neuron_names, stack, mean, df):
    # Ground truth is on its own scale; the six GNN panels share one, so their
    # raw weights stay comparable across thresholds.
    true_conn = np.asarray(true_conn, dtype=float)
    gnn_panels = []
    for t in MATRIX_THRESHOLDS:
        m = consistent_mean(stack, mean, t)
        row = df.iloc[(df["threshold"] - t).abs().idxmin()]
        title = f"{t * 100:.0f}%   F1 {row.signed_f1:.2f} / NIE {row.nie:.2f}"
        gnn_panels.append((title, m))

    truth_norm = _diverging_norm(true_conn)
    gnn_norm = _diverging_norm(*[m for _, m in gnn_panels])
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color="white")               # exact zeros render white

    panels = [("Ground truth", true_conn, truth_norm)]
    panels += [(title, m, gnn_norm) for title, m in gnn_panels]

    n = true_conn.shape[0]
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 14))
    flat = axes.ravel()
    mappables = []
    for ax, (title, mat, norm) in zip(flat, panels):
        im = ax.imshow(np.ma.masked_where(mat == 0, mat), cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal")
        mappables.append(im)
        ax.set_title(title, fontsize=8)
        _annotate_axes(ax, dataset, neuron_names, n)
        ax.set_xlabel("Postsynaptic", fontsize=6)
        ax.set_ylabel("Presynaptic", fontsize=6)

    cax_host = flat[-1]                        # 8th slot holds both colorbars
    cax_host.axis("off")
    for mappable, label, y in [(mappables[0], "Ground truth connection strength", 0.62),
                               (mappables[-1], "GNN weight (shared across thresholds)", 0.34)]:
        cb = fig.colorbar(mappable, cax=cax_host.inset_axes([0.1, y, 0.8, 0.04]),
                          orientation="horizontal")
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{TITLES[dataset]}: inferred connectivity vs consistency threshold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    pdf_path = f"{OUT_DIR}/consistency_matrices_{dataset}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for dataset in DATASETS:
        print(f"\n=== {TITLES[dataset]} ===")
        true_conn, neuron_names, stack, mean = prepare(dataset)
        df = sweep(true_conn, stack, mean)
        csv_path = f"{OUT_DIR}/metrics_consistency_{dataset}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
        plot_curves(dataset, df)
        plot_matrices(dataset, true_conn, neuron_names, stack, mean, df)


if __name__ == "__main__":
    main()
