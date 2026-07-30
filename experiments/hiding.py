"""
GNN inference vs hidden neuron fraction

Metrics are cached in figures/hiding_metrics.csv; a hide fraction already in the
CSV is read from it instead of being recomputed.

Outputs: figures/hiding/
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import load_pdcm_network, pdcm_hidden_indices
from circuit_gnn.utils import (align_matrices, get_consistent_connections, calculate_fit,
                               plot_pdcm_matrix, PDCM_POPS_ORDERED)

HIDE_FRACS = [0.0, 0.2, 0.4]
HIDE_SEED = 0

ABLATION_THRESHOLD = 0.10
CONSISTENCY_THRESHOLD = 0.8
GDI_BIN_MS = 20
DATA_DIR = "datasets/pdcm"
OUT_DIR = "figures"
CACHE_CSV = f"{OUT_DIR}/metrics_hiding.csv"

MATRIX_DIR = f"{OUT_DIR}/hiding"

MAIN_COLOR = "#1b5e20"      # dark green: the headline metric
EXTRA_COLOR = "#66bb6a"     # light green: precision / recall
MARKER_SIZE = 5


def pct(frac):
    return int(round(frac * 100))


def results_dir(frac):
    return "results/pdcm" if frac == 0 else f"results/pdcm_hide{pct(frac)}"


def load_cache():
    """{hide_frac: (signed_f1, nie, signed_precision, signed_recall)} from previous runs."""
    if not os.path.exists(CACHE_CSV):
        return {}
    df = pd.read_csv(CACHE_CSV)
    return {float(r.hide_frac): (r.signed_f1, r.nie, r.signed_precision, r.signed_recall)
            for _, r in df.iterrows()}


def save_cache(cache):
    pd.DataFrame(
        [{"hide_frac": f, "hide_pct": pct(f), "signed_f1": f1, "nie": nie,
          "signed_precision": prec, "signed_recall": rec}
         for f, (f1, nie, prec, rec) in sorted(cache.items())]
    ).to_csv(CACHE_CSV, index=False)


def load_passing_runs(results_loc):
    """Adjacency matrices of the runs that passed ablation and have a saved .npy."""
    ablation = pd.read_csv(f"{results_loc}/ablation.csv")
    mats = []
    for _, row in ablation.iterrows():
        fname = f"{results_loc}/gnn_{int(row.run_id)}.npy"
        if row.difference <= -ABLATION_THRESHOLD and os.path.exists(fname):
            mats.append(np.load(fname))
    return mats


def evaluate(frac):
    """(signed_f1, nie, signed_precision, signed_recall) of the consistent mean"""
    results_loc = results_dir(frac)

    mats = load_passing_runs(results_loc)
    print(f"  {len(mats)} runs")

    # observed-only spikes and ground truth (the loader reduces both to the kept neurons)
    binned_spikes, true_conn, _ = load_pdcm_network(downsample_factor=GDI_BIN_MS, hide_frac=frac, hide_seed=HIDE_SEED, plot=False)

    stack = np.stack(align_matrices(mats, binned_spikes=binned_spikes), axis=0)
    mask = get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)
    mean = np.round(stack.mean(axis=0), 3)
    consistent_mean = mean * mask

    np.save(f"{results_loc}/mean.npy", mean)
    np.save(f"{results_loc}/mean_consistent.npy", consistent_mean)

    pop_labels = np.load(f"{DATA_DIR}/pop_labels.npy")
    kept, hidden = pdcm_hidden_indices(len(pop_labels), frac, HIDE_SEED)
    for matrix, name in [(mean, "mean"), (consistent_mean, "mean_consistent")]:
        plot_pdcm_matrix(embed(matrix, kept, len(pop_labels)), pop_labels,
                         method_name=name, save_dir=results_loc, blackout=hidden)

    res = calculate_fit(true_conn, consistent_mean, plot=False)
    print(f"  ({len(mats)} runs)")
    return res.signed_f1, res.nie, res.signed_precision, res.signed_recall


def embed(matrix, kept, n_full):
    """Observed-only matrix back into its slot in the full N x N grid."""
    full = np.zeros((n_full, n_full))
    full[np.ix_(kept, kept)] = matrix
    return full


def plot_matrix(matrix, pop_labels, vmin, vmax, title, cbar_label, fname, blackout=None):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    matrix = np.asarray(matrix, dtype=float).copy()
    if blackout is not None and len(blackout) > 0:
        cmap.set_bad("gray")            # sns.heatmap renders NaN cells with the bad colour
        matrix[blackout, :] = np.nan
        matrix[:, blackout] = np.nan
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={"label": cbar_label, "ticks": [vmin, 0, vmax]},
                square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for p in PDCM_POPS_ORDERED:
        count = int(np.sum(pop_labels == p))
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(p)
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color="k", linewidth=0.5)
        ax.axvline(b, color="k", linewidth=0.5)
    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.title(title)
    plt.xlabel("Postsynaptic")
    plt.ylabel("Presynaptic")
    plt.tight_layout()

    os.makedirs(MATRIX_DIR, exist_ok=True)
    fig.savefig(f"{MATRIX_DIR}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {MATRIX_DIR}/{fname}")


def plot_sweep_matrix(frac):
    mat_path = f"{results_dir(frac)}/mean_consistent.npy"
    if not os.path.exists(mat_path):
        evaluate(frac)          # side effect: writes mean.npy / mean_consistent.npy

    pop_labels = np.load(f"{DATA_DIR}/pop_labels.npy")
    kept, hidden = pdcm_hidden_indices(len(pop_labels), frac, HIDE_SEED)

    plot_matrix(
        embed(np.load(mat_path), kept, len(pop_labels)),
        pop_labels,
        vmin=-0.010, vmax=0.005,
        title=f"GNN Connectivity Matrix  ({pct(frac)}% hidden, {len(kept)} of {len(pop_labels)} observed)",
        cbar_label="Connection strength",
        fname=f"mean_consistent_hide{pct(frac)}.pdf",
        blackout=hidden,
    )


def plot_ground_truth(frac):
    """Full ground truth with the hidden neurons greyed out."""
    pop_labels = np.load(f"{DATA_DIR}/pop_labels.npy")
    _, hidden = pdcm_hidden_indices(len(pop_labels), frac, HIDE_SEED)

    true_conn = np.load(f"{DATA_DIR}/weight_matrix.npy")
    true_conn = true_conn / np.max(np.abs(true_conn))

    plot_matrix(
        true_conn, pop_labels,
        vmin=-1.0, vmax=0.5,
        title=f"True Connectivity  ({pct(frac)}% hidden)",
        cbar_label="Connection strength",
        fname=f"ground_truth_hide{pct(frac)}.pdf",
        blackout=hidden,
    )


def plot_metric(vals, col, ylabel, fname, ylim, extras=()):
    """extras: (column, linestyle, label) drawn alongside the main curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [pct(f) for f in HIDE_FRACS]
    ax.plot(x, vals[:, col], marker="o", ms=MARKER_SIZE, ls="-", color=MAIN_COLOR,
            mfc=MAIN_COLOR, mec=MAIN_COLOR, lw=1.5, label=ylabel)
    for ecol, style, label in extras:
        ax.plot(x, vals[:, ecol], marker="o", ms=MARKER_SIZE, ls=style, color=EXTRA_COLOR,
                mfc=EXTRA_COLOR, mec=EXTRA_COLOR, lw=1.2, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("hidden neurons (%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if extras:
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(MATRIX_DIR, exist_ok=True)
    fig.savefig(f"{MATRIX_DIR}/{fname}")
    plt.close(fig)
    print(f"Saved {MATRIX_DIR}/{fname}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_cache()
    vals = np.full((len(HIDE_FRACS), 4), np.nan)
    for i, frac in enumerate(HIDE_FRACS):
        print(f"pdcm_hide{pct(frac)}")
        if frac not in cache:
            cache[frac] = evaluate(frac)
            save_cache(cache)
        vals[i] = cache[frac]
        print(f"  F1={vals[i, 0]:.2f}, NIE={vals[i, 1]:.2f}")

    for i, frac in enumerate(HIDE_FRACS):
        if np.isnan(vals[i, 0]):
            continue
        plot_ground_truth(frac)
        plot_sweep_matrix(frac)

    plot_metric(vals, 0, "F1 score", "hiding_f1.pdf", (0, 1.0),
                extras=[(2, "--", "precision"), (3, ":", "recall")])
    plot_metric(vals, 1, "NIE", "hiding_nie.pdf", (0, 1.0))


if __name__ == "__main__":
    main()
