"""
GNN inference vs recording duration and network size

Metrics are cached in figures/scaling_metrics.csv; a size/duration already in the
CSV is read from it instead of being recomputed.

Outputs: figures/scaling/
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import load_pdcm_network
from circuit_gnn.utils import (align_matrices, get_consistent_connections, calculate_fit,
                               plot_pdcm_matrix, PDCM_POPS_ORDERED)

SIZES = ["0p002", "0p010"]
DURATIONS_S = [60, 120, 240, 480]

ABLATION_THRESHOLD = 0.20
CONSISTENCY_THRESHOLD = 0.8
GDI_BIN_MS = 20
OUT_DIR = "figures"
CACHE_CSV = f"{OUT_DIR}/metrics_scaling.csv"
MATRIX_DIR = f"{OUT_DIR}/scaling"

CURVE_COLORS = ["#1b5e20", "#66bb6a"]     
MARKER_SIZE = 5

DEFAULT_CLIM = (-0.010, 0.005)
MATRIX_CLIM = {
    ("0p002", 60): (-0.010, 0.005),
    ("0p002", 120): (-0.010, 0.005),
    ("0p002", 240): (-0.010, 0.005),
    ("0p002", 480): (-0.010, 0.005),
    
    ("0p010", 60): (-0.002, 0.001),
    ("0p010", 120): (-0.002, 0.001),
    ("0p010", 240): (-0.002, 0.001),
    ("0p010", 480): (-0.002, 0.001),
}


def load_cache():
    """{(size, duration_s): (signed_f1, nie, signed_precision, signed_recall)} from previous runs."""
    if not os.path.exists(CACHE_CSV):
        return {}
    df = pd.read_csv(CACHE_CSV, dtype={"size": str})
    return {(r["size"], int(r.duration_s)): (r.signed_f1, r.nie, r.signed_precision, r.signed_recall)
            for _, r in df.iterrows()}


def save_cache(cache):
    pd.DataFrame(
        [{"size": s, "duration_s": d, "signed_f1": f1, "nie": nie,
          "signed_precision": prec, "signed_recall": rec}
         for (s, d), (f1, nie, prec, rec) in sorted(cache.items())]
    ).to_csv(CACHE_CSV, index=False)


def load_passing_runs(results_loc):
    """Adjacency matrices of the runs that passed ablation."""
    ablation = pd.read_csv(f"{results_loc}/ablation.csv")
    mats = []
    for _, row in ablation.iterrows():
        fname = f"{results_loc}/gnn_{int(row.run_id)}.npy"
        if row.difference <= -ABLATION_THRESHOLD and os.path.exists(fname):
            mats.append(np.load(fname))
    return mats


def evaluate(size, dur_s):
    """(signed_f1, nie, signed_precision, signed_recall) of the consistent mean"""
    results_loc = f"results/pdcm_{size}_{dur_s}s"

    mats = load_passing_runs(results_loc)

    binned_spikes, true_conn, _ = load_pdcm_network(
        downsample_factor=GDI_BIN_MS, size=size, max_time_s=dur_s)

    stack = np.stack(align_matrices(mats, binned_spikes=binned_spikes), axis=0)
    mask = get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)
    mean = np.round(stack.mean(axis=0), 3)
    consistent_mean = mean * mask

    np.save(f"{results_loc}/mean.npy", mean)
    np.save(f"{results_loc}/mean_consistent.npy", consistent_mean)

    pop_labels = np.load(f"datasets/pdcm_{size}/pop_labels.npy")
    plot_pdcm_matrix(mean, pop_labels, method_name="mean", save_dir=results_loc)
    plot_pdcm_matrix(consistent_mean, pop_labels, method_name="mean_consistent", save_dir=results_loc)

    res = calculate_fit(true_conn, consistent_mean)
    print(f"  ({len(mats)} runs)")
    return res.signed_f1, res.nie, res.signed_precision, res.signed_recall


def plot_matrix(matrix, pop_labels, vmin, vmax, title, cbar_label, fname):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
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


def plot_sweep_matrix(size, dur_s):
    results_loc = f"results/pdcm_{size}_{dur_s}s"
    mat_path = f"{results_loc}/mean_consistent.npy"
    if not os.path.exists(mat_path):
        evaluate(size, dur_s)          

    vmin, vmax = MATRIX_CLIM.get((size, dur_s), DEFAULT_CLIM)
    plot_matrix(
        np.load(mat_path),
        np.load(f"datasets/pdcm_{size}/pop_labels.npy"),
        vmin=vmin, vmax=vmax,
        title=f"GNN Connectivity Matrix  ({size_label(size)}, {dur_s} s)",
        cbar_label="Connection strength",
        fname=f"mean_consistent_{size}_{dur_s}s.pdf",
    )


def plot_ground_truth(size):
    true_conn = np.load(f"datasets/pdcm_{size}/weight_matrix.npy")
    true_conn = true_conn / np.max(np.abs(true_conn))

    vmin, vmax = -1, 0.5
    plot_matrix(
        true_conn,
        np.load(f"datasets/pdcm_{size}/pop_labels.npy"),
        vmin=vmin, vmax=vmax,
        title=f"True Connectivity  ({size_label(size)})",
        cbar_label="Connection strength",
        fname=f"ground_truth_{size}.pdf",
    )


def size_label(size):
    return f"N = {len(np.load(f'datasets/pdcm_{size}/pop_labels.npy'))}"


def plot_metric(curves, col, ylabel, fname, ylim, extras=()):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CURVE_COLORS[i % len(CURVE_COLORS)] for i in range(len(curves))]
    for c, (label, vals) in zip(colors, curves.items()):
        ax.plot(DURATIONS_S, vals[:, col], marker="o", ms=MARKER_SIZE, ls="-",
                color=c, mfc=c, mec=c, label=label, lw=1.5)
        for ecol, style, _ in extras:
            ax.plot(DURATIONS_S, vals[:, ecol], marker="o", ms=MARKER_SIZE, ls=style,
                    color=c, mfc=c, mec=c, lw=1.2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS_S)
    ax.set_xticklabels([str(d) for d in DURATIONS_S])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("recording duration (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    size_legend = ax.legend(fontsize=8, loc="lower right")
    if extras:
        styles = [(("-"), ylabel)] + [(s, lab) for _, s, lab in extras]
        ax.legend(handles=[Line2D([], [], color="0.3", ls=s, label=lab) for s, lab in styles],
                  fontsize=8, loc="upper left")
        ax.add_artist(size_legend)      # ax.legend() replaces; re-add the size one
    fig.tight_layout()
    os.makedirs(MATRIX_DIR, exist_ok=True)
    fig.savefig(f"{MATRIX_DIR}/{fname}")
    plt.close(fig)
    print(f"Saved {MATRIX_DIR}/{fname}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_cache()
    curves = {}
    for size in SIZES:
        print(f"pdcm_{size}")
        vals = np.full((len(DURATIONS_S), 4), np.nan)
        for i, dur_s in enumerate(DURATIONS_S):
            if (size, dur_s) not in cache:
                cache[(size, dur_s)] = evaluate(size, dur_s)
                save_cache(cache)
            vals[i] = cache[(size, dur_s)]
            print(f"  {dur_s}s: F1={vals[i, 0]:.2f}, NIE={vals[i, 1]:.2f}")
        curves[size_label(size)] = vals

    for size in SIZES:
        plot_ground_truth(size)
        for dur_s in DURATIONS_S:
            plot_sweep_matrix(size, dur_s)

    plot_metric(curves, 0, "F1 score", "scaling_f1.pdf", (0, 1.0),
                extras=[(2, "--", "precision"), (3, ":", "recall")])
    plot_metric(curves, 1, "NIE", "scaling_nie.pdf", (0, 1.0))


if __name__ == "__main__":
    main()
