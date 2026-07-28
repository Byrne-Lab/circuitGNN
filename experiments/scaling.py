"""
Signed F1 and NIE of the consistent-mean fit vs recording duration

Metrics are cached in figures/scaling_metrics.csv; a size/duration already in the
CSV is read from it instead of being recomputed. Delete a row to force a redo.

Outputs: figures/scaling_signed_f1.pdf, figures/scaling_nie.pdf
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import load_pdcm_network
from circuit_gnn.utils import align_matrices, get_consistent_connections, calculate_fit

SIZES = ["0p002", "0p010"]
DURATIONS_S = [60, 120, 240]

ABLATION_THRESHOLD = 0.10
CONSISTENCY_THRESHOLD = 0.8
GDI_BIN_MS = 20
OUT_DIR = "figures"
CACHE_CSV = f"{OUT_DIR}/scaling_metrics.csv"


def load_cache():
    """{(size, duration_s): (signed_f1, nie)} from previous runs."""
    if not os.path.exists(CACHE_CSV):
        return {}
    df = pd.read_csv(CACHE_CSV, dtype={"size": str})
    return {(r["size"], int(r.duration_s)): (r.signed_f1, r.nie) for _, r in df.iterrows()}


def save_cache(cache):
    pd.DataFrame(
        [{"size": s, "duration_s": d, "signed_f1": f1, "nie": nie}
         for (s, d), (f1, nie) in sorted(cache.items())]
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


def evaluate(size, dur_s):
    """(signed_f1, nie) of the consistent mean"""
    results_loc = f"results/pdcm_{size}_{dur_s}s"

    mats = load_passing_runs(results_loc)

    binned_spikes, true_conn, _ = load_pdcm_network(
        downsample_factor=GDI_BIN_MS, size=size, max_time_s=dur_s, plot=False)

    stack = np.stack(align_matrices(mats, binned_spikes=binned_spikes), axis=0)
    mask = get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)
    consistent_mean = np.round(stack.mean(axis=0), 3) * mask

    res = calculate_fit(true_conn, consistent_mean, plot=False)
    print(f"  ({len(mats)} runs)")
    return res.signed_f1, res.nie


def size_label(size):
    return f"N = {len(np.load(f'datasets/pdcm_{size}/pop_labels.npy'))}"


def plot_metric(curves, col, ylabel, fname, ylim):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    for c, (label, vals) in zip(colors, curves.items()):
        ax.plot(DURATIONS_S, vals[:, col], "o-", color=c, label=label, lw=1.5, alpha=0.8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS_S)
    ax.set_xticklabels([str(d) for d in DURATIONS_S])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("recording duration (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{fname}")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_cache()
    curves = {}
    for size in SIZES:
        print(f"pdcm_{size}")
        vals = np.full((len(DURATIONS_S), 2), np.nan)
        for i, dur_s in enumerate(DURATIONS_S):
            if (size, dur_s) not in cache:
                cache[(size, dur_s)] = evaluate(size, dur_s)
                save_cache(cache)
            vals[i] = cache[(size, dur_s)]
            print(f"  {dur_s}s: signed F1={vals[i, 0]:.2f}, NIE={vals[i, 1]:.2f}")
        curves[size_label(size)] = vals


    plot_metric(curves, 0, "signed F1", "scaling_signed_f1.pdf", (0, 1.0))
    plot_metric(curves, 1, "NIE", "scaling_nie.pdf", (0, 1.0))


if __name__ == "__main__":
    main()
