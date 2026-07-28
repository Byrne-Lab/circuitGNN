"""
Signed F1 of the consistent-mean fit vs the number of replicates (N = 1..50).

The x-axis is the run index: a run that fails ablation is not added to the
consistent-mean set, so its step carries the previous value forward (a flat
step). This keeps every dataset on a common 1..50 axis even when some runs fail.

Output: figures/replicate_signed_f1.pdf
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from circuit_gnn.data import (
    load_arbitrary_network, load_pdcm_network, load_small_circuit_A_network, load_small_circuit_B_network,
)
from circuit_gnn.utils import align_matrices, get_consistent_connections, calculate_fit

DATASETS = ["pdcm", "small_circuit_A", "small_circuit_B", "arbitrary"]
LOADERS = {
    "pdcm": load_pdcm_network,
    "arbitrary": load_arbitrary_network,
    "small_circuit_A": load_small_circuit_A_network,
    "small_circuit_B": load_small_circuit_B_network,
}

ABLATION_THRESHOLD = 0.10
CONSISTENCY_THRESHOLD = 0.8
MAX_RUNS = 50
GDI_BIN_MS = 20
NATIVE_MS = {"small_circuit_A": 0.5, "small_circuit_B": 0.5}
OUT_DIR = "figures"


def load_runs(results_loc):
    """(mats, pass_flags) in ablation.csv order; pass_flags[i] True where run i
    passed ablation and its .npy exists. len(mats) == sum(pass_flags)."""
    ablation_csv = pd.read_csv(f"{results_loc}/ablation.csv")
    mats, pass_flags = [], []
    for run_idx in ablation_csv["run_id"]:
        row = ablation_csv[ablation_csv["run_id"] == run_idx]
        passed = row.difference.values[0] <= -ABLATION_THRESHOLD
        ok = bool(passed and os.path.exists(f"{results_loc}/gnn_{run_idx}.npy"))
        pass_flags.append(ok)
        if ok:
            mats.append(np.load(f"{results_loc}/gnn_{run_idx}.npy"))
    return mats, pass_flags


def consistent_mean(aligned_subset):
    """Mean adjacency masked to connections seen consistently across the subset."""
    stack = np.stack(aligned_subset, axis=0)
    mask = get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)
    return np.round(stack.mean(axis=0), 3) * mask


def sweep_dataset(data):
    """signed-F1 value per run index N = 1..min(#runs, MAX_RUNS), or None to skip."""
    results_loc = f"results/{data}"
    if not os.path.exists(f"{results_loc}/ablation.csv"):
        print(f"  [skip] {results_loc}/ablation.csv not found")
        return None

    sign_bin = round(GDI_BIN_MS / NATIVE_MS.get(data, 1.0))
    binned_spikes, true_conn, _ = LOADERS[data](downsample_factor=sign_bin, plot=False)

    mats, pass_flags = load_runs(results_loc)
    N = min(len(pass_flags), MAX_RUNS)
    n_passed = sum(pass_flags[:N])
    print(f"  {data}: {n_passed}/{N} runs passed ablation, sweeping N = 1..{N}")

    aligned = align_matrices(mats, binned_spikes=binned_spikes)

    curve = np.full(N, np.nan)
    fit_by_count, count = {}, 0
    for i in range(N):
        if pass_flags[i]:
            count += 1
        if count == 0:
            continue  # leading failures: no consistent mean yet
        if count not in fit_by_count:
            cm = consistent_mean(aligned[:count])
            fit_by_count[count] = calculate_fit(true_conn, cm, plot=False).signed_f1
        curve[i] = fit_by_count[count]
    return {"N": N, "signed_f1": curve}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    swept = {d: out for d in DATASETS if (out := sweep_dataset(d)) is not None}
    if not swept:
        print("No datasets produced curves; check DATASETS / results dirs.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(swept)))
    for c, (data, out) in zip(colors, swept.items()):
        ax.plot(np.arange(1, out["N"] + 1), out["signed_f1"],
                color=c, label=data, lw=1.5, alpha=0.7)

    ax.set_xlabel("number of replicates (N)")
    ax.set_ylabel("signed F1")
    ax.set_xlim(1, MAX_RUNS)
    ax.set_ylim(0, 1.02)  # headroom so lines sitting at y=1 aren't clipped
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/replicate_signed_f1.pdf")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/replicate_signed_f1.pdf")


if __name__ == "__main__":
    main()
