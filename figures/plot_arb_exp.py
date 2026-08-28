"""Figures for the arbitrary SNNAP circuit and its two single-edge modifications,
saved under figures/arb_exp/.

Three conditions, eleven neurons each:

  arbitrary                baseline
  arbitrary_modification1  N3 -> N10 depressed, 1.25 -> 0.50
  arbitrary_modification2  N7 ->  N2 created,   0.00 -> 1.25

Three figures, each holding all three conditions side by side: the ground-truth
matrices, the GNN consistent means, and the spike rasters over [T_START, T_END].
Each matrix figure has one shared colour bar; the two families sit ~7x apart in
scale, so they get their own hardcoded limits.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / 'arb_exp'

NUM_NEURONS = 11
NEURON_NAMES = np.arange(NUM_NEURONS)

GT_NORMALIZER = 4.00

BASELINE_EDGES = {
    (0, 3):   0.75,   # N1 -> N4
    (2, 9):   1.25,   # N3 -> N10
    (3, 10):  0.70,   # N4 -> N11
    (4, 9):   1.25,   # N5 -> N10
    (5, 0):   0.50,   # N6 -> N1
    (5, 1):   1.25,   # N6 -> N2
    (5, 2):   0.50,   # N6 -> N3
    (5, 4):  -4.00,   # N6 -> N5
    (7, 1):  -4.00,   # N8 -> N2
    (8, 1):   1.25,   # N9 -> N2
}

CONDITIONS = [
    ('arbitrary',               'Baseline',                             BASELINE_EDGES),
    ('arbitrary_modification1', 'Mod 1: N3→N10 depressed (1.25→0.50)',  {**BASELINE_EDGES, (2, 9): 0.50}),
    ('arbitrary_modification2', 'Mod 2: N7→N2 created (0.00→1.25)',     {**BASELINE_EDGES, (6, 1): 1.25}),
]

T_START, T_END = 20.0, 60.0
SPIKE_THRESHOLD_MV = -10.0  # same threshold + re-arm as the loaders
REARM_SAMPLES = 8

GT_VMIN, GT_VMAX = -1.0, 0.4
GNN_VMIN, GNN_VMAX = -0.015, 0.015
CBAR_LABEL = 'Connection strength'


def ground_truth(edges):
    true_conn = np.zeros((NUM_NEURONS, NUM_NEURONS))
    for (pre, post), weight in edges.items():
        true_conn[pre, post] = weight
    return true_conn / GT_NORMALIZER


def load_spike_times(key):
    """Spike times (s) per neuron at native resolution, read up to T_END.

    Thresholding and the re-arm period match the loaders in data.py, and the
    scan starts at the beginning of the record, so the refractory bookkeeping is
    identical to a full-length pass -- only the tail is skipped. Spike times come
    from the file's own time column rather than index * 1 ms.
    """
    times, volts = [], []
    with open(ROOT / 'datasets' / key / 'spikes.out', encoding='utf-8') as f:
        for line in f:
            row = [float(v) for v in line.split()]
            times.append(row[0])
            volts.append(row[1:NUM_NEURONS + 1])
            if row[0] >= T_END:
                break
    times = np.asarray(times)
    volts = np.asarray(volts).T                       # (NUM_NEURONS, n_samples)

    spike_times = []
    for i in range(NUM_NEURONS):
        sp = np.where(volts[i] > SPIKE_THRESHOLD_MV)[0]
        if sp.size:
            sp = sp[np.concatenate(([True], np.diff(sp) > REARM_SAMPLES))]
        spike_times.append(times[sp])
    return spike_times


def plot_matrix_row(matrices, vmin, vmax, suptitle, fname):
    """The three matrices side by side under a single shared colour bar.

    Exact zeros are masked so absent connections render white rather than as the
    palette's centre colour.
    """
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='white')
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, matrix, (_, title, _) in zip(axes, matrices, CONDITIONS):
        matrix = np.asarray(matrix, dtype=float).copy()
        np.fill_diagonal(matrix, 0.0)
        sns.heatmap(np.ma.masked_where(matrix == 0, matrix), ax=ax,
                    xticklabels=NEURON_NAMES, yticklabels=NEURON_NAMES,
                    cmap=cmap, norm=norm, square=True, cbar=False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Postsynaptic')
    axes[0].set_ylabel('Presynaptic')

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=CBAR_LABEL, ticks=[vmin, 0, vmax])
    _save(fig, fname)


def plot_rasters(spike_times_by_cond, fname):
    """The three rasters stacked on one shared time axis."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, spike_times, (_, title, _) in zip(axes, spike_times_by_cond, CONDITIONS):
        ax.eventplot([t[t >= T_START] for t in spike_times],
                     lineoffsets=range(NUM_NEURONS), linelengths=0.8,
                     linewidths=0.5, color='black')
        ax.set_yticks(range(NUM_NEURONS))
        ax.set_yticklabels(NEURON_NAMES)
        ax.set_xlim(T_START, T_END)
        ax.set_ylim(NUM_NEURONS - 0.5, -0.5)   # neuron 0 on top
        ax.set_ylabel('Neuron')
        ax.set_title(title, fontsize=10)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel('Time (s)')

    fig.suptitle(f'Spike rasters  ({T_START:.0f}–{T_END:.0f} s)')
    plt.tight_layout()
    _save(fig, fname)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {OUT_DIR / name}')


if __name__ == '__main__':
    truths = [ground_truth(edges) for _, _, edges in CONDITIONS]
    means = [np.load(ROOT / 'results' / key / 'mean_consistent.npy') for key, _, _ in CONDITIONS]
    for (key, _, _), mean in zip(CONDITIONS, means):
        print(f'{key:<24} gnn [{mean.min():.4g}, {mean.max():.4g}]  '
              f'{np.count_nonzero(mean)} non-zero edges')

    plot_matrix_row(truths, GT_VMIN, GT_VMAX, 'True Connectivity', 'ground_truths.pdf')
    plot_matrix_row(means, GNN_VMIN, GNN_VMAX,
                    'GNN Connectivity Matrix (consistent mean)', 'consistent_means.pdf')
    plot_rasters([load_spike_times(key) for key, _, _ in CONDITIONS], 'rasters.pdf')
