"""Fig. 3: Network C and its two single-edge modifications, side by side.

  arbitrary                baseline
  arbitrary_modification1  N3 -> N10 weakened,     1.25 -> 0.50
  arbitrary_modification3  N6 ->  N2 strengthened, 1.25 -> 3.125

Three figures, each holding the three conditions: the ground-truth matrices, the GNN
consistent means, and the spike rasters over [T_START, T_END]. Each matrix figure has one
shared colour bar; the two families sit ~7x apart in scale, so they get their own limits.

Run after paper.evaluate for the three datasets:

    python -m paper.figures.plot_arb_exp

Writes paper/figures/arb_exp/.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from paper.config import FIGURES_DIR, RESULTS_DIR
from paper.data import (ARBITRARY_EDGES, ARBITRARY_NUM_NEURONS, ARBITRARY_REARM,
                        MODIFICATION1_EDGES, MODIFICATION3_EDGES, arbitrary_ground_truth,
                        read_arbitrary_voltages, spike_indices)

OUT_DIR = FIGURES_DIR / 'arb_exp'
NEURON_NAMES = np.arange(ARBITRARY_NUM_NEURONS)

CONDITIONS = [
    ('arbitrary',               'Baseline',                              ARBITRARY_EDGES),
    ('arbitrary_modification1', 'Mod 1: N3→N10 depressed (1.25→0.50)',   MODIFICATION1_EDGES),
    ('arbitrary_modification3', 'Mod 3: N6→N2 potentiated (1.25→3.125)', MODIFICATION3_EDGES),
]

T_START, T_END = 20.0, 60.0

GT_VMIN, GT_VMAX = -1.0, 0.8   # mod 3's potentiated edge reaches 3.125/4.00
GNN_VMIN, GNN_VMAX = -0.015, 0.025
CBAR_LABEL = 'Connection strength'


def load_spike_times(key):
    """Spike times (s) per neuron, detected exactly as paper.data does but read only up to
    T_END, and timed by the file's own time column rather than index * 1 ms."""
    times, volts = read_arbitrary_voltages(key, T_END)
    return [times[sp] for sp in spike_indices(volts, rearm=ARBITRARY_REARM)]


def plot_matrix_row(matrices, vmin, vmax, suptitle, fname):
    """The matrices side by side under a single shared colour bar.

    Exact zeros are masked so absent connections render white rather than as the
    palette's centre colour.
    """
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='white')
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 6))
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
    """The rasters stacked on one shared time axis."""
    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(12, 4 * len(CONDITIONS)), sharex=True)
    for ax, spike_times, (_, title, _) in zip(axes, spike_times_by_cond, CONDITIONS):
        ax.eventplot([t[t >= T_START] for t in spike_times],
                     lineoffsets=range(ARBITRARY_NUM_NEURONS), linelengths=0.8,
                     linewidths=0.5, color='black')
        ax.set_yticks(range(ARBITRARY_NUM_NEURONS))
        ax.set_yticklabels(NEURON_NAMES)
        ax.set_xlim(T_START, T_END)
        ax.set_ylim(ARBITRARY_NUM_NEURONS - 0.5, -0.5)   # neuron 0 on top
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


def main():
    truths = [arbitrary_ground_truth(edges) for _, _, edges in CONDITIONS]
    means = [np.load(RESULTS_DIR / key / 'mean_consistent.npy') for key, _, _ in CONDITIONS]
    for (key, _, _), mean in zip(CONDITIONS, means):
        print(f'{key:<24} gnn [{mean.min():.4g}, {mean.max():.4g}]  '
              f'{np.count_nonzero(mean)} non-zero edges')

    plot_matrix_row(truths, GT_VMIN, GT_VMAX, 'True Connectivity', 'ground_truths.pdf')
    plot_matrix_row(means, GNN_VMIN, GNN_VMAX,
                    'GNN Connectivity Matrix (consistent mean)', 'consistent_means.pdf')
    plot_rasters([load_spike_times(key) for key, _, _ in CONDITIONS], 'rasters.pdf')


if __name__ == '__main__':
    main()
