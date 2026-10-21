"""Ground truth next to the CircuitGNN consensus matrix at 60, 120, 240 and 480 s
for the 767-neuron PDCM network (pdcm_0p010), as one row of five heatmaps.

Run after `python -m paper.train pdcm_0p010_<duration>s` and `paper.evaluate` for every duration:

    python -m paper.figures.plot_pdcm_767

Writes paper/figures/scaling/pdcm_767_matrices.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from paper.config import DATASETS_DIR, FIGURES_DIR, RESULTS_DIR
from paper.figures.plotting import pdcm_blocks

DURATIONS_S = [60, 120, 240, 480]
DATA_DIR = DATASETS_DIR / 'pdcm_0p010'
OUT_PATH = FIGURES_DIR / 'scaling' / 'pdcm_767_matrices.pdf'


def draw(ax, matrix, pop_labels, title, vmin, vmax):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                   interpolation='nearest')
    boundaries, centers, labels = pdcm_blocks(pop_labels)
    for b in boundaries:
        ax.axhline(b - 0.5, color='k', linewidth=0.4)
        ax.axvline(b - 0.5, color='k', linewidth=0.4)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Postsynaptic')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], fraction=0.046, pad=0.04)


def main():
    pop_labels = np.load(DATA_DIR / 'pop_labels.npy')
    weights = np.load(DATA_DIR / 'weight_matrix.npy')

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    # ground truth, normalized by its largest magnitude
    draw(axes[0], weights / np.max(np.abs(weights)), pop_labels, 'Ground truth', -1, 0.5)
    for ax, dur_s in zip(axes[1:], DURATIONS_S):
        gnn = np.load(RESULTS_DIR / f'pdcm_0p010_{dur_s}s' / 'mean_consistent.npy')
        draw(ax, gnn, pop_labels, f'CircuitGNN, {dur_s} s', -0.002, 0.001)
    axes[0].set_ylabel('Presynaptic')

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches='tight')
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
