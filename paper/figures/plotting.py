"""Plot helpers shared by the paper's scripts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

PDCM_POPS = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']


def pdcm_blocks(pop_labels):
    """``(boundaries, tick_centers, tick_labels)`` of the population blocks, in PDCM_POPS order."""
    pop_labels = np.asarray(pop_labels)
    boundaries, centers, labels = [], [], []
    cursor = 0
    for pop in PDCM_POPS:
        count = int(np.sum(pop_labels == pop))
        if count == 0:
            continue
        boundaries.append(cursor)
        centers.append(cursor + count / 2)
        labels.append(pop)
        cursor += count
    boundaries.append(cursor)
    return boundaries, centers, labels


def embed(matrix, kept, n_full):
    """Observed-only matrix back into its slot in the full N x N grid."""
    full = np.zeros((n_full, n_full))
    full[np.ix_(kept, kept)] = matrix
    return full


def plot_pdcm_matrix(matrix, pop_labels, path, title, vmin=None, vmax=None, blackout=None,
                     cbar_label='Connection strength'):
    """Heatmap of a PDCM matrix with its population blocks ruled off, saved to ``path``.

    vmin / vmax default to the data range; when given they also set the colour-bar ticks.
    blackout: neuron indices (e.g. hidden neurons) drawn as grey rows and columns.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    if blackout is not None and len(blackout) > 0:
        cmap.set_bad('gray')                  # sns.heatmap renders NaN cells with the bad colour
        matrix[blackout, :] = np.nan
        matrix[:, blackout] = np.nan
    cbar_kws = {'label': cbar_label}
    if vmin is None or vmax is None:
        vmin = min(np.nanmin(matrix), -1e-6)
        vmax = max(np.nanmax(matrix), 1e-6)
    else:
        cbar_kws['ticks'] = [vmin, 0, vmax]

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False, cmap=cmap,
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                cbar_kws=cbar_kws, square=True)

    boundaries, centers, labels = pdcm_blocks(pop_labels)
    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    ax.set_xticks(centers); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(centers); ax.set_yticklabels(labels, fontsize=8, rotation=0)

    plt.title(title)
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
