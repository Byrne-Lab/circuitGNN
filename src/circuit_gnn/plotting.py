"""Connectivity-matrix figures."""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm


def plot_connectivity_matrix(matrix, neuron_names, method_name='GNN', cmap=None, save_dir=None, plot_val=True, vmin=None, vmax=None, signed=True, boundaries=None):
    # boundaries: row/column indices to rule off, for a matrix whose rows have been
    # permuted into blocks (layer, cell class, module); None draws no lines.
    # save_dir: write <method_name>.pdf there and close the figure; None returns it open.
    if signed:
        vmin = min(matrix.min(), -1e-6) if vmin is None else vmin
        vmax = max(matrix.max(),  1e-6) if vmax is None else vmax
        plot_norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plot_cmap = copy.copy(sns.diverging_palette(240, 10, as_cmap=True)) if cmap is None else copy.copy(cmap)
        cbar_label = 'Connectivity Strength'
    else:
        vmin = 0.0 if vmin is None else vmin
        vmax = max(matrix.max(), 1e-6) if vmax is None else vmax
        plot_norm = Normalize(vmin=vmin, vmax=vmax)
        plot_cmap = copy.copy(plt.cm.Greys) if cmap is None else copy.copy(cmap)
        cbar_label = 'Connectivity Strength (unsigned)'

    plot_cmap.set_bad(color='white') # mask exact zeros so they render as white
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)

    fig = plt.figure(figsize=(10, 8))
    if plot_val:
        annot_labels = np.where(matrix == 0, '', np.vectorize(lambda v: f"{v:.3f}")(matrix))
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': cbar_label},
                    square=True, annot=annot_labels, fmt='')   # fmt='' because values are already strings
    else:
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': cbar_label},
                    square=True)

    for b in (boundaries if boundaries is not None else []):
        plt.axhline(b, color='black', linewidth=1.2, zorder=3)
        plt.axvline(b, color='black', linewidth=1.2, zorder=3)

    plt.title(f"{method_name} Connectivity Matrix")
    plt.xlabel("Postsynaptic neuron")
    plt.ylabel("Presynaptic neuron")
    plt.tight_layout()
    if save_dir is None:
        return fig
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    fig.savefig(save_path, format='pdf')
    plt.close(fig)
    return fig
