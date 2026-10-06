"""Fig. 4B-C: ground truth, CircuitGNN and baseline connectivity of the Aplysia circuit.

Run after paper.evaluate OC_1 and the baselines:

    python -m paper.figures.plot_oc_1

Writes paper/figures/oc_1/.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from paper.config import DATASETS_DIR, FIGURES_DIR, RESULTS_DIR
from paper.data import neuron_names_from_head

DATA_DIR = DATASETS_DIR / 'OC_1'
OUT_DIR = FIGURES_DIR / 'oc_1'

# (matrix file, title, output, vmin, vmax); benchmarks are the best settings per method
PANELS = [
    (DATA_DIR / 'true_connectivity_0.20.npy', 'True Connectivity', 'ground_truth.pdf', -5, 2),
    (RESULTS_DIR / 'OC_1' / 'mean_consistent.npy', 'GNN Connectivity Matrix', 'gnn.pdf', -0.015, 0.015),
    (DATA_DIR / 'baseline' / 'spycon_glmpp_default.npy', 'GLMPP Connectivity Matrix', 'glmpp.pdf', -3, 12),
    (DATA_DIR / 'baseline' / 'spycon_dsttc_delta_t30ms.npy', 'DSTTC Connectivity Matrix', 'dsttc.pdf', -0.1, 0.5),
    (DATA_DIR / 'baseline' / 'CTM_DI.npy', 'CTM-DI Connectivity Matrix', 'ctm_di.pdf', -0.1, 0.6),
]

CBAR_LABEL = 'Connection strength'
ANNOTATE = False


def plot_matrix(matrix, neuron_names, vmin, vmax, title, fname):
    matrix = np.array(matrix, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='white')                       # exact zeros render white
    masked = np.ma.masked_where(matrix == 0, matrix)

    fig = plt.figure(figsize=(10, 8))
    kwargs = {}
    if ANNOTATE:
        kwargs = {'annot': np.where(matrix == 0, '', np.vectorize(lambda v: f'{v:.3f}')(matrix)),
                  'fmt': '', 'annot_kws': {'size': 6}}
    sns.heatmap(masked, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                cbar_kws={'label': CBAR_LABEL, 'ticks': [vmin, 0, vmax]},
                square=True, **kwargs)

    plt.title(title)
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {OUT_DIR / fname}')


def main():
    neuron_names = neuron_names_from_head(DATA_DIR / 'spikes.head')
    for path, title, fname, vmin, vmax in PANELS:
        if not path.exists():
            print(f'[skip] {path} not found')
            continue
        matrix = np.load(path)
        print(f'{fname:<20} data range [{np.nanmin(matrix):.4g}, {np.nanmax(matrix):.4g}]')
        plot_matrix(matrix, neuron_names, vmin, vmax, title, fname)


if __name__ == '__main__':
    main()
