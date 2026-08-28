from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / 'oc_1'

NAMES_FILE = ROOT / 'datasets/OC_1/spikes.head'

PANELS = [
    ('datasets/OC_1/true_connectivity_0.20.npy', 'True Connectivity',        'ground_truth.pdf',  -5,   2),
    ('results/OC_1/mean_consistent.npy',         'GNN Connectivity Matrix',  'gnn.pdf',           -0.015, 0.015),

    # benchmarks
    ('datasets/OC_1/baseline/spycon_glmpp_default.npy',              'GLMPP Connectivity Matrix',         'glmpp.pdf',         -3, 12),
    ('datasets/OC_1/baseline/spycon_dsttc_delta_t30ms.npy',              'DSTTC Connectivity Matrix',         'dsttc.pdf',         -0.1, 0.5),
    ('datasets/OC_1/baseline/CTM_DI.npy',                            'CTM-DI Connectivity Matrix',        'ctm_di.pdf',        -0.1, 0.6),
]

CBAR_LABEL = 'Connection strength'
ANNOTATE = False    


def load_neuron_names():
    with open(NAMES_FILE) as f:
        return [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]


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


if __name__ == '__main__':
    neuron_names = load_neuron_names()

    for rel_path, title, fname, vmin, vmax in PANELS:
        path = ROOT / rel_path
        if not path.exists():
            print(f'[skip] {rel_path} not found')
            continue

        matrix = np.load(path)
        scale = 'auto' if vmin is None or vmax is None else f'vmin={vmin}, vmax={vmax}'
        print(f'{fname:<20} data range [{matrix.min():.4g}, {matrix.max():.4g}]  ({scale})')
        plot_matrix(matrix, neuron_names, vmin, vmax, title, fname)
