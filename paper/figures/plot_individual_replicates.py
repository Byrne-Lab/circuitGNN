"""Individual GNN replicates: three random runs each of Network C (arbitrary), the Aplysia
circuit (OC_1) and PDCM, one dataset per row under a shared colour bar.

Run after paper.train for the three datasets:

    python -m paper.figures.plot_individual_training_examples

Writes paper/figures/individual_training_examples.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from circuit_gnn import load_runs
from paper.config import DATASETS_DIR, FIGURES_DIR, RESULTS_DIR
from paper.data import ARBITRARY_NUM_NEURONS, neuron_names_from_head
from paper.figures.plotting import pdcm_blocks

SEED = 2
N_PICK = 3
CBAR_LABEL = 'Connection strength'

# (dataset, row title, vmin, vmax)
ROWS = [
    ('arbitrary', 'Network C', -0.02, 0.02),
    ('OC_1', 'Aplysia', -0.02, 0.02),
    ('pdcm', 'PDCM', -0.01, 0.01),
]


def main():
    rng = np.random.default_rng(SEED)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='white')                       # exact zeros (and the diagonal) render white
    names = {'arbitrary': np.arange(ARBITRARY_NUM_NEURONS),
             'OC_1': neuron_names_from_head(DATASETS_DIR / 'OC_1' / 'spikes.head'),
             'pdcm': False}
    bounds, centers, pops = pdcm_blocks(np.load(DATASETS_DIR / 'pdcm' / 'pop_labels.npy').astype(str))

    fig, axes = plt.subplots(len(ROWS), N_PICK, figsize=(5 * N_PICK, 5 * len(ROWS)),
                             layout='constrained')
    for row, (name, title, vmin, vmax) in zip(axes, ROWS):
        matrices, _, run_ids = load_runs(RESULTS_DIR / name)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        for ax, i in zip(row, sorted(rng.choice(len(run_ids), N_PICK, replace=False))):
            m = np.asarray(matrices[i], dtype=float).copy()
            np.fill_diagonal(m, 0.0)
            sns.heatmap(np.ma.masked_equal(m, 0), ax=ax, cmap=cmap, norm=norm, square=True,
                        cbar=False, xticklabels=names[name], yticklabels=names[name])
            if name == 'pdcm':
                for b in bounds:
                    ax.axhline(b, color='k', lw=0.5)
                    ax.axvline(b, color='k', lw=0.5)
                ax.set_xticks(centers, pops, fontsize=7)
                ax.set_yticks(centers, pops, fontsize=7, rotation=0)
            ax.set_title(f'{title}, run {run_ids[i]}', fontsize=10)
            ax.set_xlabel('Postsynaptic')
        row[0].set_ylabel('Presynaptic')
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=row, shrink=0.8,
                     label=CBAR_LABEL, ticks=[vmin, 0, vmax])

    out = FIGURES_DIR / 'individual_replicates.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
