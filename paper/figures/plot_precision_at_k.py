"""Signed precision@K of CircuitGNN and the baselines: of each method's K most confident
edges (largest |weight|), the fraction that are true edges of the right sign.

Each curve stops at the method's last non-zero edge or at K = number of true edges,
whichever comes first. Run after python -m paper.baselines.compare:

    python -m paper.figures.plot_precision_at_k

Writes paper/figures/baselines/precision_at_k.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper import data
from paper.config import DATASETS_DIR, FIGURES_DIR, RESULTS_DIR

OUT_DIR = FIGURES_DIR / 'baselines'
DATASETS = {'OC_1': ('OC_1', 20), 'pdcm': ('PDCM', 200)}   # (title, x-tick step)

# (label, class in baseline_best_per_class.csv, colour)
TECHNIQUES = [
    ('CircuitGNN', 'gnn_mean_consistent', '#2a78d6'),
    ('DI',         'ctm-di',              '#eb6834'),
    ('GLMPP',      'spycon_glmpp',        '#1baf7a'),
    ('GLMCC',      'spycon_glmcc',        '#eda100'),
    ('dSTTC',      'spycon_dsttc',        '#e87ba4'),
    ('sCCG',       'spycon_sccg',         '#008300'),
    ('CoNNECT',    'connect',             '#4a3aa7'),
]


def precision_at_k(true_conn, M):
    """Signed precision of the top-K off-diagonal edges by |weight|, for K = 1..n_called."""
    mask = ~np.eye(len(true_conn), dtype=bool)
    t, m = true_conn[mask], np.nan_to_num(M[mask])
    top = np.argsort(-np.abs(m), kind='stable')[:np.count_nonzero(m)]
    hits = np.sign(m[top]) == np.sign(t[top])
    return np.cumsum(hits) / np.arange(1, len(top) + 1)


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.2))
    for ax, (dataset, (title, step)) in zip(axes, DATASETS.items()):
        true_conn = data.load(dataset)[1]
        k_max = np.count_nonzero(true_conn[~np.eye(len(true_conn), dtype=bool)])
        variants = pd.read_csv(DATASETS_DIR / dataset / 'baseline' / 'baseline_best_per_class.csv',
                               index_col='class')['variant']
        print(f'=== {title}: edges inferred ({k_max} true) ===')
        for label, cls, color in TECHNIQUES:
            path = (RESULTS_DIR / dataset / 'mean_consistent.npy' if cls == 'gnn_mean_consistent'
                    else DATASETS_DIR / dataset / 'baseline' / f'{variants[cls]}.npy')
            p = precision_at_k(true_conn, np.load(path))
            print(f'  {label:<11}{len(p)}')
            p = p[:k_max]
            ax.plot(np.arange(1, len(p) + 1), p, color=color, lw=2, label=label)

        ax.set_xlim(1, k_max)
        ax.set_xticks([*range(step, k_max - step, step), k_max])   # no tick within a step of k_max
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('K (most confident edges)')
        ax.set_title(f'{title} ({k_max} true edges)', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Signed precision@K')
    axes[-1].legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / 'precision_at_k.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
