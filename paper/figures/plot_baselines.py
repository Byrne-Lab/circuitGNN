"""Fig. 4D and 6D: signed F1, precision, recall and NIE of CircuitGNN and the baselines,
one panel per metric.

Run after python -m paper.baselines.compare:

    python -m paper.figures.plot_baselines

Writes paper/figures/baselines/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.config import DATASETS_DIR, FIGURES_DIR

OUT_DIR = FIGURES_DIR / 'baselines'

DATASETS = ['pdcm', 'OC_1']
TITLES = {'pdcm': 'PDCM', 'OC_1': 'OC_1'}

TECHNIQUES = [
    ('CircuitGNN', 'gnn_mean_consistent'),
    ('DI',         'ctm-di'),
    ('GLMPP',      'spycon_glmpp'),
    ('GLMCC',      'spycon_glmcc'),
    ('dSTTC',      'spycon_dsttc'),
    ('sCCG',       'spycon_sccg'),
    ('CoNNECT',    'connect'),
]

# (label, column, y-range). NIE (normalized inference error) is lower-is-better,
# unlike the other three, and every method lands in ~0.65-1.0, so its axis is zoomed.
METRICS = [
    ('F1',        'f1_signed',   (0.0, 1.0)),
    ('Precision', 'prec_signed', (0.0, 1.0)),
    ('Recall',    'rec_signed',  (0.0, 1.0)),
    ('NIE',       'nie',         (0.6, 1.0)),
]

GNN_COLOR = '#2a78d6'
BASELINE_COLOR = '#a6a6a6'

BAR_WIDTH = 0.75
ANNOTATE = True


def load_metrics(dataset):
    """(n_techniques, 4) array of signed F1 / precision / recall / NIE, NaN if missing."""
    csv_path = DATASETS_DIR / dataset / 'baseline' / 'baseline_best_per_class.csv'
    df = pd.read_csv(csv_path).set_index('class')

    vals = np.full((len(TECHNIQUES), len(METRICS)), np.nan)
    for i, (label, cls) in enumerate(TECHNIQUES):
        if cls not in df.index:
            print(f'  [missing] {dataset}: no "{cls}" row for {label}')
            continue
        row = df.loc[cls]
        vals[i] = [row[col] for _, col, _ in METRICS]
    return vals


def plot_dataset(dataset, vals):
    x = np.arange(len(TECHNIQUES))
    colors = [GNN_COLOR if label == 'CircuitGNN' else BASELINE_COLOR
              for label, _ in TECHNIQUES]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 3.6))
    for j, (ax, (label, col, (lo, hi))) in enumerate(zip(axes, METRICS)):
        # Bars start at the axis floor rather than 0, so a zoomed axis doesn't leave
        # clipped-off geometry in the PDF (it reappears when the mask is released).
        bars = ax.bar(x, vals[:, j] - lo, BAR_WIDTH, bottom=lo, color=colors)
        if ANNOTATE:
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals[:, j]],
                         fontsize=6.5, padding=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([t for t, _ in TECHNIQUES], fontsize=8,
                           rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylim(lo, hi + 0.06 * (hi - lo))  # headroom for the bar labels
        ax.set_yticks(np.linspace(lo, hi, 5))
        ax.set_title(f'{label} (lower is better)' if col == 'nie' else label, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Score')
    fig.suptitle(f'{TITLES[dataset]}: signed connectivity metrics', fontsize=11)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'baseline_bars_{dataset}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')


def main():
    for dataset in DATASETS:
        print(f'=== {TITLES[dataset]} ===')
        vals = load_metrics(dataset)
        print(pd.DataFrame(vals,
                           index=[label for label, _ in TECHNIQUES],
                           columns=[label for label, _, _ in METRICS]).round(3).to_string())
        plot_dataset(dataset, vals)


if __name__ == '__main__':
    main()