from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / 'baselines'

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

# NIE (normalized inference error) is lower-is-better, unlike the other three.
METRICS = [
    ('F1',                  'f1_signed',   '#1a1a1a'),
    ('Precision',           'prec_signed', '#595959'),
    ('Recall',              'rec_signed',  '#8c8c8c'),
    ('NIE',                 'nie',         '#d9d9d9'),
]

BAR_WIDTH = 0.20
ANNOTATE = True


def load_metrics(dataset):
    """(n_techniques, 4) array of signed F1 / precision / recall / NIE, NaN if missing."""
    csv_path = REPO_ROOT / 'datasets' / dataset / 'baseline' / 'baseline_best_per_class.csv'
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
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, ((label, col, color), off) in enumerate(zip(METRICS, offsets)):
        bars = ax.bar(x + off, vals[:, j], BAR_WIDTH, color=color, label=label,
                      edgecolor='#1a1a1a' if col == 'nie' else 'none',
                      linewidth=0.6 if col == 'nie' else 0)
        if ANNOTATE:
            ax.bar_label(bars, fmt='%.2f', fontsize=5.5, padding=1.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in TECHNIQUES], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'{TITLES[dataset]}: signed connectivity metrics')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper right')
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'baseline_bars_{dataset}.pdf'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    for dataset in DATASETS:
        print(f'=== {TITLES[dataset]} ===')
        vals = load_metrics(dataset)
        print(pd.DataFrame(vals,
                           index=[label for label, _ in TECHNIQUES],
                           columns=[label for label, _, _ in METRICS]).round(3).to_string())
        plot_dataset(dataset, vals)
