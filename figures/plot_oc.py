import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from circuit_gnn.data import load_control_network, load_OC_1_network, load_OC_2_network

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / 'oc'

# (condition label, dataset dir, results dir, output prefix, spike loader)
CONDITIONS = [
    ('Control', 'datasets/control', 'results/control', 'control', load_control_network),
    ('OCx1',    'datasets/OC_1',    'results/OC_1',    'oc_1',    load_OC_1_network),
    ('OCx2',    'datasets/OC_2',    'results/OC_2',    'oc_2',    load_OC_2_network),
]

TRUTH_FILE = 'true_connectivity_0.20.npy'
GNN_FILE = 'mean_consistent.npy'

TRUTH_VLIM = (-5, 2)
GNN_VLIM = (-0.015, 0.015)

CBAR_LABEL = 'Connection strength'
ANNOTATE = False

# raster settings: native sample period is 0.5 ms for all three datasets
NATIVE_MS = 0.5
BIN_MS = 1.0
T_START, T_END = 0, 240


def load_neuron_names(dataset_dir):
    with open(ROOT / dataset_dir / 'spikes.head') as f:
        return [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]


def load_raster(loader):
    """Spike times (seconds) per neuron over the [T_START, T_END] window."""
    downsample_factor = max(1, round(BIN_MS / NATIVE_MS))
    spikes, _, neuron_names = loader(downsample_factor=downsample_factor)
    bin_size_s = downsample_factor * NATIVE_MS / 1000.0
    n_neurons, n_bins = spikes.shape

    lo = 0 if T_START is None else max(0, int(T_START / bin_size_s))
    hi = n_bins if T_END is None else min(n_bins, int(np.ceil(T_END / bin_size_s)))
    spike_times_s = [(lo + np.where(spikes[i, lo:hi] > 0)[0]) * bin_size_s
                     for i in range(n_neurons)]
    return spike_times_s, neuron_names, (lo * bin_size_s, hi * bin_size_s)


def draw_raster(ax, spike_times_s, neuron_names, xlim, title=None):
    n_neurons = len(spike_times_s)
    ax.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    if n_neurons <= 40:
        ax.set_yticks(range(n_neurons))
        ax.set_yticklabels(neuron_names)
        ax.invert_yaxis()
        ax.set_ylabel('Neuron')
    else:
        ax.set_ylabel('Neuron index')
    ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel('Time (s)')
    if title:
        ax.set_title(title)


def draw_matrix(ax, matrix, neuron_names, vmin, vmax, title=None, cbar=True):
    """Heatmap of a connectivity matrix; returns the axes' mappable."""
    matrix = np.array(matrix, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap.set_bad(color='white')                       # exact zeros render white
    masked = np.ma.masked_where(matrix == 0, matrix)

    kwargs = {}
    if ANNOTATE:
        kwargs = {'annot': np.where(matrix == 0, '', np.vectorize(lambda v: f'{v:.3f}')(matrix)),
                  'fmt': '', 'annot_kws': {'size': 6}}
    sns.heatmap(masked, ax=ax, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm, cbar=cbar,
                cbar_kws={'label': CBAR_LABEL, 'ticks': [vmin, 0, vmax]},
                square=True, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel('Postsynaptic')
    ax.set_ylabel('Presynaptic')
    return ax.collections[0]


def save_matrix(matrix, neuron_names, vmin, vmax, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_matrix(ax, matrix, neuron_names, vmin, vmax, title=title)
    fig.tight_layout()
    save(fig, fname)


def save_raster(spike_times_s, neuron_names, xlim, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_raster(ax, spike_times_s, neuron_names, xlim, title=title)
    fig.tight_layout()
    save(fig, fname)


def save(fig, fname):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {OUT_DIR / fname}')


def plot_grid(panels):
    """3x3 summary: rows = conditions, columns = raster / ground truth / GNN."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))

    for row, p in enumerate(panels):
        draw_raster(axes[row][0], *p['raster'])
        axes[row][0].set_title(f"{p['label']} spike raster")

        draw_matrix(axes[row][1], p['truth'], p['names'], *TRUTH_VLIM,
                    title=f"{p['label']} true connectivity", cbar=True)
        draw_matrix(axes[row][2], p['gnn'], p['names'], *GNN_VLIM,
                    title=f"{p['label']} GNN connectivity", cbar=True)

    fig.tight_layout()
    save(fig, 'oc_summary_grid.pdf')


if __name__ == '__main__':
    panels = []

    for label, dataset_dir, results_dir, prefix, loader in CONDITIONS:
        neuron_names = load_neuron_names(dataset_dir)

        spike_times_s, raster_names, xlim = load_raster(loader)
        save_raster(spike_times_s, raster_names, xlim,
                    f'{label} Spike Raster', f'{prefix}_raster.pdf')

        panel = {'label': label, 'names': neuron_names,
                 'raster': (spike_times_s, raster_names, xlim)}

        for key, path, title, fname, vlim in [
            ('truth', ROOT / dataset_dir / TRUTH_FILE, f'{label} True Connectivity',
             f'{prefix}_ground_truth.pdf', TRUTH_VLIM),
            ('gnn', ROOT / results_dir / GNN_FILE, f'{label} GNN Connectivity Matrix',
             f'{prefix}_gnn.pdf', GNN_VLIM),
        ]:
            if not path.exists():
                print(f'[skip] {path.relative_to(ROOT)} not found')
                continue

            matrix = np.load(path)
            vmin, vmax = vlim
            print(f'{fname:<28} data range [{matrix.min():.4g}, {matrix.max():.4g}]  '
                  f'(vmin={vmin:.4g}, vmax={vmax:.4g})')
            save_matrix(matrix, neuron_names, vmin, vmax, title, fname)
            panel[key] = matrix

        if 'truth' in panel and 'gnn' in panel:
            panels.append(panel)
        else:
            print(f'[skip] {label} missing from the 3x3 grid')

    if panels:
        plot_grid(panels)
