"""Per-panel figures for the Allen visual-hierarchy recording (session 750749662,
natural_movie_three), saved under figures/allen_750749662/: the spike raster over
a chosen [T_START, T_END] window and the GNN mean-consistent connectivity matrix.

Both panels are grouped by region in REGIONS order (the order prepare_allen.py
writes the units in), so the raster rows and the matrix blocks line up. REGIONS
runs in feedforward order, LGd -> VISp -> HVAs, with the higher-order thalamic
nucleus LP last, so feedforward edges land in the matrix's upper triangle.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / 'allen_750749662'

DATA_DIR = ROOT / 'datasets/allen/prep_750749662_natural_movie_three'
GNN_FILE = ROOT / 'results/allen_750749662/mean_consistent.npy'
STRUCTURE_NAMES_FILE = ROOT / 'datasets/allen/structure_names.csv'

REGIONS = ['LGd', 'VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LP']

REGION_COLORS = {'LGd': '#1c3d73', 'VISp': '#12633a',
                 'VISl': '#8a6a12', 'VISrl': '#8a5312', 'VISal': '#8a3d12',
                 'VISpm': '#7a2233', 'VISam': '#5e1a4a', 'LP': '#571f70'}

T_START = 0.0
T_END = 30.0

GNN_VMIN = -0.001
GNN_VMAX = 0.005

CBAR_LABEL = 'Connection strength'


def load_structure_names(path=STRUCTURE_NAMES_FILE):
    """acronym -> full Allen structure name, for the matrix axis labels.

    A handful of names in the CSV start lower case ('posteromedial visual area'),
    so the first letter is normalised -- otherwise VISpm is the odd one out on an
    axis where every other label is capitalised.
    """
    with open(path, newline='', encoding='utf-8') as f:
        return {row['acronym']: row['name'][:1].upper() + row['name'][1:]
                for row in csv.DictReader(f)}


STRUCTURE_NAMES = load_structure_names()


def load_allen(data_dir=DATA_DIR):
    """Raw spike times (ms) plus the region label per neuron."""
    data_dir = Path(data_dir)
    spkt = np.load(data_dir / 'spkt.npy')                      # spike times (ms)
    spkid = np.load(data_dir / 'spkid.npy').astype(int)        # neuron index per spike
    pop_labels = np.asarray([str(p) for p in np.load(data_dir / 'pop_labels.npy')])
    sim_dur_ms = float(np.load(data_dir / 'sim_duration_ms.npy')[0])

    regions = np.asarray([n.split('_')[0] for n in pop_labels])
    return dict(spk_t=spkt, spk_id=spkid, names=pop_labels, regions=regions,
                n_neurons=len(pop_labels), sim_dur_ms=sim_dur_ms)


def plot_raster(d, t_start=T_START, t_end=T_END, fname='allen_raster.pdf'):
    t_end = d['sim_dur_ms'] / 1000 if t_end is None else min(t_end, d['sim_dur_ms'] / 1000)

    win = (d['spk_t'] >= t_start * 1000) & (d['spk_t'] < t_end * 1000)
    t, i = d['spk_t'][win] / 1000, d['spk_id'][win]

    # one event row per neuron, coloured by region
    positions = [t[i == n] for n in range(d['n_neurons'])]
    colors = [REGION_COLORS.get(r, 'black') for r in d['regions']]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.eventplot(positions, lineoffsets=range(d['n_neurons']), linelengths=0.8,
                 linewidths=0.4, colors=colors)

    # region boundaries and labels down the left edge
    boundaries, cursor = [], 0
    for region in REGIONS:
        count = int(np.sum(d['regions'] == region))
        if count == 0:
            continue
        ax.text(-0.01, cursor + (count - 1) / 2, region, ha='right', va='center',
                fontsize=8, color=REGION_COLORS.get(region, 'black'),
                transform=ax.get_yaxis_transform())
        cursor += count
        boundaries.append(cursor - 0.5)
    for b in boundaries[:-1]:
        ax.axhline(b, color='red', linewidth=0.5)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(d['n_neurons'] - 0.5, -0.5)   # LGd on top, LP at the bottom
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Spike raster  ({t_start:.0f}–{t_end:.0f} s)')
    _save(fig, fname)


def _limits(matrix, vmin=GNN_VMIN, vmax=GNN_VMAX, pct=99.0):
    """Diverging colour limits, from the non-zero entries when not set by hand.

    The consistency mask zeroes most of the matrix, so percentiles taken over
    every entry would collapse onto 0 and flatten the map. TwoSlopeNorm also
    needs vmin < 0 < vmax strictly, hence the clamps.
    """
    nz = matrix[matrix != 0]
    if vmin is None:
        vmin = min(np.percentile(nz, 100 - pct), -1e-9) if nz.size else -1e-9
    if vmax is None:
        vmax = max(np.percentile(nz, pct), 1e-9) if nz.size else 1e-9
    return float(vmin), float(vmax)


def plot_matrix(matrix, regions, vmin, vmax, title, cbar_label, fname):
    matrix = np.asarray(matrix, dtype=float)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': cbar_label, 'ticks': [vmin, 0, vmax]},
                square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for region in REGIONS:
        count = int(np.sum(regions == region))
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(STRUCTURE_NAMES.get(region, region))
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    # full structure names are long, so the postsynaptic axis is angled to fit
    ax.set_xticks(tick_centers)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(tick_centers)
    ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.title(title)
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    _save(fig, fname)


def _save(fig, name, dpi=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'saved -> {OUT_DIR / name}')


if __name__ == '__main__':
    d = load_allen()
    print(f"{d['n_neurons']} units, {d['sim_dur_ms'] / 1000:.0f} s recording")
    plot_raster(d)

    consistent_mean = np.load(GNN_FILE)
    vmin, vmax = _limits(consistent_mean)
    nonzero = int(np.count_nonzero(consistent_mean))
    print(f"mean_consistent data range [{consistent_mean.min():.4g}, "
          f"{consistent_mean.max():.4g}], {nonzero} non-zero edges "
          f"({100 * nonzero / consistent_mean.size:.1f}%)  "
          f"-> vmin={vmin:.4g}, vmax={vmax:.4g}")
    plot_matrix(consistent_mean, d['regions'], vmin=vmin, vmax=vmax,
                title='GNN Connectivity Matrix', cbar_label=CBAR_LABEL,
                fname='gnn.pdf')
