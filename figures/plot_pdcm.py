"""Per-panel figures for the base PDCM dataset, saved under figures/pdcm/:
raster (0-10 s), mean firing rate, E/I firing rate, degree distribution, and
the ground-truth / mean-consistent connectivity heatmaps.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

REAL_POPS = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
LAYERS = [2, 4, 5, 6]
POP_COLORS = {
    'L2e': '#1c3d73', 'L2i': '#4b76ba', 'L4e': '#12633a', 'L4i': '#3f9a6e',
    'L5e': '#8a3d12', 'L5i': '#b9743c', 'L6e': '#571f70', 'L6i': '#8a5cb0',
}
E_COLOR, I_COLOR = '#8c1f14', '#17506f'
OUT_DIR = Path(__file__).resolve().parent / 'pdcm'


def is_exc(pop):
    return pop.endswith('e')


def load_pdcm(data_dir):
    """Load PDCM arrays. Neurons are already stored grouped as L2e, L2i, ... L6i,
    so spkid/weights need no reordering; we just record each pop's index block."""
    data_dir = Path(data_dir)
    labels = np.asarray([str(p) for p in np.load(data_dir / 'pop_labels.npy')])
    weights = np.load(data_dir / 'weight_matrix.npy')          # (N, N) [pre, post]
    spkt = np.load(data_dir / 'spkt.npy')                      # spike times (ms)
    spkid = np.load(data_dir / 'spkid.npy').astype(int)        # neuron index per spike
    sim_dur_ms = float(np.load(data_dir / 'sim_duration_ms.npy')[0])

    ranges, cursor = {}, 0
    for p in REAL_POPS:
        n = int(np.sum(labels == p))
        ranges[p] = (cursor, cursor + n)
        cursor += n

    return dict(labels=labels, adj=(weights != 0).astype(int),
                spk_t=spkt, spk_id=spkid, ranges=ranges,
                n_neurons=len(labels), sim_dur_ms=sim_dur_ms)


def plot_raster(d, t_start=0.0, t_end=10.0):
    win = (d['spk_t'] >= t_start * 1000) & (d['spk_t'] < t_end * 1000)
    t, i = d['spk_t'][win] / 1000, d['spk_id'][win]

    # one event row per neuron so the dashes stay vector paths (~4k spikes total)
    positions, offsets, colors = [], [], []
    for p in REAL_POPS:
        s, e = d['ranges'][p]
        for n in range(s, e):
            positions.append(t[i == n])
            offsets.append(n)
            colors.append(POP_COLORS[p])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.eventplot(positions, lineoffsets=offsets, linelengths=0.8,
                 linewidths=0.8, colors=colors)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(d['n_neurons'] - 0.5, -0.5)   # L2 on top, L6 at the bottom
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_title(f'PDCM spike raster  ({t_start:.0f}–{t_end:.0f} s)')
    ax.legend(handles=[Patch(color=POP_COLORS[p], label=p) for p in REAL_POPS],
              fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              framealpha=0.9, title='Population')
    _save(fig, 'pdcm_raster.pdf')


def plot_rates(d):
    dur_s = d['sim_dur_ms'] / 1000
    rates = [int(np.sum((d['spk_id'] >= s) & (d['spk_id'] < e))) / ((e - s) * dur_s)
             for p in REAL_POPS for s, e in [d['ranges'][p]]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(REAL_POPS, rates, color=[POP_COLORS[p] for p in REAL_POPS])
    ax.set_xlabel('Firing rate (Hz)')
    ax.set_title('Mean firing rate per population')
    ax.invert_yaxis()
    _save(fig, 'pdcm_rates.pdf')


def plot_ei_rate(d, bin_ms=1000.0):
    exc = np.array([is_exc(p) for p in d['labels']])
    exc_spike = exc[d['spk_id']]
    bins = np.arange(0, d['sim_dur_ms'] + bin_ms, bin_ms)

    fig, ax = plt.subplots(figsize=(11, 5))
    for mask, n, color, name in [(exc_spike, exc.sum(), E_COLOR, 'Excitatory'),
                                 (~exc_spike, (~exc).sum(), I_COLOR, 'Inhibitory')]:
        counts, _ = np.histogram(d['spk_t'][mask], bins=bins)
        ax.plot(bins[:-1] / 1000, counts / (n * bin_ms / 1000), color=color, lw=1.2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population rate (Hz/neuron)')
    ax.set_title(f'E/I population firing rate  ({bin_ms:.0f} ms bins)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.5)
    _save(fig, 'pdcm_ei_rate.pdf')


def plot_degree(d):
    adj = d['adj']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(adj.sum(axis=0), bins=20, alpha=0.6, label='In-degree', color='steelblue')
    ax.hist(adj.sum(axis=1), bins=20, alpha=0.6, label='Out-degree', color='tomato')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Degree distribution')
    ax.legend(fontsize=9)
    _save(fig, 'pdcm_degree.pdf')


def plot_matrix(matrix, pop_labels, vmin, vmax, title, cbar_label, fname):
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': cbar_label, 'ticks': [vmin, 0, vmax]},
                square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for p in REAL_POPS:
        count = int(np.sum(pop_labels == p))
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(p)
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

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
    data_dir = 'datasets/pdcm'
    d = load_pdcm(data_dir)
    plot_raster(d)
    plot_rates(d)
    plot_ei_rate(d)
    plot_degree(d)

    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    true_conn = np.load(f'{data_dir}/weight_matrix.npy')
    true_conn = true_conn / np.max(np.abs(true_conn)) # normalize
    # originally in pA: -6000, 4000
    plot_matrix(true_conn, pop_labels, vmin=-1, vmax=0.5, title='True Connectivity', cbar_label='Connection strength', fname='ground_truth.pdf')

    consistent_mean = np.load('results/pdcm/mean_consistent.npy')
    plot_matrix(consistent_mean, pop_labels, vmin=-0.010, vmax=0.005, title='GNN Connectivity Matrix', cbar_label='Connection strength', fname='gnn.pdf')

    # benchmarks
    glmpp = np.load('datasets/pdcm/baseline/spycon_glmpp_default.npy')
    plot_matrix(glmpp, pop_labels, vmin=-2, vmax=6, title='GLMPP Connectivity Matrix', cbar_label='Connection strength', fname='glmpp.pdf')

    dsttc = np.load('datasets/pdcm/baseline/spycon_dsttc_delta_t30ms.npy')
    plot_matrix(dsttc, pop_labels, vmin=-0.1, vmax=0.4, title='DSTTC Connectivity Matrix', cbar_label='Connection strength', fname='dsttc.pdf')

    sccg = np.load('datasets/pdcm/baseline/spycon_sccg_binsize2ms_hf0.8.npy')
    plot_matrix(sccg, pop_labels, vmin=-0.1, vmax=0.3, title='SCCG Connectivity Matrix', cbar_label='Connection strength', fname='sccg.pdf')

    decision_tree = np.load('datasets/pdcm/baseline/decision_tree_mcc_threshold_signed.npy')  
    plot_matrix(decision_tree, pop_labels, vmin=-0.1, vmax=0.4, title='Decision Tree Connectivity Matrix', cbar_label='Connection strength', fname='decision_tree.pdf')
