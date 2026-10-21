"""Fig. 6A-C: the PDCM network. Raster (0-10 s), mean firing rate, E/I firing rate,
degree distribution, and the ground-truth, CircuitGNN and baseline connectivity heatmaps.

Run after paper.evaluate pdcm and the baselines:

    python -m paper.figures.plot_pdcm

Writes paper/figures/pdcm/.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paper.config import DATASETS_DIR, FIGURES_DIR, RESULTS_DIR
from paper.figures.plotting import PDCM_POPS, plot_pdcm_matrix

POP_COLORS = {
    'L2e': '#1c3d73', 'L2i': '#4b76ba', 'L4e': '#12633a', 'L4i': '#3f9a6e',
    'L5e': '#8a3d12', 'L5i': '#b9743c', 'L6e': '#571f70', 'L6i': '#8a5cb0',
}
E_COLOR, I_COLOR = '#8c1f14', '#17506f'
DATA_DIR = DATASETS_DIR / 'pdcm'
OUT_DIR = FIGURES_DIR / 'pdcm'

# (matrix file, title, output, vmin, vmax); benchmarks are the best settings per method
MATRICES = [
    (RESULTS_DIR / 'pdcm' / 'mean_consistent.npy', 'GNN Connectivity Matrix', 'gnn.pdf', -0.010, 0.005),
    (DATA_DIR / 'baseline' / 'spycon_glmpp_default.npy', 'GLMPP Connectivity Matrix', 'glmpp.pdf', -2, 6),
    (DATA_DIR / 'baseline' / 'spycon_glmcc_syn_delay1ms_beta4000_tau4ms-4ms.npy', 'GLMCC Connectivity Matrix', 'glmcc.pdf', -0.1, 0.4),
    (DATA_DIR / 'baseline' / 'spycon_dsttc_delta_t30ms.npy', 'DSTTC Connectivity Matrix', 'dsttc.pdf', -0.1, 0.4),
    (DATA_DIR / 'baseline' / 'spycon_sccg_binsize2ms_hf0.8.npy', 'SCCG Connectivity Matrix', 'sccg.pdf', -0.1, 0.3),
    (DATA_DIR / 'baseline' / 'CTM_DI.npy', 'CTM-DI Connectivity Matrix', 'ctm_di.pdf', -0.025, 0.1),
    (DATA_DIR / 'baseline' / 'CoNNECT.npy', 'Connect Connectivity Matrix', 'connect.pdf', -0.025, 0.1),
]


def is_exc(pop):
    return pop.endswith('e')


def load_pdcm():
    """Load PDCM arrays. Neurons are already stored grouped as L2e, L2i, ... L6i,
    so spkid/weights need no reordering; we just record each pop's index block."""
    labels = np.asarray([str(p) for p in np.load(DATA_DIR / 'pop_labels.npy')])
    weights = np.load(DATA_DIR / 'weight_matrix.npy')          # (N, N) [pre, post]
    spkt = np.load(DATA_DIR / 'spkt.npy')                      # spike times (ms)
    spkid = np.load(DATA_DIR / 'spkid.npy').astype(int)        # neuron index per spike
    sim_dur_ms = float(np.load(DATA_DIR / 'sim_duration_ms.npy')[0])

    ranges, cursor = {}, 0
    for p in PDCM_POPS:
        n = int(np.sum(labels == p))
        ranges[p] = (cursor, cursor + n)
        cursor += n

    return dict(labels=labels, weights=weights, adj=(weights != 0).astype(int),
                spk_t=spkt, spk_id=spkid, ranges=ranges,
                n_neurons=len(labels), sim_dur_ms=sim_dur_ms)


def plot_raster(d, t_start=0.0, t_end=10.0):
    win = (d['spk_t'] >= t_start * 1000) & (d['spk_t'] < t_end * 1000)
    t, i = d['spk_t'][win] / 1000, d['spk_id'][win]

    # one event row per neuron so the dashes stay vector paths (~4k spikes total)
    positions, offsets, colors = [], [], []
    for p in PDCM_POPS:
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
    ax.legend(handles=[Patch(color=POP_COLORS[p], label=p) for p in PDCM_POPS],
              fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              framealpha=0.9, title='Population')
    _save(fig, 'pdcm_raster.pdf')


def plot_rates(d):
    dur_s = d['sim_dur_ms'] / 1000
    rates = [int(np.sum((d['spk_id'] >= s) & (d['spk_id'] < e))) / ((e - s) * dur_s)
             for p in PDCM_POPS for s, e in [d['ranges'][p]]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(PDCM_POPS, rates, color=[POP_COLORS[p] for p in PDCM_POPS])
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


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {OUT_DIR / name}')


def main():
    d = load_pdcm()
    plot_raster(d)
    plot_rates(d)
    plot_ei_rate(d)
    plot_degree(d)

    # ground truth, normalized by its largest magnitude (originally -6000 to 4000 pA)
    plot_pdcm_matrix(d['weights'] / np.max(np.abs(d['weights'])), d['labels'],
                     OUT_DIR / 'ground_truth.pdf', 'True Connectivity', vmin=-1, vmax=0.5)
    for path, title, fname, vmin, vmax in MATRICES:
        plot_pdcm_matrix(np.load(path), d['labels'], OUT_DIR / fname, title, vmin=vmin, vmax=vmax)


if __name__ == '__main__':
    main()
