"""Loaders for the paper's datasets (see paper/README.md for where to put the files).

Every loader returns ``(binned_spikes, true_conn, neuron_names)``: spike counts (N, T)
at the requested bin, signed ground-truth connectivity [pre, post] (positive =
excitatory, negative = inhibitory), and a label per neuron.
``downsample_factor`` is the bin width in native samples, or 'auto' to choose it with
circuit_gnn.binning.optimal_bin_ms. Loaders also write raster_plot.pdf and
ground_truth.pdf into the dataset's folder.
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from circuit_gnn.binning import rebin, resolve_downsample
from paper.config import DATASETS, DATASETS_DIR
from paper.figures.plotting import plot_pdcm_matrix

SPIKE_THRESHOLD_MV = -10    # a spike is a crossing of -10 mV ...
SNNAP_REARM = 15            # ... and further crossings within 15 samples (7.5 ms) are the same spike

# Network C (arbitrary): 11 neurons, 1 ms samples
ARBITRARY_NUM_NEURONS = 11
ARBITRARY_TIME_MAX_S = 300      # limit to 300 s of simulation
ARBITRARY_REARM = 8             # re-arm period in samples (8 ms)
ARBITRARY_RASTER_MAX_S = 40     # show a readable window, not the full 300 s
ARBITRARY_NORMALIZER = 4.00     # largest |conductance|; ground truth is normalized by it

# Network C synaptic conductances {(pre, post): conductance}; neuron k here is N(k+1)
ARBITRARY_EDGES = {
    (0, 3):   0.75,   # N1 -> N4
    (2, 9):   1.25,   # N3 -> N10
    (3, 10):  0.70,   # N4 -> N11
    (4, 9):   1.25,   # N5 -> N10
    (5, 0):   0.50,   # N6 -> N1
    (5, 1):   1.25,   # N6 -> N2
    (5, 2):   0.50,   # N6 -> N3
    (5, 4):  -4.00,   # N6 -> N5
    (7, 1):  -4.00,   # N8 -> N2
    (8, 1):   1.25,   # N9 -> N2
}
# Each modification changes exactly one edge.
# 1, weakened: N3 -> N10 1.25 -> 0.50. Matched control N5 -> N10 (same target, same baseline
#    weight), so the learned ratio A[N3,N10] / A[N5,N10] goes from ~1 to a true 0.4.
MODIFICATION1_EDGES = {**ARBITRARY_EDGES, (2, 9): 0.50}
# 3, strengthened: N6 -> N2 1.25 -> 3.125 (2.5x). Matched control N9 -> N2; N6's other
#    outputs are unchanged, separating a change at this synapse from a cell-wide change.
MODIFICATION3_EDGES = {**ARBITRARY_EDGES, (5, 1): 3.125}


def load(name, downsample_factor=None):
    """Load a dataset of paper/config.py's registry; downsample_factor defaults to its own."""
    spec = DATASETS[name]
    factor = spec.downsample if downsample_factor is None else downsample_factor
    return globals()[spec.loader](downsample_factor=factor, **spec.kwargs)


def neuron_names_from_head(path):
    """Neuron labels from a SNNAP spikes.head file (lines like 'V[B20....]')."""
    with open(path) as f:
        return [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]


def spike_indices(volts, thr=SPIKE_THRESHOLD_MV, rearm=SNNAP_REARM):
    """Per neuron, the sample index of each spike in a (N, T) voltage array: the first
    sample above ``thr`` that is more than ``rearm`` samples after the previous one."""
    out = []
    for mv in volts:
        sp = np.where(mv > thr)[0]
        if sp.size > 0:
            sp = sp[np.concatenate(([True], np.diff(sp) > rearm))]
        out.append(sp)
    return out


def read_arbitrary_voltages(name, t_max):
    """``(times, volts)`` of a Network C recording, read up to the first sample at t >= t_max."""
    rows = []
    with io.open(DATASETS_DIR / name / 'spikes.out', mode="r", encoding="utf-8") as f:
        for line in f:
            rows.append([float(i) for i in line.split()])
            if rows[-1][0] >= t_max:
                break
    rows = np.asarray(rows)
    return rows[:, 0], rows[:, 1:ARBITRARY_NUM_NEURONS + 1].T


def arbitrary_ground_truth(edges):
    true_conn = np.zeros((ARBITRARY_NUM_NEURONS, ARBITRARY_NUM_NEURONS))
    for (pre, post), conductance in edges.items():
        true_conn[pre, post] = conductance
    return true_conn / ARBITRARY_NORMALIZER


def _binary_spikes(volts, rearm):
    spikes = np.zeros(volts.shape, dtype=int)
    for i, sp in enumerate(spike_indices(volts, rearm=rearm)):
        spikes[i, sp] = 1
    return spikes


def _rebin_native(spikes, downsample_factor, native_ms):
    """Rebin native-resolution spikes, resolving 'auto' via signed-edge stability."""
    factor = resolve_downsample(downsample_factor, spikes, native_ms)
    binned = rebin(spikes, factor)
    print(binned.shape)
    print("Binned spikes possibilities:", np.unique(binned))
    return binned, factor


def _bin_spike_times(spkt, spkid, n_neurons, duration_ms, bin_ms):
    """Count spikes (times in ms, neuron ids) into (n_neurons, duration_ms // bin_ms) bins."""
    n_bins = int(duration_ms // bin_ms)
    binned = np.zeros((n_neurons, n_bins), dtype=np.int32)
    t_idx = np.floor(spkt / bin_ms).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned, (spkid[valid], t_idx[valid]), 1)
    return binned


def _plot_raster(binned, bin_s, path, neuron_names=None, max_s=None):
    spike_times = [np.where(row > 0)[0] * bin_s for row in binned]
    if max_s is not None:
        spike_times = [t[t <= max_s] for t in spike_times]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    if neuron_names is not None:
        plt.yticks(range(len(neuron_names)), neuron_names)
        plt.gca().invert_yaxis()
        plt.ylabel('Neuron')
    else:
        plt.ylabel('Neuron Index')
    if max_s is not None:
        plt.xlim(0, max_s)
    plt.xlabel('Time (s)')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig(path, format='pdf')
    plt.close()


def _plot_ground_truth(true_conn, neuron_names, path, title, vmin, vmax, cbar_label,
                       cbar_ticks=None, annot_kws=None):
    cbar_kws = {'label': cbar_label}
    if cbar_ticks is not None:
        cbar_kws['ticks'] = cbar_ticks
    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=sns.diverging_palette(240, 10, as_cmap=True),
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                annot=[[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn],
                fmt='', annot_kws=annot_kws, cbar_kws=cbar_kws, square=True)
    plt.title(title)
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig(path, format='pdf')
    plt.close()


def _load_arbitrary(name, edges, title, downsample_factor):
    """Network C and its modifications: SNNAP voltages, 1 ms samples, first 300 s."""
    _, volts = read_arbitrary_voltages(name, ARBITRARY_TIME_MAX_S)
    binned, factor = _rebin_native(_binary_spikes(volts, ARBITRARY_REARM), downsample_factor, 1.0)
    true_conn = arbitrary_ground_truth(edges)
    neuron_names = np.arange(ARBITRARY_NUM_NEURONS)

    _plot_raster(binned, 1e-3 * factor, DATASETS_DIR / name / 'raster_plot.pdf',
                 neuron_names, max_s=ARBITRARY_RASTER_MAX_S)
    _plot_ground_truth(true_conn, neuron_names, DATASETS_DIR / name / 'ground_truth.pdf', title,
                       vmin=-1, vmax=0.4, cbar_label='Connection Strength',
                       cbar_ticks=[-1.0, -0.5, 0.0, 0.2, 0.4],
                       annot_kws={'size': 5, 'color': 'black'})
    return binned, true_conn, neuron_names


def load_arbitrary_network(downsample_factor='auto'):
    """Network C: 11 neurons, normalized SNNAP conductances as ground truth."""
    return _load_arbitrary('arbitrary', ARBITRARY_EDGES, 'True Connectivity', downsample_factor)


def load_arbitrary_modification1_network(downsample_factor='auto'):
    """Network C with N3 -> N10 weakened (MODIFICATION1_EDGES)."""
    return _load_arbitrary('arbitrary_modification1', MODIFICATION1_EDGES,
                           'True Connectivity (mod 1: N3->N10 depressed 1.25 -> 0.50)',
                           downsample_factor)


def load_arbitrary_modification3_network(downsample_factor='auto'):
    """Network C with N6 -> N2 strengthened (MODIFICATION3_EDGES)."""
    return _load_arbitrary('arbitrary_modification3', MODIFICATION3_EDGES,
                           'True Connectivity (mod 3: N6->N2 potentiated 1.25 -> 3.125)',
                           downsample_factor)


def _load_snnap(name, true_conn, downsample_factor):
    """SNNAP recordings with a spikes.head file: tab-separated voltages, 0.5 ms samples."""
    data_dir = DATASETS_DIR / name
    neuron_names = neuron_names_from_head(data_dir / 'spikes.head')
    _plot_ground_truth(true_conn, neuron_names, data_dir / 'ground_truth.pdf', 'True Connectivity',
                       vmin=-5, vmax=2, cbar_label='Connection Strength (mV)')

    df = pd.read_csv(data_dir / 'spikes.out', sep='\t', header=None, skipinitialspace=True)
    volts = df.iloc[:, 1:-1].values.T                      # (n_neurons, n_samples)
    spikes = _binary_spikes(volts, SNNAP_REARM)
    print(spikes.shape)
    binned, factor = _rebin_native(spikes, downsample_factor, 0.5)

    _plot_raster(binned, 0.5e-3 * factor, data_dir / 'raster_plot.pdf', neuron_names)
    return binned, true_conn, neuron_names


def load_OC_1_network(downsample_factor='auto'):
    """Aplysia feeding circuit (operant conditioning model OCx1): 14 neurons, 240 s.

    Ground truth in mV: each neuron's response to depolarizing every other neuron (mean
    potential over the 200 ms from stimulus onset minus the potential at onset), kept
    only for pairs connected by a chemical or electrical synapse.
    """
    true_conn = np.load(DATASETS_DIR / 'OC_1' / 'true_connectivity_0.20.npy')
    true_conn = np.nan_to_num(np.round(true_conn, 3), nan=0.0)
    return _load_snnap('OC_1', true_conn, downsample_factor)


def load_small_circuit_A_network(downsample_factor='auto'):
    """Network A (Cai et al., 2017): A inhibits B and excites C; 200 s."""
    true_conn = np.zeros((3, 3))
    true_conn[0, 1] = -1
    true_conn[0, 2] = 1
    return _load_snnap('small_circuit_A', true_conn, downsample_factor)


def load_small_circuit_B_network(downsample_factor='auto'):
    """Network B (Cai et al., 2017): A excites B and C, C excites B; 200 s."""
    true_conn = np.zeros((3, 3))
    true_conn[0, 1] = 0.75
    true_conn[0, 2] = 0.75
    true_conn[2, 1] = 1
    return _load_snnap('small_circuit_B', true_conn, downsample_factor)


def pdcm_hidden_indices(n_neurons, hide_frac, hide_seed=0):
    """``(kept, hidden)`` neuron indices; the hidden sets are nested across fractions."""
    rng = np.random.default_rng(hide_seed)
    perm = rng.permutation(n_neurons)
    n_hidden = int(round(hide_frac * n_neurons))
    hidden = np.sort(perm[:n_hidden])
    kept = np.sort(perm[n_hidden:])
    return kept, hidden


_PDCM_SIZES = ('0p002', '0p010')


def load_pdcm_network(downsample_factor='auto', size=None, max_time_s=None,
                      hide_frac=0.0, hide_seed=0, plot=True):
    """Potjans-Diesmann cortical microcircuit simulated in NetPyNE (1 ms resolution).

    size       : None for the 149-neuron, 200 s dataset in datasets/pdcm, or '0p002' /
                 '0p010' for the 2000 s, 149- / 767-neuron datasets in datasets/pdcm_<size>
    max_time_s : keep only the first max_time_s seconds
    hide_frac  : hide this random fraction of neurons (chosen by hide_seed); spikes,
                 ground truth and labels are reduced to the observed neurons
    plot       : write the raster and ground-truth PDFs (never for a subset)

    neuron_names are the population labels (L2e, L2i, ..., L6i). Ground truth in pA.
    """
    if size is None:
        data_dir = DATASETS_DIR / 'pdcm'
    else:
        if size not in _PDCM_SIZES:
            raise ValueError(f"unknown pdcm size {size!r}; options: {_PDCM_SIZES}")
        data_dir = DATASETS_DIR / f'pdcm_{size}'

    spkt       = np.load(data_dir / 'spkt.npy')
    spkid      = np.load(data_dir / 'spkid.npy').astype(int)
    true_conn  = np.load(data_dir / 'weight_matrix.npy')
    pop_labels = np.load(data_dir / 'pop_labels.npy')
    sim_dur_ms = np.load(data_dir / 'sim_duration_ms.npy')[0]
    n_neurons    = true_conn.shape[0]
    neuron_names = list(pop_labels)

    # Truncate to the requested recording length before binning.
    if max_time_s is not None:
        window_ms = max_time_s * 1000.0
        keep = spkt < window_ms
        spkt, spkid = spkt[keep], spkid[keep]
        sim_dur_ms = min(sim_dur_ms, window_ms)

    # resolve 'auto' from a native 1 ms binning before choosing the final bin
    if isinstance(downsample_factor, str):
        native = _bin_spike_times(spkt, spkid, n_neurons, sim_dur_ms, 1)
        downsample_factor = resolve_downsample(downsample_factor, native, 1.0)
    binned_spikes = _bin_spike_times(spkt, spkid, n_neurons, sim_dur_ms, downsample_factor)

    print(f"Loaded PDCM network ({data_dir.name}): {n_neurons} neurons, "
          f"{sim_dur_ms:.0f} ms simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  (bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")

    if hide_frac > 0:
        kept, _ = pdcm_hidden_indices(n_neurons, hide_frac, hide_seed)
        binned_spikes = binned_spikes[kept]
        true_conn     = true_conn[np.ix_(kept, kept)]
        neuron_names  = [neuron_names[i] for i in kept]
        print(f"Hiding {hide_frac:.0%} (seed={hide_seed}): {len(kept)} of {n_neurons} neurons observed")

    # subsets would overwrite the canonical raster / ground-truth PDFs for this data_dir
    if plot and hide_frac == 0 and max_time_s is None:
        _plot_raster(binned_spikes, downsample_factor / 1000, data_dir / 'raster_plot.pdf')
        plot_pdcm_matrix(true_conn, pop_labels, data_dir / 'ground_truth.pdf',
                         'True Connectivity (PDCM NetPyNE)', cbar_label='Synaptic weight (pA)')
    return binned_spikes, true_conn, neuron_names
