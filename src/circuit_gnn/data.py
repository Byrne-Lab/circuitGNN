import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_gnn.optimal_bin_size import resolve_downsample
from circuit_gnn.utils import plot_pdcm_matrix


# The four arbitrary-network conditions (baseline + three modifications) come
# from one SNNAP simulation pipeline and differ only in the dataset directory,
# the synaptic conductances, and the ground-truth plot title.

_ARBITRARY_NUM_NEURONS  = 11
_ARBITRARY_TIME_MAX_S   = 300    # limit to 300 s of simulation
_ARBITRARY_THR_MV       = -10    # spike threshold, mV
_ARBITRARY_REARM        = 8      # re-arm period in samples (8 samples × 1ms = ~7.5ms)
_ARBITRARY_RASTER_MAX_S = 40     # show a readable window, not the full 300 s
_ARBITRARY_NORMALIZER   = 4.00


def _load_arbitrary_common(dataset, edges, conn_title, downsample_factor='auto'):
    """
    Shared loader for the arbitrary SNNAP networks.

    Parameters
    ----------
    dataset : str
        Directory name under ``datasets/`` holding ``spikes.out``; also where
        ``raster_plot.pdf`` and ``ground_truth.pdf`` are written.
    edges : dict
        ``{(pre, post): conductance}`` in raw SNNAP units, before normalization.
    conn_title : str
        Title for the ground-truth connectivity heatmap.
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.

    Returns
    -------
    binned_spikes, true_conn, neuron_names
        See the public wrappers for the full description.
    """

    data_dir    = f'datasets/{dataset}'
    data_loc    = f'{data_dir}/spikes.out'
    time_max    = _ARBITRARY_TIME_MAX_S
    num_neurons = _ARBITRARY_NUM_NEURONS
    SNNAP_output = []
    with io.open(data_loc, mode="r", encoding="utf-8") as f:
        for line in f:
            SNNAP_output.append([float(i) for i in line.split()])
            if SNNAP_output[-1][0]>=time_max:
                break

    # CONVERT TO ARRAY
    SNNAP_output_array = np.asarray(SNNAP_output)

    # TIME+VOLTAGES
    volts = SNNAP_output_array[:,1:(num_neurons+1)]
    neuron_voltage_array = volts.T  # (num_neurons, n_timepoints)

    # convert voltages to spikes
    thr = _ARBITRARY_THR_MV   # mV
    ra  = _ARBITRARY_REARM    # re-arm period in samples

    n_neurons, n_timepoints = neuron_voltage_array.shape
    binned_spikes = np.zeros((n_neurons, n_timepoints), dtype=int)

    for i in range(n_neurons):
        mv = neuron_voltage_array[i]
        sp = np.where(mv > thr)[0]
        if sp.size > 0:
            mask = np.concatenate(([True], np.diff(sp) > ra))
            sp = sp[mask]
            binned_spikes[i, sp] = 1

    # downsample (native resolution is 1 ms; resolve 'auto' via signed-edge stability)
    downsample_factor = resolve_downsample(downsample_factor, binned_spikes, 1.0)
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    true_conn = np.zeros((num_neurons,num_neurons))
    for (pre, post), conductance in edges.items():
        true_conn[pre, post] = conductance
    true_conn = true_conn / _ARBITRARY_NORMALIZER

    neuron_names = np.arange(num_neurons)

    raster_max_s = _ARBITRARY_RASTER_MAX_S
    bin_size_s = 1e-3 * downsample_factor
    spike_times = [np.where(binned_spikes[i] > 0)[0] * bin_size_s for i in range(num_neurons)]
    spike_times = [t[t <= raster_max_s] for t in spike_times]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(num_neurons), neuron_names)
    plt.gca().invert_yaxis()
    plt.xlim(0, raster_max_s)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig(f"{data_dir}/raster_plot.pdf", format='pdf')
    plt.close()

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=0.4)

    # build annotation array: value if nonzero, else empty string
    annot_str = [[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn]

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                annot=annot_str, fmt='', annot_kws={'size': 5, 'color': 'black'},
                cbar_kws={'label': 'Connection Strength',
                          'ticks': [-1.0, -0.5, 0.0, 0.2, 0.4]},
                square=True)
    plt.title(conn_title)
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig(f"{data_dir}/ground_truth.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_arbitrary_network(downsample_factor = 'auto'):
    """
    Load the arbitrary SNNAP simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.

    Returns
    -------
    binned_spikes : ndarray (11, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with an 8-sample
        refractory period; limited to first 300 s of simulation.
    true_conn : ndarray (11, 11)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : ndarray of int
        Integer index 0–10 for each neuron.
    """

    # From Young et al. script
    #   true_conn[5,0] = 1;  true_conn[5,2] = 1;  true_conn[5,4] = -1
    #   true_conn[5,1] = 1;  true_conn[7,1] = -1; true_conn[8,1] = 1
    #   true_conn[2,9] = 1;  true_conn[4,9] = 1
    #   true_conn[0,3] = 1;  true_conn[3,10] = 1

    # From SNNAP conductances
    edges = {
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

    return _load_arbitrary_common(
        'arbitrary', edges, 'True Connectivity', downsample_factor)


def load_arbitrary_modification1_network(downsample_factor = 'auto'):
    """
    Load the arbitrary SNNAP network under modification 1: synaptic depression.

    Exactly ONE edge changes; the other ten are untouched and serve as the null
    distribution:

      N3 -> N10   1.25 -> 0.50   depression (2.5x weaker)

    N5 -> N10 is the matched control: same postsynaptic cell, identical
    baseline weight of 1.25. At baseline the learned ratio A[N3,N10] /
    A[N5,N10] should be ~1; here the true ratio is 0.4.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.

    Returns
    -------
    binned_spikes : ndarray (11, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with an 8-sample
        refractory period; limited to first 300 s of simulation.
    true_conn : ndarray (11, 11)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : ndarray of int
        Integer index 0–10 for each neuron.
    """

    # From SNNAP conductances; * marks the single edge changed from the baseline
    edges = {
        (0, 3):   0.75,   # N1 -> N4
        (2, 9):   0.50,   # N3 -> N10  * was 1.25 (depression)
        (3, 10):  0.70,   # N4 -> N11
        (4, 9):   1.25,   # N5 -> N10     matched control
        (5, 0):   0.50,   # N6 -> N1
        (5, 1):   1.25,   # N6 -> N2
        (5, 2):   0.50,   # N6 -> N3
        (5, 4):  -4.00,   # N6 -> N5
        (7, 1):  -4.00,   # N8 -> N2
        (8, 1):   1.25,   # N9 -> N2
    }

    return _load_arbitrary_common(
        'arbitrary_modification1', edges,
        'True Connectivity (mod 1: N3->N10 depressed 1.25 -> 0.50)',
        downsample_factor)


def load_arbitrary_modification2_network(downsample_factor = 'auto'):
    """
    Load the arbitrary SNNAP network under modification 2: synapse formation.

    Exactly ONE edge changes; the other ten are untouched and serve as the null
    distribution:

      N7 -> N2   0.00 -> 1.25   de novo synapse (true negative -> true positive)

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.

    Returns
    -------
    binned_spikes : ndarray (11, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with an 8-sample
        refractory period; limited to first 300 s of simulation.
    true_conn : ndarray (11, 11)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : ndarray of int
        Integer index 0–10 for each neuron.
    """

    # From SNNAP conductances; * marks the single edge changed from the baseline
    edges = {
        (0, 3):   0.75,   # N1 -> N4
        (2, 9):   1.25,   # N3 -> N10
        (3, 10):  0.70,   # N4 -> N11
        (4, 9):   1.25,   # N5 -> N10
        (5, 0):   0.50,   # N6 -> N1
        (5, 1):   1.25,   # N6 -> N2      within-cell control
        (5, 2):   0.50,   # N6 -> N3
        (5, 4):  -4.00,   # N6 -> N5
        (6, 1):   1.25,   # N7 -> N2   * was 0.00 (de novo synapse)
        (7, 1):  -4.00,   # N8 -> N2
        (8, 1):   1.25,   # N9 -> N2      within-cell control
    }

    return _load_arbitrary_common(
        'arbitrary_modification2', edges,
        'True Connectivity (mod 2: N7->N2 created 0.00 -> 1.25)',
        downsample_factor)


def load_arbitrary_modification3_network(downsample_factor = 'auto'):
    """
    Load the arbitrary SNNAP network under modification 3: synaptic potentiation.

    Exactly ONE edge changes; the other nine are untouched and serve as the null
    distribution:

      N6 -> N2   1.25 -> 3.125   potentiation (2.5x stronger)

    The mirror of modification 1, which depressed an edge by the same 2.5x
    factor. Two controls bracket it: N9 -> N2 is the matched control (same
    postsynaptic cell, identical baseline weight of 1.25), so the learned ratio
    A[N6,N2] / A[N9,N2] is ~1 at baseline and 2.5 here; and N6's other outputs
    (N6 -> N1, N6 -> N3, N6 -> N5) are unchanged, so a cell-wide change in N6
    can be told apart from a change at this one synapse.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.

    Returns
    -------
    binned_spikes : ndarray (11, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with an 8-sample
        refractory period; limited to first 300 s of simulation.
    true_conn : ndarray (11, 11)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : ndarray of int
        Integer index 0–10 for each neuron.
    """

    # From SNNAP conductances; * marks the single edge changed from the baseline
    edges = {
        (0, 3):   0.75,   # N1 -> N4
        (2, 9):   1.25,   # N3 -> N10
        (3, 10):  0.70,   # N4 -> N11
        (4, 9):   1.25,   # N5 -> N10
        (5, 0):   0.50,   # N6 -> N1      within-cell control
        (5, 1):   3.125,  # N6 -> N2   * was 1.25 (potentiation)
        (5, 2):   0.50,   # N6 -> N3      within-cell control
        (5, 4):  -4.00,   # N6 -> N5      within-cell control
        (7, 1):  -4.00,   # N8 -> N2
        (8, 1):   1.25,   # N9 -> N2      matched control
    }

    return _load_arbitrary_common(
        'arbitrary_modification3', edges,
        'True Connectivity (mod 3: N6->N2 potentiated 1.25 -> 3.125)',
        downsample_factor)


def load_OC_1_network(downsample_factor = 'auto'):
    """
    Load the OC_1 compartmental neuron simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 0.5 ms.

    Returns
    -------
    binned_spikes : ndarray (14, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with a 15-sample
        refractory period; 240 s total simulation.
    true_conn : ndarray (14, 14)
        Signed synaptic weight matrix [pre, post] in mV.
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Neuron label for each neuron parsed from ``spikes.head``.
    """

    GAP = 0.20

    data_loc = 'datasets/OC_1/spikes.out'
    with open('datasets/OC_1/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]

    # Load true connectivity
    true_conn = np.load(f"datasets/OC_1/true_connectivity_{GAP:.2f}.npy")
    true_conn = np.round(true_conn, 3)
    true_conn = np.nan_to_num(true_conn, nan=0.0)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)

    # build annotation array: value if nonzero, else empty string
    annot_str = [[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn]

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                annot=annot_str, fmt='',
                cbar_kws={'label': 'Connection Strength (mV)'},
                square=True)
    plt.title('True Connectivity')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig("datasets/OC_1/ground_truth.pdf", format='pdf')
    plt.close()

    df = pd.read_csv(data_loc, sep='\t', header=None, skipinitialspace=True)
    voltage_data = df.iloc[:, 1:-1]
    voltage_array_original = voltage_data.values
    neuron_voltage_array = voltage_array_original.T

    # convert voltages to spikes
    thr = -10   # mv > thr
    ra  = 15    # refractory period in samples

    n_neurons, n_timepoints = neuron_voltage_array.shape
    binned_spikes = np.zeros((n_neurons, n_timepoints), dtype=int)

    for i in range(n_neurons):
        mv = neuron_voltage_array[i]
        sp = np.where(mv > thr)[0]                             # sp = find(mv > thr)
        if sp.size > 0:
            mask = np.concatenate(([True], np.diff(sp) > ra))  # sp([1; diff(sp)>ra])
            sp = sp[mask]
            binned_spikes[i, sp] = 1

    print(binned_spikes.shape)

    # downsample (native resolution is 0.5 ms; resolve 'auto' via signed-edge stability)
    downsample_factor = resolve_downsample(downsample_factor, binned_spikes, 0.5)
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    # plot
    bin_size_s = 0.5e-3 * downsample_factor
    spike_times = []
    for i in range(binned_spikes.shape[0]):
        times = np.where(binned_spikes[i, :] > 0)[0] * bin_size_s
        spike_times.append(times)

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(len(neuron_names)), neuron_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig("datasets/OC_1/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_A_network(downsample_factor = 'auto'):
    """
    Load the small circuit A simulation dataset from Cai et al.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 0.5 ms.

    Returns
    -------
    binned_spikes : ndarray (3, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with a 15-sample
        refractory period; 200 s total simulation.
    true_conn : ndarray (3, 3)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Neuron label for each neuron parsed from ``spikes.head``.
    """

    data_loc = 'datasets/small_circuit_A/spikes.out'
    with open('datasets/small_circuit_A/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]
    
    # Load true connectivity
    true_conn = np.zeros((3,3))
    true_conn[0,1] = -1
    true_conn[0,2] = 1

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)

    # build annotation array: value if nonzero, else empty string
    annot_str = [[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn]

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                annot=annot_str, fmt='',
                cbar_kws={'label': 'Connection Strength (mV)'},
                square=True)
    plt.title('True Connectivity')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig("datasets/small_circuit_A/ground_truth.pdf", format='pdf')
    plt.close()
    
    df = pd.read_csv(data_loc, sep='\t', header=None, skipinitialspace=True)
    voltage_data = df.iloc[:, 1:-1]
    voltage_array_original = voltage_data.values
    neuron_voltage_array = voltage_array_original.T

    # convert voltages to spikes
    thr = -10   # mv > thr
    ra  = 15    # refractory period in bins

    n_neurons, n_timepoints = neuron_voltage_array.shape
    binned_spikes = np.zeros((n_neurons, n_timepoints), dtype=int)

    for i in range(n_neurons):
        mv = neuron_voltage_array[i]
        sp = np.where(mv > thr)[0]                             # sp = find(mv > thr)
        if sp.size > 0:
            mask = np.concatenate(([True], np.diff(sp) > ra))  # sp([1; diff(sp)>ra])
            sp = sp[mask]
            binned_spikes[i, sp] = 1

    print(binned_spikes.shape)

    # downsample (native resolution is 0.5 ms; resolve 'auto' via signed-edge stability)
    downsample_factor = resolve_downsample(downsample_factor, binned_spikes, 0.5)
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    # plot
    bin_size_s = 0.5e-3 * downsample_factor
    spike_times = []
    for i in range(binned_spikes.shape[0]):
        times = np.where(binned_spikes[i, :] > 0)[0] * bin_size_s
        spike_times.append(times)

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(len(neuron_names)), neuron_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig("datasets/small_circuit_A/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_B_network(downsample_factor = 'auto'):
    """
    Load the small circuit B simulation dataset froM Cai et al.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 0.5 ms.

    Returns
    -------
    binned_spikes : ndarray (3, n_timebins)
        Spike counts per bin. Voltages thresholded at -10 mV with a 15-sample
        refractory period; 200 s total simulation.
    true_conn : ndarray (3, 3)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Neuron label for each neuron parsed from ``spikes.head``.
    """

    data_loc = 'datasets/small_circuit_B/spikes.out'
    with open('datasets/small_circuit_B/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]
    
    # Load true connectivity
    true_conn = np.zeros((3,3))
    true_conn[0,1] = 0.75
    true_conn[0,2] = 0.75
    true_conn[2,1] = 1

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=2)

    # build annotation array: value if nonzero, else empty string
    annot_str = [[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn]

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                annot=annot_str, fmt='',
                cbar_kws={'label': 'Connection Strength (mV)'},
                square=True)
    plt.title('True Connectivity')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig("datasets/small_circuit_B/ground_truth.pdf", format='pdf')
    plt.close()
    
    df = pd.read_csv(data_loc, sep='\t', header=None, skipinitialspace=True)
    voltage_data = df.iloc[:, 1:-1]
    voltage_array_original = voltage_data.values
    neuron_voltage_array = voltage_array_original.T

    # convert voltages to spikes
    thr = -10   # mv > thr
    ra  = 15    # refractory period in samples

    n_neurons, n_timepoints = neuron_voltage_array.shape
    binned_spikes = np.zeros((n_neurons, n_timepoints), dtype=int)

    for i in range(n_neurons):
        mv = neuron_voltage_array[i]
        sp = np.where(mv > thr)[0]                             # sp = find(mv > thr)
        if sp.size > 0:
            mask = np.concatenate(([True], np.diff(sp) > ra))  # sp([1; diff(sp)>ra])
            sp = sp[mask]
            binned_spikes[i, sp] = 1

    print(binned_spikes.shape)

    # downsample (native resolution is 0.5 ms; resolve 'auto' via signed-edge stability)
    downsample_factor = resolve_downsample(downsample_factor, binned_spikes, 0.5)
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    # plot
    bin_size_s = 0.5e-3 * downsample_factor
    spike_times = []
    for i in range(binned_spikes.shape[0]):
        times = np.where(binned_spikes[i, :] > 0)[0] * bin_size_s
        spike_times.append(times)

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(len(neuron_names)), neuron_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig("datasets/small_circuit_B/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def pdcm_hidden_indices(n_neurons, hide_frac, hide_seed=0):
    rng = np.random.default_rng(hide_seed)
    perm = rng.permutation(n_neurons)
    n_hidden = int(round(hide_frac * n_neurons))
    hidden = np.sort(perm[:n_hidden])
    kept = np.sort(perm[n_hidden:])
    return kept, hidden


def _load_pdcm_dir(data_dir, downsample_factor='auto', max_time_s=None,
                   hide_frac=0.0, hide_seed=0, plot=True):
    """For PDCM NetPyNE dataset (base, scaling sizes, large).

    Parameters
    ----------
    data_dir : str
        Directory holding the .npy files (e.g. 'datasets/pdcm_0p010').
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.
    max_time_s : float or None
    hide_frac : float
        Fraction of neurons to hide. ``true_conn`` and ``neuron_names`` are reduced to the observed submatrix.
        The hidden set is random and deterministic in ``hide_seed``.
    hide_seed : int
        Seed selecting which neurons are hidden. The hidden sets are nested across fractions.
    plot : bool
        Write the raster / ground-truth PDFs for ``data_dir``. Set False to skip them.

    Returns
    -------
    binned_spikes, true_conn, neuron_names  (same contract as the other loaders;
    reduced to the observed neurons when ``hide_frac`` > 0)
    """

    spkt       = np.load(f'{data_dir}/spkt.npy')
    spkid      = np.load(f'{data_dir}/spkid.npy')
    true_conn  = np.load(f'{data_dir}/weight_matrix.npy')
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    sim_dur_ms = np.load(f'{data_dir}/sim_duration_ms.npy')[0]
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
        native_bins = int(sim_dur_ms // 1)
        native = np.zeros((n_neurons, native_bins), dtype=np.int32)
        nt_idx = np.floor(spkt / 1).astype(int)
        nvalid = (nt_idx >= 0) & (nt_idx < native_bins)
        np.add.at(native, (spkid.astype(int)[nvalid], nt_idx[nvalid]), 1)
        downsample_factor = resolve_downsample(downsample_factor, native, 1.0)

    n_bins = int(sim_dur_ms // downsample_factor)
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)
    t_idx = np.floor(spkt / downsample_factor).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned_spikes, (spkid.astype(int)[valid], t_idx[valid]), 1)

    print(f"Loaded PDCM network ({data_dir}): {n_neurons} neurons, "
          f"{sim_dur_ms:.0f} ms simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  "
          f"(bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")

    if hide_frac > 0:
        n_full = n_neurons
        kept, hidden = pdcm_hidden_indices(n_full, hide_frac, hide_seed)
        binned_spikes = binned_spikes[kept]
        true_conn     = true_conn[np.ix_(kept, kept)]
        pop_labels    = pop_labels[kept]
        neuron_names  = [neuron_names[i] for i in kept]
        n_neurons     = len(kept)
        print(f"Hiding {hide_frac:.0%} (seed={hide_seed}): " f"{n_neurons} of {n_full} neurons observed")

    # subsets would overwrite the canonical raster / ground-truth PDFs for this data_dir
    if not plot or hide_frac > 0 or max_time_s is not None:
        return binned_spikes, true_conn, neuron_names

    spike_times_s = [np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000 for i in range(n_neurons)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_plot.pdf', format='pdf')
    plt.close()

    plot_pdcm_matrix(true_conn, pop_labels, method_name='ground_truth', save_dir=data_dir,
                     title='True Connectivity (PDCM NetPyNE)',
                     cbar_label='Synaptic weight (pA)')

    return binned_spikes, true_conn, neuron_names


_PDCM_SIZES = ('0p002', '0p010', '0p025', '0p050')


def load_pdcm_network(downsample_factor='auto',
                      size=None, max_time_s=None,
                      hide_frac=0.0, hide_seed=0, plot=True):
    """
    Load a PDCM NetPyNE simulation dataset, optionally at a given network size
    and recording length (for time/neuron scaling studies).

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms. 'auto' (default) picks it by heuristic (see
        circuit_gnn.optimal_bin_size). Native resolution is 1 ms.
    size : {None, '0p002', '0p010', '0p025', '0p050'}
        None -> the base 'datasets/pdcm' dataset. Otherwise selects
        'datasets/pdcm_<size>' (149 / 767 / 1924 / 3854 neurons respectively).
    max_time_s : float or None
        Keep only the first ``max_time_s`` seconds (datasets are 2000 s). Use to
        sweep recording length.
    hide_frac, hide_seed : float, int
        Hide a uniform-random ``hide_frac`` of neurons.
    plot : bool
        Write the raster / ground-truth PDFs for the dataset. Set False to skip them.

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
    true_conn : ndarray (n_neurons, n_neurons)   signed weights [pre, post]
    neuron_names : list of str                   population label per neuron
    """
    if size is None:
        data_dir = 'datasets/pdcm'
    else:
        if size not in _PDCM_SIZES:
            raise ValueError(f"unknown pdcm size {size!r}; options: {_PDCM_SIZES}")
        data_dir = f'datasets/pdcm_{size}'
    return _load_pdcm_dir(data_dir, downsample_factor=downsample_factor,
                          max_time_s=max_time_s,
                          hide_frac=hide_frac, hide_seed=hide_seed, plot=plot)