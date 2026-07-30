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

def load_arbitrary_network(downsample_factor = 'auto'):
    """
    Load the arbitrary SNNAP simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 10 -> 10 ms bins). Native resolution is 1 ms.

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

    data_loc = 'datasets/arbitrary/spikes.out'
    time_max    = 300 # limit to 300 s
    num_neurons = 11
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
    thr = -10   # mV
    ra  = 8     # re-arm period in samples (8 samples × 1ms = ~7.5ms)

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
    true_conn[5,0]  = 1
    true_conn[5,2]  = 1
    true_conn[5,4]  = -1

    true_conn[5,1]  = 1
    true_conn[7,1]  = -1
    true_conn[8,1]  = 1

    true_conn[2,9] = 1
    true_conn[4,9] = 1

    true_conn[0,3]  = 1
    true_conn[3,10] = 1

    neuron_names = np.arange(num_neurons)

    bin_size_s = 1e-3 * downsample_factor
    spike_times = [np.where(binned_spikes[i] > 0)[0] * bin_size_s for i in range(num_neurons)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(num_neurons), neuron_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig("datasets/arbitrary/raster_plot.pdf", format='pdf')
    plt.close()

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # build annotation array: value if nonzero, else empty string
    annot_str = [[f'{v:.3f}' if v != 0 else '' for v in row] for row in true_conn]

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names,
                cmap=cmap, norm=norm,
                annot=annot_str, fmt='', annot_kws={'size': 5, 'color': 'black'},
                cbar_kws={'label': 'Connection Strength (mV)'},
                square=True)
    plt.title('True Connectivity')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig("datasets/arbitrary/ground_truth.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_OC_2_network(downsample_factor = 'auto'):
    """
    Load the OC_2 compartmental neuron simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 20 -> 10 ms bins, 10 -> 5 ms bins).
        Native resolution is 0.5 ms.

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

    Notes
    -----
    The synaptic gap/delay used to pick the ground-truth connectivity file is
    fixed below (``GAP``); edit it here to switch to a different
    ``true_connectivity_*.npy``.
    """

    # synaptic gap/delay selecting the ground-truth file
    GAP = 0.20

    data_loc = 'datasets/OC_2/spikes.out'
    with open('datasets/OC_2/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]

    # Load true connectivity
    true_conn = np.load(f"datasets/OC_2/true_connectivity_{GAP:.2f}.npy")
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
    plt.savefig("datasets/OC_2/ground_truth.pdf", format='pdf')
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
    plt.savefig("datasets/OC_2/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_A_network(downsample_factor = 'auto'):
    """
    Load the small circuit A simulation dataset froM Cai et al.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 20 -> 10 ms bins, 10 -> 5 ms bins).
        Native resolution is 0.5 ms.

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
    plt.savefig("datasets/small_circuit_A/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_B_network(downsample_factor = 'auto'):
    """
    Load the small circuit B simulation dataset froM Cai et al.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 20 -> 10 ms bins, 10 -> 5 ms bins).
        Native resolution is 0.5 ms.

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


def load_pdcm_network(downsample_factor='auto'):
    """
    Load the PDCM NetPyNE simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 10 -> 10 ms bins)

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Population label for each neuron (e.g. 'L2e', 'L4i', ...).
    """

    data_dir = 'datasets/pdcm'

    spkt       = np.load(f'{data_dir}/spkt.npy')
    spkid      = np.load(f'{data_dir}/spkid.npy')
    true_conn  = np.load(f'{data_dir}/weight_matrix.npy')
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    sim_dur_ms = np.load(f'{data_dir}/sim_duration_ms.npy')[0]
    n_neurons    = true_conn.shape[0]
    neuron_names = list(pop_labels)

    print(spkt.shape, spkid.shape)

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

    print(f"Loaded PDCM network: {n_neurons} neurons, "
          f"{sim_dur_ms:.0f} ms simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  "
          f"(bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")

    spike_times_s = [np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000 for i in range(n_neurons)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_plot.pdf', format='pdf')
    plt.close()

    # Plot ground truth connectivity
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Synaptic weight (pA)'},
                square=True)

    # Draw population boundary lines
    pops_ordered = ['L2e','L2i','L4e','L4i','L5e','L5i','L6e','L6i']
    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in pops_ordered:
        count = np.sum(pop_labels == pop)
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(pop)
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.title('True Connectivity (PDCM NetPyNE)')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig('datasets/pdcm/ground_truth.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_pdcm_large_network(downsample_factor='auto'):
    """
    Load the PDCM NetPyNE simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
        Bin size in ms (e.g. 10 -> 10 ms bins)

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        Signed synaptic weight matrix [pre, post].
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Population label for each neuron (e.g. 'L2e', 'L4i', ...).
    """

    data_dir = 'datasets/pdcm_large'

    spkt       = np.load(f'{data_dir}/spkt.npy')
    spkid      = np.load(f'{data_dir}/spkid.npy')
    true_conn  = np.load(f'{data_dir}/weight_matrix.npy')
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    sim_dur_ms = np.load(f'{data_dir}/sim_duration_ms.npy')[0]
    n_neurons    = true_conn.shape[0]
    neuron_names = list(pop_labels)

    print(spkt.shape, spkid.shape)

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

    print(f"Loaded PDCM network: {n_neurons} neurons, "
          f"{sim_dur_ms:.0f} ms simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  "
          f"(bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")

    spike_times_s = [np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000 for i in range(n_neurons)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_plot.pdf', format='pdf')
    plt.close()

    # Plot ground truth connectivity
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Synaptic weight (pA)'},
                square=True)

    # Draw population boundary lines
    pops_ordered = ['L2e','L2i','L4e','L4i','L5e','L5i','L6e','L6i']
    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in pops_ordered:
        count = np.sum(pop_labels == pop)
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(pop)
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.title('True Connectivity (PDCM NetPyNE)')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig('datasets/pdcm_large/ground_truth.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names
