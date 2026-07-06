import json

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import warnings
import os
import re
import tarfile
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

from circuit_gnn.optimal_bin_size import resolve_downsample

def load_arbitrary_network(downsample_factor = 10, gap=-1, step=-1):
    """
    Load the arbitrary SNNAP simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins). Native resolution is 1 ms.
    gap : ignored
        Kept for API compatibility with other loaders.
    step : ignored
        Kept for API compatibility with other loaders.

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

    # downsample
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


def load_yuto_network(downsample_factor = 20, gap=0.20, step=1):
    """
    Load the Yuto compartmental neuron simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 20 -> 10 ms bins, 10 -> 5 ms bins).
        Native resolution is 0.5 ms.
    gap : float
        Synaptic gap/delay used to select the ground-truth connectivity file
        (e.g. 0.20 loads ``true_connectivity_0.20.npy``).
    step : {1, 2}
        1 for direct (monosynaptic) connections, 2 for 1-hop (disynaptic)
        connections. Selects between the two connectivity .npy files.

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

    data_loc = 'datasets/yuto/spikes.out'
    with open('datasets/yuto/spikes.head') as f:
        neuron_names = [line.split('[')[1].split('.')[0] for line in f if line.startswith('V[')]
    
    # Load true connectivity
    if step == 1:
        true_conn = np.load(f"datasets/yuto/true_connectivity_{gap:.2f}.npy")
    elif step == 2:
        true_conn = np.load(f"datasets/yuto/true_connectivity_1_hop_{gap:.2f}.npy")
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
    plt.savefig("datasets/yuto/ground_truth.pdf", format='pdf')
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

    # downsample
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
    plt.savefig("datasets/yuto/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_A_network(downsample_factor = 20, gap=0.20, step=1):
    """
    Load the small circuit A simulation dataset froM Cai et al.

    Parameters
    ----------
    downsample_factor : int
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

    # downsample
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


def load_small_circuit_B_network(downsample_factor = 20, gap=0.20, step=1):
    """
    Load the small circuit B simulation dataset froM Cai et al.

    Parameters
    ----------
    downsample_factor : int
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

    # downsample
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


def load_pdcm_network(downsample_factor=2, gap=0, step=0):
    """
    Load the PDCM NetPyNE simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins)
    gap : ignored
        Kept for API compatibility with other loaders.

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

def load_v1_network(downsample_factor=2, gap=0, step=0):
    """
    Load the multi-area-model V1 simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins).
    gap : ignored
        Kept for API compatibility with other loaders.

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        Signed synaptic weight matrix [pre, post] (rows=source, cols=target).
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Population label for each neuron (e.g. '23E', '4I', ...).
    """

    # Load raw files
    data_dir = 'datasets/v1'

    W_saved = np.load(f'{data_dir}/ground_truth_weights.npy')   # (N, N)
    true_conn = W_saved.T.astype(np.float64)                    # (N, N) [pre, post]

    spike_data = np.load(f'{data_dir}/spikes.npy')              # (S, 2): [neuron_idx, time_ms]
    gids = np.load(f'{data_dir}/neuron_gids.npy')               # (N,)

    with open(f'{data_dir}/neuron_info.json') as f:
        neuron_info = json.load(f)                              # gid_str -> {pop, local_idx}

    n_neurons = true_conn.shape[0]

    # Build neuron index -> population name

    idx_to_pop = {}
    for gid_str, info in neuron_info.items():
        gid = int(gid_str)
        idx = int(np.searchsorted(gids, gid))
        if idx < len(gids) and gids[idx] == gid:
            idx_to_pop[idx] = info['pop']

    pop_order = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    neuron_names = [idx_to_pop.get(i, 'unknown') for i in range(n_neurons)]

    # Bin spikes

    spike_idx = spike_data[:, 0].astype(int)
    spike_times_ms = spike_data[:, 1]
    sim_dur_ms = spike_times_ms.max()

    n_bins = int(sim_dur_ms // downsample_factor)
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)

    t_idx = np.floor(spike_times_ms / downsample_factor).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned_spikes, (spike_idx[valid], t_idx[valid]), 1)

    print(f"Loaded V1 network: {n_neurons} neurons, {sim_dur_ms/1000:.1f} s simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  (bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")
    mean_rate = spike_data.shape[0] / n_neurons / (sim_dur_ms / 1000.) if sim_dur_ms > 0 else 0
    print(f"Mean firing rate: {mean_rate:.2f} Hz")

    # Raster plot

    pop_colors = {
        '23E': '#4878CF', '23I': '#D65F5F',
        '4E':  '#6ACC65', '4I':  '#B47CC7',
        '5E':  '#C4AD66', '5I':  '#77BEDB',
        '6E':  '#F7910A', '6I':  '#999999',
    }

    spike_times_by_neuron = [
        np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000.
        for i in range(n_neurons)
    ]
    colors_per_neuron = [pop_colors.get(neuron_names[i], '#aaaaaa') for i in range(n_neurons)]

    plt.figure(figsize=(12, 5))
    plt.eventplot(spike_times_by_neuron, orientation='horizontal',
                  linelengths=0.8, colors=colors_per_neuron)

    # Population boundary lines + labels
    cursor = 0
    for pop in pop_order:
        count = sum(1 for name in neuron_names if name == pop)
        if count == 0:
            continue
        if cursor > 0:
            plt.axhline(cursor - 0.5, color='k', linewidth=0.5, alpha=0.5)
        plt.text(sim_dur_ms / 1000. * 1.001, cursor + count / 2,
                 pop, va='center', fontsize=7,
                 color=pop_colors.get(pop, 'k'))
        cursor += count

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title(f'V1 Spike Raster  ({n_neurons} neurons, bin={downsample_factor} ms)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster.pdf', format='pdf')

    # Connectivity heatmap

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Synaptic weight (mV)'},
                square=True)

    # Population boundary lines
    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in pop_order:
        count = sum(1 for name in neuron_names if name == pop)
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

    plt.title('True Connectivity (V1, multi-area-model)')
    plt.xlabel('Postsynaptic (target)')
    plt.ylabel('Presynaptic (source)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/ground_truth.pdf', format='pdf')

    return binned_spikes, true_conn, neuron_names


def load_pdcm_large_network(downsample_factor=2, gap=0, step=0):
    """
    Load the large PDCM NetPyNE simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins)
    gap : ignored
        Kept for API compatibility with other loaders.

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

    n_bins = int(sim_dur_ms // downsample_factor)
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)
    t_idx = np.floor(spkt / downsample_factor).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned_spikes, (spkid.astype(int)[valid], t_idx[valid]), 1)

    print(f"Loaded PDCM large network: {n_neurons} neurons, "
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


def load_mdp_network(downsample_factor=6, gap=0, step=0):
    """
    Load the multi-area-model mdp simulation dataset.

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins).
    gap : ignored
        Kept for API compatibility with other loaders.

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        Signed synaptic weight matrix [pre, post] (rows=source, cols=target).
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Population label for each neuron (e.g. '23E', '4I', ...).
    """

    # Load raw files
    data_dir = 'datasets/mdp'

    W_saved = np.load(f'{data_dir}/ground_truth_weights.npy')   # (N, N)
    true_conn = W_saved.T.astype(np.float64)                    # (N, N) [pre, post]

    spike_data = np.load(f'{data_dir}/spikes.npy')              # (S, 2): [neuron_idx, time_ms]
    gids = np.load(f'{data_dir}/neuron_gids.npy')               # (N,)

    with open(f'{data_dir}/neuron_info.json') as f:
        neuron_info = json.load(f)                              # gid_str -> {pop, local_idx}

    n_neurons = true_conn.shape[0]

    # Build neuron index -> population name

    idx_to_pop = {}
    for gid_str, info in neuron_info.items():
        gid = int(gid_str)
        idx = int(np.searchsorted(gids, gid))
        if idx < len(gids) and gids[idx] == gid:
            idx_to_pop[idx] = info['pop']

    pop_order = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
    neuron_names = [idx_to_pop.get(i, 'unknown') for i in range(n_neurons)]

    # Bin spikes

    spike_idx = spike_data[:, 0].astype(int)
    spike_times_ms = spike_data[:, 1]
    sim_dur_ms = spike_times_ms.max()

    n_bins = int(sim_dur_ms // downsample_factor)
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)

    t_idx = np.floor(spike_times_ms / downsample_factor).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned_spikes, (spike_idx[valid], t_idx[valid]), 1)

    print(f"Loaded mdp network: {n_neurons} neurons, {sim_dur_ms/1000:.1f} s simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  (bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")
    mean_rate = spike_data.shape[0] / n_neurons / (sim_dur_ms / 1000.) if sim_dur_ms > 0 else 0
    print(f"Mean firing rate: {mean_rate:.2f} Hz")

    # Raster plot

    pop_colors = {
        '23E': '#4878CF', '23I': '#D65F5F',
        '4E':  '#6ACC65', '4I':  '#B47CC7',
        '5E':  '#C4AD66', '5I':  '#77BEDB',
        '6E':  '#F7910A', '6I':  '#999999',
    }

    spike_times_by_neuron = [
        np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000.
        for i in range(n_neurons)
    ]
    colors_per_neuron = [pop_colors.get(neuron_names[i], '#aaaaaa') for i in range(n_neurons)]

    plt.figure(figsize=(12, 5))
    plt.eventplot(spike_times_by_neuron, orientation='horizontal',
                  linelengths=0.8, colors=colors_per_neuron)

    # Population boundary lines + labels
    cursor = 0
    for pop in pop_order:
        count = sum(1 for name in neuron_names if name == pop)
        if count == 0:
            continue
        if cursor > 0:
            plt.axhline(cursor - 0.5, color='k', linewidth=0.5, alpha=0.5)
        plt.text(sim_dur_ms / 1000. * 1.001, cursor + count / 2,
                 pop, va='center', fontsize=7,
                 color=pop_colors.get(pop, 'k'))
        cursor += count

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title(f'MDP Spike Raster  ({n_neurons} neurons, bin={downsample_factor} ms)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster.pdf', format='pdf')
    plt.close()

    # Connectivity heatmap

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Synaptic weight (mV)'},
                square=True)

    # Population boundary lines
    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in pop_order:
        count = sum(1 for name in neuron_names if name == pop)
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

    plt.title('True Connectivity (MDP, multi-area-model)')
    plt.xlabel('Postsynaptic (target)')
    plt.ylabel('Presynaptic (source)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/ground_truth.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_v1_v2_network(downsample_factor=6, gap=0, step=0):
    """
    Load the multi-area-model V1+V2 two-area simulation dataset.

    Both areas are simulated together, so the connectivity matrix has four
    blocks: V1->V1, V2->V1, V1->V2, V2->V2 (the off-diagonal blocks are the real
    cortico-cortical synapses between the two areas). Neurons are ordered
    area-by-area then population-by-population (V1 23E ... V1 6I, V2 23E ... V2 6I).

    Parameters
    ----------
    downsample_factor : int
        Bin size in ms (e.g. 10 -> 10 ms bins).
    gap : ignored
        Kept for API compatibility with other loaders.

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins)
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        Signed synaptic weight matrix [pre, post] (rows=source, cols=target).
        Positive = excitatory, negative = inhibitory, 0 = no connection.
    neuron_names : list of str
        Area-qualified population label for each neuron (e.g. 'V1_23E', 'V2_4I').
    """

    # Load raw files
    data_dir = 'datasets/v1_v2'
    areas = ['V1', 'V2']
    pop_order = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

    W_saved = np.load(f'{data_dir}/ground_truth_weights.npy')   # (N, N)
    true_conn = W_saved.T.astype(np.float64)                    # (N, N) [pre, post]

    spike_data = np.load(f'{data_dir}/spikes.npy')              # (S, 2): [neuron_idx, time_ms]
    gids = np.load(f'{data_dir}/neuron_gids.npy')               # (N,) index order

    with open(f'{data_dir}/neuron_info.json') as f:
        neuron_info = json.load(f)                              # gid_str -> {area, pop, local_idx}

    n_neurons = true_conn.shape[0]

    # Build neuron index -> 'AREA_POP' label.
    # gids is stored in index order (block layout), so position == neuron index.
    neuron_names = ['unknown'] * n_neurons
    for idx, gid in enumerate(gids):
        info = neuron_info[str(int(gid))]
        neuron_names[idx] = f"{info['area']}_{info['pop']}"

    # Ordered list of populated 'AREA_POP' blocks, matching the run-script layout
    block_labels = []
    for area in areas:
        for pop in pop_order:
            label = f'{area}_{pop}'
            if any(name == label for name in neuron_names):
                block_labels.append(label)

    # Bin spikes
    spike_idx = spike_data[:, 0].astype(int)
    spike_times_ms = spike_data[:, 1]
    sim_dur_ms = spike_times_ms.max()

    n_bins = int(sim_dur_ms // downsample_factor)
    binned_spikes = np.zeros((n_neurons, n_bins), dtype=np.int32)

    t_idx = np.floor(spike_times_ms / downsample_factor).astype(int)
    valid = (t_idx >= 0) & (t_idx < n_bins)
    np.add.at(binned_spikes, (spike_idx[valid], t_idx[valid]), 1)

    print(f"Loaded V1+V2 network: {n_neurons} neurons, {sim_dur_ms/1000:.1f} s simulation")
    print(f"Binned spikes shape: {binned_spikes.shape}  (bin size = {downsample_factor} ms)")
    print(f"Binned spikes possibilities: {np.unique(binned_spikes)}")
    mean_rate = spike_data.shape[0] / n_neurons / (sim_dur_ms / 1000.) if sim_dur_ms > 0 else 0
    print(f"Mean firing rate: {mean_rate:.2f} Hz")

    # Raster plot

    pop_colors = {
        '23E': '#4878CF', '23I': '#D65F5F',
        '4E':  '#6ACC65', '4I':  '#B47CC7',
        '5E':  '#C4AD66', '5I':  '#77BEDB',
        '6E':  '#F7910A', '6I':  '#999999',
    }

    def color_for(label):
        # label is 'AREA_POP'; color by population
        return pop_colors.get(label.split('_', 1)[1], '#aaaaaa')

    spike_times_by_neuron = [
        np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000.
        for i in range(n_neurons)
    ]
    colors_per_neuron = [color_for(neuron_names[i]) for i in range(n_neurons)]

    plt.figure(figsize=(12, 6))
    plt.eventplot(spike_times_by_neuron, orientation='horizontal',
                  linelengths=0.8, colors=colors_per_neuron)

    # Thin population boundary lines + labels
    cursor = 0
    for label in block_labels:
        count = sum(1 for name in neuron_names if name == label)
        if cursor > 0:
            plt.axhline(cursor - 0.5, color='k', linewidth=0.4, alpha=0.4)
        plt.text(sim_dur_ms / 1000. * 1.001, cursor + count / 2,
                 label, va='center', fontsize=6, color=color_for(label))
        cursor += count
    # Thick area boundary lines
    cursor = 0
    for area in areas:
        a_count = sum(1 for name in neuron_names if name.startswith(area + '_'))
        if cursor > 0:
            plt.axhline(cursor - 0.5, color='k', linewidth=1.6)
        cursor += a_count

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title(f'V1+V2 Spike Raster  ({n_neurons} neurons, bin={downsample_factor} ms)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster.pdf', format='pdf')
    plt.close()

    # Connectivity heatmap

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(true_conn.min(), -1e-6)
    vmax = max(true_conn.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(true_conn,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Synaptic weight (mV)'},
                square=True)

    ax = plt.gca()

    # Thin population boundary lines + tick labels
    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for label in block_labels:
        count = sum(1 for name in neuron_names if name == label)
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(label)
        cursor += count
    boundaries.append(cursor)

    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.4, alpha=0.5)
        ax.axvline(b, color='k', linewidth=0.4, alpha=0.5)

    # Thick area boundary lines
    cursor = 0
    for area in areas:
        a_count = sum(1 for name in neuron_names if name.startswith(area + '_'))
        if cursor > 0:
            ax.axhline(cursor, color='k', linewidth=1.8)
            ax.axvline(cursor, color='k', linewidth=1.8)
        cursor += a_count

    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=6, rotation=90)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=6, rotation=0)

    plt.title('True Connectivity (V1+V2, multi-area-model)')
    plt.xlabel('Postsynaptic (target)')
    plt.ylabel('Presynaptic (source)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/ground_truth.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_hc3_session(downsample_factor='auto', gap=-1, step=-1,
                     tar_path='datasets/hc3/ec016.1016.tar.gz',
                     topdir='ec016.58',
                     cell_csv='datasets/hc3/hc3-cell.csv',
                     regions=['CA3', 'DG', 'EC2', 'EC3'],
                     quality_filter=True,
                     min_rate_hz=0.0,
                     duration_s=2981.888,   # ec016.1016, from hc3-session.csv (id 1427)
                     max_time_s=None,
                     data_dir='datasets/hc3'):
    """
    Load spikes from one CRCNS hc-3 recording session into the same format as
    the simulation loaders: (binned_spikes, true_conn, neuron_names).

    Unlike the simulated datasets, this is in-vivo data, so:
      * there is NO ground-truth connectivity -> ``true_conn`` is an all-zeros
        placeholder. The putative excitatory/inhibitory identity from the
        metadata (cellType 'p'/'i'/'n') is encoded in ``neuron_names`` and is
        the reference Kobayashi et al. validate against.
      * spikes come from Klusters output: ``.res.N`` holds spike times (in
        samples) and ``.clu.N`` holds the cluster id per spike, for shank N.
        Clusters 0 (noise) and 1 (unsortable) are never neurons and are dropped.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms, or 'auto' to pick it by occupancy via
        circuit_gnn.optimal_bin_size (native resolution here is 1 ms).
    gap, step : ignored
        Kept for API compatibility with the other loaders.
    tar_path : str
        Path to the downloaded session archive, e.g. ec016.1016.tar.gz.
        Only .res.N/.clu.N/.xml are read; the large .spk/.eeg/.fet members are
        streamed past, never extracted.
    topdir : str
        Top-level directory the session belongs to (ec016.1016 -> 'ec016.58').
        Used to select this session's units from the metadata cell table.
    cell_csv : str or None
        Path to hc3-cell.csv from crcns-hc3-metadata-tables.zip. Needed for
        region / cell-type labels and quality filtering. If None, every sorted
        cluster (>=2) on every shank is returned, labelled 'NA'/'n'.
    regions : list of str or None
        Restrict to these regions, e.g. ['CA3','DG','EC2','EC3']. Requires
        cell_csv. None keeps all regions present.
    quality_filter : bool
        If True (requires cell_csv), keep only eDist>14 & RefracRatio<0.2 &
        RefracViol<0.01.
    min_rate_hz : float
        Drop units whose mean rate over the session is below this. Kobayashi
        used 0.5 Hz for the connectivity analysis; default 0.0 keeps everything.
    duration_s : float or None
        Exact recording length (from the session metadata table). If None it is
        estimated from the last spike, which slightly inflates firing rates.
        Ignored when ``max_time_s`` is set (the window becomes ``max_time_s``).
    max_time_s : float or None
        If set, keep only spikes in the first ``max_time_s`` seconds of the
        session and use that as the window for rates and binning. None uses the
        whole session.

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins) of int
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        All-zeros placeholder (no ground truth for in-vivo data).
    neuron_names : list of str
        '<region>_<cellType>_e<shank>c<cluster>' per neuron, ordered by region
        then type so the raster / matrix are block-structured.
    """
    session = os.path.basename(tar_path)
    for suf in ('.tar.gz', '.tgz', '.tar'):
        if session.endswith(suf):
            session = session[:-len(suf)]
            break

    # --- pull only the small files we need straight out of the archive ---
    res_re = re.compile(r'\.res\.(\d+)$')
    clu_re = re.compile(r'\.clu\.(\d+)$')
    res_raw, clu_raw, xml_raw = {}, {}, None
    with tarfile.open(tar_path, 'r|*') as tf:          # streaming: single pass, no 1.3 GB extract
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if not base.startswith(session):
                continue
            mres, mclu = res_re.search(base), clu_re.search(base)
            if mres:
                res_raw[int(mres.group(1))] = tf.extractfile(m).read()
            elif mclu:
                clu_raw[int(mclu.group(1))] = tf.extractfile(m).read()
            elif base == f'{session}.xml' or (xml_raw is None and base.endswith('.xml')):
                xml_raw = tf.extractfile(m).read()
            # .spk / .eeg / .fet / .whl are skipped -> never read into memory

    if not res_raw:
        raise FileNotFoundError(f'No .res.N files found in {tar_path}')

    # --- sampling rate from the .xml (.res values are sample indices) ---
    fs = 20000.0
    if xml_raw is not None:
        root = ET.fromstring(xml_raw)
        node = root.find('.//acquisitionSystem/samplingRate')
        if node is None:
            node = root.find('.//samplingRate')     # NB: don't use `a or b`; leaf Elements are falsy
        if node is not None and node.text:
            fs = float(node.text)
    samples_per_ms = fs / 1000.0

    def read_ints(buf):
        if not buf.strip():
            return np.empty(0, np.int64)
        return pd.read_csv(io.BytesIO(buf), header=None).iloc[:, 0].to_numpy(np.int64)

    # --- optional metadata: region / cellType / quality per (ele, clu) ---
    # The CRCNS hc-3 CSVs are headerless; column order is fixed by hc3-tables.sql.
    cell_cols = ['id', 'topdir', 'animal', 'ele', 'clu', 'region',
                 'nexciting', 'ninhibiting', 'exciting', 'inhibiting',
                 'excited', 'inhibited', 'fireRate', 'totalFireRate',
                 'cellType', 'eDist', 'RefracRatio', 'RefracViol']
    meta = None
    if cell_csv is not None:
        meta = pd.read_csv(cell_csv, header=None, names=cell_cols)
        meta = meta[meta['topdir'] == topdir].copy()
        if regions is not None:
            meta = meta[meta['region'].isin(regions)]
        if quality_filter:
            meta = meta[(meta['eDist'] > 14) &
                        (meta['RefracRatio'] < 0.2) &
                        (meta['RefracViol'] < 0.01)]
    elif regions is not None or quality_filter:
        warnings.warn('regions / quality_filter need cell_csv; ignoring them.')

    # --- one unit per sorted cluster ---
    units = []
    for shank in sorted(res_raw):
        if shank not in clu_raw:
            continue
        res = read_ints(res_raw[shank])
        clu = read_ints(clu_raw[shank])
        if clu.size == 0:
            continue
        clu = clu[1:]                                 # first line is the cluster count
        if len(res) != len(clu):
            warnings.warn(f'shank {shank}: res/clu length mismatch '
                          f'({len(res)} vs {len(clu)}); skipped')
            continue

        if meta is not None:
            wanted = [(int(r.clu), r.region, r.cellType)
                      for r in meta[meta['ele'] == shank].itertuples()]
        else:
            wanted = [(int(c), 'NA', 'n') for c in np.unique(clu) if c >= 2]

        for c, region, ctype in wanted:
            samples = res[clu == c]
            if max_time_s is not None:                    # keep only the first max_time_s seconds
                samples = samples[samples < max_time_s * fs]
            if samples.size:
                units.append(dict(region=str(region), ctype=str(ctype),
                                  ele=shank, clu=c, samples=samples))

    if not units:
        raise RuntimeError('No units selected. If you passed cell_csv, check '
                           'topdir / regions / quality filters '
                           '(or max_time_s if you truncated the session).')

    # truncating the session shortens the window used for rates and binning
    if max_time_s is not None:
        duration_s = float(max_time_s)

    # --- duration and firing-rate filter (Kobayashi: 0.5 Hz) ---
    last_sample = max(u['samples'].max() for u in units)
    if duration_s is None:
        duration_s = last_sample / fs
        warnings.warn(f'duration_s not given; estimated {duration_s:.1f}s from the '
                      f'last spike. Pass the exact value from the session table '
                      f'for accurate rates.')
    for u in units:
        u['rate'] = u['samples'].size / duration_s
    kept = [u for u in units if u['rate'] >= min_rate_hz]
    if len(kept) != len(units):
        print(f'Dropped {len(units) - len(kept)}/{len(units)} units below {min_rate_hz} Hz.')
    units = kept

    # --- block-order by region then type ---
    units.sort(key=lambda u: (u['region'], u['ctype'], u['ele'], u['clu']))
    neuron_names = [f"{u['region']}_{u['ctype']}_e{u['ele']}c{u['clu']}" for u in units]
    n = len(units)

    # --- pick bin size ---
    dur_ms = duration_s * 1000.0
    if isinstance(downsample_factor, str):
        # native 1 ms grid, sized to cover the last spike, then let the resolver
        # choose the factor by occupancy (matches the pdcm/mdp loaders)
        n_native = max(int(np.ceil(dur_ms)), int(last_sample // samples_per_ms) + 1)
        native = np.zeros((n, n_native), dtype=np.int32)   # ~n * dur_s * 1000 * 4 bytes; sizable on long sessions
        for i, u in enumerate(units):
            np.add.at(native[i], np.floor(u['samples'] / samples_per_ms).astype(np.int64), 1)
        bin_ms = int(resolve_downsample(downsample_factor, native, 1.0))
        del native
    else:
        bin_ms = int(downsample_factor)

    # --- bin (single path; n_bins covers both the stated duration and the last spike) ---
    step = samples_per_ms * bin_ms
    n_bins = max(int(np.ceil(dur_ms / bin_ms)), int(last_sample // step) + 1)
    binned_spikes = np.zeros((n, n_bins), dtype=np.int32)
    for i, u in enumerate(units):
        np.add.at(binned_spikes[i], np.floor(u['samples'] / step).astype(np.int64), 1)

    print(f'{session}: {n} units, {duration_s:.1f}s, fs={fs:.0f} Hz, bin={bin_ms} ms')
    print(f'Binned spikes shape: {binned_spikes.shape}')
    print(f'Binned spikes possibilities: {np.unique(binned_spikes)}')
    print(f'Units per region: {pd.Series([u["region"] for u in units]).value_counts().to_dict()}')

    # --- no ground truth for in-vivo data ---
    true_conn = np.zeros((n, n))
    warnings.warn('In-vivo data: no ground-truth connectivity exists. true_conn is an '
                  "all-zeros placeholder; use the putative E/I labels in neuron_names "
                  '(cellType p/i/n) as the validation reference, as Kobayashi et al. do.')

    # --- raster, same style as the other loaders ---
    os.makedirs(data_dir, exist_ok=True)
    bin_s = bin_ms / 1000.0
    spike_times = [np.where(binned_spikes[i] > 0)[0] * bin_s for i in range(n)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(n), neuron_names, fontsize=5)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)'); plt.ylabel('Neuron'); plt.title(f'Spike Raster ({session})')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_plot.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names