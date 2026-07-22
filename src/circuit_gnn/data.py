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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_gnn.optimal_bin_size import resolve_downsample

def load_arbitrary_network(downsample_factor = 'auto', gap=-1, step=-1):
    """
    Load the arbitrary SNNAP simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
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

    # downsample (native resolution is 1 ms; resolve 'auto' via signed-edge stability)
    downsample_factor = resolve_downsample(downsample_factor, binned_spikes, 1.0)
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    true_conn = np.zeros((num_neurons,num_neurons))
    
    # From Young et al. script
    """    
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
    """
    # From SNNAP conductances
    true_conn[0, 3]  =  0.75   # N1 -> N4
    true_conn[2, 9]  =  1.25   # N3 -> N10
    true_conn[3, 10] =  0.70   # N4 -> N11
    true_conn[4, 9]  =  1.25   # N5 -> N10
    true_conn[5, 0]  =  0.50   # N6 -> N1
    true_conn[5, 1]  =  1.25   # N6 -> N2
    true_conn[5, 2]  =  0.50   # N6 -> N3
    true_conn[5, 4]  = -4.00   # N6 -> N5
    true_conn[7, 1]  = -4.00   # N8 -> N2
    true_conn[8, 1]  =  1.25   # N9 -> N2
    true_conn = true_conn / np.abs(true_conn).max()


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
    plt.title('True Connectivity')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    plt.savefig("datasets/arbitrary/ground_truth.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_yuto_network(downsample_factor = 'auto', gap=0.20, step=1):
    """
    Load the Yuto compartmental neuron simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
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
    plt.savefig("datasets/yuto/raster_plot.pdf", format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit_A_network(downsample_factor = 'auto', gap=0.20, step=1):
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


def load_small_circuit_B_network(downsample_factor = 'auto', gap=0.20, step=1):
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


def _load_pdcm_dir(data_dir, downsample_factor='auto', max_time_s=None, plot=True):
    """Shared engine for every PDCM NetPyNE dataset (base, scaling sizes, large).

    All PDCM datasets share an identical on-disk layout
    (spkt/spkid/weight_matrix/pop_labels/sim_duration_ms), differing only in
    neuron count and simulated duration, so binning/plotting lives here once.

    Parameters
    ----------
    data_dir : str
        Directory holding the .npy files (e.g. 'datasets/pdcm_0p010').
    downsample_factor : int or 'auto'
        Bin size in ms, or 'auto' to pick it by signed-edge stability. For a
        scaling sweep pass an explicit int so the bin is fixed across runs;
        'auto' also materialises a 1 ms native array over the (possibly
        truncated) recording, which is large for the big networks.
    max_time_s : float or None
        If set, keep only spikes in the first ``max_time_s`` seconds and treat
        that as the simulated duration. Used to sweep recording length.
    plot : bool
        Write the raster / ground-truth PDFs. Turn off inside a replicate sweep
        (the figures are identical across seeds and slow for large N).

    Returns
    -------
    binned_spikes, true_conn, neuron_names  (same contract as the other loaders)
    """

    spkt       = np.load(f'{data_dir}/spkt.npy')
    spkid      = np.load(f'{data_dir}/spkid.npy')
    true_conn  = np.load(f'{data_dir}/weight_matrix.npy')
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    sim_dur_ms = np.load(f'{data_dir}/sim_duration_ms.npy')[0]
    n_neurons    = true_conn.shape[0]
    neuron_names = list(pop_labels)

    print(spkt.shape, spkid.shape)

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

    if not plot:
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
    plt.savefig(f'{data_dir}/ground_truth.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


# Neuron count per scaling size, for reference (derived from weight_matrix.npy):
#   0p002 -> 149,  0p010 -> 767,  0p025 -> 1924,  0p050 -> 3854 neurons.
_PDCM_SIZES = ('0p002', '0p010', '0p025', '0p050')


def load_pdcm_network(downsample_factor='auto', gap=0, step=0,
                      size=None, max_time_s=None, plot=True):
    """
    Load a PDCM NetPyNE simulation dataset, optionally at a given network size
    and recording length (for time/neuron scaling studies).

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms, or 'auto' (signed-edge stability). Pass an explicit int
        for a scaling sweep so the bin is held fixed across sizes/durations.
    gap, step : ignored
        Kept for API compatibility with other loaders.
    size : {None, '0p002', '0p010', '0p025', '0p050'}
        None -> the base 'datasets/pdcm' dataset. Otherwise selects
        'datasets/pdcm_<size>' (149 / 767 / 1924 / 3854 neurons respectively).
    max_time_s : float or None
        Keep only the first ``max_time_s`` seconds (datasets are 2000 s). Use to
        sweep recording length.
    plot : bool
        Write raster / ground-truth PDFs (turn off during a replicate sweep).

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
                          max_time_s=max_time_s, plot=plot)


def load_allen_network(downsample_factor='auto', gap=0, step=0,
                       max_time_s=None, data_dir='datasets/allen'):
    """
    Load the Allen Visual-Coding Neuropixels hippocampal dataset used by
    Medvedeva et al. 2025 (session 715093703, probe 810755803, regions
    CA1/CA3/DG). Built by datasets/allen/prepare_allen.py and stored in the same
    .npy layout as the simulated datasets (spkt/spkid/pop_labels/sim_duration_ms),
    with spike times in ms.

    Like the hc-3 recordings this is in-vivo data, so there is NO ground-truth
    connectivity: ``true_conn`` is an all-zeros placeholder. Region identity is
    encoded in ``neuron_names`` ('<region>_u<unit_id>'), ordered CA1->CA3->DG so
    the raster and learned matrix are block-structured.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        Bin size in ms, or 'auto' to pick it by signed-edge stability (see
        circuit_gnn.optimal_bin_size). NB: 'auto' builds a 1 ms native array over
        the whole ~2.68 h recording (~3 GB for 92 neurons); pass an explicit bin
        (or max_time_s) to avoid that.
    gap, step : ignored
        Kept for API compatibility with the other loaders.
    max_time_s : float or None
        If set, keep only spikes in the first ``max_time_s`` seconds. The full
        recording is ~2.68 h; the paper also fits on 500 s / 1000 s subsets.
    data_dir : str
        Directory holding the .npy files (default 'datasets/allen').

    Returns
    -------
    binned_spikes : ndarray (n_neurons, n_timebins) of int
        Spike counts per bin.
    true_conn : ndarray (n_neurons, n_neurons)
        All-zeros placeholder (no ground truth for in-vivo data).
    neuron_names : list of str
        '<region>_u<unit_id>' per neuron, ordered by region.
    """
    spkt       = np.load(f'{data_dir}/spkt.npy')          # ms
    spkid      = np.load(f'{data_dir}/spkid.npy')
    pop_labels = np.load(f'{data_dir}/pop_labels.npy')
    sim_dur_ms = np.load(f'{data_dir}/sim_duration_ms.npy')[0]

    n_neurons    = len(pop_labels)
    neuron_names = list(pop_labels)
    true_conn    = np.zeros((n_neurons, n_neurons))       # in-vivo: no ground truth

    if max_time_s is not None:
        window_ms = max_time_s * 1000.0
        keep = spkt < window_ms
        spkt, spkid = spkt[keep], spkid[keep]
        sim_dur_ms = min(sim_dur_ms, window_ms)

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

    print(f"Loaded Allen hippocampal network: {n_neurons} neurons, "
          f"{sim_dur_ms:.0f} ms recording")
    print(f"Binned spikes shape: {binned_spikes.shape}  "
          f"(bin size = {downsample_factor} ms)")
    print("Units per region: "
          f"{pd.Series([n.split('_')[0] for n in neuron_names]).value_counts().to_dict()}")

    # Region-blocked spike raster.
    spike_times_s = [np.where(binned_spikes[i] > 0)[0] * downsample_factor / 1000
                     for i in range(n_neurons)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    regions = [n.split('_')[0] for n in neuron_names]
    ax = plt.gca()
    i = 0
    while i < len(regions):
        j = i
        while j < len(regions) and regions[j] == regions[i]:
            j += 1
        ax.axhline(j - 0.5, color='red', linewidth=0.5)
        ax.text(-0.01, (i + j - 1) / 2, regions[i], ha='right', va='center',
                fontsize=8, transform=ax.get_yaxis_transform())
        i = j
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Allen hippocampal spike raster (CA1/CA3/DG)')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_plot.pdf', format='pdf')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_mdp_network(downsample_factor='auto', gap=0, step=0):
    """
    Load the multi-area-model mdp simulation dataset.

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
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

    # resolve 'auto' from a native 1 ms binning before choosing the final bin
    if isinstance(downsample_factor, str):
        native_bins = int(sim_dur_ms // 1)
        native = np.zeros((n_neurons, native_bins), dtype=np.int32)
        nt_idx = np.floor(spike_times_ms / 1).astype(int)
        nvalid = (nt_idx >= 0) & (nt_idx < native_bins)
        np.add.at(native, (spike_idx[nvalid], nt_idx[nvalid]), 1)
        downsample_factor = resolve_downsample(downsample_factor, native, 1.0)

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


def load_pdcm_large_network(downsample_factor='auto', gap=0, step=0,
                            max_time_s=None, plot=True):
    """
    Load the large PDCM NetPyNE simulation dataset. Thin wrapper over
    :func:`_load_pdcm_dir`; see :func:`load_pdcm_network` for the shared
    parameter/return contract.
    """
    return _load_pdcm_dir('datasets/pdcm_large', downsample_factor=downsample_factor,
                          max_time_s=max_time_s, plot=plot)


def load_v1_v2_network(downsample_factor='auto', gap=0, step=0):
    """
    Load the multi-area-model V1+V2 two-area simulation dataset.

    Both areas are simulated together, so the connectivity matrix has four
    blocks: V1->V1, V2->V1, V1->V2, V2->V2 (the off-diagonal blocks are the real
    cortico-cortical synapses between the two areas). Neurons are ordered
    area-by-area then population-by-population (V1 23E ... V1 6I, V2 23E ... V2 6I).

    Parameters
    ----------
    downsample_factor : int or 'auto'
        If 'auto' (default), the bin is chosen by signed-edge stability (see
        circuit_gnn.optimal_bin_size); otherwise used as below.
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

    # resolve 'auto' from a native 1 ms binning before choosing the final bin
    if isinstance(downsample_factor, str):
        native_bins = int(sim_dur_ms // 1)
        native = np.zeros((n_neurons, native_bins), dtype=np.int32)
        nt_idx = np.floor(spike_times_ms / 1).astype(int)
        nvalid = (nt_idx >= 0) & (nt_idx < native_bins)
        np.add.at(native, (spike_idx[nvalid], nt_idx[nvalid]), 1)
        downsample_factor = resolve_downsample(downsample_factor, native, 1.0)

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


# ---------------------------------------------------------------------------
# CRCNS hc-3 loaders. One engine (_load_hc3) backs all of them: a single session
# is just the one-element case of the multi-session aggregator, so tar-reading,
# metadata, binning and the raster live in exactly one place.
# ---------------------------------------------------------------------------

# hc3-cell.csv is headerless; column order is fixed by hc3-tables.sql.
_HC3_CELL_COLS = ['id', 'topdir', 'animal', 'ele', 'clu', 'region',
                  'nexciting', 'ninhibiting', 'exciting', 'inhibiting',
                  'excited', 'inhibited', 'fireRate', 'totalFireRate',
                  'cellType', 'eDist', 'RefracRatio', 'RefracViol']


def _hc3_cell_neurons(cell_csv, topdir, regions=None, quality_filter=True):
    """Ordered (ele, clu, region, cellType) list for a topdir from hc3-cell.csv,
    sorted region->type->ele->clu so rasters / matrices are block-structured."""
    meta = pd.read_csv(cell_csv, header=None, names=_HC3_CELL_COLS)
    meta = meta[meta['topdir'] == topdir]
    if regions is not None:
        meta = meta[meta['region'].isin(regions)]
    if quality_filter:
        meta = meta[(meta['eDist'] > 14) & (meta['RefracRatio'] < 0.2) & (meta['RefracViol'] < 0.01)]
    meta = meta.sort_values(['region', 'cellType', 'ele', 'clu'])
    return [(int(e), int(c), str(r), str(ct)) for e, c, r, ct in
            meta[['ele', 'clu', 'region', 'cellType']].itertuples(index=False, name=None)]


def _bin_hc3(samples, fs, duration_s, downsample_factor):
    """Bin a list of per-neuron spike-sample arrays -> (binned_spikes, bin_ms).
    'auto' picks the bin by signed-edge stability on a native 1 ms grid (as the
    pdcm/mdp loaders do); an int is used directly as the bin width in ms."""
    n = len(samples)
    samples_per_ms = fs / 1000.0
    dur_ms = duration_s * 1000.0
    last_sample = max((s.max() for s in samples if s.size), default=0.0)
    if isinstance(downsample_factor, str):
        n_native = max(int(np.ceil(dur_ms)), int(last_sample // samples_per_ms) + 1)
        native = np.zeros((n, n_native), dtype=np.int32)   # sizable on long sessions; freed below
        for i, s in enumerate(samples):
            if s.size:
                np.add.at(native[i], np.floor(s / samples_per_ms).astype(np.int64), 1)
        bin_ms = int(resolve_downsample(downsample_factor, native, 1.0))
        del native
    else:
        bin_ms = int(downsample_factor)
    step = samples_per_ms * bin_ms
    n_bins = max(int(np.ceil(dur_ms / bin_ms)), int(last_sample // step) + 1)
    binned = np.zeros((n, n_bins), dtype=np.int32)
    for i, s in enumerate(samples):
        if s.size:
            np.add.at(binned[i], np.floor(s / step).astype(np.int64), 1)
    return binned, bin_ms


def _save_hc3_raster(binned_spikes, neuron_names, tag, bin_ms, data_dir='datasets/hc3'):
    """Spike raster in the same style as the other loaders; per-tag filename so
    loaders / stages don't clobber each other's PDF."""
    os.makedirs(data_dir, exist_ok=True)
    bin_s = bin_ms / 1000.0
    n = len(neuron_names)
    spike_times = [np.where(binned_spikes[i] > 0)[0] * bin_s for i in range(n)]
    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black')
    plt.yticks(range(n), neuron_names, fontsize=5)
    plt.gca().invert_yaxis()
    plt.xlabel('Time (s)'); plt.ylabel('Neuron'); plt.title(f'Spike Raster ({tag})')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/raster_{tag}.pdf', format='pdf')
    plt.close()


def _load_hc3(session_tars, durations, cell_csv=None, topdir=None, regions=None,
              quality_filter=True, downsample_factor='auto', min_rate_hz=0.0,
              max_time_s=None, keep_silent=False, tag=None, data_dir='datasets/hc3'):
    """Load one or more hc-3 sessions -> (binned_spikes, true_conn(zeros), neuron_names).

    session_tars : list of .tar.gz paths (len 1 == single session). Only .res/.clu/.xml
        are streamed out; the big .spk/.eeg/.fet members are never read into memory.
    durations    : seconds per session, aligned to session_tars (None -> estimate from
        that session's last spike). Several sessions are concatenated end-to-end with a
        cumulative sample offset, so short blocks pool into one longer train.
    cell_csv     : hc3-cell.csv. If given, the neuron set is fixed by the cell table
        (region/quality filtered, sorted) -> identical neurons/order across sessions, so
        learned-A matrices are comparable element by element. If None, every sorted
        cluster (>=2) found in the archives is used, labelled 'NA'/'n'.
    min_rate_hz  : drop neurons below this mean rate (ignored when keep_silent).
    max_time_s   : keep only the first max_time_s seconds of each session.
    keep_silent  : keep spikeless neurons as all-zero rows (preserves index alignment
        across stages); otherwise silent / sub-rate neurons are dropped.

    No ground truth exists for in-vivo data -> true_conn is all zeros; the E/I identity
    (cellType p/i/n) lives in neuron_names, which is the Kobayashi et al. reference.
    """
    reads = [_read_hc3_res_clu(t) for t in session_tars]
    fs = reads[0][2]

    if cell_csv is not None:
        neurons = _hc3_cell_neurons(cell_csv, topdir, regions, quality_filter)
        if not neurons:
            raise RuntimeError(f'No neurons for topdir {topdir!r} after region/quality filter.')
    else:
        if regions is not None or quality_filter:
            warnings.warn('regions / quality_filter need cell_csv; ignoring them.')
        keys = set()
        for res, clu, _ in reads:
            for sh, cl in clu.items():
                if sh in res and len(res[sh]) == len(cl):
                    keys |= {(sh, int(c)) for c in np.unique(cl) if c >= 2}
        neurons = [(e, c, 'NA', 'n') for e, c in sorted(keys)]
        if not neurons:
            raise FileNotFoundError('No sorted clusters (>=2) found in the archive(s).')
    n = len(neurons)

    # concatenate each neuron's spikes across sessions, offsetting by elapsed time
    per_neuron = [[] for _ in range(n)]
    offset = 0.0
    total_duration = 0.0
    for (res, clu, _), dur, tar in zip(reads, durations, session_tars):
        rows, sess_last = [], 0.0
        for i, (e, c, r, ct) in enumerate(neurons):
            if e in res and e in clu and len(res[e]) == len(clu[e]):
                s = res[e][clu[e] == c]
                if max_time_s is not None:                 # keep only the first max_time_s seconds
                    s = s[s < max_time_s * fs]
                if s.size:
                    rows.append((i, s))
                    sess_last = max(sess_last, float(s.max()))
        if max_time_s is not None:
            dur = float(max_time_s)
        if dur is None:
            dur = sess_last / fs
            warnings.warn(f'duration not given for {os.path.basename(tar)}; estimated '
                          f'{dur:.1f}s from its last spike (inflates rates slightly).')
        for i, s in rows:
            per_neuron[i].append(s.astype(np.float64) + offset)
        offset += dur * fs
        total_duration += dur
    samples = [np.concatenate(p) if p else np.empty(0) for p in per_neuron]

    # rate / silence filter (silent neurons kept only when alignment is needed)
    if keep_silent:
        keep = list(range(n))
    else:
        keep = [i for i in range(n)
                if samples[i].size and samples[i].size / total_duration >= min_rate_hz]
    if len(keep) != n:
        print(f'Kept {len(keep)}/{n} neurons (dropped silent / < {min_rate_hz} Hz).')
    neurons = [neurons[i] for i in keep]
    samples = [samples[i] for i in keep]
    n = len(neurons)
    if n == 0:
        raise RuntimeError('No neurons left after the rate / silence filter.')
    neuron_names = [f'{r}_{ct}_e{e}c{c}' for e, c, r, ct in neurons]

    binned_spikes, bin_ms = _bin_hc3(samples, fs, total_duration, downsample_factor)

    if tag is None:
        tag = os.path.basename(session_tars[0])
        for suf in ('.tar.gz', '.tgz', '.tar'):
            if tag.endswith(suf):
                tag = tag[:-len(suf)]
                break

    silent = sum(1 for s in samples if s.size == 0)
    print(f'{tag}: {n} neurons ({silent} silent), {total_duration:.0f}s, '
          f'fs={fs:.0f} Hz, bin={bin_ms} ms, shape={binned_spikes.shape}')
    print(f'Units per region: {pd.Series([r for _, _, r, _ in neurons]).value_counts().to_dict()}')
    warnings.warn('In-vivo data: no ground-truth connectivity. true_conn is all-zeros; use '
                  'the E/I labels in neuron_names (cellType p/i/n) as the reference, as '
                  'Kobayashi et al. do.')
    _save_hc3_raster(binned_spikes, neuron_names, tag, bin_ms, data_dir)
    return binned_spikes, np.zeros((n, n)), neuron_names


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
        Bin size in ms, or 'auto' to pick it by signed-edge stability via
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
    return _load_hc3([tar_path], [duration_s],
                     cell_csv=cell_csv, topdir=topdir, regions=regions,
                     quality_filter=quality_filter, downsample_factor=downsample_factor,
                     min_rate_hz=min_rate_hz, max_time_s=max_time_s,
                     keep_silent=False, data_dir=data_dir)


def load_hc3_gor01(downsample_factor='auto', gap=-1, step=-1, **overrides):
    """CRCNS hc-3 session gor01-6-7 / 2006-6-7_11-26-53 (linearOne maze).

    The densest single-region recording in the data set: ~60 quality-passing
    CA3 cells recorded together, so a large fraction of one recurrent microcircuit
    is observed -> much less confounding by unrecorded neurons than a multi-region
    session. Within CA3, 42 pyramidal cells are physiologically flagged 'exciting'
    and 6 interneurons 'inhibiting' (CCG analysis), giving an E/I reference.

    Thin wrapper over load_hc3_session with this session baked in; pass any
    load_hc3_session kwarg (e.g. regions, max_time_s) via **overrides.
    """
    params = dict(
        tar_path='datasets/hc3/2006-6-7_11-26-53.tar.gz',
        topdir='gor01-6-7',
        cell_csv='datasets/hc3/hc3-cell.csv',
        regions=['CA3'],
        quality_filter=True,
        duration_s=2692.247,          # from hc3-session.csv (id 1455)
    )
    params.update(overrides)
    return load_hc3_session(downsample_factor=downsample_factor, gap=gap, step=step, **params)


def load_hc3_pin01(downsample_factor='auto', gap=-1, step=-1, **overrides):
    """CRCNS hc-3 session pin01-11-04 / 11-04_22-31-40 (Tmaze).

    A CA1+CA3 recording (20 CA1 + 27 quality-passing CA3 cells) with monosynaptic
    labels in both regions: CA1 has 4 'exciting' pyramidal + 1 'inhibiting'
    interneuron, CA3 has 12 'exciting' pyramidal + 1 'inhibiting' interneuron.
    CA1 pyramidal<->interneuron pairs are the best-characterised in-vivo
    monosynaptic connections, so this trades some sampling density for a cleaner
    validation reference than the CA3-only gor01 session.

    Thin wrapper over load_hc3_session; pass any load_hc3_session kwarg via
    **overrides.
    """
    params = dict(
        tar_path='datasets/hc3/11-04_22-31-40.tar.gz',
        topdir='pin01-11-04',
        cell_csv='datasets/hc3/hc3-cell.csv',
        regions=['CA1', 'CA3'],
        quality_filter=True,
        duration_s=1811.692,          # from hc3-session.csv (id 1477)
    )
    params.update(overrides)
    return load_hc3_session(downsample_factor=downsample_factor, gap=gap, step=step, **params)


def load_hc3_ec014(downsample_factor='auto', gap=-1, step=-1, **overrides):
    """CRCNS hc-3 session ec014.333 / topdir ec014.24 (bigSquare open field).

    This is the in-vivo session used for Fig. 5 of Kobayashi et al. 2019
    (Reconstructing neuronal circuitry from parallel spike trains): CA1 recorded
    while the rat explored a square open field, with entorhinal cortex (EC) units
    also present (Fig. 5d). 5608.653 s (~93 min) covers their 90-min window; they
    also show 10- and 30-min windows (pass max_time_s=600 or 1800 to reproduce).

    Defaults mirror the paper: regions CA1 + EC and min_rate_hz=0.5 (Fig. 5a only
    plots neurons firing >0.5 Hz). Quality-passing cells: CA1 47, EC2 7, EC3 6,
    EC5 2; monosynaptic labels are populated in every region (e.g. CA1 17 exciting
    + 3 inhibiting). Thin wrapper over load_hc3_session; override via **overrides.
    """
    params = dict(
        tar_path='datasets/hc3/ec014.333.tar.gz',
        topdir='ec014.24',
        cell_csv='datasets/hc3/hc3-cell.csv',
        regions=['CA1', 'EC2', 'EC3', 'EC5'],
        quality_filter=True,
        min_rate_hz=0.0,              # Kobayashi et al. Fig. 5a: firing rate > 0.5 Hz
        duration_s=5608.653,          # from hc3-session.csv (id 854)
    )
    params.update(overrides)
    return load_hc3_session(downsample_factor=downsample_factor, gap=gap, step=step, **params)


def load_hc3_ec016(downsample_factor='auto', gap=-1, step=-1, **overrides):
    """CRCNS hc-3 session ec016.1016 / topdir ec016.58 (multi-region).

    The multi-region reference session used across the standard baselines and the
    variant/architecture sweeps: CA3 + DG + EC2 + EC3 recorded together, capped at
    the first 2000 s so every method sees the same neuron set and window. Because
    the neurons span several regions, it is more confounded by unrecorded units
    than the single-region gor01 session, but it exercises the cross-region case.

    Thin wrapper over load_hc3_session with this session baked in; pass any
    load_hc3_session kwarg (e.g. regions, max_time_s) via **overrides.
    """
    params = dict(
        tar_path='datasets/hc3/ec016.1016.tar.gz',
        topdir='ec016.58',
        cell_csv='datasets/hc3/hc3-cell.csv',
        regions=['CA3', 'DG', 'EC2', 'EC3'],
        quality_filter=True,
        max_time_s=2000,
    )
    params.update(overrides)
    return load_hc3_session(downsample_factor=downsample_factor, gap=gap, step=step, **params)


def _read_hc3_res_clu(tar_path):
    """Stream the .res.N / .clu.N / .xml members out of one hc-3 session tarball.

    Returns (res_by_shank, clu_by_shank, fs). The big .spk/.eeg/.fet members are
    never read into memory. The .clu count line (first entry) is already dropped,
    so res_by_shank[N] and clu_by_shank[N] are spike-aligned.
    """
    session = os.path.basename(tar_path)
    for suf in ('.tar.gz', '.tgz', '.tar'):
        if session.endswith(suf):
            session = session[:-len(suf)]
            break

    res_re = re.compile(r'\.res\.(\d+)$')
    clu_re = re.compile(r'\.clu\.(\d+)$')
    res_raw, clu_raw, xml_raw = {}, {}, None
    with tarfile.open(tar_path, 'r|*') as tf:
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

    if not res_raw:
        raise FileNotFoundError(f'No .res.N files found in {tar_path}')

    fs = 20000.0
    if xml_raw is not None:
        root = ET.fromstring(xml_raw)
        node = root.find('.//acquisitionSystem/samplingRate')
        if node is None:
            node = root.find('.//samplingRate')     # NB: leaf Elements are falsy; use explicit None
        if node is not None and node.text:
            fs = float(node.text)

    def read_ints(buf):
        if not buf.strip():
            return np.empty(0, np.int64)
        return pd.read_csv(io.BytesIO(buf), header=None).iloc[:, 0].to_numpy(np.int64)

    res = {sh: read_ints(b) for sh, b in res_raw.items()}
    clu = {sh: read_ints(b)[1:] for sh, b in clu_raw.items()}   # drop the cluster-count line
    return res, clu, fs


def load_hc3_topdir_stage(topdir, sessions, session_durations,
                          downsample_factor='auto', gap=-1, step=-1,
                          data_dir='datasets/hc3',
                          cell_csv='datasets/hc3/hc3-cell.csv',
                          regions=None, quality_filter=True, tag=None):
    """Aggregate several sessions of one topdir into a single index-aligned train.

    The neuron set/order are fixed by the cell table for ``topdir`` (region/quality
    filtered, sorted), so every stage returns the SAME neurons in the SAME order ->
    learned-A matrices from different stages are comparable element by element.
    ``sessions`` are concatenated end-to-end with a cumulative time offset; neurons
    silent within the stage stay as all-zero rows. Thin wrapper over _load_hc3
    (keep_silent=True). Returns (binned_spikes, true_conn(zeros), neuron_names).
    """
    tars = [f'{data_dir}/{s}.tar.gz' for s in sessions]
    durs = [session_durations[s] for s in sessions]
    return _load_hc3(tars, durs, cell_csv=cell_csv, topdir=topdir, regions=regions,
                     quality_filter=quality_filter, downsample_factor=downsample_factor,
                     keep_silent=True, tag=tag or topdir, data_dir=data_dir)


# ec015.03: interleaved task/sleep, same neurons throughout (CA1 63 + EC ~20 quality).
# Grouped so each stage pools its same-behaviour sessions into one longer train.
# ec015.03 recording order (interleaved task/rest):
#   ec015.041 midSquare  -> ec015.042 sleep  -> ec015.043 wheel  -> ec015.044 sleep
#   -> ec015.045 wheel   -> ec015.046 sleep  -> ec015.047 midSquare
# NB the 3 sleep blocks are NOT interchangeable: 042 follows midSquare, 044 & 046
# follow wheel. Pool them only when you don't care about the preceding task; keep
# them separate (pass explicit session IDs) to look for task-specific reorganisation.
EC015_03_STAGE_SESSIONS = {
    'midSquare': ['ec015.041', 'ec015.047'],
    'wheel':     ['ec015.043', 'ec015.045'],
    'sleep':     ['ec015.042', 'ec015.044', 'ec015.046'],
}
EC015_03_DURATIONS = {                      # seconds, from hc3-session.csv
    'ec015.041': 2826.0, 'ec015.042': 608.0, 'ec015.043': 2889.0, 'ec015.044': 623.0,
    'ec015.045': 2922.0, 'ec015.046': 600.0, 'ec015.047': 2967.0,
}


def load_hc3_ec015(stage='wheel', sessions=None,
                   downsample_factor='auto', gap=-1, step=-1, **overrides):
    """Load part of topdir ec015.03 for state-dependent connectivity inference.

    Specify EITHER:
      stage : 'wheel' (default) | 'midSquare' | 'sleep'
          A named behaviour; its sessions are pooled into one train. 'sleep' pools
          the 3 interleaved sleep blocks (~30 min); 'wheel'/'midSquare' pool their
          2 task sessions (~95-100 min each).
      sessions : list[str] or str
          Exact session ID(s), e.g. ['ec015.044', 'ec015.046'] or 'ec015.042'.
          Overrides `stage`. Use this to keep the sleep blocks apart -- 042 follows
          midSquare, 044 & 046 follow wheel -- so pooling doesn't wash out any
          task-specific reorganisation. Valid IDs: ec015.041 .. ec015.047.

    Every call returns the SAME CA1+EC neuron set in the SAME order (fixed by the
    cell table), so learned A matrices from different stages/sessions are directly
    comparable element by element. Thin wrapper over load_hc3_topdir_stage.
    """
    if sessions is not None:
        if isinstance(sessions, str):
            sessions = [sessions]
        unknown = [s for s in sessions if s not in EC015_03_DURATIONS]
        if unknown:
            raise ValueError(f"unknown session IDs {unknown}; valid: {list(EC015_03_DURATIONS)}")
        tag = 'ec015.03_' + '+'.join(s.split('.')[-1] for s in sessions)
    else:
        if stage not in EC015_03_STAGE_SESSIONS:
            raise ValueError(f"stage must be one of {list(EC015_03_STAGE_SESSIONS)}, got {stage!r}")
        sessions = EC015_03_STAGE_SESSIONS[stage]
        tag = f'ec015.03_{stage}'
    params = dict(
        topdir='ec015.03',
        sessions=sessions,
        session_durations=EC015_03_DURATIONS,
        regions=['CA1', 'EC2', 'EC3', 'EC5'],
        quality_filter=True,
        tag=tag,
    )
    params.update(overrides)
    return load_hc3_topdir_stage(downsample_factor=downsample_factor, gap=gap, step=step, **params)