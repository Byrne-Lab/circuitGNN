import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import find_peaks
import pickle
import pandas as pd
from functools import partial
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def load_cpg(downsample_factor = 10, show=True, quiet=False):
    # aplysia cpg
    data_loc = 'data/cpg.mat'
    binned_spikes = sio.loadmat(data_loc)['spike_times_binned'] # load cpg spikes (binned at 0.2 ms resolutions)
    time = sio.loadmat(data_loc)['time'][0,:]
    neuron_names = ["CBI2","B4","B8","B20","B30","B31a","B34","B40", "B51a","B52","B63","B64a","B65"]

    f, ax = plt.subplots(figsize=(20, 9)) 
    for i in range(binned_spikes.shape[1]):
        plt.plot(time, i+1.5-binned_spikes[:,i], label='Neuron = %i'%(i+1)) 
    plt.xlabel('Time (s)') 
    plt.ylabel('Neuron') 
    plt.gca().invert_yaxis() 
    plt.yticks(np.arange(binned_spikes.shape[1])+1,neuron_names) 
    plt.title('CPG Raster') 
    if show:
        plt.show()

    binned_spikes = binned_spikes.T

    downsample_factor = downsample_factor
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)

    if not quiet:
        print(binned_spikes.shape)
        print("Binned spikes possibilities:", np.unique(binned_spikes))
        print("Bin resolution (s): ", 0.0002 * downsample_factor)

    # True connectivity
    true_conn = sio.loadmat(data_loc)['C']

    return binned_spikes, true_conn, neuron_names


def load_arbitrary_network(downsample_factor = 10):
    data_loc = 'data/arb.out'
    time_max    = 300
    num_neurons = 11
    # bin_width   = 35
    bin_width = downsample_factor

    SNNAP_output = []
    with io.open(data_loc, mode="r", encoding="utf-8") as f:
        for line in f:
            SNNAP_output.append([float(i) for i in line.split()])
            if SNNAP_output[-1][0]>=time_max:
                break

    # CONVERT TO ARRAY
    SNNAP_output_array = np.asarray(SNNAP_output)

    # TIME+VOLTAGES
    time = SNNAP_output_array[:,0]
    time_step = time[1]-time[0]
    volts = SNNAP_output_array[:,1:(num_neurons+1)]
    
    binned_spikes = np.zeros(volts.shape)
    for ii in range(volts.shape[1]):
        peaks = find_peaks(volts[:,ii], height=0)
        binned_spikes[peaks[0],ii] = 1

    # FURTHER BIN SPIKE TIMES
    num_bins = np.ceil(time_max/(bin_width*0.001))
    range_max = num_bins*(bin_width*0.001)
    further_binned_spikes = np.zeros((int(num_bins),volts.shape[1]))
    for ii in range(volts.shape[1]):
        further_binned_spikes[:,ii], time_further_binned = np.histogram(time[binned_spikes[:,ii].astype(bool)],
                        bins=np.arange(0,range_max+((bin_width*0.001)/2),(bin_width*0.001)))

    # CAP BINS TO 1 SPIKE
    further_binned_spikes[further_binned_spikes>0]=1

    # PLOT further_binned_spikes
    f, ax = plt.subplots(figsize=(11, 9))
    for ii in range(further_binned_spikes.shape[1]):
        plt.plot(time_further_binned[:-1]+((bin_width*0.001)/2), ii+0.5+further_binned_spikes[:,ii], label='Neuron = %i'%(ii+1))
    plt.xlim((100,105))
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron')
    plt.yticks(np.arange(11)+1)
    plt.show()
    
    binned_spikes = further_binned_spikes.T
    print(binned_spikes.shape) # (11, 8572)

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

    return binned_spikes, true_conn, np.arange(num_neurons)

def apply_refractory_period(spike_train, ref_len=19):
    processed_train = spike_train.copy()
    i = 0
    while i < len(processed_train):
        if processed_train[i] == 1:
            ref_end = i + 1 + ref_len
            processed_train[i+1 : ref_end] = 0
            i = ref_end
        else:
            i += 1
    return processed_train

def load_snnap(snnap_output_loc, downsample_factor=10):
    df = pd.read_csv(snnap_output_loc, sep='\t', header=None, skipinitialspace=True)
    voltage_data = df.iloc[:, 1:4]
    voltage_array_original = voltage_data.values
    neuron_voltage_array = voltage_array_original.T

    binned_spikes = (neuron_voltage_array >= 0).astype(int)
    binned_spikes = np.apply_along_axis(func1d=apply_refractory_period, axis=1, arr=binned_spikes, ref_len=9)
    print(binned_spikes.shape)
    first_spike_indices = np.argmax(binned_spikes, axis=1)
    print(first_spike_indices)

    # downsample
    binned_spikes = binned_spikes[:, :binned_spikes.shape[1] // downsample_factor * downsample_factor]
    binned_spikes = binned_spikes.reshape(binned_spikes.shape[0], -1, downsample_factor).sum(2)
    print(binned_spikes.shape)
    print("Binned spikes possibilities:", np.unique(binned_spikes))

    # plot
    spike_times = []
    for i in range(binned_spikes.shape[0]):
        times = np.where(binned_spikes[i, :] > 0)[0]
        spike_times.append(times)

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black') 
    plt.xlabel(f'Time Step (Binned by {downsample_factor})')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster')
    plt.show()

    return binned_spikes


def load_yuto_network(downsample_factor = 20, gap=0.20):
    # downsample_factor: 20 -> 10 ms bins, 10 -> 5 ms bins
    # 0.5 ms bins
    # 240 s total
    # 13 neurons

    data_loc = 'datasets/yuto/spikes.out'
    neuron_names = ['CBI2','B31a','B20', 'B30', 'B40', 'B34', 'B35', 'B63', 'B4', 'B8', 'B51a', 'B64a', 'B52']

    # Load true connectivity
    neuron_names = ['CBI2','B31a','B20', 'B30', 'B40', 'B34', 'B35', 'B63', 'B4', 'B8', 'B51a', 'B64a', 'B52']
    true_conn = np.load(f"datasets/yuto/true_connectivity_{gap:.2f}.npy")
    true_conn = np.round(true_conn, 3)

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
                cbar_kws={'label': 'Connection Strength (shifted mV)'},
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
    spike_times = []
    for i in range(binned_spikes.shape[0]):
        times = np.where(binned_spikes[i, :] > 0)[0]
        spike_times.append(times)

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times, orientation='horizontal', linelengths=0.8, color='black') 
    plt.xlabel(f'Time Step (Binned by {downsample_factor})')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster')
    plt.close()

    return binned_spikes, true_conn, neuron_names


def load_small_circuit(circuit_number, downsample_factor=10):

    if circuit_number == 1:
        num_neurons = 3
        neuron_names = ['A','B','C']
        true_conn = np.zeros((num_neurons,num_neurons))

        binned_spikes = load_snnap('data/SNAAP_cai_et_al/Figure5/networkA/networkA.smu.out', downsample_factor)
        true_conn[0,1] = -1
        true_conn[0,2] =  1
        return binned_spikes, true_conn, neuron_names
    
    elif circuit_number == 2:
        num_neurons = 3
        neuron_names = ['A','B','C']
        true_conn = np.zeros((num_neurons,num_neurons))
    
        binned_spikes = load_snnap('data/SNAAP_cai_et_al/Figure5/networkB/networkB.smu.out', downsample_factor)
        true_conn[0,1] =  1
        true_conn[0,2] =  1
        true_conn[2,1] =  1
        return binned_spikes, true_conn, neuron_names
    
    elif circuit_number == 3:
        num_neurons = 3
        neuron_names = ['A','B','C']
        true_conn = np.zeros((num_neurons,num_neurons))
    
        binned_spikes = load_snnap('data/SNAAP_cai_et_al/Figure5/networkC/networkC.smu.out', downsample_factor)
        true_conn[0,1] =  1
        true_conn[1,2] =  1

        return binned_spikes, true_conn, neuron_names
    
    else:
        raise ValueError("Invalid circuit number. Choose 1, 2, or 3.")
    
load_small_circuit_1 = partial(load_small_circuit, 1)
load_small_circuit_2 = partial(load_small_circuit, 2)
load_small_circuit_3 = partial(load_small_circuit, 3)