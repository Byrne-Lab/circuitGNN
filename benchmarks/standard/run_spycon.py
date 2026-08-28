# Spycon spike-connectivity inference methods.

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

import numpy as np
import os
import time
import itertools
import traceback
from circuit_gnn.data import *
from circuit_gnn.utils import *
from spycon.coninf import CoincidenceIndex, Smoothed_CCG, directed_STTC, GLMCC, GLMPP

DATASETS = ['allen']

data_map = {
    'OC_1':      load_OC_1_network,
    'OC_2':      load_OC_2_network,
    'control':   load_control_network,
    'arbitrary': load_arbitrary_network,
    'pdcm':      load_pdcm_network,
    'allen':     load_allen_network,
}

# native recording resolution, ms
native_res_ms = {'OC_1': 0.5, 'OC_2': 0.5, 'control': 0.5,
                 'arbitrary': 1.0, 'pdcm': 1.0, 'allen': 1.0}

loader_kwargs = {
    'allen': {'data_dir': 'datasets/allen/prep_793224716', 'max_time_s': 600},
}

BIN_MS = 1.0

ALLEN_REGIONS = ['DG', 'CA3', 'CA1', 'SUB']


def binned_to_spikes(binned_spikes, bin_size_sec):
    # Convert (channels × time) binned spike matrix to (times, ids) arrays.

    # Check for bins with multiple spikes => will become duplicates
    n_multi = int(np.sum(binned_spikes > 1))
    if n_multi > 0:
        print(f"  WARNING: {n_multi} bins hold >1 spike (max {int(binned_spikes.max())}); ")
        
    times_list, ids_list = [], []
    for ch in range(binned_spikes.shape[0]):
        bins = np.where(binned_spikes[ch] > 0)[0]
        for b in bins:
            count = int(binned_spikes[ch, b])
            times_list.extend([b * bin_size_sec] * count)
            ids_list.extend([ch] * count)
    times = np.array(times_list, dtype=np.float64)
    ids   = np.array(ids_list,   dtype=np.int64)
    order = np.argsort(times, kind='stable')
    return times[order], ids[order]


def result_to_matrices(result, num_channels):
    weight_mat = np.zeros((num_channels, num_channels), dtype=np.float64)
    score_mat = np.zeros((num_channels, num_channels), dtype=np.float64)
    weights = np.asarray(result.all_weights).reshape(-1)
    nodes = np.asarray(result.nodes).astype(int)
    assert nodes.min() >= 0 and nodes.max() < num_channels, \
        f"unit ID out of range [0, {num_channels}): {nodes.min()}..{nodes.max()}"
    for k, (pre, post, stat) in enumerate(result.stats):
        i, j = int(pre), int(post)
        weight_mat[i, j] = weights[k]
        score_mat[i, j] = stat
    return weight_mat, score_mat


def threshold_weights(weight_mat, score_mat, threshold):
    return np.where(score_mat > threshold, weight_mat, 0.0)


def plot_matrix(data, matrix, neuron_names, title, save_dir):
    if data == 'allen':
        plot_region_matrix(matrix, [n.split('_')[0] for n in neuron_names],
                           ALLEN_REGIONS, method_name=title, save_dir=save_dir)
    else:
        plot_connectivity_matrix(matrix, neuron_names, title,
                                 save_dir=save_dir, plot_val=False)


# Axes measured in seconds
_TIME_AXES = {'binsize', 'syn_tau', 'syn_delay', 'delta_t', 'tau',
              'gauss_std', 'ccg_tau', 'syn_window'}


def _fmt(k, v):
    if isinstance(v, bool):          # before the numeric branch: bool is an int
        return str(v)
    if isinstance(v, (tuple, list)):  # e.g. syn_window=(0.8e-3, 5.8e-3)
        return "-".join(_fmt(k, x) for x in v)
    return f"{v * 1e3:g}ms" if k in _TIME_AXES else f"{v:g}"


def factorial(cls, prefix, **axes):
    # full Cartesian grid over the given parameter axes
    names = list(axes)
    out = {}
    for combo in itertools.product(*(axes[k] for k in names)):
        params = dict(zip(names, combo))
        tag = "_".join(f"{k}{_fmt(k, v)}" for k, v in params.items())
        out[f"{prefix}_{tag}"] = cls(params)
    return out


def ofat(cls, prefix, **axes):
    # one-factor-at-a-time
    out = {}
    for key, values in axes.items():
        for v in values:
            out[f"{prefix}_{key}{_fmt(key, v)}"] = cls({key: v})
    return out


for data in DATASETS:

    # 1 ms bins to accurately get spike times
    downsample_factor = int(BIN_MS / native_res_ms[data])
    bin_size_sec = BIN_MS / 1000.0

    save_dir = f"datasets/{data}/baseline"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](
        downsample_factor=downsample_factor, **loader_kwargs.get(data, {}))
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    times, ids = binned_to_spikes(binned_spikes, bin_size_sec)
    print(f'  Total spikes: {len(times)},  bin size: {bin_size_sec*1e3:.1f} ms')

    methods = {
        # 
        # 'ci_default':    CoincidenceIndex(),
        # 'sccg_default':  Smoothed_CCG(),
        # 'dsttc_default': directed_STTC(),
        # 'glmcc_default': GLMCC(),
        # 'glmpp_default': GLMPP(),
    }
    
    """
    methods.update(factorial(Smoothed_CCG, 'sccg',
        binsize=[1e-3, 2e-3, 5e-3],
        hf=[0.4, 0.6, 0.8]))
    methods.update(ofat(Smoothed_CCG, 'sccg',
        syn_window=[(0.5e-3, 4e-3), (1e-3, 10e-3), (2e-3, 20e-3)],
        alpha=[0.01, 0.05],
        deconv_ccg=[True]))
    
    methods.update(ofat(directed_STTC, 'dsttc',
        delta_t=[3e-3, 5e-3, 7e-3, 10e-3, 20e-3, 30e-3],
        num_surrogates=[20, 100],
        alpha=[1e-2, 5e-2]))
    methods.update(factorial(directed_STTC, 'dsttc',
        jitter=[True],
        jitter_factor=[3.0, 7.0, 15.0]))
    """
    methods.update(factorial(GLMCC, 'glmcc',
        syn_delay=[1e-3, 3e-3, 5e-3, 10e-3],
        beta=[500, 4000],
        tau=[[1e-3, 1e-3], [4e-3, 4e-3]]))
    methods.update(factorial(GLMCC, 'glmcc',
        syn_delay=[10e-3],
        ccg_tau=[25e-3, 100e-3],
        deconv_ccg=[False, True]))
    

    print(f'  {len(methods)} variants queued')
    for name, method in methods.items():
        out_npy = f"{save_dir}/spycon_{name}.npy"

        start = time.time()
        result = method.infer_connectivity(times, ids)
        elapsed = time.time() - start

        weight_mat, score_mat = result_to_matrices(result, num_channels)
        threshold = float(result.threshold)
        weight_thr = threshold_weights(weight_mat, score_mat, threshold)
        n_edges = int(np.count_nonzero(weight_thr))
        print(f"  {name} time: {elapsed / 60:.2f} min")
        print(f"    threshold ({result.method}) = {threshold:.4g}, {n_edges} edges")

        np.save(out_npy, weight_thr)
        plot_matrix(data, weight_thr, neuron_names, f"Spycon {name.upper()}", save_dir)