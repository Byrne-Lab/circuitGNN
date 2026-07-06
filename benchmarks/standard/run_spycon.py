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
from functools import partial
from circuit_gnn.data import *
from circuit_gnn.utils import *
from spycon.coninf import CoincidenceIndex, Smoothed_CCG, directed_STTC, GLMCC, GLMPP

# hc3 is in-vivo (no ground truth); load_hc3_session needs session-specific
# kwargs but returns the same (binned_spikes, true_conn, neuron_names) shape and
# takes downsample_factor first, so partial slots it into the loader map.
load_hc3 = partial(
    load_hc3_session,
    tar_path='datasets/hc3/ec016.1016.tar.gz',
    topdir='ec016.58',
    cell_csv='datasets/hc3/hc3-cell.csv',
    regions=['CA3', 'DG', 'EC2', 'EC3'],
    quality_filter=True,
    max_time_s=2000,
)

data_map = {
    'yuto':      load_yuto_network,
    'arbitrary': load_arbitrary_network,
    'pdcm':      load_pdcm_network,
    'hc3':       load_hc3,
}

bin_size_map = {
    'yuto':      lambda f: 0.5e-3 * f,
    'arbitrary': lambda f: 1e-3 * f,
    'pdcm':      lambda f: f / 1000.0,
    'hc3':       lambda f: f / 1000.0,   # native resolution 1 ms
}


def binned_to_spikes(binned_spikes, bin_size_sec):
    # Convert (channels × time) binned spike matrix to (times, ids) arrays.
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


def result_to_matrices(result):
    # Each method returns BOTH graphs the paper describes (Donner et al. 2024):
    #   - weight graph W: result.all_weights, the graded connection strength.
    #   - score graph  S: result.stats[:, 2], the decision statistic. Its meaning
    #     is method-specific (|z| for CI/dSTTC/GLMCC/TE, -log p for sCCG, z for
    #     GLMPP); higher always means an edge is more probable.
    # all_weights is aligned row-for-row with stats (spycon's own set_threshold
    # masks all_weights with a stats-derived mask), and both span the full
    # directed graph (every ordered pair).
    n = len(result.nodes)
    node_idx = {int(node): i for i, node in enumerate(result.nodes)}
    weight_mat = np.zeros((n, n), dtype=np.float64)
    score_mat = np.zeros((n, n), dtype=np.float64)
    weights = np.asarray(result.all_weights).reshape(-1)
    for k, (pre, post, stat) in enumerate(result.stats):
        i, j = node_idx[int(pre)], node_idx[int(post)]
        weight_mat[i, j] = weights[k]
        score_mat[i, j] = stat
    return weight_mat, score_mat


def threshold_weights(weight_mat, score_mat, threshold):
    # Keep a weight only where its connectivity score exceeds the method's
    # significance threshold; everything else is treated as an unconnected pair.
    # The threshold is method-specific (result.threshold), derived from a
    # significance level alpha exactly as in the paper; pass `alpha` in a
    # method's params to sweep it (paper uses 0.05 / 0.01 / 0.005 / 0.001).
    return np.where(score_mat > threshold, weight_mat, 0.0)


def _ms(v):
    return f"{v * 1e3:g}ms"


def factorial(cls, prefix, **axes):
    # full Cartesian grid over the given parameter axes
    names = list(axes)
    out = {}
    for combo in itertools.product(*(axes[k] for k in names)):
        params = dict(zip(names, combo))
        tag = "_".join(f"{k}{_ms(v)}" for k, v in params.items())
        out[f"{prefix}_{tag}"] = cls(params)
    return out


def ofat(cls, prefix, **axes):
    # one-factor-at-a-time: vary each axis, holding the others at their defaults
    out = {}
    for key, values in axes.items():
        for v in values:
            out[f"{prefix}_{key}{_ms(v)}"] = cls({key: v})
    return out


for data in ['hc3']:

    # 1 ms bins to accurately get spike times
    if data == 'yuto':
        downsample_factor = 2  # 1 ms
    elif data == 'arbitrary':
        downsample_factor = 1  # 1 ms
    elif data == 'pdcm':
        downsample_factor = 1  # 1 ms
    elif data == 'hc3':
        downsample_factor = 1  # 1 ms (native resolution)

    bin_size_sec = bin_size_map[data](downsample_factor)

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    times, ids = binned_to_spikes(binned_spikes, bin_size_sec)
    print(f'  Total spikes: {len(times)},  bin size: {bin_size_sec*1e3:.1f} ms')

    # 1 ms is the resolution floor
    methods = {
        'ci_default':    CoincidenceIndex(),
        'sccg_default':  Smoothed_CCG(),
        'dsttc_default': directed_STTC(),
        'glmcc_default': GLMCC(),
        'glmpp_default': GLMPP(),
    }
    """
    methods.update(factorial(CoincidenceIndex, 'ci',
        binsize=[1e-3, 2e-3, 5e-3, 10e-3],
        syn_tau=[3e-3, 6e-3, 10e-3, 20e-3]))

    methods.update(ofat(Smoothed_CCG, 'sccg',
        binsize=[1e-3, 2e-3, 5e-3]))
    methods.update({f'sccg_hf{hf}': Smoothed_CCG({'hf': hf}) for hf in (0.4, 0.8)})

    methods.update(ofat(directed_STTC, 'dsttc',
        delta_t=[3e-3, 5e-3, 7e-3, 10e-3, 20e-3, 30e-3]))

    methods.update(factorial(GLMCC, 'glmcc',
        binsize=[1e-3],
        syn_delay=[1e-3, 3e-3, 5e-3, 10e-3]))

    methods.update(ofat(GLMPP, 'glmpp',
        tau=[2e-3, 5e-3, 10e-3, 20e-3]))
    """
    for name, method in methods.items():
        start = time.time()
        result = method.infer_connectivity(times, ids)
        elapsed = time.time() - start
        print(f"  {name} time: {elapsed / 60:.2f} min")

        weight_mat, score_mat = result_to_matrices(result)
        threshold = float(result.threshold)
        weight_thr = threshold_weights(weight_mat, score_mat, threshold)
        print(f"    threshold ({result.method}) = {threshold:.4g}")

        np.save(f"{save_dir}/spycon_{name}_W.npy", weight_mat)
        np.save(f"{save_dir}/spycon_{name}_S.npy", score_mat)
        np.save(f"{save_dir}/spycon_{name}_Wthr.npy", weight_thr)
        plot_connectivity_matrix(weight_mat, neuron_names, f"Spycon {name.upper()} W", save_dir=save_dir, plot_val=False)
        plot_connectivity_matrix(score_mat, neuron_names, f"Spycon {name.upper()} S", save_dir=save_dir, plot_val=False)
        plot_connectivity_matrix(weight_thr, neuron_names, f"Spycon {name.upper()} Wthr", save_dir=save_dir, plot_val=False)