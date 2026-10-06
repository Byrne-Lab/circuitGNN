"""sCCG, dSTTC, GLMCC and GLMPP baselines via spycon (Donner et al., 2024).

Runs every parameter setting listed in the Methods and writes each thresholded weight
matrix to datasets/<dataset>/baseline/spycon_<setting>.npy ([pre, post]):

    python -m paper.baselines.run_spycon

Then score them with python -m paper.baselines.compare.
"""
import itertools
import time

import numpy as np
from spycon.coninf import GLMCC, GLMPP, Smoothed_CCG, directed_STTC

from circuit_gnn.plotting import plot_connectivity_matrix
from paper import data
from paper.config import DATASETS, DATASETS_DIR

DATASET_NAMES = ['OC_1', 'pdcm']
BIN_MS = 1.0   # spike times are taken from 1 ms bins

# Axes measured in seconds
_TIME_AXES = {'binsize', 'syn_tau', 'syn_delay', 'delta_t', 'tau',
              'gauss_std', 'ccg_tau', 'syn_window'}


def binned_to_spikes(binned_spikes, bin_size_sec):
    # Convert (channels × time) binned spike matrix to (times, ids) arrays.
    n_multi = int(np.sum(binned_spikes > 1))
    if n_multi > 0:
        print(f"  WARNING: {n_multi} bins hold >1 spike (max {int(binned_spikes.max())}); "
              "their spikes share one time stamp")

    times_list, ids_list = [], []
    for ch in range(binned_spikes.shape[0]):
        bins = np.where(binned_spikes[ch] > 0)[0]
        for b in bins:
            count = int(binned_spikes[ch, b])
            times_list.extend([b * bin_size_sec] * count)
            ids_list.extend([ch] * count)
    times = np.array(times_list, dtype=np.float64)
    ids = np.array(ids_list, dtype=np.int64)
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


def _fmt(k, v):
    if isinstance(v, bool):           # before the numeric branch: bool is an int
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


def methods():
    """Every spycon setting in the Methods, keyed by output-file tag."""
    out = {
        'sccg_default': Smoothed_CCG(),
        'dsttc_default': directed_STTC(),
        'glmcc_default': GLMCC(),
        'glmpp_default': GLMPP(),
    }
    out.update(factorial(Smoothed_CCG, 'sccg',
                         binsize=[1e-3, 2e-3, 5e-3],
                         hf=[0.4, 0.6, 0.8]))
    out.update(ofat(Smoothed_CCG, 'sccg',
                    syn_window=[(0.5e-3, 4e-3), (1e-3, 10e-3), (2e-3, 20e-3)],
                    alpha=[0.01, 0.05],
                    deconv_ccg=[True]))
    out.update(ofat(directed_STTC, 'dsttc',
                    delta_t=[3e-3, 5e-3, 7e-3, 10e-3, 20e-3, 30e-3],
                    num_surrogates=[20, 100],
                    alpha=[1e-2, 5e-2]))
    out.update(factorial(directed_STTC, 'dsttc',
                         jitter=[True],
                         jitter_factor=[3.0, 7.0, 15.0]))
    out.update(factorial(GLMCC, 'glmcc',
                         syn_delay=[1e-3, 3e-3, 5e-3, 10e-3],
                         beta=[500, 4000],
                         tau=[[1e-3, 1e-3], [4e-3, 4e-3]]))
    out.update(factorial(GLMCC, 'glmcc',
                         syn_delay=[10e-3],
                         ccg_tau=[25e-3, 100e-3],
                         deconv_ccg=[False, True]))
    return out


def main():
    for name in DATASET_NAMES:
        save_dir = DATASETS_DIR / name / "baseline"
        save_dir.mkdir(parents=True, exist_ok=True)

        binned_spikes, _, neuron_names = data.load(
            name, downsample_factor=int(BIN_MS / DATASETS[name].native_ms))
        num_channels, num_time = binned_spikes.shape
        print(f'\n=== {name}: {num_channels} channels, {num_time} time points ===')

        times, ids = binned_to_spikes(binned_spikes, BIN_MS / 1000.0)
        print(f'  Total spikes: {len(times)},  bin size: {BIN_MS:.1f} ms')

        settings = methods()
        print(f'  {len(settings)} variants queued')
        for tag, method in settings.items():
            start = time.time()
            result = method.infer_connectivity(times, ids)
            elapsed = time.time() - start

            weight_mat, score_mat = result_to_matrices(result, num_channels)
            threshold = float(result.threshold)
            weight_thr = np.where(score_mat > threshold, weight_mat, 0.0)
            print(f"  {tag} time: {elapsed / 60:.2f} min")
            print(f"    threshold ({result.method}) = {threshold:.4g}, "
                  f"{int(np.count_nonzero(weight_thr))} edges")

            np.save(save_dir / f"spycon_{tag}.npy", weight_thr)
            plot_connectivity_matrix(weight_thr, neuron_names, f"Spycon {tag.upper()}",
                                     save_dir=save_dir, plot_val=False)


if __name__ == "__main__":
    main()
