"""Raster plot of a dataset's spike train, with the time axis in seconds."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from circuit_gnn.data import (
    load_arbitrary_network, load_yuto_network, load_pdcm_network,
    load_mdp_network, load_v1_v2_network, load_allen_network,
    load_small_circuit_A_network, load_small_circuit_B_network,
)

# dataset name -> (loader, native sample period in ms). seconds-per-bin =
# downsample_factor * native_ms / 1000.
DATASETS = {
    'small_circuit_A': (load_small_circuit_A_network, 0.5),
    'small_circuit_B': (load_small_circuit_B_network, 0.5),
    'arbitrary':       (load_arbitrary_network, 1.0),
    'yuto':            (load_yuto_network, 0.5),
    'pdcm':            (load_pdcm_network, 1.0),
    'mdp':             (load_mdp_network, 1.0),
    'v1_v2':           (load_v1_v2_network, 1.0),
    'allen':           (load_allen_network, 1.0),
}


def plot_dataset_raster(dataset, bin_ms=1.0, save_path=None, t_start=None, t_end=None):

    loader, native_ms = DATASETS[dataset]
    downsample_factor = max(1, round(bin_ms / native_ms))
    spikes, _, neuron_names = loader(downsample_factor=downsample_factor)
    bin_size_s = downsample_factor * native_ms / 1000.0
    n_neurons, n_bins = spikes.shape

    # clip the bin range to the requested [t_start, t_end] window (in seconds)
    lo = 0 if t_start is None else max(0, int(t_start / bin_size_s))
    hi = n_bins if t_end is None else min(n_bins, int(np.ceil(t_end / bin_size_s)))
    spike_times_s = [(lo + np.where(spikes[i, lo:hi] > 0)[0]) * bin_size_s for i in range(n_neurons)]

    plt.figure(figsize=(10, 5))
    plt.eventplot(spike_times_s, orientation='horizontal', linelengths=0.8, color='black')
    if n_neurons <= 40:
        plt.yticks(range(n_neurons), neuron_names)
        plt.gca().invert_yaxis()
        plt.ylabel('Neuron')
    else:
        plt.ylabel('Neuron index')
    plt.xlim(lo * bin_size_s, hi * bin_size_s)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.xlabel('Time (s)')
    plt.title('Spike Raster')
    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).resolve().parent / f'{dataset}_raster.pdf'
    plt.savefig(save_path)
    print(f"saved raster -> {save_path}")


if __name__ == '__main__':
    plot_dataset_raster('small_circuit_A', t_start=45, t_end=85)
    plot_dataset_raster('small_circuit_B', t_start=45, t_end=85)
    plot_dataset_raster('arbitrary',       t_start=45, t_end=65)
