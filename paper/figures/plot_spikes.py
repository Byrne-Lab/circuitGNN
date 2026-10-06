"""Fig. 2 and 4A rasters: a dataset's spike train over a time window, in seconds.

    python -m paper.figures.plot_spikes

Writes paper/figures/<dataset>_raster.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np

from paper import data
from paper.config import DATASETS, FIGURES_DIR


def plot_dataset_raster(dataset, bin_ms=1.0, save_path=None, t_start=None, t_end=None):
    native_ms = DATASETS[dataset].native_ms
    downsample_factor = max(1, round(bin_ms / native_ms))
    spikes, _, neuron_names = data.load(dataset, downsample_factor=downsample_factor)
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
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f'{dataset}_raster.pdf'
    plt.savefig(save_path)
    plt.close()
    print(f"saved raster -> {save_path}")


def main():
    for name in ('small_circuit_A', 'small_circuit_B', 'arbitrary', 'OC_1'):
        plot_dataset_raster(name, t_start=0, t_end=240)


if __name__ == '__main__':
    main()
