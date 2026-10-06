"""Fig. 5: rasters of the default OC_1 simulation and its two single-synapse ablations
(B63 -> B31, CBI-2 -> B63) side by side, first 40 s.

    python -m paper.figures.plot_ablation_spikes

Writes paper/figures/OC_1_ablation_rasters.pdf.
"""
import matplotlib.pyplot as plt
import pandas as pd

from paper.config import DATASETS_DIR, FIGURES_DIR
from paper.data import neuron_names_from_head, spike_indices

OC1 = DATASETS_DIR / 'OC_1'
RUNS = {
    'Default': OC1 / 'spikes.out',
    'B63 → B31': OC1 / 'OCx1_ablation_B63_to_B31' / 'smu' / 'BMP.smu.out',
    'CBI-2 → B63': OC1 / 'OCx1_ablation_CBI-2_to_B63' / 'smu' / 'BMP.smu.out',
}
T_MAX_S = 40
SAMPLE_S = 0.5e-3


def main():
    names = neuron_names_from_head(OC1 / 'spikes.head')
    fig, axes = plt.subplots(1, len(RUNS), figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, (title, path) in zip(axes, RUNS.items()):
        df = pd.read_csv(path, sep='\t', header=None, nrows=round(T_MAX_S / SAMPLE_S))
        t, volts = df[0].values, df.iloc[:, 1:-1].values.T   # rows end in a tab -> empty last column
        ax.eventplot([t[sp] for sp in spike_indices(volts)], linelengths=0.8, linewidths=0.5, color='black')
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.tick_params(labelleft=True, labelbottom=True)   # shared axes hide inner labels by default
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
    axes[0].set_xlim(0, T_MAX_S)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    fig.tight_layout()

    out = FIGURES_DIR / 'OC_1_ablation_rasters.pdf'
    fig.savefig(out)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
