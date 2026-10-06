"""Fig. 7A: CircuitGNN vs recording duration and network size (PDCM, 149 and 767 neurons).

Run after `python -m paper.train pdcm_<size>_<duration>s` for every size and duration:

    python -m paper.experiments.scaling

Writes paper/figures/metrics_scaling.csv and paper/figures/scaling/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from paper.config import DATASETS_DIR, FIGURES_DIR
from paper.evaluate import evaluate
from paper.figures.plotting import plot_pdcm_matrix

SIZES = ["0p002", "0p010"]
DURATIONS_S = [60, 120, 240, 480]
MATRIX_CLIM = {"0p002": (-0.010, 0.005), "0p010": (-0.002, 0.001)}
OUT_DIR = FIGURES_DIR / "scaling"

CURVE_COLORS = ["#1b5e20", "#66bb6a"]
MARKER_SIZE = 5


def plot_metric(curves, col, ylabel, fname, ylim, extras=()):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CURVE_COLORS[i % len(CURVE_COLORS)] for i in range(len(curves))]
    for c, (label, vals) in zip(colors, curves.items()):
        ax.plot(DURATIONS_S, vals[:, col], marker="o", ms=MARKER_SIZE, ls="-",
                color=c, mfc=c, mec=c, label=label, lw=1.5)
        for ecol, style, _ in extras:
            ax.plot(DURATIONS_S, vals[:, ecol], marker="o", ms=MARKER_SIZE, ls=style,
                    color=c, mfc=c, mec=c, lw=1.2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(DURATIONS_S)
    ax.set_xticklabels([str(d) for d in DURATIONS_S])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("recording duration (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    size_legend = ax.legend(fontsize=8, loc="lower right")
    if extras:
        styles = [("-", ylabel)] + [(s, lab) for _, s, lab in extras]
        ax.legend(handles=[Line2D([], [], color="0.3", ls=s, label=lab) for s, lab in styles],
                  fontsize=8, loc="upper left")
        ax.add_artist(size_legend)      # ax.legend() replaces; re-add the size one
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"Saved {OUT_DIR / fname}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for size in SIZES:
        data_dir = DATASETS_DIR / f"pdcm_{size}"
        pop_labels = np.load(data_dir / "pop_labels.npy")
        label = f"N = {len(pop_labels)}"
        vmin, vmax = MATRIX_CLIM[size]

        vals = []
        for dur_s in DURATIONS_S:
            c, _, fits = evaluate(f"pdcm_{size}_{dur_s}s")
            res = fits["mean_consistent"]
            vals.append([res.signed_f1, res.nie, res.signed_precision, res.signed_recall])
            rows.append({"size": size, "duration_s": dur_s, "signed_f1": res.signed_f1,
                         "nie": res.nie, "signed_precision": res.signed_precision,
                         "signed_recall": res.signed_recall})
            plot_pdcm_matrix(c.A_consistent, pop_labels,
                             OUT_DIR / f"mean_consistent_{size}_{dur_s}s.pdf",
                             f"GNN Connectivity Matrix  ({label}, {dur_s} s)", vmin=vmin, vmax=vmax)
        curves[label] = np.array(vals)

        true_conn = np.load(data_dir / "weight_matrix.npy")
        plot_pdcm_matrix(true_conn / np.max(np.abs(true_conn)), pop_labels,
                         OUT_DIR / f"ground_truth_{size}.pdf", f"True Connectivity  ({label})",
                         vmin=-1, vmax=0.5)

    pd.DataFrame(rows).to_csv(FIGURES_DIR / "metrics_scaling.csv", index=False)
    plot_metric(curves, 0, "F1 score", "scaling_f1.pdf", (0, 1.0),
                extras=[(2, "--", "precision"), (3, ":", "recall")])
    plot_metric(curves, 1, "NIE", "scaling_nie.pdf", (0, 1.0))


if __name__ == "__main__":
    main()
