"""Fig. 7B-C: CircuitGNN vs the fraction of hidden (unrecorded) PDCM neurons.

Run after `python -m paper.train pdcm`, `... pdcm_hide20` and `... pdcm_hide40`:

    python -m paper.experiments.hiding

Writes paper/figures/metrics_hiding.csv and paper/figures/hiding/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.config import DATASETS_DIR, FIGURES_DIR
from paper.data import pdcm_hidden_indices
from paper.evaluate import evaluate
from paper.figures.plotting import embed, plot_pdcm_matrix

HIDE_PCTS = [0, 20, 40]
HIDE_SEED = 0
OUT_DIR = FIGURES_DIR / "hiding"

MAIN_COLOR = "#1b5e20"      
EXTRA_COLOR = "#66bb6a"     
MARKER_SIZE = 5


def plot_metric(vals, col, ylabel, fname, ylim, extras=()):
    """extras: (column, linestyle, label) drawn alongside the main curve."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(HIDE_PCTS, vals[:, col], marker="o", ms=MARKER_SIZE, ls="-", color=MAIN_COLOR,
            mfc=MAIN_COLOR, mec=MAIN_COLOR, lw=1.5, label=ylabel)
    for ecol, style, label in extras:
        ax.plot(HIDE_PCTS, vals[:, ecol], marker="o", ms=MARKER_SIZE, ls=style, color=EXTRA_COLOR,
                mfc=EXTRA_COLOR, mec=EXTRA_COLOR, lw=1.2, label=label)

    ax.set_xticks(HIDE_PCTS)
    ax.set_xticklabels([str(v) for v in HIDE_PCTS])
    ax.set_xlabel("hidden neurons (%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if extras:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"Saved {OUT_DIR / fname}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pop_labels = np.load(DATASETS_DIR / "pdcm" / "pop_labels.npy")
    true_conn = np.load(DATASETS_DIR / "pdcm" / "weight_matrix.npy")
    true_conn = true_conn / np.max(np.abs(true_conn))
    n_full = len(pop_labels)

    rows = []
    for pct in HIDE_PCTS:
        kept, hidden = pdcm_hidden_indices(n_full, pct / 100, HIDE_SEED)
        c, _, fits = evaluate("pdcm" if pct == 0 else f"pdcm_hide{pct}")
        res = fits["mean_consistent"]
        rows.append({"hide_frac": pct / 100, "hide_pct": pct, "signed_f1": res.signed_f1,
                     "nie": res.nie, "signed_precision": res.signed_precision,
                     "signed_recall": res.signed_recall})

        plot_pdcm_matrix(true_conn, pop_labels, OUT_DIR / f"ground_truth_hide{pct}.pdf",
                         f"True Connectivity  ({pct}% hidden)", vmin=-1.0, vmax=0.5, blackout=hidden)
        plot_pdcm_matrix(embed(c.A_consistent, kept, n_full), pop_labels,
                         OUT_DIR / f"mean_consistent_hide{pct}.pdf",
                         f"GNN Connectivity Matrix  ({pct}% hidden, {len(kept)} of {n_full} observed)",
                         vmin=-0.010, vmax=0.005, blackout=hidden)

    df = pd.DataFrame(rows)
    df.to_csv(FIGURES_DIR / "metrics_hiding.csv", index=False)
    vals = df[["signed_f1", "nie", "signed_precision", "signed_recall"]].to_numpy()
    plot_metric(vals, 0, "F1 score", "hiding_f1.pdf", (0, 1.0),
                extras=[(2, "--", "precision"), (3, ":", "recall")])
    plot_metric(vals, 1, "NIE", "hiding_nie.pdf", (0, 1.0))


if __name__ == "__main__":
    main()
