"""Consensus connectivity for one of the paper's datasets, and its fit to ground truth.

Run after paper.train:

    python -m paper.evaluate OC_1

Writes results/<dataset>/mean.npy and mean_consistent.npy (with heatmaps) and prints the
metrics of both against ground truth.
"""
import argparse

import numpy as np

from circuit_gnn import calculate_fit, consensus, load_runs
from circuit_gnn.consensus import ABLATION_THRESHOLD, SIGN_BIN_MS
from paper import data
from paper.config import DATASETS, RESULTS_DIR
from paper.train import plotter


def gnn_consensus(name, ablation_threshold=ABLATION_THRESHOLD):
    """``(Consensus, true_conn, neuron_names)`` from the replicates in results/<name>/."""
    spec = DATASETS[name]
    sign_spikes, true_conn, neuron_names = data.load(
        name, downsample_factor=round(SIGN_BIN_MS / spec.native_ms))
    matrices, ablation, run_ids = load_runs(RESULTS_DIR / name)
    c = consensus(matrices, ablation, sign_spikes, run_ids=run_ids,
                  ablation_threshold=ablation_threshold, round_decimals=spec.round_decimals)
    passed = [r for r, d in zip(run_ids, ablation) if d <= -ablation_threshold]
    failed = [r for r in run_ids if r not in passed]
    fallback = "" if passed else f"; none passed, so all {len(run_ids)} were kept"
    print(f"\n{name}: {len(passed)} of {len(run_ids)} replicates passed the ablation test "
          f"(threshold {ablation_threshold}); failed {failed}{fallback}; sign-flipped {c.flipped}")
    return c, true_conn, neuron_names


def evaluate(name, ablation_threshold=ABLATION_THRESHOLD):
    """Save and score ``gnn_consensus``. Returns ``(Consensus, true_conn, fits)``, where fits
    maps 'mean' and 'mean_consistent' to a FitResult."""
    c, true_conn, neuron_names = gnn_consensus(name, ablation_threshold)
    out_dir = RESULTS_DIR / name
    np.save(out_dir / "mean.npy", c.A)
    np.save(out_dir / "mean_consistent.npy", c.A_consistent)

    plot = plotter(name, neuron_names, out_dir)
    kw = {"vmin": -0.015, "vmax": 0.010} if DATASETS[name].kind == "small" else {}
    plot(c.A, "mean", **kw)
    plot(c.A_consistent, "mean_consistent", **kw)

    fits = {"mean": calculate_fit(true_conn, c.A),
            "mean_consistent": calculate_fit(true_conn, c.A_consistent)}
    for label, res in fits.items():
        print(f"{label}: Cosine={res.cosine:.2f}, Pearson={res.pearson:.2f}, "
              f"Spearman={res.spearman:.2f}, Overall Precision={res.overall_precision:.2f}, "
              f"Overall Recall={res.overall_recall:.2f}, Overall F1={res.overall_f1:.2f}, "
              f"Signed Precision={res.signed_precision:.2f}, Signed Recall={res.signed_recall:.2f}, "
              f"Signed F1={res.signed_f1:.2f}, AUROC={res.auroc:.2f}, NIE={res.nie:.2f}")
    return c, true_conn, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--ablation-threshold", type=float, default=ABLATION_THRESHOLD)
    args = parser.parse_args()
    evaluate(args.dataset, args.ablation_threshold)


if __name__ == "__main__":
    main()
