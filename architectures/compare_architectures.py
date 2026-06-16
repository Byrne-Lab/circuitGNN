"""Compare trained architecture results for circuitGNN.

Run from repo root after running run_architectures.py:
    python architectures/compare_architectures.py --datasets yuto
    python architectures/compare_architectures.py --datasets pdcm --architectures baseline dales
    python architectures/compare_architectures.py --datasets yuto --ablation_threshold 0.10

Prints a table and saves a CSV (architecture_comparison_<dataset>_thr<threshold>.csv) with all
metrics (cosine, pearson, spearman, unsigned/signed F1, unsigned/signed MCC, AUROC, NIE) for both
the mean and the consistent-mean matrix of each architecture.
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

ARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARCH_DIR.parent
sys.path.insert(0, str(ARCH_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.data import (load_yuto_network, load_pdcm_network, load_arbitrary_network,
                               load_small_circuit_A_network, load_small_circuit_B_network)
from circuit_gnn.utils import (calculate_fit, align_matrices,
                                get_consistent_connections, get_top_k, sign_inference,
                                plot_connectivity_matrix, plot_pdcm_matrix,
                                plot_metric_verification)

RESULTS_BASE = ARCH_DIR / "results"

DATASET_REGISTRY = {
    "yuto":            {"loader": load_yuto_network,            "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1}},
    "pdcm":            {"loader": load_pdcm_network,            "loader_kwargs": {"downsample_factor": 2}},
    "arbitrary":       {"loader": load_arbitrary_network,       "loader_kwargs": {"downsample_factor": 10}},
    "small_circuit_A": {"loader": load_small_circuit_A_network, "loader_kwargs": {"downsample_factor": 20}},
    "small_circuit_B": {"loader": load_small_circuit_B_network, "loader_kwargs": {"downsample_factor": 20}},
}

ALL_ARCHITECTURES = [
    "baseline", "dales", "indirect_filter",
    "skip_residual", "edge_cond", "split_path", "multi_step", "learned_scale", "hadamard_mod", "diff",
    "softmax", "per_neuron_self", "gated", "adaptive", "relu_message",
]


def load_architecture_results(
    dataset: str,
    architecture: str,
    ablation_threshold: float = 0.20,
    excitatory: bool = True,
):
    arch_dir = RESULTS_BASE / dataset / architecture
    csv_path = arch_dir / "ablation.csv"

    if not csv_path.exists():
        return None, None, 0, 0

    df = pd.read_csv(csv_path)
    total = len(df)
    passed_ids = df[df["difference"].abs() >= ablation_threshold]["run_id"].tolist()

    matrices = []
    for run_id in passed_ids:
        p = arch_dir / f"gnn_{run_id}.npy"
        if p.exists():
            matrices.append(np.load(p))

    if not matrices:
        return None, None, 0, total

    matrices = align_matrices(matrices, excitatory=excitatory)
    stacked = np.stack(matrices)
    mean_A = np.round(stacked.mean(axis=0), 3)
    consistency_mask = get_consistent_connections(stacked, threshold=0.8)
    consistent_mean_A = mean_A * consistency_mask
    return mean_A, consistent_mean_A, len(matrices), total


def main():
    parser = argparse.ArgumentParser(description="Compare circuitGNN architecture results")
    parser.add_argument("--datasets", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--architectures", nargs="+", default=ALL_ARCHITECTURES,
                        help="Architectures to compare (default: all)")
    parser.add_argument("--ablation_threshold", type=float, default=0.20,
                        help="Min |difference| to keep a run (default: 0.20)")
    parser.add_argument("--excitatory", action="store_true", default=True,
                        help="Assume excitatory connections dominate for sign alignment")
    args = parser.parse_args()

    reg = DATASET_REGISTRY[args.dataset]
    _, true_conn, neuron_names = reg["loader"](**reg["loader_kwargs"])

    # Plot function: pdcm uses population-boundary format, others use standard
    if args.dataset == "pdcm":
        pop_labels = np.load("datasets/pdcm/pop_labels.npy")
        def do_plot(matrix, method_name, save_dir):
            plot_pdcm_matrix(matrix, pop_labels, method_name=method_name, save_dir=save_dir)
    else:
        def do_plot(matrix, method_name, save_dir):
            plot_connectivity_matrix(matrix, neuron_names=neuron_names, method_name=method_name, save_dir=save_dir, plot_val=True)

    rows = []
    for arch in args.architectures:
        mean_A, consistent_mean_A, n_passed, n_total = load_architecture_results(
            args.dataset, arch, args.ablation_threshold, args.excitatory
        )

        if mean_A is None:
            rows.append({
                "architecture": arch, "n_passed": 0, "n_total": n_total,
                "cosine": "-", "pearson": "-", "spearman": "-",
                "f1_unsign": "-", "f1_signed": "-",
                "mcc_unsign": "-", "mcc_signed": "-", "auroc": "-", "nie": "-",
                "cosine_cons": "-", "pearson_cons": "-", "spearman_cons": "-",
                "f1_unsign_cons": "-", "f1_signed_cons": "-",
                "mcc_unsign_cons": "-", "mcc_signed_cons": "-", "auroc_cons": "-", "nie_cons": "-",
            })
            continue

        arch_dir = RESULTS_BASE / args.dataset / arch
        do_plot(mean_A, "mean", str(arch_dir))
        do_plot(consistent_mean_A, "mean_consistent", str(arch_dir))
        plot_metric_verification(true_conn, consistent_mean_A, str(arch_dir),
                                 method_name=f"{arch} — consistent mean")

        res   = calculate_fit(true_conn, mean_A, plot=False)
        res_c = calculate_fit(true_conn, consistent_mean_A, plot=False)
        rows.append({
            "architecture":    arch,
            "n_passed":        n_passed,
            "n_total":         n_total,
            "cosine":          f"{res.cosine:.3f}",
            "pearson":         f"{res.pearson:.3f}",
            "spearman":        f"{res.spearman:.3f}",
            "f1_unsign":       f"{res.overall_f1:.3f}",
            "f1_signed":       f"{res.signed_f1:.3f}",
            "mcc_unsign":      f"{res.mcc_unsigned:.3f}",
            "mcc_signed":      f"{res.mcc_signed:.3f}",
            "auroc":           f"{res.auroc:.3f}",
            "nie":             f"{res.nie:.3f}",
            "cosine_cons":     f"{res_c.cosine:.3f}",
            "pearson_cons":    f"{res_c.pearson:.3f}",
            "spearman_cons":   f"{res_c.spearman:.3f}",
            "f1_unsign_cons":  f"{res_c.overall_f1:.3f}",
            "f1_signed_cons":  f"{res_c.signed_f1:.3f}",
            "mcc_unsign_cons": f"{res_c.mcc_unsigned:.3f}",
            "mcc_signed_cons": f"{res_c.mcc_signed:.3f}",
            "auroc_cons":      f"{res_c.auroc:.3f}",
            "nie_cons":        f"{res_c.nie:.3f}",
        })

    df = pd.DataFrame(rows).set_index("architecture")
    print(f"\n=== {args.dataset} — architecture comparison (ablation threshold={args.ablation_threshold}) ===")
    print(df.to_string())
    print()

    csv_path = ARCH_DIR / f"architecture_{args.dataset}.csv"
    df.to_csv(csv_path)
    print(f"Saved full metrics table to {csv_path}")


if __name__ == "__main__":
    main()
