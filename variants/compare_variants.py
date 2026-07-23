"""Compare trained variant results for circuitGNN.

Run from repo root after running run_variants.py:
    python variants/compare_variants.py --dataset X --variants Y
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # suppress plot windows when loading data

import numpy as np
import pandas as pd

# path setup 
VARIANTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VARIANTS_DIR.parent
sys.path.insert(0, str(VARIANTS_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.data import (load_yuto_network, load_pdcm_network, load_pdcm_large_network,
                              load_arbitrary_network,
                              load_small_circuit_A_network, load_small_circuit_B_network,
                              load_mdp_network, load_v1_v2_network)
from circuit_gnn.utils import (calculate_fit, align_matrices, robust_sign_inference,
                                get_consistent_connections, get_top_k, sign_inference,
                                plot_connectivity_matrix, plot_pdcm_matrix,
                                plot_mdp_matrix, plot_v1_v2_matrix,
                                plot_metric_verification)

RESULTS_BASE = VARIANTS_DIR / "results"

DATASET_REGISTRY = {
    "yuto":            {"loader": load_yuto_network,            "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1}},
    "arbitrary":       {"loader": load_arbitrary_network,       "loader_kwargs": {"downsample_factor": 10}},
    "pdcm":            {"loader": load_pdcm_network,            "loader_kwargs": {"downsample_factor": 2}},
    "pdcm_large":      {"loader": load_pdcm_large_network,      "loader_kwargs": {"downsample_factor": 6}},
    "mdp":             {"loader": load_mdp_network,             "loader_kwargs": {"downsample_factor": 6}},
    "v1_v2":           {"loader": load_v1_v2_network,           "loader_kwargs": {"downsample_factor": 6}},
    "small_circuit_A": {"loader": load_small_circuit_A_network, "loader_kwargs": {"downsample_factor": 20}},
    "small_circuit_B": {"loader": load_small_circuit_B_network, "loader_kwargs": {"downsample_factor": 20}},
}

ALL_VARIANTS = [
    # eps sweep — rate-norm Poisson floor (mem_dim fixed at 16); eps_1e-3 == mem_16
    "eps_1e-4", "eps_1e-3", "eps_1e-2", "eps_1e-1",
    # mem_dim sweep — per-neuron memory width (eps fixed at 1e-3)
    "mem_4", "mem_8", "mem_16", "mem_32",
]


def load_variant_results(
    dataset: str,
    variant: str,
    ablation_threshold: float = 0.20,
    bias: str = None,
    gdi_sign=None,
):
    """Load and align adjacency matrices for a single variant, filtering by ablation test.

    bias=None uses the consensus + sign-inference orientation; pass a precomputed
    gdi_sign (same for every variant of a dataset) so robust_sign_inference runs once.
    bias='excitatory'/'inhibitory' reverts to the legacy dominance-assumption method.
    """
    variant_dir = RESULTS_BASE / dataset / variant
    csv_path = variant_dir / "ablation.csv"

    if not csv_path.exists():
        return None, None, 0, 0

    df = pd.read_csv(csv_path)
    total = len(df)
    passed_ids = df[df["difference"] <= -ablation_threshold]["run_id"].tolist()

    matrices = []
    for run_id in passed_ids:
        p = variant_dir / f"gnn_{run_id}.npy"
        if p.exists():
            matrices.append(np.load(p))

    if not matrices:
        return None, None, 0, total

    matrices = align_matrices(matrices, bias=bias, gdi_sign=gdi_sign)
    stacked = np.stack(matrices)
    mean_A = np.round(stacked.mean(axis=0), 3)
    consistency_mask = get_consistent_connections(stacked, threshold=0.8)
    consistent_mean_A = mean_A * consistency_mask
    return mean_A, consistent_mean_A, len(matrices), total


def main():
    parser = argparse.ArgumentParser(description="Compare circuitGNN variant results")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        help="Variants to compare (default: all)")
    parser.add_argument("--ablation_threshold", type=float, default=0.20,
                        help="Min |difference| to keep a run (default: 0.20)")
    parser.add_argument("--bias", default=None, choices=["excitatory", "inhibitory"],
                        help="Legacy sign method assuming excitatory/inhibitory dominance. "
                             "Omit for the new consensus + sign-inference orientation.")
    args = parser.parse_args()

    # Load ground truth (plots suppressed by Agg backend); binned_spikes feeds the
    # consensus sign-inference orientation.
    reg = DATASET_REGISTRY[args.dataset]
    binned_spikes, true_conn, neuron_names = reg["loader"](**reg["loader_kwargs"])

    # gdi_sign depends only on the spikes, not the variant -> compute once and reuse
    # across every variant (skips re-running robust_sign_inference per variant).
    gdi_sign = None if args.bias is not None else robust_sign_inference(binned_spikes)

    if args.dataset in ("pdcm", "pdcm_large"):
        pop_labels = np.load(f"datasets/{args.dataset}/pop_labels.npy")
        def do_plot(matrix, method_name, save_dir):
            plot_pdcm_matrix(matrix, pop_labels, method_name=method_name, save_dir=save_dir)
    elif args.dataset == "mdp":
        def do_plot(matrix, method_name, save_dir):
            plot_mdp_matrix(matrix, neuron_names, method_name=method_name, save_dir=save_dir)
    elif args.dataset == "v1_v2":
        def do_plot(matrix, method_name, save_dir):
            plot_v1_v2_matrix(matrix, neuron_names, method_name=method_name, save_dir=save_dir)
    else:
        def do_plot(matrix, method_name, save_dir):
            plot_connectivity_matrix(matrix, neuron_names=neuron_names, method_name=method_name, save_dir=save_dir, plot_val=True)

    rows = []
    for variant in args.variants:
        mean_A, consistent_mean_A, n_passed, n_total = load_variant_results(
            args.dataset, variant, args.ablation_threshold, args.bias, gdi_sign
        )

        if mean_A is None:
            rows.append({
                "variant": variant, "n_passed": 0, "n_total": n_total,
                "cosine": "-", "pearson": "-", "spearman": "-",
                "prec_unsign": "-", "rec_unsign": "-", "f1_unsign": "-",
                "prec_signed": "-", "rec_signed": "-", "f1_signed": "-",
                "mcc_unsign": "-", "mcc_signed": "-", "auroc": "-", "nie": "-",
                "cosine_cons": "-", "pearson_cons": "-", "spearman_cons": "-",
                "prec_unsign_cons": "-", "rec_unsign_cons": "-", "f1_unsign_cons": "-",
                "prec_signed_cons": "-", "rec_signed_cons": "-", "f1_signed_cons": "-",
                "mcc_unsign_cons": "-", "mcc_signed_cons": "-", "auroc_cons": "-", "nie_cons": "-",
            })
            continue

        variant_dir = RESULTS_BASE / args.dataset / variant
        do_plot(mean_A, "mean", str(variant_dir))
        do_plot(consistent_mean_A, "mean_consistent", str(variant_dir))
        plot_metric_verification(true_conn, consistent_mean_A, str(variant_dir),
                                 method_name=f"{variant} — consistent mean")

        res   = calculate_fit(true_conn, mean_A, plot=False)
        res_c = calculate_fit(true_conn, consistent_mean_A, plot=False)
        rows.append({
            "variant":         variant,
            "n_passed":        n_passed,
            "n_total":         n_total,
            "cosine":          f"{res.cosine:.3f}",
            "pearson":         f"{res.pearson:.3f}",
            "spearman":        f"{res.spearman:.3f}",
            "prec_unsign":     f"{res.overall_precision:.3f}",
            "rec_unsign":      f"{res.overall_recall:.3f}",
            "f1_unsign":       f"{res.overall_f1:.3f}",
            "prec_signed":     f"{res.signed_precision:.3f}",
            "rec_signed":      f"{res.signed_recall:.3f}",
            "f1_signed":       f"{res.signed_f1:.3f}",
            "mcc_unsign":      f"{res.mcc_unsigned:.3f}",
            "mcc_signed":      f"{res.mcc_signed:.3f}",
            "auroc":           f"{res.auroc:.3f}",
            "nie":             f"{res.nie:.3f}",
            "cosine_cons":     f"{res_c.cosine:.3f}",
            "pearson_cons":    f"{res_c.pearson:.3f}",
            "spearman_cons":   f"{res_c.spearman:.3f}",
            "prec_unsign_cons": f"{res_c.overall_precision:.3f}",
            "rec_unsign_cons":  f"{res_c.overall_recall:.3f}",
            "f1_unsign_cons":  f"{res_c.overall_f1:.3f}",
            "prec_signed_cons": f"{res_c.signed_precision:.3f}",
            "rec_signed_cons":  f"{res_c.signed_recall:.3f}",
            "f1_signed_cons":  f"{res_c.signed_f1:.3f}",
            "mcc_unsign_cons": f"{res_c.mcc_unsigned:.3f}",
            "mcc_signed_cons": f"{res_c.mcc_signed:.3f}",
            "auroc_cons":      f"{res_c.auroc:.3f}",
            "nie_cons":        f"{res_c.nie:.3f}",
        })

    df = pd.DataFrame(rows).set_index("variant")
    print(f"\n=== {args.dataset} — variant comparison (ablation threshold={args.ablation_threshold}) ===")
    print(df.to_string())
    print()

    csv_path = VARIANTS_DIR / f"variant_{args.dataset}.csv"
    df.to_csv(csv_path)
    print(f"Saved full metrics table to {csv_path}")


if __name__ == "__main__":
    main()
