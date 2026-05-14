"""Compare trained variant results for circuitGNN.

Run from repo root after running run_variants.py:
    python variants/compare_variants.py --dataset yuto
    python variants/compare_variants.py --dataset yuto --variants baseline dales layers_7
    python variants/compare_variants.py --dataset arbitrary --ablation_threshold 0.10

Prints a table: cosine, pearson, spearman, unsigned F1, signed F1 for each variant.
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

from circuit_gnn.data import load_yuto_network, load_pdcm_network, load_arbitrary_network
from circuit_gnn.utils import (calculate_fit, align_matrices,
                                get_consistent_connections, get_top_k, sign_inference)

RESULTS_BASE = VARIANTS_DIR / "results"

DATASET_REGISTRY = {
    "yuto":      {"loader": load_yuto_network,     "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1}},
    "arbitrary": {"loader": load_arbitrary_network, "loader_kwargs": {"downsample_factor": 10}},
    "pdcm":      {"loader": load_pdcm_network,      "loader_kwargs": {"downsample_factor": 2}},
}

ALL_VARIANTS = [
    # batch size
    "batch_4", "batch_16", "batch_32", "batch_64",
    # GNN layers
    "layers_1", "layers_3", "layers_5", "layers_7",
    # bin size
    "bin_1ms", "bin_2ms", "bin_5ms", "bin_10ms",
    # window size
    "window_5", "window_10", "window_50", "window_100", "window_200",
    # encoder channels
    "channels_8", "channels_16", "channels_32", "channels_64",
    # encoder kernel size
    "kernel_size_2", "kernel_size_3", "kernel_size_4",
    # L1 regularisation
    "l1_0", "l1_1e-3", "l1_1e-2", "l1_5e-2", "l1_1e-1", "l1_1",
    # competitive lambda
    "comp_0", "comp_1e-2", "comp_1e-1", "comp_1",
    # self_scale
    "self_0", "self_0p1", "self_0p25", "self_0p5", "self_1",
    # embedding dim
    "emb_16", "emb_32", "emb_64", "emb_128",
    # hidden dim
    "hidden_32", "hidden_64", "hidden_128",
    # learning rate
    "lr_1e-5", "lr_5e-5", "lr_1e-4", "lr_5e-4",
]


def load_variant_results(
    dataset: str,
    variant: str,
    ablation_threshold: float = 0.20,
    excitatory: bool = True,
):
    """Load and align adjacency matrices for a single variant, filtering by ablation test."""
    variant_dir = RESULTS_BASE / dataset / variant
    csv_path = variant_dir / "ablation.csv"

    if not csv_path.exists():
        return None, None, 0, 0

    df = pd.read_csv(csv_path)
    total = len(df)
    passed_ids = df[df["difference"].abs() >= ablation_threshold]["run_id"].tolist()

    matrices = []
    for run_id in passed_ids:
        p = variant_dir / f"gnn_{run_id}.npy"
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
    parser = argparse.ArgumentParser(description="Compare circuitGNN variant results")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        help="Variants to compare (default: all)")
    parser.add_argument("--ablation_threshold", type=float, default=0.20,
                        help="Min |difference| to keep a run (default: 0.20)")
    parser.add_argument("--excitatory", action="store_true", default=True,
                        help="Assume excitatory connections dominate for sign alignment")
    args = parser.parse_args()

    # Load ground truth (plots suppressed by Agg backend)
    reg = DATASET_REGISTRY[args.dataset]
    _, true_conn, _ = reg["loader"](**reg["loader_kwargs"])

    rows = []
    for variant in args.variants:
        mean_A, consistent_mean_A, n_passed, n_total = load_variant_results(
            args.dataset, variant, args.ablation_threshold, args.excitatory
        )

        if mean_A is None:
            rows.append({
                "variant": variant, "n_passed": 0, "n_total": n_total,
                "cosine": "-", "pearson": "-", "spearman": "-",
                "f1_unsign": "-", "f1_signed": "-",
                "pearson_cons": "-", "f1_unsign_cons": "-", "f1_signed_cons": "-",
            })
            continue

        cos, pear, spear, _, _, f1, _, _, sf1 = calculate_fit(true_conn, mean_A, plot=False)
        _, pear_c, _, _, _, f1_c, _, _, sf1_c = calculate_fit(true_conn, consistent_mean_A, plot=False)
        rows.append({
            "variant":        variant,
            "n_passed":       n_passed,
            "n_total":        n_total,
            "cosine":         f"{cos:.3f}",
            "pearson":        f"{pear:.3f}",
            "spearman":       f"{spear:.3f}",
            "f1_unsign":      f"{f1:.3f}",
            "f1_signed":      f"{sf1:.3f}",
            "pearson_cons":   f"{pear_c:.3f}",
            "f1_unsign_cons": f"{f1_c:.3f}",
            "f1_signed_cons": f"{sf1_c:.3f}",
        })

    df = pd.DataFrame(rows).set_index("variant")
    print(f"\n=== {args.dataset} — variant comparison (ablation threshold={args.ablation_threshold}) ===")
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
