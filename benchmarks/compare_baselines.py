"""
python benchmarks/compare_baselines.py                      # all datasets with a baseline/ folder
python benchmarks/compare_baselines.py --dataset pdcm       # one dataset
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(str(REPO_ROOT))  # loaders use dataset paths relative to the repo root

from circuit_gnn.data import (load_OC_2_network, load_pdcm_network, load_arbitrary_network, load_small_circuit_A_network, load_small_circuit_B_network)
from circuit_gnn.utils import calculate_fit

DATASET_REGISTRY = {
    "OC_2":            {"loader": load_OC_2_network,            "loader_kwargs": {"downsample_factor": 20}},
    "pdcm":            {"loader": load_pdcm_network,            "loader_kwargs": {"downsample_factor": 2}},
    "arbitrary":       {"loader": load_arbitrary_network,       "loader_kwargs": {"downsample_factor": 10}},
    "small_circuit_A": {"loader": load_small_circuit_A_network, "loader_kwargs": {"downsample_factor": 20}},
    "small_circuit_B": {"loader": load_small_circuit_B_network, "loader_kwargs": {"downsample_factor": 20}},
}


def score_baseline_folder(baseline_dir, true_conn):
    rows, skipped, n_nan_zeroed = [], [], 0
    for p in sorted(baseline_dir.glob("*.npy")):
        M = np.load(p)

        # NaN for pairs with no inferred connection -> treat as 0 (no edge)
        if np.isnan(M).any():
            M = np.where(np.isnan(M), 0.0, M)
            n_nan_zeroed += 1

        res = calculate_fit(true_conn, M, plot=False)
        rows.append({
            "variant":     p.stem,
            "cosine":      round(res.cosine, 4),
            "pearson":     round(res.pearson, 4),
            "spearman":    round(res.spearman, 4),
            "prec_unsign": round(res.overall_precision, 4),
            "rec_unsign":  round(res.overall_recall, 4),
            "f1_unsign":   round(res.overall_f1, 4),
            "prec_signed": round(res.signed_precision, 4),
            "rec_signed":  round(res.signed_recall, 4),
            "f1_signed":   round(res.signed_f1, 4),
            "mcc_unsign":  round(res.mcc_unsigned, 4),
            "mcc_signed":  round(res.mcc_signed, 4),
            "auroc":       round(res.auroc, 4),
            "nie":         round(res.nie, 4),
        })
    return rows, skipped, n_nan_zeroed


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for init-baseline matrices.")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Datasets to process (default: all in the registry that have a baseline/ folder).")
    args = parser.parse_args()

    datasets = args.dataset or list(DATASET_REGISTRY.keys())
    for dataset in datasets:
        baseline_dir = REPO_ROOT / "datasets" / dataset / "baseline"

        reg = DATASET_REGISTRY[dataset]
        _, true_conn, _ = reg["loader"](**reg["loader_kwargs"])

        rows, skipped, n_nan_zeroed = score_baseline_folder(baseline_dir, true_conn)

        df = pd.DataFrame(rows).set_index("variant").sort_index()
        csv_path = baseline_dir / f"baseline_comparison.csv"
        df.to_csv(csv_path)

        print(f"\n=== {dataset}: scored {len(rows)} variants (ground truth {true_conn.shape}) ===")
        print(df.to_string())
        print(f"Saved -> {csv_path}")

if __name__ == "__main__":
    main()
