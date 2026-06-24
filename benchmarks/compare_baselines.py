"""
python benchmarks/compare_baselines.py                      # all datasets with an init/ folder
python benchmarks/compare_baselines.py --dataset yuto       # one dataset
python benchmarks/compare_baselines.py --dataset yuto pdcm  # a subset
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display; calculate_fit is called with plot=False anyway

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(str(REPO_ROOT))  # loaders use dataset paths relative to the repo root

from circuit_gnn.data import (load_yuto_network, load_pdcm_network, load_arbitrary_network,
                              load_small_circuit_A_network, load_small_circuit_B_network,
                              load_v1_network)
from circuit_gnn.utils import calculate_fit

# Dense baseline matrices binarize to a single class, which makes sklearn's
# matthews_corrcoef warn; the value (0) is still correct, so silence the noise.
warnings.filterwarnings("ignore", message="A single label was found")

DATASET_REGISTRY = {
    "yuto":            {"loader": load_yuto_network,            "loader_kwargs": {"downsample_factor": 20, "gap": 0.20, "step": 1}},
    "pdcm":            {"loader": load_pdcm_network,            "loader_kwargs": {"downsample_factor": 2}},
    "arbitrary":       {"loader": load_arbitrary_network,       "loader_kwargs": {"downsample_factor": 10}},
    "small_circuit_A": {"loader": load_small_circuit_A_network, "loader_kwargs": {"downsample_factor": 20}},
    "small_circuit_B": {"loader": load_small_circuit_B_network, "loader_kwargs": {"downsample_factor": 20}},
    "v1":              {"loader": load_v1_network,              "loader_kwargs": {"downsample_factor": 2}},
}


def score_init_folder(init_dir, true_conn):
    """Run calculate_fit on every .npy in init_dir; return (rows, skipped, n_nan_zeroed)."""
    rows, skipped, n_nan_zeroed = [], [], 0
    for p in sorted(init_dir.glob("*.npy")):
        try:
            M = np.load(p)
        except Exception as e:
            skipped.append((p.name, f"load error: {e}"))
            continue
        if M.shape != true_conn.shape:
            skipped.append((p.name, f"shape {M.shape} != ground truth {true_conn.shape}"))
            continue
        # Some methods leave NaN for pairs with no inferred connection -> treat as 0 (no edge).
        if np.isnan(M).any():
            M = np.where(np.isnan(M), 0.0, M)
            n_nan_zeroed += 1
        try:
            res = calculate_fit(true_conn, M, plot=False)
        except Exception as e:
            skipped.append((p.name, f"calculate_fit error: {e}"))
            continue
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
                        help="Datasets to process (default: all in the registry that have an init/ folder).")
    args = parser.parse_args()

    datasets = args.dataset or list(DATASET_REGISTRY.keys())
    for dataset in datasets:
        if dataset not in DATASET_REGISTRY:
            print(f"[skip] unknown dataset '{dataset}' (known: {', '.join(DATASET_REGISTRY)})")
            continue
        init_dir = REPO_ROOT / "datasets" / dataset / "init"
        if not init_dir.is_dir():
            print(f"[skip] {dataset}: no init/ folder at {init_dir}")
            continue

        reg = DATASET_REGISTRY[dataset]
        try:
            _, true_conn, _ = reg["loader"](**reg["loader_kwargs"])
        except Exception as e:
            print(f"[skip] {dataset}: loader failed: {e}")
            continue

        rows, skipped, n_nan_zeroed = score_init_folder(init_dir, true_conn)
        if not rows:
            print(f"[skip] {dataset}: no scorable .npy files in {init_dir}")
            continue

        df = pd.DataFrame(rows).set_index("variant").sort_index()
        csv_path = init_dir / f"compare_baselines_{dataset}.csv"
        df.to_csv(csv_path)

        print(f"\n=== {dataset}: scored {len(rows)} variants (ground truth {true_conn.shape}) ===")
        print(df.to_string())
        print(f"Saved -> {csv_path}")
        if n_nan_zeroed:
            print(f"({n_nan_zeroed} file(s) had NaN entries treated as 0 / no edge.)")
        if skipped:
            print(f"Skipped {len(skipped)} file(s) (not square / shape mismatch / error):")
            for name, why in skipped:
                print(f"   {name}: {why}")


if __name__ == "__main__":
    main()
