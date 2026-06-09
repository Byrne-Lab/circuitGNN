"""Compare trained architecture results for circuitGNN.

Run from repo root after running run_architectures.py:
    python architectures/compare_architectures.py --dataset yuto
    python architectures/compare_architectures.py --dataset pdcm --architectures baseline dales
    python architectures/compare_architectures.py --dataset yuto --ablation_threshold 0.10

Prints a table: cosine, pearson, spearman, unsigned F1, signed F1 for each architecture.
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
                                get_consistent_connections, get_top_k, sign_inference)

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
    "skip_residual", "edge_cond", "split_path", "multi_step", "learned_scale", "hadamard_mod",
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
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--architectures", nargs="+", default=ALL_ARCHITECTURES,
                        help="Architectures to compare (default: all)")
    parser.add_argument("--ablation_threshold", type=float, default=0.20,
                        help="Min |difference| to keep a run (default: 0.20)")
    parser.add_argument("--excitatory", action="store_true", default=True,
                        help="Assume excitatory connections dominate for sign alignment")
    args = parser.parse_args()

    reg = DATASET_REGISTRY[args.dataset]
    _, true_conn, _ = reg["loader"](**reg["loader_kwargs"])

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
                "pearson_cons": "-", "f1_unsign_cons": "-", "f1_signed_cons": "-",
            })
            continue

        cos, pear, spear, _, _, f1, _, _, sf1 = calculate_fit(true_conn, mean_A, plot=False)
        _, pear_c, _, _, _, f1_c, _, _, sf1_c = calculate_fit(true_conn, consistent_mean_A, plot=False)
        rows.append({
            "architecture":   arch,
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

    df = pd.DataFrame(rows).set_index("architecture")
    print(f"\n=== {args.dataset} — architecture comparison (ablation threshold={args.ablation_threshold}) ===")
    print(df.to_string())
    print()


if __name__ == "__main__":
    main()
