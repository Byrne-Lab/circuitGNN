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

from circuit_gnn.data import (load_OC_1_network, load_OC_2_network, load_control_network, load_pdcm_network, load_arbitrary_network, load_small_circuit_A_network, load_small_circuit_B_network)
from circuit_gnn.utils import calculate_fit, align_matrices, get_consistent_connections

DATASET_REGISTRY = {
    "OC_1":            {"loader": load_OC_1_network,            "loader_kwargs": {"downsample_factor": 20}},
    "OC_2":            {"loader": load_OC_2_network,            "loader_kwargs": {"downsample_factor": 20}},
    "control":         {"loader": load_control_network,         "loader_kwargs": {"downsample_factor": 20}},
    "pdcm":            {"loader": load_pdcm_network,            "loader_kwargs": {"downsample_factor": 2}},
}


RESULTS_ROOT = REPO_ROOT / "results"
ABLATION_THRESHOLD = 0.10      
CONSISTENCY_THRESHOLD = 0.8
SIGN_BIN_MS = 20                
NATIVE_MS = {"OC_1": 0.5, "OC_2": 0.5, "control": 0.5, "pdcm": 1.0,
             "arbitrary": 1.0, "small_circuit_A": 0.5, "small_circuit_B": 0.5}
NUM_RUNS = 50

CLASS_PREFIXES = [
    ("gnn_mean",                     "gnn_mean"),
    ("gnn_mean_consistent",          "gnn_mean_consistent"),
    ("conditional_transfer_entropy", "conditional_transfer_entropy"),
    ("connect",                      "CoNNECT"),
    ("correlation",                  "correlation"),
    ("ctm-di",                       "CTM_DI"),
    ("decision_tree",                "decision_tree"),
    ("mutual_information",           "mutual_information"),
    ("spycon_ci",                    "spycon_ci"),
    ("spycon_dsttc",                 "spycon_dsttc"),
    ("spycon_glmcc",                 "spycon_glmcc"),
    ("spycon_glmpp",                 "spycon_glmpp"),
    ("spycon_sccg",                  "spycon_sccg"),
    ("transfer_entropy",             "transfer_entropy"),
]

BEST_COLUMNS = ["prec_signed", "rec_signed", "f1_signed", "nie"]


def metric_row(name, true_conn, M):
    res = calculate_fit(true_conn, M)
    return {
        "variant":     name,
        "cosine":      round(res.cosine, 4),
        "pearson":     round(res.pearson, 4),
        "spearman":    round(res.spearman, 4),
        "prec_unsign": round(res.overall_precision, 4),
        "rec_unsign":  round(res.overall_recall, 4),
        "f1_unsign":   round(res.overall_f1, 4),
        "prec_signed": round(res.signed_precision, 4),
        "rec_signed":  round(res.signed_recall, 4),
        "f1_signed":   round(res.signed_f1, 4),
        "auroc":       round(res.auroc, 4),
        "nie":         round(res.nie, 4),
    }


def score_baseline_folder(baseline_dir, true_conn):
    rows, skipped, n_nan_zeroed = [], [], 0
    for p in sorted(baseline_dir.glob("*.npy")):
        M = np.load(p)

        # NaN for pairs with no inferred connection -> treat as 0 (no edge)
        if np.isnan(M).any():
            M = np.where(np.isnan(M), 0.0, M)
            n_nan_zeroed += 1

        rows.append(metric_row(p.stem, true_conn, M))
    return rows, skipped, n_nan_zeroed


def variant_class(variant):
    for cls, prefix in sorted(CLASS_PREFIXES, key=lambda cp: -len(cp[1])):
        if variant.startswith(prefix):
            return cls
    return None


def rank_key(row):
    return (-row["f1_signed"], row["nie"], row["variant"])


def best_per_class(rows):
    best, unclassified = {}, []
    for row in rows:
        cls = variant_class(row["variant"])
        if cls is None:
            unclassified.append(row["variant"])
            continue
        incumbent = best.get(cls)
        if incumbent is None or rank_key(row) < rank_key(incumbent):
            best[cls] = row

    out = [{"class": cls, **best[cls]} for cls, _ in CLASS_PREFIXES if cls in best]
    return out, sorted(unclassified)


def score_gnn_runs(dataset, true_conn):
    results_loc = RESULTS_ROOT / dataset
    ablation_loc = results_loc / "ablation.csv"
    if not ablation_loc.exists():
        print(f"  no {ablation_loc}, skipping GNN rows")
        return []

    reg = DATASET_REGISTRY[dataset]
    sign_bin = round(SIGN_BIN_MS / NATIVE_MS.get(dataset, 1.0))
    binned_spikes, _, _ = reg["loader"](**{**reg["loader_kwargs"],
                                           "downsample_factor": sign_bin})

    ablation_csv = pd.read_csv(ablation_loc)
    mats = []
    for run_idx in range(1, NUM_RUNS + 1):
        ablation_row = ablation_csv[ablation_csv["run_id"] == run_idx]
        if ablation_row.difference.values[0] > -ABLATION_THRESHOLD:
            continue
        mats.append(np.load(results_loc / f"gnn_{run_idx}.npy"))
    if not mats:
        print(f"  no runs passed ablation in {ablation_loc}, skipping GNN rows")
        return []

    mats = align_matrices(mats, bias=None, binned_spikes=binned_spikes)
    stack = np.stack(mats, axis=0)
    mean_A = np.round(stack.mean(axis=0), 3)
    consistent_A = mean_A * get_consistent_connections(stack, threshold=CONSISTENCY_THRESHOLD)

    print(f"  GNN rows from {len(mats)} runs passing ablation (sign bin {sign_bin})")
    return [metric_row("gnn_mean", true_conn, mean_A),
            metric_row("gnn_mean_consistent", true_conn, consistent_A)]


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for init-baseline matrices.")
    parser.add_argument("--dataset", nargs="+", default=None,
                        help="Datasets to process")
    args = parser.parse_args()

    datasets = args.dataset or list(DATASET_REGISTRY.keys())
    for dataset in datasets:
        baseline_dir = REPO_ROOT / "datasets" / dataset / "baseline"

        reg = DATASET_REGISTRY[dataset]
        _, true_conn, _ = reg["loader"](**reg["loader_kwargs"])

        rows, skipped, n_nan_zeroed = score_baseline_folder(baseline_dir, true_conn)

        df = pd.DataFrame(rows).set_index("variant").sort_index()

        gnn_rows = score_gnn_runs(dataset, true_conn)
        if gnn_rows:
            df = pd.concat([pd.DataFrame(gnn_rows).set_index("variant"), df])

        csv_path = baseline_dir / f"baseline_comparison.csv"
        df.to_csv(csv_path)

        print(f"\n=== {dataset}: scored {len(rows)} variants + {len(gnn_rows)} GNN rows "
              f"(ground truth {true_conn.shape}) ===")
        print(df.to_string())
        print(f"Saved -> {csv_path}")

        best_rows, unclassified = best_per_class(gnn_rows + rows)
        best_df = pd.DataFrame(best_rows).set_index("class")[["variant"] + BEST_COLUMNS]
        best_path = baseline_dir / "baseline_best_per_class.csv"
        best_df.to_csv(best_path)

        missing = [cls for cls, _ in CLASS_PREFIXES if cls not in set(best_df.index)]
        if missing:
            print(f"  no variants found for: {', '.join(missing)}")
        if unclassified:
            print(f"  unclassified variants (excluded): {', '.join(unclassified)}")
        print(f"\n--- {dataset}: best variant per class by signed F1, ties on lower NIE ---")
        print(best_df.to_string())
        print(f"Saved -> {best_path}")

if __name__ == "__main__":
    main()
