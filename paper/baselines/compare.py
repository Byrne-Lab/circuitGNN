"""Score every baseline matrix in datasets/<dataset>/baseline/ against ground truth.

    python -m paper.baselines.compare                   # OC_1 and pdcm
    python -m paper.baselines.compare --dataset pdcm

Writes baseline_comparison.csv (every setting) and baseline_best_per_class.csv (best
setting per method by signed F1, ties broken by lower NIE) next to the matrices. The
CircuitGNN rows are the consensus of the replicates in results/<dataset>/.
"""
import argparse

import numpy as np
import pandas as pd

from circuit_gnn.metrics import calculate_fit
from paper.config import DATASETS_DIR
from paper.evaluate import gnn_consensus

DATASET_NAMES = ["OC_1", "pdcm"]

# (class, file-name prefix) for CircuitGNN and the six baselines in the manuscript
CLASS_PREFIXES = [
    ("gnn_mean",            "gnn_mean"),
    ("gnn_mean_consistent", "gnn_mean_consistent"),
    ("connect",             "CoNNECT"),
    ("ctm-di",              "CTM_DI"),
    ("spycon_dsttc",        "spycon_dsttc"),
    ("spycon_glmcc",        "spycon_glmcc"),
    ("spycon_glmpp",        "spycon_glmpp"),
    ("spycon_sccg",         "spycon_sccg"),
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
    rows, n_nan_zeroed = [], 0
    for p in sorted(baseline_dir.glob("*.npy")):
        M = np.load(p)
        # NaN for pairs with no inferred connection -> treat as 0 (no edge)
        if np.isnan(M).any():
            M = np.where(np.isnan(M), 0.0, M)
            n_nan_zeroed += 1
        rows.append(metric_row(p.stem, true_conn, M))
    return rows, n_nan_zeroed


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=DATASET_NAMES)
    args = parser.parse_args()

    for dataset in args.dataset:
        baseline_dir = DATASETS_DIR / dataset / "baseline"
        if not baseline_dir.is_dir():
            print(f"\n=== {dataset}: no {baseline_dir}, skipped ===")
            continue

        c, true_conn, _ = gnn_consensus(dataset)
        gnn_rows = [metric_row("gnn_mean", true_conn, c.A),
                    metric_row("gnn_mean_consistent", true_conn, c.A_consistent)]
        rows, n_nan_zeroed = score_baseline_folder(baseline_dir, true_conn)
        rows = sorted(rows, key=lambda r: r["variant"])
        if n_nan_zeroed:
            print(f"  {n_nan_zeroed} baseline matrices had NaN entries, scored as no edge")

        df = pd.DataFrame(gnn_rows + rows).set_index("variant")
        csv_path = baseline_dir / "baseline_comparison.csv"
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
