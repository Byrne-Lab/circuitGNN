"""Test sign-inference-based alignment as an alternative to align_matrices.

For one results folder, aligns the runs three ways and compares the fit to ground
truth, writing plots + a metrics CSV to a separate output folder:

  align_matrices : baseline (flip each run toward a reference matrix)
  consensus      : flip each run to agree with the others, orient once via gdi_sign
  abs_x_gdi_sign : |consensus mean| with every edge's sign taken from gdi_sign

Run from repo root, e.g.:
    python variants/test_sign_inference_alignment.py \
        --dataset mdp --results_dir results/mdp/no_init \
        --output_dir variants/results_sign/mdp
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # suppress plot windows when loading data

import numpy as np
import pandas as pd

# path setup (mirror compare_variants.py)
VARIANTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = VARIANTS_DIR.parent
sys.path.insert(0, str(VARIANTS_DIR))
os.chdir(str(REPO_ROOT))

from circuit_gnn.utils import (calculate_fit, align_matrices, get_consistent_connections,
                               plot_connectivity_matrix, plot_pdcm_matrix,
                               plot_mdp_matrix, plot_v1_v2_matrix, plot_hc3_matrix)
from compare_variants import DATASET_REGISTRY


def load_matrices(results_dir, ablation_threshold=0.20):
    """Load gnn_*.npy from a results folder, keeping only runs that pass ablation."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "ablation.csv")
    passed_ids = df[df["difference"].abs() >= ablation_threshold]["run_id"].tolist()
    matrices, run_ids = [], []
    for run_id in passed_ids:
        p = results_dir / f"gnn_{run_id}.npy"
        if p.exists():
            matrices.append(np.load(p))
            run_ids.append(run_id)
    return matrices, run_ids, len(df)


def robust_sign_inference(binned_spikes, num_taus, ridge=1e-3):
    """GDI sign per edge (partial-correlation sign at best lag), robust on large N.

    Silent (zero-variance) neurons are dropped (their edges stay sign 0), and a ridge
    keeps the correlation invertible even when it is singular/collinear -- both of
    which otherwise make np.linalg.inv return NaN or raise. binned_spikes is (N, T).
    """
    def partial_corr(X):
        corr = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        prec = np.linalg.inv(corr + ridge * np.eye(corr.shape[0]))
        d = np.diag(prec)
        return -prec / np.sqrt(np.outer(d, d))

    N = binned_spikes.shape[0]
    active = np.where(binned_spikes.std(axis=1) > 0)[0]
    gdi = np.zeros((N, N))
    if len(active) < N:
        print(f"sign inference: dropped {N - len(active)} silent neurons, "
              f"{len(active)} active (ridge={ridge}).")
    if len(active) < 2:
        return gdi

    X = binned_spikes[active].T
    C = X.shape[1]
    sign = np.zeros((C, C))
    for j in range(C):
        pcorr = np.zeros((C, num_taus))
        for tt in range(num_taus):
            Xs = X.copy()
            Xs[:-(tt + 1), j] = Xs[tt + 1:, j]
            pcorr[:, tt] = partial_corr(Xs[:-(tt + 1), :])[:, j]
        best = pcorr[np.arange(C), np.argmax(np.abs(pcorr), axis=1)]  # value at max-|lag|
        sign[:, j] = np.sign(best)
    gdi[np.ix_(active, active)] = sign
    return gdi


def _cosine_offdiag(a, b):
    """Off-diagonal cosine similarity (matches calculate_fit's convention)."""
    v1, v2 = a[~np.eye(a.shape[0], dtype=bool)], b[~np.eye(b.shape[0], dtype=bool)]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


def consensus_flips(matrices, gdi_sign, max_iter=100):
    """Fix each run's arbitrary global sign, then orient the whole cluster.

    1. Iteratively flip each run to agree with the mean of the OTHER runs until
       nothing changes, so real edges reinforce instead of cancelling in the mean.
    2. Flip the whole consensus if its off-diagonal signs mostly disagree with
       gdi_sign; if gdi_sign has no usable edges, fall back to excitatory-dominant.

    Returns (aligned_matrices, flip_flags).
    """
    mats = [A.copy() for A in matrices]
    n = len(mats)
    flip = [False] * n

    total = np.stack(mats).sum(axis=0)
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            loo = (total - mats[i]) / max(n - 1, 1)  # mean of the other runs
            if _cosine_offdiag(-mats[i], loo) > _cosine_offdiag(mats[i], loo):
                total = total - 2 * mats[i]
                mats[i] = -mats[i]
                flip[i] = not flip[i]
                changed = True
        if not changed:
            break

    # global orientation
    mean = np.stack(mats).mean(axis=0)
    off = ~np.eye(mean.shape[0], dtype=bool)
    cur, ref = np.sign(mean[off]), gdi_sign[off]
    usable = (cur != 0) & (ref != 0)
    if usable.sum() == 0:
        flip_all = mean.sum() < 0  # excitatory-dominant fallback
        print("Warning: gdi_sign gave no usable orientation edges; using excitatory fallback.")
    else:
        agree = np.sum(cur[usable] == ref[usable])
        flip_all = agree < usable.sum() - agree
    if flip_all:
        mats = [-A for A in mats]
        flip = [not f for f in flip]
    return mats, flip


def mean_and_consistent(matrices, threshold=0.8):
    stacked = np.stack(matrices)
    mean_A = np.round(stacked.mean(axis=0), 3)
    return mean_A, mean_A * get_consistent_connections(stacked, threshold)


def summarize(true_conn, mean_A, cons_A, label):
    """Fit metrics for the mean and the consistency-masked mean."""
    row = {"approach": label}
    for suffix, m in (("", mean_A), ("_cons", cons_A)):
        r = calculate_fit(true_conn, m, plot=False)
        row.update({
            f"cosine{suffix}":      round(r.cosine, 3),
            f"pearson{suffix}":     round(r.pearson, 3),
            f"spearman{suffix}":    round(r.spearman, 3),
            f"prec_signed{suffix}": round(r.signed_precision, 3),
            f"rec_signed{suffix}":  round(r.signed_recall, 3),
            f"f1_signed{suffix}":   round(r.signed_f1, 3),
            f"mcc_unsign{suffix}":  round(r.mcc_unsigned, 3),
            f"mcc_signed{suffix}":  round(r.mcc_signed, 3),
            f"auroc{suffix}":       round(r.auroc, 3),
            f"nie{suffix}":         round(r.nie, 3),
        })
    return row


def main():
    parser = argparse.ArgumentParser(description="Compare sign-alignment approaches on one results folder")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--results_dir", required=True, help="Folder with gnn_*.npy and ablation.csv")
    parser.add_argument("--output_dir", required=True, help="Separate folder for plots + metrics")
    parser.add_argument("--num_taus", type=int, default=3, help="Number of lags for sign inference")
    parser.add_argument("--ridge", type=float, default=1e-3, help="Ridge for the sign-inference inverse")
    parser.add_argument("--ablation_threshold", type=float, default=0.20, help="Min |difference| to keep a run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = DATASET_REGISTRY[args.dataset]
    binned_spikes, true_conn, neuron_names = reg["loader"](**reg["loader_kwargs"])

    # per-dataset plot helper (mirrors compare_variants.py)
    if args.dataset in ("pdcm", "pdcm_large"):
        pop_labels = np.load(f"datasets/{args.dataset}/pop_labels.npy")
        def do_plot(m, name):
            plot_pdcm_matrix(m, pop_labels, method_name=name, save_dir=str(output_dir))
    elif args.dataset == "mdp":
        def do_plot(m, name):
            plot_mdp_matrix(m, neuron_names, method_name=name, save_dir=str(output_dir))
    elif args.dataset == "v1_v2":
        def do_plot(m, name):
            plot_v1_v2_matrix(m, neuron_names, method_name=name, save_dir=str(output_dir))
    elif args.dataset == "hc3":
        def do_plot(m, name):
            plot_hc3_matrix(m, neuron_names, method_name=name, save_dir=str(output_dir))
    else:
        def do_plot(m, name):
            plot_connectivity_matrix(m, neuron_names=neuron_names, method_name=name,
                                     save_dir=str(output_dir), plot_val=True)

    gdi_sign = robust_sign_inference(binned_spikes, args.num_taus, ridge=args.ridge)

    matrices, run_ids, n_total = load_matrices(args.results_dir, args.ablation_threshold)
    if not matrices:
        print(f"No usable runs in {args.results_dir} (0 / {n_total} passed ablation).")
        return
    print(f"Loaded {len(matrices)}/{n_total} runs that passed ablation: {run_ids}")

    rows = []

    # baseline: align_matrices (mutates its input, so pass copies)
    mean_b, cons_b = mean_and_consistent(align_matrices([A.copy() for A in matrices]))
    do_plot(mean_b, "align_matrices_mean")
    do_plot(cons_b, "align_matrices_mean_consistent")
    rows.append(summarize(true_conn, mean_b, cons_b, "align_matrices"))

    # consensus flips, oriented once by gdi_sign
    signed, flip_flags = consensus_flips(matrices, gdi_sign)
    print(f"consensus flipped runs: {[run_ids[i] for i, f in enumerate(flip_flags) if f]}")
    mean_s, cons_s = mean_and_consistent(signed)
    do_plot(mean_s, "consensus_mean")
    do_plot(cons_s, "consensus_mean_consistent")
    rows.append(summarize(true_conn, mean_s, cons_s, "consensus"))

    # |consensus mean| * gdi_sign (same as eval.py signed_mean_A)
    do_plot(np.abs(mean_s) * gdi_sign, "abs_x_gdi_sign_mean")
    do_plot(np.abs(cons_s) * gdi_sign, "abs_x_gdi_sign_mean_consistent")
    rows.append(summarize(true_conn, np.abs(mean_s) * gdi_sign, np.abs(cons_s) * gdi_sign, "abs_x_gdi_sign"))

    df = pd.DataFrame(rows).set_index("approach")
    print(f"\n=== {args.dataset} | {args.results_dir} ===")
    print(df.to_string())
    df.to_csv(output_dir / "sign_inference_comparison.csv")
    print(f"\nSaved plots + metrics to {output_dir}")


if __name__ == "__main__":
    main()
