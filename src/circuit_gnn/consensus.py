"""Combining independently trained replicates into one consensus connectivity estimate.

Each replicate recovers the connectivity only up to a global sign. ``consensus`` drops
replicates that fail the ablation test, orients the rest against one another
(``orient_runs``) and, as a set, against a data-derived sign estimate
(``robust_sign_inference``), averages them, and masks edges whose sign is unstable.
"""
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

ABLATION_THRESHOLD = 0.10     # keep a replicate only if A lowers its validation loss by >= this
CONSISTENCY_THRESHOLD = 0.8   # fraction of kept replicates that must agree on an edge's sign
SIGN_BIN_MS = 20.0            # bin of the spike counts used for the global sign estimate
ROUND_DECIMALS = 3            # the mean is rounded, so |w| < 0.0005 becomes 0


@dataclass
class Consensus:
    A: np.ndarray             # (N, N) mean of the kept, sign-aligned replicates, [pre, post]
    A_consistent: np.ndarray  # A masked to edges whose sign is stable across replicates
    consistency: np.ndarray   # (N, N) 0/1 mask behind A_consistent
    runs: np.ndarray          # (n_kept, N, N) kept replicates, sign-aligned
    run_ids: List[int]        # ids of the kept replicates
    flipped: List[int]        # ids of the kept replicates whose sign was flipped


def consensus(matrices, ablation, sign_spikes, *, run_ids=None,
              ablation_threshold=ABLATION_THRESHOLD,
              consistency_threshold=CONSISTENCY_THRESHOLD,
              round_decimals=ROUND_DECIMALS) -> Consensus:
    """Consensus connectivity from independently trained replicates.

    matrices              : per-replicate (N, N) connectivity, [pre, post]
    ablation              : per-replicate validation loss with A minus without A
    sign_spikes           : (N, T) spike counts at ~SIGN_BIN_MS, for the global sign estimate
    run_ids               : replicate ids, for reporting (default 1..len(matrices))
    ablation_threshold    : keep a replicate only if ablation <= -ablation_threshold. An
                            absolute BCE-loss difference, so dataset-dependent; if no
                            replicate passes, all are kept and a warning is issued.
    consistency_threshold : fraction of kept replicates that must agree on an edge's sign
    round_decimals        : decimals the mean is rounded to; None disables rounding
    """
    run_ids = list(range(1, len(matrices) + 1)) if run_ids is None else list(run_ids)
    ablation = np.asarray(ablation, dtype=float)
    keep = [i for i, d in enumerate(ablation) if d <= -ablation_threshold]
    if not keep:
        warnings.warn(
            f"ablation_threshold={ablation_threshold} rejected all {len(ablation)} runs "
            f"(differences: {np.round(ablation, 4).tolist()}); keeping all runs. "
            "This threshold is an absolute BCE-loss difference and is dataset-dependent.",
            stacklevel=2)
        keep = list(range(len(ablation)))

    oriented, flip = orient_runs([matrices[i] for i in keep], robust_sign_inference(sign_spikes))
    runs = np.stack(oriented)
    mask = get_consistent_connections(runs, consistency_threshold)
    A = runs.mean(axis=0)
    if round_decimals is not None:
        A = np.round(A, round_decimals)
    kept_ids = [run_ids[i] for i in keep]
    return Consensus(A=A, A_consistent=A * mask, consistency=mask, runs=runs, run_ids=kept_ids,
                     flipped=[r for r, f in zip(kept_ids, flip) if f])


def load_runs(run_dir, run_ids=None):
    """Read back the replicates that ``fit_runs`` wrote to ``run_dir``.

    Returns ``(matrices, ablation, run_ids)`` in run-id order. A run listed more than once
    in ablation.csv (it was retrained) takes its last row. ``run_ids`` selects a subset.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "ablation.csv", newline="") as f:
        gap = {int(row["run_id"]): float(row["difference"]) for row in csv.DictReader(f)}
    ids = sorted(gap) if run_ids is None else list(run_ids)
    missing = sorted(set(ids) - set(gap))
    if missing:
        raise FileNotFoundError(f"{run_dir / 'ablation.csv'} has runs {sorted(gap)}; missing {missing}")
    return [np.load(run_dir / f"gnn_{i}.npy") for i in ids], np.array([gap[i] for i in ids]), ids


# Adapted from Young et al. (2021)
def robust_sign_inference(binned_spikes, num_taus=3, ridge=1e-3):
    # GDI sign per edge (partial-correlation sign at best lag), robust on large N.
    # Zero-variance (silent) neurons are dropped (edges = 0)
    # Ridge keeps the correlation invertible even when it is singular/collinear

    def _partial_corr(X):
        corr = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        prec = np.linalg.inv(corr + ridge * np.eye(corr.shape[0])) # add ridge
        d = np.diag(prec)
        return -prec / np.sqrt(np.outer(d, d))

    N = binned_spikes.shape[0]
    active = np.where(binned_spikes.std(axis=1) > 0)[0] # drop silent neurons
    gdi = np.zeros((N, N))
    if len(active) < N:
        warnings.warn(f"sign inference: dropped {N - len(active)} silent neurons, "
                      f"{len(active)} active (ridge={ridge}).", stacklevel=2)
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
            pcorr[:, tt] = _partial_corr(Xs[:-(tt + 1), :])[:, j]
        best = pcorr[np.arange(C), np.argmax(np.abs(pcorr), axis=1)]  # value at max-|lag|
        sign[:, j] = np.sign(best)
    gdi[np.ix_(active, active)] = sign
    return gdi


def _cosine_offdiag(a, b):
    # Off-diagonal cosine similarity
    v1, v2 = a[~np.eye(a.shape[0], dtype=bool)], b[~np.eye(b.shape[0], dtype=bool)]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 0 else 0.0


def orient_runs(matrices, gdi_sign, reference_idx=0):
    # Fix each run's arbitrary global sign by aligning to a reference run

    mats = [A.copy() for A in matrices]
    n = len(mats)
    flip = [False] * n

    ref = mats[reference_idx]
    for i in range(n):
        if i == reference_idx:
            continue
        if _cosine_offdiag(-mats[i], ref) > _cosine_offdiag(mats[i], ref):
            mats[i] = -mats[i]
            flip[i] = True

    # global orientation
    mean = np.stack(mats).mean(axis=0)
    off = ~np.eye(mean.shape[0], dtype=bool)
    score = np.sum(mean[off] * gdi_sign[off])
    if score == 0:
        flip_all = mean.sum() < 0  # excitatory-dominant fallback
        warnings.warn("gdi_sign gave no usable orientation edges; using excitatory fallback.",
                      stacklevel=2)
    else:
        flip_all = score < 0
    if flip_all:
        mats = [-A for A in mats]
        flip = [not f for f in flip]
    return mats, flip


def get_consistent_connections(adjacency_matrices, threshold):
    n_matrices = adjacency_matrices.shape[0]

    positive_count = np.sum(adjacency_matrices > 0, axis=0)
    negative_count = np.sum(adjacency_matrices < 0, axis=0)
    binary_mask = ((positive_count / n_matrices) >= threshold) | ((negative_count / n_matrices) >= threshold)

    return binary_mask.astype(float)
