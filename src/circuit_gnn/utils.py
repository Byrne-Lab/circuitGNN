import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize
import copy
from scipy.optimize import minimize_scalar
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from collections import namedtuple

# Result of calculate_fit. Fields are accessed by name (e.g. res.overall_f1).
FitResult = namedtuple("FitResult", [
    "cosine", "pearson", "spearman",
    "overall_precision", "overall_recall", "overall_f1",
    "signed_precision", "signed_recall", "signed_f1",
    "auroc", "nie",
])


def plot_connectivity_matrix(matrix, neuron_names, method_name='GNN', cmap=None, save_dir='benchmarking_results', plot_val=True, vmin=None, vmax=None, signed=True):
    os.makedirs(save_dir, exist_ok=True)

    if signed:
        vmin = min(matrix.min(), -1e-6) if vmin is None else vmin
        vmax = max(matrix.max(),  1e-6) if vmax is None else vmax
        plot_norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plot_cmap = copy.copy(sns.diverging_palette(240, 10, as_cmap=True)) if cmap is None else copy.copy(cmap)
        cbar_label = 'Connectivity Strength'
    else:
        vmin = 0.0 if vmin is None else vmin
        vmax = max(matrix.max(), 1e-6) if vmax is None else vmax
        plot_norm = Normalize(vmin=vmin, vmax=vmax)
        plot_cmap = copy.copy(plt.cm.Greys) if cmap is None else copy.copy(cmap)
        cbar_label = 'Connectivity Strength (unsigned)'

    plot_cmap.set_bad(color='white') # mask exact zeros so they render as white
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)

    plt.figure(figsize=(10, 8))
    if plot_val:
        annot_labels = np.where(matrix == 0, '', np.vectorize(lambda v: f"{v:.3f}")(matrix))
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': cbar_label},
                    square=True, annot=annot_labels, fmt='')   # fmt='' because values are already strings
    else:
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': cbar_label},
                    square=True)

    plt.title(f"{method_name} Connectivity Matrix")
    plt.xlabel("Postsynaptic neuron")
    plt.ylabel("Presynaptic neuron")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()


PDCM_POPS_ORDERED = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

def plot_pdcm_matrix(matrix, pop_labels, method_name='GNN', save_dir='results', blackout=None,
                     title=None, cbar_label='Connection strength'):
    # title defaults to '<method_name> Connectivity Matrix'; the file is always
    # <save_dir>/<method_name>.pdf, so callers can retitle without renaming.

    os.makedirs(save_dir, exist_ok=True)
    matrix = np.asarray(matrix, dtype=float).copy()
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    if blackout is not None and len(blackout) > 0:
        cmap.set_bad('gray')                  # sns.heatmap renders NaN cells with the bad colour
        matrix[blackout, :] = np.nan
        matrix[:, blackout] = np.nan
    vmin = min(np.nanmin(matrix), -1e-6)
    vmax = max(np.nanmax(matrix),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': cbar_label},
                square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in PDCM_POPS_ORDERED:
        count = int(np.sum(pop_labels == pop))
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(pop)
        cursor += count
    boundaries.append(cursor)

    ax = plt.gca()
    for b in boundaries:
        ax.axhline(b, color='k', linewidth=0.5)
        ax.axvline(b, color='k', linewidth=0.5)
    ax.set_xticks(tick_centers); ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_centers); ax.set_yticklabels(tick_labels, fontsize=8, rotation=0)

    plt.title(title if title is not None else f'{method_name} Connectivity Matrix')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()


def calculate_fit(true_conn, learned_conn):

    # rescaling learned_conn following Yoon et al. 2025
    def objective(s):
        return np.sum(np.abs(true_conn - s * learned_conn))

    res = minimize_scalar(objective, bounds=(0, 1e9), method='bounded')
    s_opt = res.x
    learned_conn_rescaled = s_opt * learned_conn

    mask = ~np.eye(true_conn.shape[0], dtype=bool)

    # Continuous metrics
    v1 = true_conn[mask].flatten()
    v2 = learned_conn[mask].flatten()
    cos_sim  = np.dot(v1, v2) / (norm(v1) * norm(v2))
    pearson  = np.corrcoef(v1, v2)[0, 1]
    rho, _   = spearmanr(v1, v2)
    nie = np.linalg.norm((true_conn - learned_conn_rescaled)[mask]) / np.linalg.norm(true_conn[mask]) # normalized inference error

    # binary masks (off-diagonal)
    lb_m = (learned_conn[mask] != 0).astype(int)
    tb_m = (true_conn[mask]    != 0).astype(int)

    # Unsigned metrics via sklearn (standard binary classification)
    overall_precision = precision_score(tb_m, lb_m, zero_division=0)
    overall_recall    = recall_score(tb_m, lb_m, zero_division=0)
    overall_f1        = f1_score(tb_m, lb_m, zero_division=0)
    auroc = roc_auc_score(tb_m, np.abs(learned_conn[mask])) if 0 < tb_m.sum() < tb_m.size else float('nan')

    # three-class
    true_lbl = np.sign(true_conn[mask]).astype(int)
    pred_lbl = np.sign(learned_conn[mask]).astype(int)
    signed_precision = precision_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)
    signed_recall    = recall_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)
    signed_f1        = f1_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)

    return FitResult(
        cosine=cos_sim, pearson=pearson, spearman=rho,
        overall_precision=overall_precision, overall_recall=overall_recall, overall_f1=overall_f1,
        signed_precision=signed_precision, signed_recall=signed_recall, signed_f1=signed_f1,
        auroc=auroc, nie=nie,
    )


def align_matrices(adjacency_matrices, bias=None, binned_spikes=None, gdi_sign=None, num_taus=3, ridge=1e-3):
    if bias is not None:
        b = bias.lower()
        if b not in ("excitatory", "inhibitory"):
            raise ValueError(f"bias must be 'excitatory', 'inhibitory', or None; got {bias!r}")
        return _align_matrices_legacy(adjacency_matrices, excitatory=(b == "excitatory"))

    if gdi_sign is None: # pass a precomputed gdi_sign to skip robust_sign_inference
        gdi_sign = robust_sign_inference(binned_spikes, num_taus, ridge=ridge)
    aligned, _ = orient_runs(adjacency_matrices, gdi_sign)
    return aligned


def _align_matrices_legacy(adjacency_matrices, excitatory=True):
    """Legacy sign alignment: flip each run toward a reference matrix, orienting the
    whole cluster by assuming excitatory (or inhibitory) connections dominate."""
    excitation = 1 if np.sum(adjacency_matrices[0]) >= 0 else 0
    if (excitatory and not excitation) or (not excitatory and excitation):
        adjacency_matrices[0] = -adjacency_matrices[0]
    reference_matrix = adjacency_matrices[0]

    for i in range(1, len(adjacency_matrices)):
        current_matrix = adjacency_matrices[i]

        cos_sim_reg = calculate_fit(reference_matrix, current_matrix).cosine
        cos_sim_rev = calculate_fit(reference_matrix, -current_matrix).cosine
        if cos_sim_rev > cos_sim_reg:
            adjacency_matrices[i] = -current_matrix
    return adjacency_matrices

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
        print(f"sign inference: dropped {N - len(active)} silent neurons, " f"{len(active)} active (ridge={ridge}).")
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


def apply_gdi_sign(matrix, gdi_sign):
    return np.abs(matrix) * gdi_sign


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
        print("Warning: gdi_sign gave no usable orientation edges; using excitatory fallback.")
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
