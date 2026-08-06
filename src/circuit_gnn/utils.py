import numpy as np
from scipy.stats import spearmanr, rankdata
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import copy
import numpy.matlib as matlib
from scipy.optimize import minimize_scalar
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from collections import namedtuple

# Result of calculate_fit. Fields are accessed by name (e.g. res.overall_f1).
FitResult = namedtuple("FitResult", [
    "cosine", "pearson", "spearman",
    "overall_precision", "overall_recall", "overall_f1",
    "signed_precision", "signed_recall", "signed_f1",
    "auroc", "nie",
])


def overlay_zeros(ax, matrix):
    # Paint exact-zero cells bright white on top of an existing imshow
    white_overlay = np.ma.masked_where(matrix != 0, matrix)
    ax.imshow(white_overlay, cmap=ListedColormap(["white"]),
            aspect="auto", vmin=-1, vmax=1, zorder=2)


def make_saturated_rdbu(cut=0.15):
    # RdBu_r with the pale center [0.5-cut, 0.5+cut] removed.

    rdbu = plt.cm.RdBu_r
    colors = np.vstack([
        rdbu(np.linspace(0,        0.5 - cut, 128)),   # blue side
        rdbu(np.linspace(0.5 + cut, 1.0,      128)),   # red side
    ])
    return LinearSegmentedColormap.from_list("RdBu_saturated", colors)


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


def plot_region_matrix(matrix, group_labels, groups_ordered, method_name='GNN',
                       save_dir='results', title=None, cbar_label='Connection strength'):
    os.makedirs(save_dir, exist_ok=True)
    matrix = np.asarray(matrix, dtype=float)
    group_labels = np.asarray(group_labels)

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(np.nanmin(matrix), -1e-6)
    vmax = max(np.nanmax(matrix),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': cbar_label}, square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for group in groups_ordered:
        count = int(np.sum(group_labels == group))
        if count == 0:
            continue
        boundaries.append(cursor)
        tick_centers.append(cursor + count / 2)
        tick_labels.append(group)
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


def plot_metric_verification(true_conn, learned_conn, save_dir, method_name="consistent mean"):
    """Save one multi-panel figure showing how each reported metric is computed.

    Mirrors the computations in calculate_fit so the table/CSV numbers can be
    visually verified: scatter plots for cosine/pearson/spearman, classification
    maps for the F1s, an ROC curve for AUROC, and the rescaled residual map for NIE.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = true_conn.shape[0]
    mask = ~np.eye(n, dtype=bool)

    # continuous metrics
    v1 = true_conn[mask].flatten()
    v2 = learned_conn[mask].flatten()
    cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
    pearson = np.corrcoef(v1, v2)[0, 1]
    rho, _ = spearmanr(v1, v2)

    # rescaling (L1-optimal) used by NIE
    def objective(s):
        return np.sum(np.abs(true_conn - s * learned_conn))
    s_opt = minimize_scalar(objective, bounds=(0, 1e9), method="bounded").x
    residual_full = true_conn - s_opt * learned_conn
    nie = np.linalg.norm(residual_full[mask]) / np.linalg.norm(true_conn[mask])

    # unsigned confusion
    lb_m = (learned_conn[mask] != 0).astype(int)
    tb_m = (true_conn[mask] != 0).astype(int)
    tp = int(np.sum((lb_m == 1) & (tb_m == 1)))
    fp = int(np.sum((lb_m == 1) & (tb_m == 0)))
    fn = int(np.sum((lb_m == 0) & (tb_m == 1)))
    prec = precision_score(tb_m, lb_m, zero_division=0)
    rec = recall_score(tb_m, lb_m, zero_division=0)
    f1 = f1_score(tb_m, lb_m, zero_division=0)

    # signed: micro-averaged over the two edge classes (see calculate_fit)
    true_lbl = np.sign(true_conn[mask]).astype(int)
    pred_lbl = np.sign(learned_conn[mask]).astype(int)
    precs = precision_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)
    recs = recall_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)
    f1s = f1_score(true_lbl, pred_lbl, labels=[1, -1], average='micro', zero_division=0)

    # cell-partition counts, for the map colours
    tp_s = ((learned_conn[mask] > 0) & (true_conn[mask] > 0)) | \
           ((learned_conn[mask] < 0) & (true_conn[mask] < 0))
    fp_s = (lb_m == 1) & ((tb_m == 0) | (np.sign(learned_conn[mask]) != np.sign(true_conn[mask])))
    fn_s = (lb_m == 0) & (tb_m == 1)
    tps, fps, fns = int(np.sum(tp_s)), int(np.sum(fp_s)), int(np.sum(fn_s))

    # AUROC (rank edges by |learned|)
    if 0 < tb_m.sum() < tb_m.size:
        auroc = roc_auc_score(tb_m, np.abs(learned_conn[mask]))
        fpr, tpr, _ = roc_curve(tb_m, np.abs(learned_conn[mask]))
    else:
        auroc, fpr, tpr = float("nan"), None, None

    # classification maps (full n x n, NaN on diagonal)
    clf_map = np.full((n, n), np.nan)
    flat = np.zeros(lb_m.shape, dtype=int)
    flat[(lb_m == 1) & (tb_m == 1)] = 1
    flat[(lb_m == 1) & (tb_m == 0)] = 2
    flat[(lb_m == 0) & (tb_m == 1)] = 3
    clf_map[mask] = flat
    sign_map = np.full((n, n), np.nan)
    sflat = np.zeros(lb_m.shape, dtype=int)
    sflat[tp_s] = 1
    sflat[fp_s] = 2
    sflat[fn_s] = 3
    sign_map[mask] = sflat
    clf_colors = ["#d0d0d0", "#2ecc71", "#e74c3c", "#e67e22"]  # TN, TP, FP, FN
    clf_cmap = ListedColormap(clf_colors)
    clf_cmap.set_bad(color="white")
    clf_patches = [mpatches.Patch(color=clf_colors[k], label=lbl)
                   for k, lbl in enumerate(["TN", "TP", "FP", "FN"])]

    # row 0: continuous metrics, row 1: classification metrics
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    fig.suptitle(f"Metric verification ({method_name})", fontsize=15, fontweight="bold")

    # cosine
    ax = axes[0, 0]
    ax.scatter(v1, v2, s=10, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("true (off-diag)"); ax.set_ylabel("learned (off-diag)")
    ax.set_title(f"Cosine = {cos_sim:.3f}\ncos = (t·l) / (‖t‖ ‖l‖)")

    # pearson
    ax = axes[0, 1]
    ax.scatter(v1, v2, s=10, alpha=0.5)
    if np.std(v1) > 0:
        slope, intercept = np.polyfit(v1, v2, 1)
        xs = np.linspace(v1.min(), v1.max(), 50)
        ax.plot(xs, slope * xs + intercept, "r-", lw=1)
    ax.set_xlabel("true"); ax.set_ylabel("learned")
    ax.set_title(f"Pearson r = {pearson:.3f}\n(centered correlation + LS fit)")

    # spearman
    ax = axes[0, 2]
    ax.scatter(rankdata(v1), rankdata(v2), s=10, alpha=0.5)
    ax.set_xlabel("rank(true)"); ax.set_ylabel("rank(learned)")
    ax.set_title(f"Spearman ρ = {rho:.3f}\n(Pearson on ranks)")

    # NIE
    ax = axes[0, 3]
    resid_map = np.where(np.eye(n, dtype=bool), np.nan, residual_full)
    vmax = np.nanmax(np.abs(resid_map))
    vmax = vmax if vmax > 0 else 1e-9
    im = ax.imshow(resid_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("residual  t − s·l")
    ax.set_title(f"NIE = {nie:.3f}   (scale s = {s_opt:.3g})\n‖t − s·l‖ / ‖t‖   (off-diag residual)")

    # unsigned F1
    ax = axes[1, 0]
    ax.imshow(clf_map, cmap=clf_cmap, vmin=0, vmax=3, aspect="auto")
    ax.legend(handles=clf_patches, loc="upper right", fontsize=7)
    ax.set_title(f"Unsigned F1 = {f1:.3f}\nTP={tp} FP={fp} FN={fn}  P={prec:.2f} R={rec:.2f}")

    # signed F1
    ax = axes[1, 1]
    ax.imshow(sign_map, cmap=clf_cmap, vmin=0, vmax=3, aspect="auto")
    ax.legend(handles=clf_patches, loc="upper right", fontsize=7)
    n_flipped = int(np.sum((lb_m == 1) & (tb_m == 1) & (pred_lbl != true_lbl)))
    ax.set_title(f"Signed F1 = {f1s:.3f}\nTP={tps} FP={fps} FN={fns} (flipped={n_flipped})  "
                 f"P={precs:.2f} R={recs:.2f}")

    # AUROC
    ax = axes[1, 2]
    if fpr is not None:
        ax.plot(fpr, tpr, lw=1.5)
        ax.fill_between(fpr, tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"AUROC = {auroc:.3f}\n(rank edges by |learned|)")

    axes[1, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, "metric_verification.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    return save_path


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


def scale_ablation(A, ablation_matrix):
    ablation_matrix[ablation_matrix < 0] = 0 # negative values means loss went up
    normalized_matrix = A * ablation_matrix
    return normalized_matrix    


def get_consistent_connections(adjacency_matrices, threshold):
    n_matrices = adjacency_matrices.shape[0]
    
    positive_count = np.sum(adjacency_matrices > 0, axis=0)
    negative_count = np.sum(adjacency_matrices < 0, axis=0)
    binary_mask = ((positive_count / n_matrices) >= threshold) | ((negative_count / n_matrices) >= threshold)
    
    return binary_mask.astype(float)


def get_top_k(A, k_val=0.1):
    # if k is fraction
    flattened_A = A.flatten()
    if isinstance(k_val, float):
        k = int(len(flattened_A) * k_val)
    if isinstance(k_val, int):
        k = min(k_val, len(flattened_A))

    if k == 0:
        return np.zeros_like(A)
    threshold_value = np.partition(np.abs(flattened_A), -k)[-k]
    top_k_mask = np.abs(A) >= threshold_value
    top_k_A = A * top_k_mask

    return top_k_A