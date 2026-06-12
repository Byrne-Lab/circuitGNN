import numpy as np
from scipy.stats import spearmanr, rankdata
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import copy
import numpy.matlib as matlib
from scipy.optimize import minimize_scalar
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from collections import namedtuple

# Result of calculate_fit. Fields are accessed by name (e.g. res.overall_f1).
FitResult = namedtuple("FitResult", [
    "cosine", "pearson", "spearman",
    "overall_precision", "overall_recall", "overall_f1",
    "signed_precision", "signed_recall", "signed_f1",
    "mcc_unsigned", "mcc_signed", "auroc", "nie",
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


def plot_connectivity_matrix(matrix, neuron_names, method_name='GNN', cmap=None, save_dir='benchmarking_results', plot_val=True):
    os.makedirs(save_dir, exist_ok=True)

    all_nonneg = matrix.min() >= 0
    if all_nonneg:
        plot_norm = plt.Normalize(vmin=0, vmax=max(matrix.max(), 1e-6))
        plot_cmap = copy.copy(plt.cm.Greys) if cmap is None else copy.copy(cmap)
    else:
        vmin = min(matrix.min(), -1e-6)
        vmax = max(matrix.max(),  1e-6)
        plot_norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plot_cmap = copy.copy(sns.diverging_palette(240, 10, as_cmap=True)) if cmap is None else copy.copy(cmap)

    # Mask exact zeros so they render as white
    plot_cmap.set_bad(color='white')
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)

    plt.figure(figsize=(10, 8))
    if plot_val:
        # Build annotation array: empty string where value is 0
        annot_labels = np.where(matrix == 0, '',
                                np.vectorize(lambda v: f"{v:.3f}")(matrix))
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': 'Connectivity Strength'},
                    square=True, annot=annot_labels, fmt='')   # fmt='' because values are already strings
    else:
        sns.heatmap(masked_matrix, xticklabels=neuron_names, yticklabels=neuron_names,
                    cmap=plot_cmap, norm=plot_norm, cbar_kws={'label': 'Connectivity Strength'},
                    square=True)

    plt.title(f"{method_name} Connectivity Matrix")
    plt.xlabel("Postsynaptic neuron")
    plt.ylabel("Presynaptic neuron")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()


PDCM_POPS_ORDERED = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']

def plot_pdcm_matrix(matrix, pop_labels, method_name='GNN', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(matrix.min(), -1e-6)
    vmax = max(matrix.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Connection strength'},
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

    plt.title(f'{method_name} Connectivity Matrix')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()


V1_POPS_ORDERED = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

def plot_v1_matrix(matrix, neuron_names, method_name='GNN', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmin = min(matrix.min(), -1e-6)
    vmax = max(matrix.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                xticklabels=False, yticklabels=False,
                cmap=cmap, norm=norm,
                cbar_kws={'label': 'Connection strength'},
                square=True)

    boundaries, tick_centers, tick_labels = [], [], []
    cursor = 0
    for pop in V1_POPS_ORDERED:
        count = sum(1 for name in neuron_names if name == pop)
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

    plt.title(f'{method_name} Connectivity Matrix')
    plt.xlabel('Postsynaptic')
    plt.ylabel('Presynaptic')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{method_name.replace(' ', '_').lower()}.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()


def calculate_fit(true_conn, learned_conn, plot=True):

    # rescaling learned_conn following Yoon et al. 2025
    def objective(s):
        return np.sum(np.abs(true_conn - s * learned_conn))

    res = minimize_scalar(objective, bounds=(0, 1e9), method='bounded')
    s_opt = res.x
    learned_conn_rescaled = s_opt * learned_conn

    mask = ~np.eye(true_conn.shape[0], dtype=bool)
    n = true_conn.shape[0]

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
    mcc_unsigned      = matthews_corrcoef(tb_m, lb_m)
    auroc = roc_auc_score(tb_m, np.abs(learned_conn[mask])) if 0 < tb_m.sum() < tb_m.size else float('nan')

    # raw counts retained for the classification-map plot
    tp = int(np.sum((lb_m == 1) & (tb_m == 1)))
    fp = int(np.sum((lb_m == 1) & (tb_m == 0)))
    fn = int(np.sum((lb_m == 0) & (tb_m == 1)))

    # signed TP/FP/FN
    tp_pos = (learned_conn[mask] > 0) & (true_conn[mask] > 0)
    tp_neg = (learned_conn[mask] < 0) & (true_conn[mask] < 0)
    tp_s   = tp_pos | tp_neg
    fp_s   = (lb_m == 1) & ((tb_m == 0) | (np.sign(learned_conn[mask]) != np.sign(true_conn[mask])))
    fn_s   = (lb_m == 0) & (tb_m == 1)

    tp_signed = np.sum(tp_s)
    fp_signed = np.sum(fp_s)
    fn_signed = np.sum(fn_s)

    signed_precision = tp_signed / (tp_signed + fp_signed) if (tp_signed + fp_signed) > 0 else 0
    signed_recall    = tp_signed / (tp_signed + fn_signed) if (tp_signed + fn_signed) > 0 else 0
    signed_f1        = 2 * (signed_precision * signed_recall) / (signed_precision + signed_recall) if (signed_precision + signed_recall) > 0 else 0

    # MCC
    tn_signed = mask.sum() - tp_signed - fp_signed - fn_signed
    denom = np.sqrt((tp_signed + fp_signed) * (tp_signed + fn_signed) *
                    (tn_signed + fp_signed) * (tn_signed + fn_signed))
    mcc_signed = (tp_signed * tn_signed - fp_signed * fn_signed) / denom if denom > 0 else 0.0

    # plot
    if plot:
        # classification maps
        clf_map  = np.full((n, n), np.nan)
        sign_map = np.full((n, n), np.nan)

        flat = np.zeros(lb_m.shape, dtype=int)
        flat[(lb_m == 1) & (tb_m == 1)] = 1   # TP
        flat[(lb_m == 1) & (tb_m == 0)] = 2   # FP
        flat[(lb_m == 0) & (tb_m == 1)] = 3   # FN
        clf_map[mask] = flat

        signed_flat = np.zeros(lb_m.shape, dtype=int)
        signed_flat[tp_s] = 1
        signed_flat[fp_s] = 2
        signed_flat[fn_s] = 3
        sign_map[mask] = signed_flat

        colors = ["#d0d0d0", "#2ecc71", "#e74c3c", "#e67e22"]  # TN, TP, FP, FN
        cmap   = ListedColormap(colors)
        cmap.set_bad(color="white")  # diagonal NaN → white

        vmax = max(np.abs(true_conn).max(), np.abs(learned_conn).max())

        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        fig.suptitle("calculate_fit — Visual Verification", fontsize=14, fontweight="bold")

        saturated_cmap = make_saturated_rdbu(cut=0.05)
        # true connectivity
        ax = axes[0, 0]
        im = ax.imshow(true_conn, cmap=saturated_cmap, aspect="auto", vmin=-0.5, vmax=0.5)
        overlay_zeros(ax, true_conn)
        ax.set_title("True Connectivity")
        plt.colorbar(im, ax=ax)

        # learned connectivity
        ax = axes[0, 1]
        im = ax.imshow(learned_conn, cmap=saturated_cmap, aspect="auto", vmin=-0.01, vmax=0.01)
        overlay_zeros(ax, learned_conn)
        ax.set_title("Learned Connectivity")
        plt.colorbar(im, ax=ax)

        # unsigned classification map
        ax = axes[1, 0]
        ax.imshow(clf_map, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_title(f"Unsigned Classification Map\n"
                     f"TP={tp}  FP={fp}  FN={fn}\n"
                     f"P={overall_precision:.3f}  R={overall_recall:.3f}  F1={overall_f1:.3f}")

        # signed classification map
        ax = axes[1, 1]
        ax.imshow(sign_map, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_title(f"Signed Classification Map\n"
                     f"TP_s={tp_signed}  FP_s={fp_signed}  FN_s={fn_signed}\n"
                     f"P={signed_precision:.3f}  R={signed_recall:.3f}  F1={signed_f1:.3f}")

        patches = [
            mpatches.Patch(color=colors[0], label="TN"),
            mpatches.Patch(color=colors[1], label="TP"),
            mpatches.Patch(color=colors[2], label="FP"),
            mpatches.Patch(color=colors[3], label="FN"),
        ]
        for ax in axes[1]:
            ax.legend(handles=patches, loc="upper right", fontsize=8)

        fig.text(0.5, 0.01,
                 f"Cosine={cos_sim:.4f}   Pearson r={pearson:.4f}   Spearman ρ={rho:.4f}",
                 ha="center", fontsize=11, style="italic",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

    return FitResult(
        cosine=cos_sim, pearson=pearson, spearman=rho,
        overall_precision=overall_precision, overall_recall=overall_recall, overall_f1=overall_f1,
        signed_precision=signed_precision, signed_recall=signed_recall, signed_f1=signed_f1,
        mcc_unsigned=mcc_unsigned, mcc_signed=mcc_signed, auroc=auroc, nie=nie,
    )


def _draw_confusion(ax, tp, fp, fn, tn, mcc, title):
    # 2x2 confusion matrix (rows = actual +/-, cols = predicted +/-) annotated with the MCC formula.
    M = np.array([[tp, fn], [fp, tn]], dtype=float)
    ax.imshow(M, cmap="Blues", aspect="auto")
    cells = [[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", f"TN\n{tn}"]]
    thresh = M.max() / 2 if M.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cells[i][j], ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > thresh else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred +", "pred −"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual +", "actual −"])
    ax.set_title(f"{title} = {mcc:.3f}\n(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
                 fontsize=8)


def plot_metric_verification(true_conn, learned_conn, save_dir, method_name="consistent mean"):
    """Save one multi-panel figure showing how each reported metric is computed.

    Mirrors the computations in calculate_fit so the table/CSV numbers can be
    visually verified: scatter plots for cosine/pearson/spearman, classification
    maps for the F1s, confusion matrices for the MCCs, an ROC curve for AUROC, and
    the rescaled residual map for NIE.
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
    tn = int(mask.sum() - tp - fp - fn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    du = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_u = (tp * tn - fp * fn) / du if du > 0 else 0.0

    # signed confusion (sign flip counts as FP only)
    tp_s = ((learned_conn[mask] > 0) & (true_conn[mask] > 0)) | \
           ((learned_conn[mask] < 0) & (true_conn[mask] < 0))
    fp_s = (lb_m == 1) & ((tb_m == 0) | (np.sign(learned_conn[mask]) != np.sign(true_conn[mask])))
    fn_s = (lb_m == 0) & (tb_m == 1)
    tps, fps, fns = int(np.sum(tp_s)), int(np.sum(fp_s)), int(np.sum(fn_s))
    tns = int(mask.sum() - tps - fps - fns)
    precs = tps / (tps + fps) if (tps + fps) else 0.0
    recs = tps / (tps + fns) if (tps + fns) else 0.0
    f1s = 2 * precs * recs / (precs + recs) if (precs + recs) else 0.0
    ds = np.sqrt((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns))
    mcc_s = (tps * tns - fps * fns) / ds if ds > 0 else 0.0

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

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
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

    # unsigned F1
    ax = axes[1, 0]
    ax.imshow(clf_map, cmap=clf_cmap, vmin=0, vmax=3, aspect="auto")
    ax.legend(handles=clf_patches, loc="upper right", fontsize=7)
    ax.set_title(f"Unsigned F1 = {f1:.3f}\nTP={tp} FP={fp} FN={fn}  P={prec:.2f} R={rec:.2f}")

    # signed F1
    ax = axes[1, 1]
    ax.imshow(sign_map, cmap=clf_cmap, vmin=0, vmax=3, aspect="auto")
    ax.legend(handles=clf_patches, loc="upper right", fontsize=7)
    ax.set_title(f"Signed F1 = {f1s:.3f}\nTP={tps} FP={fps} FN={fns}  P={precs:.2f} R={recs:.2f}")

    # AUROC
    ax = axes[1, 2]
    if fpr is not None:
        ax.plot(fpr, tpr, lw=1.5)
        ax.fill_between(fpr, tpr, alpha=0.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"AUROC = {auroc:.3f}\n(rank edges by |learned|)")

    # unsigned / signed MCC
    _draw_confusion(axes[2, 0], tp, fp, fn, tn, mcc_u, "Unsigned MCC")
    _draw_confusion(axes[2, 1], tps, fps, fns, tns, mcc_s, "Signed MCC")

    # NIE
    ax = axes[2, 2]
    resid_map = np.where(np.eye(n, dtype=bool), np.nan, residual_full)
    vmax = np.nanmax(np.abs(resid_map))
    vmax = vmax if vmax > 0 else 1e-9
    im = ax.imshow(resid_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("residual  t − s·l")
    ax.set_title(f"NIE = {nie:.3f}   (scale s = {s_opt:.3g})\n‖t − s·l‖ / ‖t‖   (off-diag residual)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(save_dir, "metric_verification.pdf")
    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    return save_path


def align_matrices(adjacency_matrices, excitatory=True):
    excitation = 1 if np.sum(adjacency_matrices[0]) >= 0 else 0
    if (excitatory and not excitation) or (not excitatory and excitation):
        adjacency_matrices[0] = -adjacency_matrices[0]
    reference_matrix = adjacency_matrices[0]

    for i in range(1, len(adjacency_matrices)):
        current_matrix = adjacency_matrices[i]

        cos_sim_reg = calculate_fit(reference_matrix, current_matrix, plot=False).cosine
        cos_sim_rev = calculate_fit(reference_matrix, -current_matrix, plot=False).cosine
        if cos_sim_rev > cos_sim_reg:
            adjacency_matrices[i] = -current_matrix
    return adjacency_matrices


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


# From Young et al. 2021
def partial_corr(X):

    # COMPUTE CORRELATION COEFFICIENT MATRIX
    corr_matrix = np.corrcoef(X.T)

    # COMPUTE PRECISION MATRIX
    precision_matrix = np.linalg.inv(corr_matrix)

    # NORMALIZE PRECISION MATRIX BY DIAGONAL VALUES TO GET PARTIAL CORRELATIONS
    precision_matrix_diagonal = np.diag(precision_matrix)
    precision_col = matlib.repmat(precision_matrix_diagonal,corr_matrix.shape[0],1)
    precision_row = precision_col.T
    normalization_matrix = np.sqrt(precision_row*precision_col)
    partial_corr_matrix = -precision_matrix/normalization_matrix

    # RETURN PARTIAL & REGULAR CORRELATION MATRICES
    return partial_corr_matrix, corr_matrix


def sign_inference(X,M):

    # MAKE SURE INTEGERS
    M = int(M)

    # PARAMETER(S)
    num_channels = X.shape[1]

    # LOOP THROUGH TAUS & CHANNELS
    X_pcorr_sign  = np.zeros((num_channels,num_channels))
    X_corr_sign   = np.zeros((num_channels,num_channels))
    current_pcorr = np.zeros((num_channels,num_channels,M))
    current_corr  = np.zeros((num_channels,num_channels,M))
    for chan2 in range(num_channels):
        for tt in range(M):
            X_copy = copy.deepcopy(X)
            X_copy[:(-(tt+1)),chan2] = X_copy[(tt+1):,chan2]
            X_copy = X_copy[:(-(tt+1)),:]
            current_pcorr[:,:,tt], current_corr[:,:,tt] = partial_corr(X_copy)
        # EXTRACT RELEVANT COLUMN
        current_pcorr_col = np.squeeze(current_pcorr[:,chan2,:])
        current_corr_col  = np.squeeze(current_corr[:,chan2,:])

        # TAKE MAX ABS VAL ACROSS TAUS
        max_pcorr_val = np.zeros((num_channels,))
        max_corr_val  = np.zeros((num_channels,))
        for chan1 in range(num_channels):
            max_pcorr_val[chan1] = current_pcorr_col[chan1,np.argmax(np.absolute(current_pcorr_col[chan1,:]))]
            max_corr_val[chan1]  = current_corr_col[chan1,np.argmax(np.absolute(current_corr_col[chan1,:]))]

        # TAKE SIGN AT LOWEST LAG THAT WAS FOUND TO PRODUCE MAX ABS VAL
        X_pcorr_sign[:,chan2] = np.sign(max_pcorr_val)
        X_corr_sign[:,chan2]  = np.sign(max_corr_val)

    # RETURN CORR & PCORR SIGNS
    return X_pcorr_sign, X_corr_sign