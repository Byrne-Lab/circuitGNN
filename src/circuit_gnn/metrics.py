"""Agreement metrics between an inferred connectivity matrix and ground truth."""
from collections import namedtuple

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Result of calculate_fit. Fields are accessed by name (e.g. res.overall_f1).
FitResult = namedtuple("FitResult", [
    "cosine", "pearson", "spearman",
    "overall_precision", "overall_recall", "overall_f1",
    "signed_precision", "signed_recall", "signed_f1",
    "auroc", "nie",
])


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
