# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import numpy as np
import infomeasure as im

# Coincidence Index from Ito et al. (2011)
def calculate_coincidence_index(te_curve, coincidence_window=5):    
    # CI = sum(M(t) for t in [tp - τ/2, tp + τ/2]) / sum(M(t) for t in [1, T])
    
    te_curve = np.asarray(te_curve, dtype=float)
    
    denominator = np.sum(te_curve) # entire lag window (1 to T)
    if denominator == 0:
        return 0.0

    tp = int(np.argmax(te_curve)) # peak time
    half = coincidence_window // 2

    # Clamp window to valid indices
    lower = max(0, tp - half)
    upper = min(len(te_curve), tp + half + 1)

    numerator = np.sum(te_curve[lower:upper])
    return float(numerator / denominator)


def get_top_neighbors(pairwise_te, j, exclude, k=5):
    te_into_j = pairwise_te[:, j].copy()
    te_into_j[exclude] = -np.inf
    te_into_j[j] = -np.inf
    top_k = np.argsort(te_into_j)[-k:]
    return top_k[te_into_j[top_k] > 0]


def H(prob):
    p = prob[prob>0]
    return -np.sum(p * np.log2(p))


# mutual information
def calculate_mutual_information(binned_spikes, max_lag=30, n_surrogates=0, alpha=0.05):
    print('Calculating mutual information...')
    num_channels, ntime = binned_spikes.shape
    mi_curves = np.zeros((num_channels, num_channels, max_lag))

    for tau in range(1, max_lag+1):
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue

                mi = im.mutual_information(
                    binned_spikes[i],
                    binned_spikes[j],
                    approach='miller_madow',
                    offset=tau,
                )

                mi_curves[i, j, tau-1] = max(0, mi)

    ci_matrix = np.zeros((num_channels, num_channels))
    pk_matrix = np.zeros((num_channels, num_channels))

    for src in range(num_channels):
        for tgt in range(num_channels):
            if src == tgt:
                continue

            curve = mi_curves[src, tgt, :]
            pk_matrix[src, tgt] = np.max(curve)
            ci_matrix[src, tgt] = calculate_coincidence_index(curve)

    return ci_matrix, pk_matrix


# transfer entropy
def calculate_transfer_entropy(binned_spikes, L=3, max_lag=30, n_surrogates=0, alpha=0.05):
    print('Calculating transfer entropy...')
    num_channels, ntime = binned_spikes.shape
    te_curves = np.zeros((num_channels, num_channels, max_lag))

    for tau in range(1, max_lag+1):
        if tau % 2 == 0:
            print(f'Processing lag {tau}')
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue

                if ntime <= tau + L:
                    continue

                te = im.transfer_entropy(
                    binned_spikes[i],
                    binned_spikes[j],
                    approach='miller_madow',
                    step_size=1,
                    prop_time=tau,
                    src_hist_len=L,
                    dest_hist_len=L,
                )

                te_curves[i, j, tau-1] = max(0, te)

    ci_matrix = np.zeros((num_channels, num_channels))
    pk_matrix = np.zeros((num_channels, num_channels))

    for src in range(num_channels):
        for tgt in range(num_channels):
            if src == tgt:
                continue

            curve = te_curves[src, tgt, :]
            pk_matrix[src, tgt] = np.max(curve)
            ci_matrix[src, tgt] = calculate_coincidence_index(curve)

    return ci_matrix, pk_matrix


# conditional transfer entropy
def calculate_conditional_transfer_entropy(binned_spikes, pairwise_te, L=3, n_surrogates=0, alpha=0.05):
    print('Calculating conditional transfer entropy...')
    num_channels, ntime = binned_spikes.shape
    cte_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue

            top_neighbors = get_top_neighbors(pairwise_te, j, exclude=[i, j], k=3)
            if len(top_neighbors) == 0:
                continue

            cond_vars = tuple(binned_spikes[k] for k in top_neighbors)

            cte = im.conditional_transfer_entropy(
                binned_spikes[i],
                binned_spikes[j],
                cond=cond_vars,
                approach='discrete',
                step_size=1,
                src_hist_len=L,
                dest_hist_len=L,
            )

            cte_matrix[i, j] = max(0, cte)

    np.fill_diagonal(cte_matrix, 0)

    return cte_matrix