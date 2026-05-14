# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
import numpy.matlib as matlib
import infomeasure as im
from statsmodels.tsa.vector_ar.var_model import VAR
from joblib import Parallel, delayed

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


def get_top_neighbors(pairwise_te, j, exclude, k=3):
    te_into_j = pairwise_te[:, j].copy()
    te_into_j[exclude] = -np.inf
    te_into_j[j] = -np.inf
    top_k = np.argsort(te_into_j)[-k:]
    return top_k[te_into_j[top_k] > 0]


def H(prob):
    p = prob[prob>0]
    return -np.sum(p * np.log2(p))


# correlation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
def calculate_correlation(binned_spikes, max_lag = 100):
    print('Calculating correlation...')
    num_channels, num_time = binned_spikes.shape
    corr_curves = np.zeros((num_channels, num_channels, max_lag))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue
            x = binned_spikes[i, :]
            y = binned_spikes[j, :]
            corr = scipy.signal.correlate(x - np.mean(x), y - np.mean(y), mode='full', method='fft')
            corr = corr / (np.std(x) * np.std(y) * len(x))
            zero_lag = len(corr) // 2
            corr_curves[i, j, :] = corr[zero_lag + 1 : zero_lag + max_lag + 1]

    ci_matrix = np.zeros((num_channels, num_channels))
    pk_matrix = np.zeros((num_channels, num_channels))

    for src in range(num_channels):
        for tgt in range(num_channels):
            if src == tgt:
                continue
            curve = corr_curves[src, tgt, :]
            pk_matrix[src, tgt] = curve[np.argmax(np.abs(curve))]  # signed peak
            ci_matrix[src, tgt] = calculate_coincidence_index(np.abs(curve))

    # partial correlation on pk: condition each pair on all other channels
    np.fill_diagonal(pk_matrix, 1)
    inverted_matrix = -np.linalg.inv(pk_matrix)
    diag_sqrt = np.sqrt(np.abs(np.diag(inverted_matrix)))
    pk_matrix = inverted_matrix / diag_sqrt[:, None] / diag_sqrt[None, :]
    np.fill_diagonal(pk_matrix, 0)

    return ci_matrix, pk_matrix


# mutual information
def calculate_mutual_information(binned_spikes, max_lag = 100):
    print('Calculating mutual information...')
    num_channels, ntime = binned_spikes.shape
    # mutual_information = np.zeros((num_channels, num_channels, max_lag+1))
    mi_curves = np.zeros((num_channels, num_channels, max_lag))

    for tau in range(1, max_lag+1):
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue

                mi = im.mutual_information(
                    binned_spikes[i], # variable 1
                    binned_spikes[j], # variable 2
                    approach='miller_madow', # bias correction
                    offset=tau, # lag
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
def calculate_transfer_entropy(binned_spikes, L=5, max_lag = 100):
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
                    binned_spikes[i], # source
                    binned_spikes[j], # target
                    approach='miller_madow', # bias correction
                    step_size=1, 
                    prop_time=tau, # lag
                    src_hist_len=L, # number of past observations
                    dest_hist_len=L, # number of past observations
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
def calculate_conditional_transfer_entropy(binned_spikes, pairwise_te, L=5):
    print('Calculating conditional transfer entropy...')
    num_channels, ntime = binned_spikes.shape
    conditional_transfer_entropy = np.zeros((num_channels, num_channels))
    
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

            conditional_transfer_entropy[i, j] = max(0, cte)

    np.fill_diagonal(conditional_transfer_entropy, 0)
    return conditional_transfer_entropy


# Granger causality
def calculate_granger_causality(binned_spikes, max_lag = 100):
    print('Calculating Granger causality...')
    num_channels, ntime = binned_spikes.shape

    model = VAR(binned_spikes.T)
    result = model.fit(maxlags=max_lag, ic='aic')
    granger_causality = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue

            test = result.test_causality(caused=j, causing=i, kind='f')
            granger_causality[i, j] = test.test_statistic

    return granger_causality