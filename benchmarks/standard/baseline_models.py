# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
import numpy.matlib as matlib

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

def H(prob):
    p = prob[prob>0]
    return -np.sum(p * np.log2(p))


# correlation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
def calculate_correlation(binned_spikes, max_lag = 1000):
    print('Calculating correlation...')
    num_channels, num_time = binned_spikes.shape
    directed_correlation = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            x = binned_spikes[i, :]
            y = binned_spikes[j, :]

            corr = scipy.signal.correlate(x - np.mean(x), y - np.mean(y), mode='full', method='fft')
            corr = corr / (np.std(x) * np.std(y) * len (x)) # approxiation of energy normalization

            zero_lag = len(corr) // 2
            Cij = corr[zero_lag : zero_lag + max_lag + 1]
            Cji = corr[zero_lag - max_lag : zero_lag + 1]
            Dij = Cij[np.argmax(np.abs(Cij))]
            Dji = Cji[np.argmax(np.abs(Cji))]

            directed_correlation[i, j] = Dij
            directed_correlation[j, i] = Dji

    # conditional matrix
    np.fill_diagonal(directed_correlation, 1)
    inverted_matrix = -np.linalg.inv(directed_correlation) # precision matrix
    diag_sqrt = np.sqrt(np.abs(np.diag(inverted_matrix)))
    directed_correlation = inverted_matrix / diag_sqrt[:, None] / diag_sqrt[None, :]  # normalize
    np.fill_diagonal(directed_correlation, 0)

    return directed_correlation


# mutual information
def calculate_mutual_information(binned_spikes, max_lag = 1000):
    print('Calculating mutual information...')
    num_channels, ntime = binned_spikes.shape
    mutual_information = np.zeros((num_channels, num_channels, max_lag+1))

    for tau in range(max_lag+1):
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue

                x = binned_spikes[i, :ntime-tau]
                y = binned_spikes[j, tau:]

                # empirical joint distribution
                p00 = np.mean((x==0) & (y==0))
                p01 = np.mean((x==0) & (y==1))
                p10 = np.mean((x==1) & (y==0))
                p11 = np.mean((x==1) & (y==1))

                Pxy = np.array([[p00, p01], [p10, p11]])
                Px = Pxy.sum(axis=1)
                Py = Pxy.sum(axis=0)

                Hx, Hy, Hxy = H(Px), H(Py), H(Pxy)
                mutual_information[i, j, tau] = max(0, Hx + Hy - Hxy)

    mutual_information_max = np.max(mutual_information, axis=2)
    np.fill_diagonal(mutual_information_max, 0)

    return mutual_information_max


# transfer entropy
# https://pablomflores.com/transfer-entropy/
def calculate_transfer_entropy(binned_spikes, L=1, max_lag = 1000):
    print('Calculating transfer entropy...')
    num_channels, ntime = binned_spikes.shape
    transfer_entropy = np.zeros((num_channels, num_channels, max_lag+1))
    for tau in range(1, max_lag+1):
        if tau % 2 == 0:
            print(f'Processing lag {tau}')
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue
                
                if ntime <= tau + L:
                    continue

                n_samples = ntime - tau - L # how many values can y_next take

                # create data structures
                x = np.array([binned_spikes[i, l:l+n_samples] for l in range(L)]).T # source
                y = np.array([binned_spikes[j, tau+l:tau+l+n_samples] for l in range(L)]).T # target
                y_next = binned_spikes[j, tau+L:tau+L+n_samples] # target future step

                # reshape(-1, 1) -> single column, multiple rows
                # 0 for y_next, 1 to L for y, L+1 to 2L+1 for x
                # marginalizing over A means we remove the dependency on A by summing across all its possible states
                counts, _ = np.histogramdd(np.column_stack([y_next.reshape(-1, 1), y, x]), bins=(2,)*(1+2*L), range=[(0,2)]*(1+2*L)) # 2L + 1 dimensions, (0,, 2) for binary -> 0 and 1
                P = counts / counts.sum() # number of samples in each bin / total number of samples, sigma(P) = 1
                P_yt_ytn = P.sum(axis=tuple(range(L+1, 2*L+1))) # sum over all x
                P_yt_xt = P.sum(axis=0) # sum over ytn
                P_yt = P.sum(axis=(0,) + tuple(range(L+1, 2*L+1))) # sum over yn and x

                H_yt_ytn = H(P_yt_ytn)
                H_yt_xt = H(P_yt_xt)
                H_yt = H(P_yt)
                H_yt_xt_ytn = H(P)
            
                te_xy = H_yt_ytn - H_yt - H_yt_xt_ytn + H_yt_xt
                transfer_entropy[i, j, tau] = max(0, te_xy)
                    
    transfer_entropy_max = np.max(transfer_entropy, axis=2)
    np.fill_diagonal(transfer_entropy_max, 0)

    return transfer_entropy_max


# conditional transfer entropy
# https://github.com/angelikipapana/PTE-variants
def calculate_conditional_transfer_entropy(binned_spikes, L=1, max_lag = 200):
    print(f'Calculating {L} step conditional transfer entropy...')
    num_channels, ntime = binned_spikes.shape
    conditional_transfer_entropy = np.zeros((num_channels, num_channels, max_lag+1))
    for tau in range(1, max_lag+1):
        if tau % 2 == 0:
            print(f'Processing lag {tau}')
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    continue
                
                if ntime <= tau + L:
                    continue

                conditioning_indices = None

                if conditioning_indices is None:
                    cond_idx = [k for k in range(num_channels)if k not in (i, j)]
                else:
                    cond_idx = conditioning_indices

                n_samples = ntime - tau - L # how many values can y_next take
            
                x = np.array([binned_spikes[i, l:l+n_samples] for l in range(L)]).T # source
                y = np.array([binned_spikes[j, tau+l:tau+l+n_samples] for l in range(L)]).T # target
                y_next = binned_spikes[j, tau+L:tau+L+n_samples] # target future step
                z_list = []
                for z_idx in cond_idx:
                    z_i = np.array([binned_spikes[z_idx, l:l+n_samples] for l in range(L)]).T # conditioned variable
                    z_list.append(z_i)
                z = np.hstack(z_list) # conditioned variables

                data = np.column_stack([y_next.reshape(-1, 1), y, x, z])
                n_dims = data.shape[1]
                counts, _ = np.histogramdd(data, bins=(2,)*n_dims, range=[(0,2)]*n_dims) 
                P = counts / counts.sum()

                # Marginals
                idx_y = tuple(range(1, 1+L))
                idx_x = tuple(range(1+L, 1+2*L))
                idx_z = tuple(range(1+2*L, n_dims))

                # Compute probabilities
                P_ytn_yt_z = P.sum(axis=idx_x) # p(y_next, y, z)
                P_yt_z = P.sum(axis=(0,) + idx_x) # p(y, z)
                P_ytn_yt_xt_z = P # p(y_next, y, x, z)
                P_yt_xt_z = P.sum(axis=0) # p(y, x, z)

                H_ytn_yt_zt = H(P_ytn_yt_z)
                H_yt_z = H(P_yt_z)
                H_ytn_yt_xt_z = H(P_ytn_yt_xt_z)
                H_yt_xt_z = H(P_yt_xt_z)

                pte_xy = H_ytn_yt_zt - H_yt_z - H_ytn_yt_xt_z + H_yt_xt_z
                conditional_transfer_entropy[i, j, tau] = max(0, pte_xy)

    conditional_transfer_entropy_max = np.max(conditional_transfer_entropy, axis=2)
    np.fill_diagonal(conditional_transfer_entropy_max, 0)

    return conditional_transfer_entropy_max