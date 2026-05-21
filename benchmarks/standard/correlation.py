import numpy as np
import scipy
from benchmarks.standard.information_theory import calculate_coincidence_index 

# cross-correlation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
def calculate_correlation(binned_spikes, max_lag=30):
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
            corr = corr / (np.std(x) * np.std(y) * (len(x)-1))
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

    return ci_matrix, pk_matrix