import numpy as np
import scipy
from benchmarks.standard.information_theory import calculate_coincidence_index 

# cross-correlation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
def calculate_correlation(binned_spikes, max_lag=30):
    print('Calculating correlation...')
    num_channels, num_time = binned_spikes.shape
    corr_curves = np.zeros((num_channels, num_channels, max_lag))

    for pre in range(num_channels):
        for post in range(num_channels):
            if pre == post:
                continue

            x_pre = binned_spikes[pre, :]
            x_post = binned_spikes[post, :]

            sd_pre, sd_post = np.std(x_pre), np.std(x_post)
            if sd_pre == 0 or sd_post == 0:  # silent unit: correlation undefined
                continue

            # correlate(a, b) puts the SECOND argument's lead at positive lag index,
            # so pass (post, pre) to read out pre -> post at lags 1..max_lag.
            corr = scipy.signal.correlate(x_post - np.mean(x_post), x_pre - np.mean(x_pre), mode='full', method='fft')
            corr = corr / (sd_pre * sd_post * (len(x_pre) - 1))
            zero_lag = len(corr) // 2
            corr_curves[pre, post, :] = corr[zero_lag + 1 : zero_lag + max_lag + 1]

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