import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

# Granger causality
def calculate_granger_causality(binned_spikes, max_lag = 2):
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