# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.correlation import *
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from circuit_gnn.data import *
from circuit_gnn.utils import *
data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network}

for data in ['yuto', 'arbitrary', 'pdcm']:

    # 1 ms bins following Ito et al. 2011
    if data == 'yuto':
        downsample_factor = 2 # 1 ms
    elif data == 'arbitrary':
        downsample_factor = 1 # 1 ms
    elif data == 'pdcm':
        downsample_factor = 1 # 1 ms
    
    max_lag = 30

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'{num_channels} channels, {num_time} time points')

    # Correlation
    start_time = time.time()
    ci_correlation, pk_correlation = calculate_correlation(binned_spikes, max_lag=max_lag)
    end_time = time.time()
    print(f"Correlation calculation time: {(end_time - start_time)/60:.2f} minutes")

    np.save(f"{save_dir}/correlation_pk.npy", pk_correlation)
    np.save(f"{save_dir}/correlation_ci.npy", ci_correlation)
    plot_connectivity_matrix(pk_correlation, neuron_names, "Correlation PK", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(ci_correlation, neuron_names, "Correlation CI", save_dir=save_dir, plot_val=False)