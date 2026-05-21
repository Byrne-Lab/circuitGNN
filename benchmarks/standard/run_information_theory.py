# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.information_theory import *
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
    L = 3

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'{num_channels} channels, {num_time} time points')

    # Mutual information
    start_time = time.time()
    ci_mi, pk_mi = calculate_mutual_information(binned_spikes, max_lag=max_lag)
    end_time = time.time()
    print(f"Mutual information calculation time: {(end_time - start_time)/60:.2f} minutes")

    np.save(f"{save_dir}/mutual_information_pk.npy", pk_mi)
    np.save(f"{save_dir}/mutual_information_ci.npy", ci_mi)
    plot_connectivity_matrix(pk_mi, neuron_names, "Mutual Information PK", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(ci_mi, neuron_names, "Mutual Information CI", save_dir=save_dir, plot_val=False)

    # Transfer entropy
    start_time = time.time()
    ci_te, pk_te = calculate_transfer_entropy(binned_spikes, L=L, max_lag=max_lag)
    end_time = time.time()
    print(f"Transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")

    np.save(f"{save_dir}/transfer_entropy_pk.npy", pk_te)
    np.save(f"{save_dir}/transfer_entropy_ci.npy", ci_te)
    plot_connectivity_matrix(pk_te, neuron_names, "Transfer Entropy PK", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(ci_te, neuron_names, "Transfer Entropy CI", save_dir=save_dir, plot_val=False)

    # Conditional transfer entropy
    start_time = time.time()
    pairwise_te = np.load(f"{save_dir}/transfer_entropy_pk.npy")
    conditional_transfer_entropy = calculate_conditional_transfer_entropy(binned_spikes, pairwise_te, L=L)
    end_time = time.time()
    print(f"conditional transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")

    np.save(os.path.join(f"{save_dir}/conditional_transfer_entropy.npy"), conditional_transfer_entropy)
    plot_connectivity_matrix(conditional_transfer_entropy, neuron_names, "Conditional Transfer Entropy", save_dir=save_dir, plot_val=False)