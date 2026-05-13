# http://bactra.org/notebooks/directed-information-and-transfer-entropy.html
# https://www.statlect.com/asymptotic-theory/plug-in-principle
# https://github.com/cbueth/infomeasure/

import sys
from pathlib import Path
from baseline_models import *
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from circuit_gnn.data import *
from circuit_gnn.utils import *
data_map = {'yuto': load_yuto_network, 'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network}

data = 'yuto'

if data == 'yuto':
    downsample_factor = 20
elif data == 'arbitrary':
    downsample_factor = 10
elif data == 'pdcm':
    downsample_factor = 2

save_dir = f"datasets/{data}/init"
os.makedirs(save_dir, exist_ok=True)

# load data
binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
num_channels, num_time = binned_spikes.shape
print(f'{num_channels} channels, {num_time} time points')

# visualize true connectivity
cmap = sns.diverging_palette(240, 10, as_cmap=True)
plt.figure(figsize=(10, 8))
sns.heatmap(true_conn, xticklabels=neuron_names, yticklabels=neuron_names, cmap=cmap, center=0, cbar_kws={'label': 'Connectivity Strength'})
plt.title("True Connectivity Matrix")
plt.xlabel("Postsynaptic neuron")
plt.ylabel("Presynaptic neuron")
plt.tight_layout()
plt.show()

run_models = ['mutual_information', 'transfer_entropy', 'conditional_transfer_entropy'] # options: 'correlation', 'mutual_information', 'transfer_entropy', 'conditional_transfer_entropy', 'granger_causality'
gdi_sign, di_sign = sign_inference(binned_spikes.T, 3) # obtain sign

if 'correlation' in run_models:
    start_time = time.time()
    directed_correlation = calculate_correlation(binned_spikes)
    signed_directed_correlation = directed_correlation * gdi_sign
    np.save(os.path.join(f"{save_dir}/correlation.npy"), directed_correlation)
    np.save(os.path.join(f"{save_dir}/signed_correlation.npy"), signed_directed_correlation)

    end_time = time.time()
    print(f"Correlation calculation time: {(end_time - start_time)/60:.2f} minutes")
    plot_connectivity_matrix(directed_correlation, neuron_names, "Correlation", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(signed_directed_correlation, neuron_names, "Signed Correlation", save_dir=save_dir, plot_val=False)

if 'mutual_information' in run_models:
    start_time = time.time()
    mutual_information = calculate_mutual_information(binned_spikes, max_lag=100)
    signed_mutual_information = mutual_information * gdi_sign
    np.save(os.path.join(f"{save_dir}/mutual_information.npy"), mutual_information)
    np.save(os.path.join(f"{save_dir}/signed_mutual_information.npy"), signed_mutual_information)

    end_time = time.time()
    print(f"Mutual information calculation time: {(end_time - start_time)/60:.2f} minutes")
    plot_connectivity_matrix(mutual_information, neuron_names, "Mutual Information", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(signed_mutual_information, neuron_names, "Signed Mutual Information", save_dir=save_dir, plot_val=False)

if 'transfer_entropy' in run_models:
    start_time = time.time()
    transfer_entropy = calculate_transfer_entropy(binned_spikes, L=1, max_lag = 100)
    signed_transfer_entropy = transfer_entropy * gdi_sign
    np.save(os.path.join(f"{save_dir}/transfer_entropy.npy"), transfer_entropy)
    np.save(os.path.join(f"{save_dir}/signed_transfer_entropy.npy"), signed_transfer_entropy)

    end_time = time.time()
    print(f"Transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")
    plot_connectivity_matrix(transfer_entropy, neuron_names, "Transfer Entropy", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(signed_transfer_entropy, neuron_names, "Signed Transfer Entropy", save_dir=save_dir, plot_val=False)

if 'conditional_transfer_entropy' in run_models:
    start_time = time.time()
    pairwise_te = np.load(f"{save_dir}/transfer_entropy.npy")
    conditional_transfer_entropy = calculate_conditional_transfer_entropy(binned_spikes, pairwise_te, L=1)
    np.save(os.path.join(f"{save_dir}/conditional_transfer_entropy.npy"), conditional_transfer_entropy)

    end_time = time.time()
    print(f"conditional transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")
    plot_connectivity_matrix(conditional_transfer_entropy, neuron_names, "Conditional Transfer Entropy", save_dir=save_dir, plot_val=False)

if 'granger_causality' in run_models:
    start_time = time.time()
    granger_causality = calculate_granger_causality(binned_spikes, max_lag=10) # max_lag = 100 for small circuits, 10 for large circuits
    signed_granger_causality = granger_causality * gdi_sign
    np.save(os.path.join(f"{save_dir}/granger_causality.npy"), granger_causality)
    np.save(os.path.join(f"{save_dir}/signed_granger_causality.npy"), signed_granger_causality)

    end_time = time.time()
    print(f"Granger causality calculation time: {(end_time - start_time)/60:.2f} minutes")
    plot_connectivity_matrix(granger_causality, neuron_names, "Granger Causality", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(signed_granger_causality, neuron_names, "Signed Granger Causality", save_dir=save_dir, plot_val=False)