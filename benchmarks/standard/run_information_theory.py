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

data_map = {'OC_1': load_OC_1_network, 'OC_2': load_OC_2_network,
            'control': load_control_network,
            'arbitrary': load_arbitrary_network, 'pdcm': load_pdcm_network}

native_res_ms = {'OC_1': 0.5, 'OC_2': 0.5, 'control': 0.5, 'arbitrary': 1.0, 'pdcm': 1.0} # in ms
bin_sizes_ms = [50, 10, 5, 2, 1]
GDI_BIN_MS = 20

for data in ['OC_1', 'OC_2', 'control']: # next pdcm

    sign_downsample_factor = int(GDI_BIN_MS / native_res_ms[data])
    sign_spikes, _, _ = data_map[data](downsample_factor=sign_downsample_factor)
    gdi_sign = robust_sign_inference(sign_spikes)

    for bin_size_ms in bin_sizes_ms:

        downsample_factor = int(bin_size_ms / native_res_ms[data])
        suffix = f"_{bin_size_ms}ms"
        max_lag = 30
        L = 3

        save_dir = f"datasets/{data}/baseline"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n {data} | {bin_size_ms} ms bins (downsample_factor={downsample_factor})")

        # Load data
        binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
        num_channels, num_time = binned_spikes.shape
        print(f'{num_channels} channels, {num_time} time points')

        # Mutual information
        
        start_time = time.time()
        ci_mi, pk_mi = calculate_mutual_information(binned_spikes, max_lag=max_lag)
        end_time = time.time()
        print(f"Mutual information calculation time: {(end_time - start_time)/60:.2f} minutes")

        np.save(f"{save_dir}/mutual_information_pk{suffix}.npy", pk_mi)
        np.save(f"{save_dir}/mutual_information_ci{suffix}.npy", ci_mi)
        plot_connectivity_matrix(pk_mi, neuron_names, f"Mutual Information PK {suffix}", save_dir=save_dir, plot_val=False, signed=False)
        plot_connectivity_matrix(ci_mi, neuron_names, f"Mutual Information CI {suffix}", save_dir=save_dir, plot_val=False, signed=False)

        pk_mi_signed = apply_gdi_sign(pk_mi, gdi_sign)
        ci_mi_signed = apply_gdi_sign(ci_mi, gdi_sign)
        np.save(f"{save_dir}/mutual_information_pk{suffix}_signed.npy", pk_mi_signed)
        np.save(f"{save_dir}/mutual_information_ci{suffix}_signed.npy", ci_mi_signed)
        plot_connectivity_matrix(pk_mi_signed, neuron_names, f"Mutual Information PK {suffix} GDI-signed", save_dir=save_dir, plot_val=False)
        plot_connectivity_matrix(ci_mi_signed, neuron_names, f"Mutual Information CI {suffix} GDI-signed", save_dir=save_dir, plot_val=False)
        
        # Transfer entropy

        start_time = time.time()
        ci_te, pk_te = calculate_transfer_entropy(binned_spikes, L=L, max_lag=max_lag)
        end_time = time.time()
        print(f"Transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")

        np.save(f"{save_dir}/transfer_entropy_pk{suffix}.npy", pk_te)
        np.save(f"{save_dir}/transfer_entropy_ci{suffix}.npy", ci_te)
        plot_connectivity_matrix(pk_te, neuron_names, f"Transfer Entropy PK {suffix}", save_dir=save_dir, plot_val=False, signed=False)
        plot_connectivity_matrix(ci_te, neuron_names, f"Transfer Entropy CI {suffix}", save_dir=save_dir, plot_val=False, signed=False)

        pk_te_signed = apply_gdi_sign(pk_te, gdi_sign)
        ci_te_signed = apply_gdi_sign(ci_te, gdi_sign)
        np.save(f"{save_dir}/transfer_entropy_pk{suffix}_signed.npy", pk_te_signed)
        np.save(f"{save_dir}/transfer_entropy_ci{suffix}_signed.npy", ci_te_signed)
        plot_connectivity_matrix(pk_te_signed, neuron_names, f"Transfer Entropy PK {suffix} GDI-signed", save_dir=save_dir, plot_val=False)
        plot_connectivity_matrix(ci_te_signed, neuron_names, f"Transfer Entropy CI {suffix} GDI-signed", save_dir=save_dir, plot_val=False)
        
        # Conditional transfer entropy
        
        start_time = time.time()
        pairwise_te = np.load(f"{save_dir}/transfer_entropy_pk{suffix}.npy")
        conditional_transfer_entropy = calculate_conditional_transfer_entropy(binned_spikes, pairwise_te, L=L)
        end_time = time.time()
        print(f"conditional transfer entropy calculation time: {(end_time - start_time)/60:.2f} minutes")

        np.save(f"{save_dir}/conditional_transfer_entropy{suffix}.npy", conditional_transfer_entropy)
        plot_connectivity_matrix(conditional_transfer_entropy, neuron_names, f"Conditional Transfer Entropy {suffix}", save_dir=save_dir, plot_val=False, signed=False)

        cte_signed = apply_gdi_sign(conditional_transfer_entropy, gdi_sign)
        np.save(f"{save_dir}/conditional_transfer_entropy{suffix}_signed.npy", cte_signed)
        plot_connectivity_matrix(cte_signed, neuron_names, f"Conditional Transfer Entropy {suffix} GDI-signed", save_dir=save_dir, plot_val=False)