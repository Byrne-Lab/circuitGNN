# Pozo-Jimenez et al. (2021): decision-tree connectivity benchmark.

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
from benchmarks.standard.decision_tree import (
    calculate_mcc_matrix,
    calculate_variable_importance_matrix,
)
from benchmarks.standard.utils import threshold_top_k
import numpy as np
import os
import time
from circuit_gnn.data import *
from circuit_gnn.utils import *

data_map = {
    'OC_1': load_OC_1_network,
    'OC_2': load_OC_2_network,
    'arbitrary': load_arbitrary_network,
    'pdcm': load_pdcm_network,
}

for data in ['pdcm']:

    # 1 ms bins to accurately get spike times
    if data in ('OC_1', 'OC_2'):
        downsample_factor = 2  # 1 ms
    elif data == 'arbitrary':
        downsample_factor = 1  # 1 ms
    elif data == 'pdcm':
        downsample_factor = 1  # 1 ms

    window_bins = 50        # 50 ms history
    subinterval_bins = 10   # 10 ms subintervals
    n_runs = 30             # random-seed repetitions

    save_dir = f"datasets/{data}/baseline"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    gdi_sign = robust_sign_inference(binned_spikes)

    # Individual MCC matrix (one model per source–target pair)
    start = time.time()
    mcc_matrix = calculate_mcc_matrix(binned_spikes, window_bins=window_bins, subinterval_bins=subinterval_bins, n_runs=n_runs)
    print(f"MCC matrix time: {(time.time() - start) / 60:.2f} min")

    mcc_threshold = threshold_top_k(mcc_matrix, true_conn)

    np.save(f"{save_dir}/decision_tree_mcc.npy", mcc_matrix)
    np.save(f"{save_dir}/decision_tree_mcc_threshold.npy", mcc_threshold)
    plot_connectivity_matrix(mcc_matrix, neuron_names, "Decision Tree MCC", save_dir=save_dir, plot_val=False, signed=False)
    plot_connectivity_matrix(mcc_threshold, neuron_names, "Decision Tree MCC threshold", save_dir=save_dir, plot_val=False, signed=False)

    mcc_signed = apply_gdi_sign(mcc_matrix, gdi_sign)
    mcc_threshold_signed = apply_gdi_sign(mcc_threshold, gdi_sign)
    np.save(f"{save_dir}/decision_tree_mcc_signed.npy", mcc_signed)
    np.save(f"{save_dir}/decision_tree_mcc_threshold_signed.npy", mcc_threshold_signed)
    plot_connectivity_matrix(mcc_signed, neuron_names, "Decision Tree MCC GDI-signed", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(mcc_threshold_signed, neuron_names, "Decision Tree MCC threshold GDI-signed", save_dir=save_dir, plot_val=False)

    # Variable importance matrix (full model per target)
    start = time.time()
    vi_matrix = calculate_variable_importance_matrix(binned_spikes, window_bins=window_bins, subinterval_bins=subinterval_bins, n_runs=n_runs)
    print(f"Variable importance matrix time: {(time.time() - start) / 60:.2f} min")

    vi_threshold = np.where(vi_matrix > 0, vi_matrix, 0)

    np.save(f"{save_dir}/decision_tree_vi.npy", vi_matrix)
    np.save(f"{save_dir}/decision_tree_vi_threshold.npy", vi_threshold)
    plot_connectivity_matrix(vi_matrix, neuron_names, "Decision Tree Variable Importance", save_dir=save_dir, plot_val=False, signed=False)
    plot_connectivity_matrix(vi_threshold, neuron_names, "Decision Tree Variable Importance threshold", save_dir=save_dir, plot_val=False, signed=False)

    vi_signed = apply_gdi_sign(vi_matrix, gdi_sign)
    vi_threshold_signed = apply_gdi_sign(vi_threshold, gdi_sign)
    np.save(f"{save_dir}/decision_tree_vi_signed.npy", vi_signed)
    np.save(f"{save_dir}/decision_tree_vi_threshold_signed.npy", vi_threshold_signed)
    plot_connectivity_matrix(vi_signed, neuron_names, "Decision Tree Variable Importance GDI-signed", save_dir=save_dir, plot_val=False)
    plot_connectivity_matrix(vi_threshold_signed, neuron_names, "Decision Tree Variable Importance threshold GDI-signed", save_dir=save_dir, plot_val=False)