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
from functools import partial
from circuit_gnn.data import *
from circuit_gnn.utils import *

# hc3 is in-vivo (no ground truth); load_hc3_session needs session-specific
# kwargs but returns the same shape and takes downsample_factor first.
load_hc3 = partial(
    load_hc3_session,
    tar_path='datasets/hc3/ec016.1016.tar.gz',
    topdir='ec016.58',
    cell_csv='datasets/hc3/hc3-cell.csv',
    regions=['CA3', 'DG', 'EC2', 'EC3'],
    quality_filter=True,
    max_time_s=2000,
)

data_map = {
    'yuto': load_yuto_network,
    'arbitrary': load_arbitrary_network,
    'pdcm': load_pdcm_network,
    'hc3': load_hc3,
}

for data in ['hc3']:

    # 1 ms bins to accurately get spike times
    if data == 'yuto':
        downsample_factor = 2  # 1 ms
    elif data == 'arbitrary':
        downsample_factor = 1  # 1 ms
    elif data == 'pdcm':
        downsample_factor = 1  # 1 ms
    elif data == 'hc3':
        downsample_factor = 1  # 1 ms (native resolution)

    window_bins = 50        # 50 ms history
    subinterval_bins = 10   # 10 ms subintervals
    n_runs = 30             # random-seed repetitions

    save_dir = f"datasets/{data}/init"
    os.makedirs(save_dir, exist_ok=True)

    binned_spikes, true_conn, neuron_names = data_map[data](downsample_factor=downsample_factor)
    num_channels, num_time = binned_spikes.shape
    print(f'\n=== {data}: {num_channels} channels, {num_time} time points ===')

    # Individual MCC matrix (one model per source–target pair)
    start = time.time()
    mcc_matrix = calculate_mcc_matrix(binned_spikes, window_bins=window_bins, subinterval_bins=subinterval_bins, n_runs=n_runs)
    print(f"MCC matrix time: {(time.time() - start) / 60:.2f} min")

    mcc_threshold = threshold_top_k(mcc_matrix, true_conn)

    np.save(f"{save_dir}/decision_tree_mcc.npy", mcc_matrix)
    np.save(f"{save_dir}/decision_tree_mcc_threshold.npy", mcc_threshold)
    plot_connectivity_matrix(mcc_matrix, neuron_names, "Decision Tree MCC", save_dir=save_dir, plot_val=False,)
    plot_connectivity_matrix(mcc_threshold, neuron_names, "Decision Tree MCC threshold", save_dir=save_dir, plot_val=False,)

    # Variable importance matrix (full model per target)
    start = time.time()
    vi_matrix = calculate_variable_importance_matrix(binned_spikes, window_bins=window_bins, subinterval_bins=subinterval_bins, n_runs=n_runs)
    print(f"Variable importance matrix time: {(time.time() - start) / 60:.2f} min")

    vi_threshold = np.where(vi_matrix > 0, vi_matrix, 0)

    np.save(f"{save_dir}/decision_tree_vi.npy", vi_matrix)
    np.save(f"{save_dir}/decision_tree_vi_threshold.npy", vi_threshold)
    plot_connectivity_matrix(vi_matrix, neuron_names, "Decision Tree Variable Importance", save_dir=save_dir, plot_val=False,)
    plot_connectivity_matrix(vi_threshold, neuron_names, "Decision Tree Variable Importance threshold", save_dir=save_dir, plot_val=False,)